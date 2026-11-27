#!/usr/bin/env bash
# Bounded checkpoint/restore benchmark for the segmented conversation archive.
#
# The benchmark starts one --parallel 1 server with a small archive, sends a
# long prompt (default target 32768 tokens; set BENCH_TOKENS=100000 for the
# large case), then measures explicit
# `POST /slots/{id}?action=checkpoint` requests while the conversation grows by
# one short continuation between checkpoints. It records:
#
#   * request wall time and the server's `timings.save_ms`;
#   * n_saved and n_written for each checkpoint;
#   * manifest.json, segments.json, STRSTATE and .seg file sizes/counts;
#   * the archive's used_bytes/budget_bytes and entry count;
#   * a restore continuation's wall time and cached_tokens.
#
# It prints one row per checkpoint and one row per artifact. It does not run
# automatically in CI: it needs a CUDA GPU and a model.
#
# Usage: tools/bench_slot_archive.sh MODEL BIN SERVER_DIR [PORT]
#
#   MODEL       GGUF model to serve.
#   BIN         rustingstrata binary (a path, or a name on PATH).
#   SERVER_DIR  directory the server runs in.
#   PORT        benchmark server port (default 8080).
#
# Environment:
#   BENCH_TOKENS       target prompt tokens (default 32768; 100000 for the
#                      large bounded run)
#   BENCH_CTX          --ctx (default 131072; must exceed BENCH_TOKENS)
#   BENCH_REPEATS      checkpoint/continuation pairs (default 3)
#   BENCH_CHARS_TOKEN  approximate prompt characters per token (default 4)
#   LOAD_TIMEOUT       seconds to wait for a model load (default 600)
#   REQUEST_TIMEOUT    seconds per HTTP request (default 1200)
#
# The default 32k case is bounded to one prompt, BENCH_REPEATS short
# continuations and one restore; it is not a soak. Use
# tools/soak_slot_archive.sh for repeated switching.
#
# Requires python3 (stdlib only) and a CUDA GPU.

set -uo pipefail

usage() {
  cat <<'EOF'
Usage: tools/bench_slot_archive.sh MODEL BIN SERVER_DIR [PORT]

  MODEL       GGUF model to serve
  BIN         rustingstrata binary (path or name on PATH)
  SERVER_DIR  directory the server runs in
  PORT        benchmark server port (default 8080)

Environment: BENCH_TOKENS (default 32768), BENCH_CTX (default 131072),
             BENCH_REPEATS (default 3), BENCH_CHARS_TOKEN (default 4),
             LOAD_TIMEOUT (default 600), REQUEST_TIMEOUT (default 1200)
EOF
}

fail() {
  echo "bench_slot_archive: $*" >&2
  exit 1
}

if (( $# < 3 || $# > 4 )); then
  usage >&2
  exit 2
fi

MODEL_ARG=$1
BIN_ARG=$2
SERVER_DIR_ARG=$3
PORT=${4:-8080}
TOKENS=${BENCH_TOKENS:-32768}
CTX=${BENCH_CTX:-131072}
REPEATS=${BENCH_REPEATS:-3}
CHARS_TOKEN=${BENCH_CHARS_TOKEN:-4}
LOAD_TIMEOUT=${LOAD_TIMEOUT:-600}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-1200}

[[ "$PORT" =~ ^[0-9]+$ ]] || fail "PORT must be a number: $PORT"
(( PORT >= 1 && PORT <= 65534 )) || fail "PORT must be between 1 and 65534"
[[ "$TOKENS" =~ ^[0-9]+$ ]] && (( TOKENS >= 1024 )) || fail "BENCH_TOKENS must be >= 1024"
[[ "$CTX" =~ ^[0-9]+$ ]] && (( CTX > TOKENS )) || fail "BENCH_CTX must be > BENCH_TOKENS"
[[ "$REPEATS" =~ ^[0-9]+$ ]] && (( REPEATS >= 1 && REPEATS <= 16 )) \
  || fail "BENCH_REPEATS must be between 1 and 16"
[[ "$CHARS_TOKEN" =~ ^[0-9]+$ ]] && (( CHARS_TOKEN >= 1 )) \
  || fail "BENCH_CHARS_TOKEN must be >= 1"

abspath() {
  python3 -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$1"
}

MODEL=$(abspath "$MODEL_ARG")
SERVER_DIR=$(abspath "$SERVER_DIR_ARG")
if [[ "$BIN_ARG" == */* ]]; then
  BIN=$(abspath "$BIN_ARG")
else
  BIN=$(command -v -- "$BIN_ARG" || true)
fi

[[ -f "$MODEL" ]] || fail "MODEL is not a file: $MODEL_ARG"
[[ -n "$BIN" && -x "$BIN" ]] || fail "BIN is not executable: $BIN_ARG"
[[ -d "$SERVER_DIR" ]] || fail "SERVER_DIR is not a directory: $SERVER_DIR_ARG"

WORK=$(mktemp -d "${TMPDIR:-/tmp}/strata-slot-bench.XXXXXX") \
  || fail "could not create a temporary directory"
ARCHIVE="$WORK/archive"
LOG="$WORK/server.log"
PID1=""

alive() {
  local state
  state=$(ps -o stat= -p "$1" 2>/dev/null) || return 1
  [[ "$state" != *Z* ]]
}

stop_server() {
  local pid=$1 i
  [[ -n "$pid" ]] || return 0
  kill "$pid" 2>/dev/null || true
  for ((i = 0; i < 100; i++)); do
    alive "$pid" || break
    sleep 0.1
  done
  kill -9 "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

cleanup() {
  stop_server "${PID1:-}"
  rm -rf -- "$WORK"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

run_py() {
  python3 - "$@" <<'PY'
import json
import os
import sys
import time
import urllib.error
import urllib.request

CONTINUE = "Continue with one short sentence."


def http(base, method, path, payload=None, timeout=1200.0):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        base.rstrip("/") + path, data=data, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            status = response.status
    except urllib.error.HTTPError as err:
        raw = err.read()
        status = err.code
    except Exception as err:
        return None, str(err)
    text = raw.decode("utf-8", "replace")
    try:
        return status, json.loads(text)
    except ValueError:
        return status, text


def brief(value, limit=240):
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    return text if len(text) <= limit else text[:limit] + "..."


def as_obj(body):
    return body if isinstance(body, dict) else {}


def as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def reply_of(body):
    try:
        message = body["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return ""
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str) and content:
        return content
    reasoning = message.get("reasoning_content")
    return reasoning if isinstance(reasoning, str) else ""


def usage_of(body):
    return as_obj(as_obj(body).get("usage"))


def cached_of(body):
    try:
        return as_int(body["usage"]["prompt_tokens_details"]["cached_tokens"])
    except (KeyError, TypeError):
        return 0


class Report:
    def __init__(self):
        self.total = 0
        self.failed = 0

    def check(self, name, ok, detail=""):
        self.total += 1
        if ok:
            print("PASS " + name, flush=True)
            return
        self.failed += 1
        text = "FAIL " + name
        if detail:
            text += " [" + detail + "]"
        print(text, flush=True)

    def summary(self):
        print()
        if self.failed:
            print("%d of %d checks failed" % (self.failed, self.total))
        else:
            print("all %d checks passed" % self.total)
        return 1 if self.failed else 0


PARAGRAPH = (
    "A long conversation is a sequence of turns whose attention state depends "
    "on every earlier token. The archive writer copies the recurrent snapshot "
    "once and then walks the history in fixed token segments, writing only the "
    "segments that do not already exist under the same model fingerprint and "
    "parent chain. A restore validates the state header, the contiguous range "
    "cover and every segment payload before it installs a row, so a truncated "
    "or swapped file falls back to a normal prefill instead of corrupting the "
    "model. The checkpoint path is synchronous in this wave: it holds the "
    "archive mutex and the model thread while the segment and state files are "
    "written. The benchmark measures that cost for a bounded prompt and records "
    "the file decomposition, not just one wall-clock number. "
)


def build_prompt(target_tokens, chars_per_token):
    target_chars = max(4096, target_tokens * chars_per_token)
    parts = ["Archive benchmark prompt. "]
    total = len(parts[0])
    while total < target_chars:
        parts.append(PARAGRAPH)
        total += len(PARAGRAPH)
    parts.append("\n\nAnswer with one short sentence.")
    return "".join(parts)


def chat(base, messages, timeout, max_tokens=16):
    return http(
        base,
        "POST",
        "/v1/chat/completions",
        {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout,
    )


def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, ValueError):
        return None


def archive_stats(archive_dir):
    manifest_path = os.path.join(archive_dir, "manifest.json")
    segments_path = os.path.join(archive_dir, "segments.json")
    manifest = load_json(manifest_path)
    segments = load_json(segments_path)
    entries = manifest.get("entries", []) if isinstance(manifest, dict) else []
    rows = segments.get("segments", []) if isinstance(segments, dict) else []
    entries = [entry for entry in entries if isinstance(entry, dict)]
    rows = [row for row in rows if isinstance(row, dict)]

    state_bytes = 0
    state_count = 0
    for entry in entries:
        name = entry.get("state_file")
        if not isinstance(name, str) or not name:
            continue
        try:
            state_bytes += os.path.getsize(os.path.join(archive_dir, name))
            state_count += 1
        except OSError:
            pass

    seg_bytes = 0
    seg_count = 0
    seg_sizes = []
    for row in rows:
        row_id = row.get("id")
        if not isinstance(row_id, str):
            continue
        path = os.path.join(archive_dir, "segments", row_id[:2], row_id + ".seg")
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        seg_bytes += size
        seg_count += 1
        seg_sizes.append(size)

    def size(path):
        try:
            return os.path.getsize(path)
        except OSError:
            return 0

    return {
        "manifest_bytes": size(manifest_path),
        "segments_json_bytes": size(segments_path),
        "entries": len(entries),
        "segment_rows": len(rows),
        "segment_files": seg_count,
        "segment_bytes": seg_bytes,
        "segment_max": max(seg_sizes) if seg_sizes else 0,
        "segment_avg": (seg_bytes // seg_count) if seg_count else 0,
        "state_files": state_count,
        "state_bytes": state_bytes,
    }


def run(base, archive_dir, target_tokens, repeats, chars_per_token, timeout):
    report = Report()
    prompt = build_prompt(target_tokens, chars_per_token)
    print(
        "bench: target_tokens=%d prompt_chars=%d repeats=%d archive=%s"
        % (target_tokens, len(prompt), repeats, archive_dir),
        flush=True,
    )

    t0 = time.monotonic()
    status, body = chat(base, [{"role": "user", "content": prompt}], timeout)
    initial_wall = (time.monotonic() - t0) * 1e3
    obj = as_obj(body)
    reply = reply_of(obj)
    report.check(
        "initial prompt answered",
        status == 200 and bool(reply),
        "status=%s wall_ms=%.1f reply=%s" % (status, initial_wall, brief(reply)),
    )
    prompt_tokens = as_int(usage_of(obj).get("prompt_tokens"))
    report.check(
        "initial prompt is close to the target token count",
        prompt_tokens >= max(1024, target_tokens // 2),
        "prompt_tokens=%d target=%d" % (prompt_tokens, target_tokens),
    )
    print(
        "bench: initial prompt_tokens=%d wall_ms=%.1f"
        % (prompt_tokens, initial_wall),
        flush=True,
    )

    status, body = http(base, "GET", "/slots", None, timeout)
    slots = as_obj(body).get("slots", [])
    used = [
        as_int(slot.get("id"))
        for slot in slots
        if isinstance(slot, dict) and as_int(slot.get("n_past")) > 0
    ]
    slot = used[0] if used else 0
    report.check(
        "GET /slots identifies the active slot",
        status == 200 and bool(used),
        "status=%s slots=%s" % (status, brief(slots)),
    )

    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": reply or "..."},
    ]
    rows = []
    for index in range(repeats):
        t0 = time.monotonic()
        status, body = http(
            base, "POST", "/slots/%d?action=checkpoint" % slot, {}, timeout
        )
        wall = (time.monotonic() - t0) * 1e3
        obj = as_obj(body)
        timings = obj.get("timings")
        save_ms = timings.get("save_ms") if isinstance(timings, dict) else None
        row = {
            "phase": "checkpoint %d" % (index + 1),
            "wall_ms": wall,
            "save_ms": save_ms,
            "n_saved": as_int(obj.get("n_saved")),
            "n_written": as_int(obj.get("n_written")),
            "id": obj.get("id"),
            "cached": 0,
        }
        rows.append(row)
        report.check(
            "checkpoint %d returns 200" % (index + 1),
            status == 200,
            "status=%s body=%s" % (status, brief(obj)),
        )
        report.check(
            "checkpoint %d reports numeric save_ms" % (index + 1),
            is_number(save_ms) and save_ms >= 0,
            "timings=%s" % brief(timings),
        )
        report.check(
            "checkpoint %d saved a non-empty history" % (index + 1),
            as_int(obj.get("n_saved")) > 0,
            "body=%s" % brief(obj),
        )

        if index + 1 >= repeats:
            break

        messages.append(
            {
                "role": "user",
                "content": "%s (turn %d)" % (CONTINUE, index + 1),
            }
        )
        status, body = chat(base, messages, timeout)
        obj = as_obj(body)
        reply = reply_of(obj)
        report.check(
            "growth turn %d answered" % (index + 1),
            status == 200 and bool(reply),
            "status=%s body=%s" % (status, brief(obj)),
        )
        messages.append({"role": "assistant", "content": reply or "..."})

    stats = archive_stats(archive_dir)
    status, body = http(base, "GET", "/slots/archive", None, timeout)
    archive = as_obj(body)
    report.check(
        "GET /slots/archive returns 200",
        status == 200,
        "status=%s body=%s" % (status, brief(archive)),
    )
    report.check(
        "archive reports entries and a positive budget",
        is_number(archive.get("used_bytes"))
        and as_int(archive.get("entries")) >= 1
        and is_number(archive.get("budget_bytes"))
        and archive.get("budget_bytes") > 0,
        "body=%s" % brief(archive),
    )
    report.check(
        "segments.json rows match the files on disk",
        stats["segment_rows"] == stats["segment_files"] and stats["segment_rows"] > 0,
        "rows=%d files=%d" % (stats["segment_rows"], stats["segment_files"]),
    )

    # Switch to a different conversation in the single slot, then continue the
    # benchmarked one. No slot holds the original history, so a positive
    # cached_tokens can only come from the archive.
    switch_prompt = (
        "Restore probe. Lighthouse keepers wrote wind, visibility and the "
        "height of the swell in a daily log. Answer with one short sentence."
    )
    status, body = chat(base, [{"role": "user", "content": switch_prompt}], timeout)
    report.check(
        "restore: switch prompt answered",
        status == 200 and bool(reply_of(as_obj(body))),
        "status=%s body=%s" % (status, brief(body)),
    )
    restore_messages = list(messages)
    restore_messages.append(
        {"role": "user", "content": "Continue with one short sentence."}
    )
    t0 = time.monotonic()
    status, body = chat(base, restore_messages, timeout)
    restore_wall = (time.monotonic() - t0) * 1e3
    obj = as_obj(body)
    cached = cached_of(obj)
    report.check(
        "restore: continuation returns 200",
        status == 200,
        "status=%s wall_ms=%.1f body=%s" % (status, restore_wall, brief(obj)),
    )
    report.check(
        "restore: continuation reuses the archived prefix",
        status == 200 and cached > 0,
        "cached_tokens=%d wall_ms=%.1f body=%s"
        % (cached, restore_wall, brief(obj)),
    )

    print()
    print("%-16s %10s %10s %9s %11s %10s %8s" % (
        "phase", "wall_ms", "save_ms", "n_saved", "n_written", "cached", "id"
    ))
    print("-" * 80)
    for row in rows:
        save_ms = row["save_ms"]
        print("%-16s %10.1f %10s %9d %11d %10d %8s" % (
            row["phase"],
            row["wall_ms"],
            ("%.1f" % save_ms) if is_number(save_ms) else "?",
            row["n_saved"],
            row["n_written"],
            row["cached"],
            str(row["id"])[:8],
        ))
    print("%-16s %10.1f %10s %9s %11s %10d %8s" % (
        "restore", restore_wall, "-", "-", "-", cached, "-"
    ))
    print()
    print("%-24s %14s" % ("artifact", "bytes"))
    print("-" * 40)
    for name in (
        "manifest_bytes",
        "segments_json_bytes",
        "state_bytes",
        "segment_bytes",
    ):
        print("%-24s %14d" % (name, stats[name]))
    print("%-24s %14d" % ("archive_used_bytes", as_int(archive.get("used_bytes"))))
    print("%-24s %14d" % ("archive_budget_bytes", as_int(archive.get("budget_bytes"))))
    print()
    print("%-24s %14d" % ("entries", stats["entries"]))
    print("%-24s %14d" % ("state_files", stats["state_files"]))
    print("%-24s %14d" % ("segment_rows", stats["segment_rows"]))
    print("%-24s %14d" % ("segment_files", stats["segment_files"]))
    print("%-24s %14d" % ("segment_max_bytes", stats["segment_max"]))
    print("%-24s %14d" % ("segment_avg_bytes", stats["segment_avg"]))
    return report.summary()


def tail(path, limit=65536):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as file:
            file.seek(0, os.SEEK_END)
            size = file.tell()
            file.seek(max(0, size - limit))
            return file.read().rstrip()
    except OSError as err:
        return "could not read %s: %s" % (path, err)


def log_loaded(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as file:
            file.seek(0, os.SEEK_END)
            size = file.tell()
            file.seek(max(0, size - 65536))
            return "loaded " in file.read()
    except OSError:
        return False


def alive(pid):
    try:
        with open("/proc/%d/status" % pid, "r", encoding="utf-8") as file:
            for line in file:
                if line.startswith("State:"):
                    return not line.split()[1].startswith("Z")
    except (OSError, IndexError):
        pass
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def wait_ready(base, log_path, pid, timeout):
    deadline = time.monotonic() + timeout
    loaded = False
    while time.monotonic() < deadline:
        if not alive(pid):
            print("server exited while loading; last log lines:", file=sys.stderr)
            print(tail(log_path), file=sys.stderr)
            return 1
        if not loaded and log_loaded(log_path):
            loaded = True
        if loaded:
            status, _ = http(base, "GET", "/health", None, 2.0)
            if status == 200:
                return 0
        time.sleep(0.5)
    print("timed out waiting for %s; last log lines:" % base, file=sys.stderr)
    print(tail(log_path), file=sys.stderr)
    return 1


def seconds(text, default):
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


def main(argv):
    if len(argv) < 3:
        print("usage: check.py MODE BASE_URL ...", file=sys.stderr)
        return 2
    mode = argv[1]
    base = argv[2]
    if mode == "wait":
        if len(argv) < 6:
            print("usage: check.py wait BASE_URL LOG PID TIMEOUT", file=sys.stderr)
            return 2
        return wait_ready(base, argv[3], int(argv[4]), seconds(argv[5], 600.0))
    if mode == "bench":
        if len(argv) < 8:
            print(
                "usage: check.py bench BASE_URL ARCHIVE TOKENS REPEATS CHARS TIMEOUT",
                file=sys.stderr,
            )
            return 2
        return run(
            base,
            argv[3],
            int(argv[4]),
            int(argv[5]),
            int(argv[6]),
            seconds(argv[7], 1200.0),
        )
    print("unknown mode: %s" % mode, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
PY
}

FAILED=0

echo "bench_slot_archive: model=$MODEL"
echo "bench_slot_archive: binary=$BIN"
echo "bench_slot_archive: server dir=$SERVER_DIR"
echo "bench_slot_archive: port=$PORT"
echo "bench_slot_archive: temporary archive dir=$ARCHIVE"
echo

( cd "$SERVER_DIR" && exec "$BIN" serve \
    --model "$MODEL" --gpu --backend cuda --parallel 1 --ctx "$CTX" \
    --kv q8 --slot-archive-dir "$ARCHIVE" --slot-archive-gb 16 \
    --slot-archive-min-tokens 64 --slot-archive-min-delta 1 \
    --slot-archive-versions 2 \
    --addr "127.0.0.1:$PORT" ) >"$LOG" 2>&1 &
PID1=$!

if ! run_py wait "http://127.0.0.1:$PORT" "$LOG" "$PID1" "$LOAD_TIMEOUT"; then
  fail "benchmark server did not become ready"
fi

if ! run_py bench "http://127.0.0.1:$PORT" "$ARCHIVE" "$TOKENS" "$REPEATS" "$CHARS_TOKEN" "$REQUEST_TIMEOUT"; then
  FAILED=1
fi

stop_server "$PID1"
PID1=""

exit "$FAILED"
