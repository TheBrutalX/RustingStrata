#!/usr/bin/env bash
# Soak harness for the segmented conversation archive.
#
# It starts one --parallel 2 server with a small archive and repeatedly switches
# among a handful of conversations. Each switch forces a checkpoint and the
# next turn for that conversation normally restores the archived prefix. The
# harness checks:
#
#   * every chat returns 200 with a reply and no 5xx is observed;
#   * at least one continuation reports cached_tokens > 0 (the archive restore
#     actually ran, not just the slot cache);
#   * periodic forced GC returns 200 and `used_bytes <= budget_bytes`;
#   * the final manifest/segments.json is valid, every segment row has its file,
#     every v2 entry has a contiguous range, and refcounts match the manifest;
#   * no writer temp files are left, the process is alive, and the log has no
#     panic;
#   * host RSS (from /proc/$PID/status) and GPU memory (nvidia-smi, optional)
#     do not grow past a generous bound over the run. A missing nvidia-smi is
#     tolerated and reported as a NOTE, not a failure.
#
# Usage: tools/soak_slot_archive.sh MODEL BIN SERVER_DIR [PORT]
#
#   MODEL       GGUF model to serve.
#   BIN         rustingstrata binary (a path, or a name on PATH).
#   SERVER_DIR  directory the server runs in.
#   PORT        soak server port (default 8080).
#
# Environment:
#   SOAK_ITERATIONS    chat/switch iterations (default 40; 30-50 recommended)
#   SOAK_BUDGET_GB     archive budget (default 4)
#   SOAK_PROMPT_CHARS  approximate prompt size (default 6000)
#   SOAK_RSS_GROWTH_MB allowed host RSS growth (default 1024)
#   SOAK_GPU_GROWTH_MB allowed GPU memory growth (default 2048)
#   LOAD_TIMEOUT       seconds to wait for a model load (default 600)
#   REQUEST_TIMEOUT    seconds per HTTP request (default 600)
#
# Requires python3 (stdlib only), a CUDA GPU, and enough VRAM for
# --parallel 2 with a 32k q8 KV cache.

set -uo pipefail

usage() {
  cat <<'EOF'
Usage: tools/soak_slot_archive.sh MODEL BIN SERVER_DIR [PORT]

  MODEL       GGUF model to serve
  BIN         rustingstrata binary (path or name on PATH)
  SERVER_DIR  directory the server runs in
  PORT        soak server port (default 8080)

Environment: SOAK_ITERATIONS (default 40), SOAK_BUDGET_GB (default 4),
             SOAK_PROMPT_CHARS (default 6000), SOAK_RSS_GROWTH_MB (default 1024),
             SOAK_GPU_GROWTH_MB (default 2048), LOAD_TIMEOUT (default 600),
             REQUEST_TIMEOUT (default 600)
EOF
}

fail() {
  echo "soak_slot_archive: $*" >&2
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
ITERATIONS=${SOAK_ITERATIONS:-40}
BUDGET_GB=${SOAK_BUDGET_GB:-4}
PROMPT_CHARS=${SOAK_PROMPT_CHARS:-6000}
RSS_GROWTH_MB=${SOAK_RSS_GROWTH_MB:-1024}
GPU_GROWTH_MB=${SOAK_GPU_GROWTH_MB:-2048}
LOAD_TIMEOUT=${LOAD_TIMEOUT:-600}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-600}

[[ "$PORT" =~ ^[0-9]+$ ]] || fail "PORT must be a number: $PORT"
(( PORT >= 1 && PORT <= 65534 )) || fail "PORT must be between 1 and 65534"
[[ "$ITERATIONS" =~ ^[0-9]+$ ]] && (( ITERATIONS >= 2 )) \
  || fail "SOAK_ITERATIONS must be >= 2"
[[ "$BUDGET_GB" =~ ^[0-9]+$ ]] && (( BUDGET_GB >= 2 )) \
  || fail "SOAK_BUDGET_GB must be >= 2"
[[ "$PROMPT_CHARS" =~ ^[0-9]+$ ]] && (( PROMPT_CHARS >= 2000 )) \
  || fail "SOAK_PROMPT_CHARS must be >= 2000"
[[ "$RSS_GROWTH_MB" =~ ^[0-9]+$ ]] || fail "SOAK_RSS_GROWTH_MB must be a number"
[[ "$GPU_GROWTH_MB" =~ ^[0-9]+$ ]] || fail "SOAK_GPU_GROWTH_MB must be a number"

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

WORK=$(mktemp -d "${TMPDIR:-/tmp}/strata-slot-soak.XXXXXX") \
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
import subprocess
import sys
import time
import urllib.error
import urllib.request

CONTINUE = "Continue with one short sentence."


def http(base, method, path, payload=None, timeout=600.0):
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


def brief(value, limit=200):
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    return text if len(text) <= limit else text[:limit] + "..."


def as_obj(body):
    return body if isinstance(body, dict) else {}


def as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_id(value):
    return (
        isinstance(value, str)
        and len(value) == 16
        and all(char in "0123456789abcdef" for char in value)
    )


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
    "A conversation switch is the common case for an inference server: the "
    "next request belongs to a different history, so the scheduler checkpoints "
    "the resident slot and later restores the requested prefix from the "
    "segment store. Each checkpoint writes one recurrent state file and the "
    "KV segments that do not already exist under the same parent chain; a "
    "restore validates every range and payload before installing a row. A "
    "long soak repeats that cycle until an allocator leak, a stale refcount or "
    "a stranded temp file would show up. The harness samples host and GPU "
    "memory, forces GC, and re-reads the manifests rather than trusting only "
    "the HTTP counters. "
)


def long_prompt(label, chars):
    parts = [label + ". "]
    total = len(parts[0])
    while total < chars:
        parts.append(PARAGRAPH)
        total += len(PARAGRAPH)
    parts.append("\n\nAnswer with one short sentence.")
    return "".join(parts)


def chat(base, messages, timeout):
    return http(
        base,
        "POST",
        "/v1/chat/completions",
        {
            "messages": messages,
            "max_tokens": 32,
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


def entries_of(body):
    entries = as_obj(body).get("entries")
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def temp_files(root):
    found = []
    for current, _, files in os.walk(root):
        for name in files:
            if ".tmp" in name:
                found.append(os.path.join(current, name))
    return found


def parse_archive(report, archive_dir, min_entries=0):
    manifest = load_json(os.path.join(archive_dir, "manifest.json"))
    segments = load_json(os.path.join(archive_dir, "segments.json"))
    ok = isinstance(manifest, dict) and isinstance(segments, dict)
    report.check(
        "soak: manifest.json and segments.json are readable",
        ok,
        "dir=%s" % archive_dir,
    )
    if not ok:
        return None
    rows = segments.get("segments")
    entries = manifest.get("entries")
    if not isinstance(rows, list) or not isinstance(entries, list):
        report.check("soak: archive JSON shapes are valid", False)
        return None
    entries = [entry for entry in entries if isinstance(entry, dict)]
    rows = [row for row in rows if isinstance(row, dict)]
    report.check(
        "soak: manifest lists at least %d entries" % min_entries,
        len(entries) >= min_entries,
        "entries=%d" % len(entries),
    )
    report.check(
        "soak: every surviving entry is v2 (format 2)",
        bool(entries) and all(as_int(entry.get("format")) == 2 for entry in entries),
        "formats=%r" % ([entry.get("format") for entry in entries],),
    )
    bad_meta = [
        entry.get("id")
        for entry in entries
        if as_int(entry.get("n_tokens")) <= 0 or as_int(entry.get("bytes")) <= 0
    ]
    report.check(
        "soak: entries report positive n_tokens and bytes",
        bool(entries) and not bad_meta,
        "bad=%s" % brief(bad_meta),
    )

    by_id = {}
    for row in rows:
        row_id = row.get("id")
        if is_id(row_id):
            by_id[row_id] = row
    bad_rows = []
    for row in rows:
        row_id = row.get("id")
        if (
            not is_int(row.get("bytes"))
            or row.get("bytes") <= 0
            or not is_int(row.get("n_tokens"))
            or row.get("n_tokens") <= 0
            or not is_int(row.get("refs"))
            or row.get("refs") < 0
            or as_int(row.get("format")) != 1
        ):
            bad_rows.append(row_id)
    report.check(
        "soak: segment rows have positive sizes and refs",
        not bad_rows,
        "bad=%s" % brief(bad_rows[:8]),
    )

    bad_files = []
    for row_id, row in by_id.items():
        path = os.path.join(archive_dir, "segments", row_id[:2], row_id + ".seg")
        try:
            size = os.path.getsize(path)
        except OSError:
            bad_files.append(row_id)
            continue
        if size != as_int(row.get("bytes")):
            bad_files.append(row_id)
    report.check(
        "soak: every segment row has its file on disk",
        not bad_files,
        "bad=%s" % brief(bad_files[:8]),
    )

    ref_counts = {}
    bad_refs = []
    for entry in entries:
        position = 0
        segment_list = entry.get("segments")
        if not isinstance(segment_list, list) or not segment_list:
            bad_refs.append("entry %r has no segments" % (entry.get("id"),))
            continue
        for segment in segment_list:
            if not isinstance(segment, dict):
                continue
            row_id = segment.get("id")
            start = segment.get("start")
            end = segment.get("end")
            if isinstance(row_id, str):
                ref_counts[row_id] = ref_counts.get(row_id, 0) + 1
            if not is_id(row_id) or row_id not in by_id:
                bad_refs.append(
                    "entry %r references unknown segment %r"
                    % (entry.get("id"), row_id)
                )
            if (
                not is_int(start)
                or not is_int(end)
                or start != position
                or end <= start
            ):
                bad_refs.append(
                    "entry %r has range %r..%r at %d"
                    % (entry.get("id"), start, end, position)
                )
                if is_int(end) and end > position:
                    position = end
                continue
            position = end
        if position != as_int(entry.get("n_tokens")):
            bad_refs.append(
                "entry %r covers %d of %r tokens"
                % (entry.get("id"), position, entry.get("n_tokens"))
            )
    report.check(
        "soak: entry ranges are contiguous and resolve",
        not bad_refs,
        "problems=%s" % brief(bad_refs[:6]),
    )

    mismatched = []
    for row_id, count in ref_counts.items():
        row = by_id.get(row_id)
        if row is None or as_int(row.get("refs")) != count:
            mismatched.append(
                (row_id, None if row is None else row.get("refs"), count)
            )
    for row_id, row in by_id.items():
        if row_id not in ref_counts and as_int(row.get("refs")) != 0:
            mismatched.append((row_id, row.get("refs"), 0))
    report.check(
        "soak: refcounts match the manifest",
        not mismatched,
        "mismatch=%s" % brief(mismatched[:6]),
    )
    return {"entries": entries, "by_id": by_id}


def read_rss_mb(pid):
    try:
        with open("/proc/%d/status" % pid, "r", encoding="utf-8") as file:
            for line in file:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except (OSError, IndexError, ValueError):
        pass
    return None


def read_gpu_mb():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    values = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(int(line.split()[0]))
        except (IndexError, ValueError):
            return None
    return max(values) if values else None


def log_has_panic(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as file:
            file.seek(0, os.SEEK_END)
            size = file.tell()
            file.seek(max(0, size - 262144))
            return "panicked at" in file.read()
    except OSError:
        return False


def process_alive(pid):
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


def run(base, pid, archive_dir, iterations, prompt_chars, rss_limit, gpu_limit, timeout):
    report = Report()
    prompts = [
        long_prompt("Soak conversation %d" % index, prompt_chars)
        for index in range(6)
    ]
    histories = {}
    errors = []
    restored = 0
    gc_checks = 0
    gc_failures = []
    status_codes = []
    rss_samples = []
    gpu_samples = []
    gpu_missing = False

    def sample(label):
        rss = read_rss_mb(pid)
        gpu = read_gpu_mb()
        if rss is not None:
            rss_samples.append((label, rss))
        if gpu is None:
            return
        gpu_samples.append((label, gpu))

    for index in range(iterations):
        prompt_index = index % len(prompts)
        prompt = prompts[prompt_index]
        if prompt_index in histories:
            messages = list(histories[prompt_index])
            messages.append({"role": "user", "content": CONTINUE})
            kind = "continuation"
        else:
            messages = [{"role": "user", "content": prompt}]
            kind = "new"
        status, body = chat(base, messages, timeout)
        obj = as_obj(body)
        reply = reply_of(obj)
        status_codes.append(status)
        cached = cached_of(obj)
        if cached > 0:
            restored += 1
        if status != 200 or not reply:
            errors.append(
                "iteration %d %s: status=%r body=%s"
                % (index, kind, status, brief(obj))
            )
        else:
            history = list(messages)
            history.append({"role": "assistant", "content": reply})
            histories[prompt_index] = history

        if index % 5 == 0:
            status, body = http(base, "POST", "/slots/archive/gc", {}, timeout)
            obj = as_obj(body)
            status_codes.append(status)
            gc_checks += 1
            if status != 200:
                gc_failures.append("iteration %d: status=%r body=%s" % (index, status, brief(obj)))
            else:
                status, body = http(base, "GET", "/slots/archive", None, timeout)
                obj = as_obj(body)
                status_codes.append(status)
                used = obj.get("used_bytes")
                budget = obj.get("budget_bytes")
                if (
                    not is_number(used)
                    or not is_number(budget)
                    or used > budget
                ):
                    gc_failures.append(
                        "iteration %d: used=%r budget=%r body=%s"
                        % (index, used, budget, brief(obj))
                    )

        if index % 5 == 0:
            sample("iteration %d" % index)
        if (index + 1) % 10 == 0:
            print(
                "NOTE soak: %d/%d iterations, restored=%d, rss_mb=%r, gpu_mb=%r"
                % (index + 1, iterations, restored, read_rss_mb(pid), read_gpu_mb()),
                flush=True,
            )

    sample("final")
    report.check(
        "soak: every chat answered with a reply",
        not errors,
        "errors=%s" % brief(errors[:4]),
    )
    report.check(
        "soak: no HTTP 5xx was returned",
        all(code is None or code < 500 for code in status_codes),
        "codes=%s" % brief(status_codes[-12:]),
    )
    report.check(
        "soak: at least one continuation restored an archived prefix",
        restored >= 1,
        "restored=%d of %d iterations" % (restored, iterations),
    )
    report.check(
        "soak: forced GC ran and stayed within the budget",
        gc_checks >= 1 and not gc_failures,
        "checks=%d failures=%s" % (gc_checks, brief(gc_failures[:4])),
    )

    status, body = http(base, "GET", "/slots/archive", None, timeout)
    archive = as_obj(body)
    report.check(
        "soak: final GET /slots/archive returns 200",
        status == 200,
        "status=%s body=%s" % (status, brief(archive)),
    )
    used = archive.get("used_bytes")
    budget = archive.get("budget_bytes")
    report.check(
        "soak: final used_bytes <= budget_bytes",
        is_number(used) and is_number(budget) and used <= budget,
        "used=%r budget=%r" % (used, budget),
    )
    analysis = parse_archive(report, archive_dir, 1)
    if analysis is not None:
        report.check(
            "soak: the archive kept at least one restorable entry",
            len(analysis["entries"]) >= 1,
            "entries=%d" % len(analysis["entries"]),
        )

    temps = temp_files(archive_dir)
    report.check(
        "soak: no writer temp files are left behind",
        not temps,
        "temps=%s" % brief(temps[:8]),
    )
    report.check(
        "soak: the server process is alive",
        process_alive(pid),
        "pid=%d" % pid,
    )
    report.check(
        "soak: the log has no panic",
        not log_has_panic(os.path.join(os.path.dirname(archive_dir), "server.log")),
        "archive_dir=%s" % archive_dir,
    )

    if rss_samples:
        baseline_label, baseline = rss_samples[0]
        peak_label, peak = max(rss_samples, key=lambda item: item[1])
        growth = peak - baseline
        report.check(
            "soak: host RSS growth stays within %d MiB" % rss_limit,
            growth <= rss_limit,
            "baseline=%dMiB (%s) peak=%dMiB (%s) growth=%dMiB"
            % (baseline, baseline_label, peak, peak_label, growth),
        )
        print(
            "NOTE soak: RSS samples %s" % brief(rss_samples, 500),
            flush=True,
        )
    else:
        report.check(
            "soak: host RSS was readable from /proc",
            False,
            "could not read VmRSS for pid %d" % pid,
        )

    if gpu_samples:
        baseline = gpu_samples[0][1]
        peak = max(value for _, value in gpu_samples)
        growth = peak - baseline
        report.check(
            "soak: GPU memory growth stays within %d MiB" % gpu_limit,
            growth <= gpu_limit,
            "baseline=%dMiB peak=%dMiB growth=%dMiB" % (baseline, peak, growth),
        )
        print("NOTE soak: GPU samples %s" % brief(gpu_samples, 500), flush=True)
    else:
        gpu_missing = True
        print(
            "NOTE soak: nvidia-smi is unavailable; GPU memory growth was not sampled",
            flush=True,
        )
        report.check(
            "soak: GPU memory check is skipped cleanly without nvidia-smi",
            gpu_missing,
            "install nvidia-smi or set SOAK_GPU_GROWTH_MB to require it",
        )

    print(
        "NOTE soak: iterations=%d restored=%d gc_checks=%d final_entries=%d"
        % (
            iterations,
            restored,
            gc_checks,
            len(entries_of(archive)),
        ),
        flush=True,
    )
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
    if mode == "soak":
        if len(argv) < 10:
            print(
                "usage: check.py soak BASE_URL PID ARCHIVE ITERATIONS PROMPT_CHARS RSS_MB GPU_MB TIMEOUT",
                file=sys.stderr,
            )
            return 2
        return run(
            base,
            int(argv[3]),
            argv[4],
            int(argv[5]),
            int(argv[6]),
            int(argv[7]),
            int(argv[8]),
            seconds(argv[9], 600.0),
        )
    print("unknown mode: %s" % mode, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
PY
}

FAILED=0

echo "soak_slot_archive: model=$MODEL"
echo "soak_slot_archive: binary=$BIN"
echo "soak_slot_archive: server dir=$SERVER_DIR"
echo "soak_slot_archive: port=$PORT"
echo "soak_slot_archive: iterations=$ITERATIONS budget=${BUDGET_GB}GiB"
echo "soak_slot_archive: temporary archive dir=$ARCHIVE"
echo

( cd "$SERVER_DIR" && exec "$BIN" serve \
    --model "$MODEL" --gpu --backend cuda --parallel 2 --ctx 32768 \
    --kv q8 --slot-archive-dir "$ARCHIVE" --slot-archive-gb "$BUDGET_GB" \
    --slot-archive-min-tokens 64 --slot-archive-min-delta 1 \
    --slot-archive-versions 2 \
    --addr "127.0.0.1:$PORT" ) >"$LOG" 2>&1 &
PID1=$!

if ! run_py wait "http://127.0.0.1:$PORT" "$LOG" "$PID1" "$LOAD_TIMEOUT"; then
  fail "soak server did not become ready"
fi

if ! run_py soak "http://127.0.0.1:$PORT" "$PID1" "$ARCHIVE" "$ITERATIONS" \
    "$PROMPT_CHARS" "$RSS_GROWTH_MB" "$GPU_GROWTH_MB" "$REQUEST_TIMEOUT"; then
  FAILED=1
fi

stop_server "$PID1"
PID1=""

exit "$FAILED"
