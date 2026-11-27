#!/usr/bin/env bash
# End-to-end test for the phase-3 shutdown checkpoint and the periodic archive
# GC (docs/SLOT-ARCHIVE-PHASE2.md §8, §9 and §15).
#
# Phase 1 (graceful shutdown):
#   Start a save-enabled server with --slot-archive-on-shutdown, chat a long
#   prompt A, check that the archive is still empty, then stop the server with
#   SIGTERM (not SIGKILL) and wait for it to exit. The shutdown checkpoint must
#   log a save and write a format-2 entry. Restart the same server on the same
#   archive directory, check that GET /slots/archive lists an entry with
#   n_tokens > 0 and format 2, and send the continuation of A: it must answer
#   200 with cached_tokens > 0, so the prefix came from the shutdown entry.
#
# Phase 2 (periodic GC, no manual call):
#   Start a fresh server with --slot-archive-gc-interval 1s and
#   --slot-archive-ttl 1s. Checkpoint conversation A explicitly and pin it,
#   then checkpoint a diverging conversation B and leave it unpinned. Wait
#   about 3.5 s without calling POST /slots/archive/gc: the timer must drop the
#   unpinned TTL entry and keep the pinned one, and the server log must show a
#   periodic GC line with a TTL expiry.
#
# Usage: tools/test_slot_archive_shutdown.sh MODEL BIN SERVER_DIR [PORT]
#
#   MODEL       GGUF model to serve.
#   BIN         rustingstrata binary (a path, or a name on PATH).
#   SERVER_DIR  directory the server runs in.
#   PORT        port of the servers (default 8080), reused in turn.
#
# Environment:
#   LOAD_TIMEOUT     seconds to wait for a model load (default 600).
#   REQUEST_TIMEOUT  seconds per HTTP test request (default 600).
#   STOP_TIMEOUT     seconds to wait for SIGTERM to end a server (default 180).
#                    The binary gives a graceful stop 3 s plus the configured
#                    shutdown deadline (30 s here) before its watchdog exits.
#
# Assumptions:
#   - SIGTERM is delivered after the chat response, so the slot is idle and its
#     pos equals its token count; the shutdown checkpoint then saves it.
#   - The TTL is 1 s with a 1 s GC interval. The pinned entry is created and
#     pinned in two back-to-back local requests, well inside one TTL, before
#     the next timer run; the unpinned entry is left alone for > 1 s.
#   - Every stop uses stop_server (SIGTERM, wait); kill -9 is only a bounded
#     cleanup fallback if a server ignores the signal.
#
# Requires python3 (stdlib urllib only), a CUDA GPU, and enough VRAM for a
# 64k q8 KV cache with one slot.

set -uo pipefail

usage() {
  cat <<'EOF'
Usage: tools/test_slot_archive_shutdown.sh MODEL BIN SERVER_DIR [PORT]

  MODEL       GGUF model to serve
  BIN         rustingstrata binary (path or name on PATH)
  SERVER_DIR  directory the server runs in
  PORT        port of the servers (default 8080), reused in turn

Environment: LOAD_TIMEOUT (default 600), REQUEST_TIMEOUT (default 600),
             STOP_TIMEOUT (default 180)
EOF
}

fail() {
  echo "test_slot_archive_shutdown: $*" >&2
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
LOAD_TIMEOUT=${LOAD_TIMEOUT:-600}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-600}
STOP_TIMEOUT=${STOP_TIMEOUT:-180}

[[ "$PORT" =~ ^[0-9]+$ ]] || fail "PORT must be a number: $PORT"
(( PORT >= 1 && PORT <= 65535 )) || fail "PORT must be between 1 and 65535"
[[ "$STOP_TIMEOUT" =~ ^[0-9]+$ ]] || fail "STOP_TIMEOUT must be a number: $STOP_TIMEOUT"
(( STOP_TIMEOUT > 0 )) || fail "STOP_TIMEOUT must be positive"

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

WORK=$(mktemp -d "${TMPDIR:-/tmp}/strata-slot-archive-shutdown.XXXXXX") \
  || fail "could not create a temporary directory"
# Two fresh archive directories: the restart must read only the first one, and
# the GC phase must start with an empty manifest so the TTL check is exact.
ARCHIVE="$WORK/archive-shutdown"
ARCHIVE_GC="$WORK/archive-gc"
STATE="$WORK/shutdown-state.json"
LOG1="$WORK/server-shutdown.log"
LOG2="$WORK/server-shutdown-restart.log"
LOG3="$WORK/server-gc.log"
CHECKS="$WORK/checks.log"
: >"$CHECKS"

PID1=""
PID2=""
SERVER_PID=""
STOP_STATUS=""

alive() {
  local state
  state=$(ps -o stat= -p "$1" 2>/dev/null) || return 1
  [[ "$state" != *Z* ]]
}

# Graceful stop: SIGTERM first and wait for the process to leave on its own.
# The binary handles SIGTERM (drains requests, runs the optional shutdown
# checkpoint, saves the expert profile) and exits 0. SIGKILL is used only when
# the deadline expires, so a broken build cannot wedge the harness.
stop_server() {
  local pid=$1 wait_s=$2 label=${3:-server} i
  [[ -n "$pid" ]] || return 0
  STOP_STATUS=""
  if ! alive "$pid"; then
    wait "$pid" 2>/dev/null
    STOP_STATUS=$?
    return 0
  fi
  echo "test_slot_archive_shutdown: stopping $label (pid $pid) with SIGTERM, waiting up to ${wait_s}s"
  kill -TERM "$pid" 2>/dev/null || true
  for ((i = 0; i < wait_s * 10; i++)); do
    if ! alive "$pid"; then
      wait "$pid" 2>/dev/null
      STOP_STATUS=$?
      return 0
    fi
    sleep 0.1
  done
  echo "test_slot_archive_shutdown: $label did not exit after ${wait_s}s; sending SIGKILL" >&2
  kill -KILL "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  STOP_STATUS="killed"
  return 1
}

stop_and_report() {
  local pid=$1 label=$2
  if stop_server "$pid" "$STOP_TIMEOUT" "$label"; then
    return 0
  fi
  echo "FAIL $label: server did not exit after SIGTERM" | tee -a "$CHECKS"
  FAILED=1
  return 1
}

cleanup() {
  # Do not turn a broken run into a long wait: TERM, a short grace, then KILL.
  stop_server "${PID1:-}" 20 "phase-1 server (cleanup)" || true
  stop_server "${PID2:-}" 20 "phase-2 server (cleanup)" || true
  rm -rf -- "$WORK"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

start_server() {
  local port=$1 log=$2 mode=$3
  case "$mode" in
    shutdown)
      # --parallel 1 keeps one slot, so the shutdown checkpoint has exactly one
      # non-empty slot to save. The low thresholds keep the test's checkpoint.
      ( cd "$SERVER_DIR" && exec "$BIN" serve \
          --model "$MODEL" --gpu --backend cuda --parallel 1 --ctx 65536 \
          --kv q8 --slot-archive-dir "$ARCHIVE" --slot-archive-gb 5 \
          --slot-archive-on-shutdown --slot-archive-shutdown-timeout 30s \
          --slot-archive-min-tokens 64 --slot-archive-min-delta 1 \
          --addr "127.0.0.1:$port" ) >"$log" 2>&1 &
      ;;
    gc)
      # A one-second TTL and a one-second timer make the periodic expiry
      # deterministic without a manual GC request. The pin cap keeps the small
      # test entry pinnable.
      ( cd "$SERVER_DIR" && exec "$BIN" serve \
          --model "$MODEL" --gpu --backend cuda --parallel 1 --ctx 65536 \
          --kv q8 --slot-archive-dir "$ARCHIVE_GC" --slot-archive-gb 5 \
          --slot-archive-gc-interval 1s --slot-archive-ttl 1s \
          --slot-archive-min-tokens 64 --slot-archive-min-delta 1 \
          --slot-archive-pin-gb 1 \
          --addr "127.0.0.1:$port" ) >"$log" 2>&1 &
      ;;
    *)
      fail "unknown server mode: $mode"
      ;;
  esac
  SERVER_PID=$!
}

# run_py MODE ...: stdlib-only HTTP checks and startup wait. Its output is
# duplicated into CHECKS so the final summary counts every PASS/FAIL line.
run_py() {
  python3 - "$@" <<'PY'
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


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
    except Exception as err:  # connection refused, timeout, reset, ...
        return None, str(err)
    text = raw.decode("utf-8", "replace")
    try:
        return status, json.loads(text)
    except ValueError:
        return status, text


def brief(value, limit=220):
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    # One line per check: a log tail in a detail must not add PASS/FAIL lines
    # of its own to the harness summary.
    text = text.replace("\r", "").replace("\n", "\\n")
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

    def positive(self, name, value):
        self.check(name, value > 0, "got %r" % (value,))

    def summary(self):
        print()
        if self.failed:
            print("%d of %d checks failed" % (self.failed, self.total))
            return 1
        print("all %d checks passed" % self.total)
        return 0


# A and B do not share their first tokens: B must diverge from the slot history
# so the scheduler checkpoints A before it prefills B.
PARAGRAPH_A = (
    "A shared inference server keeps several conversations in GPU memory at "
    "once. Each slot owns its attention keys and values, the recurrent state "
    "of its layers, and the token history that explains where the state came "
    "from. When a slot is saved, every one of those buffers is streamed to "
    "disk in bounded chunks, so a very long conversation can be written and "
    "later restored without holding the whole state in host memory. The file "
    "records the model fingerprint, the number of tokens, the owner of the "
    "slot, and a framed copy of each buffer. A restore checks all of that "
    "before it changes a single GPU row, which keeps a truncated file or a "
    "file from another model from corrupting the server. "
)

PARAGRAPH_B = (
    "Lighthouse keepers wrote the weather in a daily log before radio reached "
    "the coast. The log recorded wind, visibility, the height of the swell and "
    "the ships that passed the reef, so a later keeper could compare one "
    "winter with another and spot a storm that arrived without warning. The "
    "books were bound in oilcloth, stored above the spray line, and copied by "
    "hand when a station closed. Historians now read them for storms, bird "
    "migration and the slow change of the shoreline, which the routine entries "
    "capture better than any official survey. "
)


def long_prompt(paragraph, size=8000):
    parts = []
    total = 0
    while total < size:  # roughly 2000 tokens, well above --slot-archive-min-tokens
        parts.append(paragraph)
        total += len(paragraph)
    return "".join(parts) + "\n\nAnswer with one short sentence."


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


def entries_of(body):
    entries = as_obj(body).get("entries")
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def ids_of(body):
    return [entry.get("id") for entry in entries_of(body)]


def entry_by_id(body, entry_id):
    return next(
        (entry for entry in entries_of(body) if entry.get("id") == entry_id),
        None,
    )


def check_metadata(report, obj, archive_dir):
    directory = obj.get("dir")
    report.check(
        "archive reports a directory",
        isinstance(directory, str) and bool(directory),
        "dir=%r" % (directory,),
    )
    same = (
        isinstance(directory, str)
        and bool(directory)
        and os.path.realpath(directory) == os.path.realpath(archive_dir)
    )
    report.check(
        "archive reports the configured directory",
        same,
        "dir=%r want=%r" % (directory, archive_dir),
    )
    report.check(
        "archive budget_bytes > 0",
        is_number(obj.get("budget_bytes")) and obj.get("budget_bytes") > 0,
        "budget_bytes=%r" % (obj.get("budget_bytes"),),
    )
    report.check(
        "archive used_bytes is numeric",
        is_number(obj.get("used_bytes")) and obj.get("used_bytes") >= 0,
        "used_bytes=%r" % (obj.get("used_bytes"),),
    )


def write_state(path, data):
    try:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file)
    except OSError as err:
        print("could not write %s: %s" % (path, err), file=sys.stderr)


def read_state(path):
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError) as err:
        print("could not read %s: %s" % (path, err), file=sys.stderr)
        return {}


def run_prime(base, state_path, archive_dir, timeout):
    """Chat A on the first server and prove nothing is archived yet."""
    report = Report()
    prompt_a = long_prompt(PARAGRAPH_A)

    status, body = chat(base, [{"role": "user", "content": prompt_a}], timeout)
    reply = reply_of(as_obj(body))
    report.check(
        "prompt A answered",
        status == 200 and bool(reply),
        "status=%s reply=%s" % (status, brief(reply)),
    )

    status, body = http(base, "GET", "/slots", None, timeout)
    slots = as_obj(body).get("slots", [])
    used = [s for s in slots if isinstance(s, dict) and as_int(s.get("n_past")) > 0]
    report.check(
        "GET /slots returns 200",
        status == 200,
        "status=%s body=%s" % (status, brief(body)),
    )
    report.check(
        "prompt A leaves a non-empty resident slot",
        status == 200 and bool(used),
        "status=%s used=%s body=%s" % (status, brief(used), brief(body)),
    )

    status, body = http(base, "GET", "/slots/archive", None, timeout)
    archive_body = as_obj(body)
    report.check(
        "GET /slots/archive returns 200 before the stop",
        status == 200,
        "status=%s body=%s" % (status, brief(body)),
    )
    report.check(
        "archive is empty before the graceful stop",
        status == 200 and not entries_of(body),
        "status=%s body=%s" % (status, brief(body)),
    )
    check_metadata(report, archive_body, archive_dir)

    # The continuation must use the exact strings the first server tokenized.
    write_state(
        state_path,
        {"assistant_reply": reply, "archive_dir": archive_dir},
    )
    return report.summary()


def run_shutdown(log_path, stop_ok, stop_status):
    """After SIGTERM: the server exited and logged a shutdown checkpoint."""
    report = Report()
    exited = stop_ok == "0"
    report.check(
        "server exited after SIGTERM",
        exited,
        "stopped=%r status=%r" % (stop_ok, stop_status),
    )
    report.check(
        "server exit status is 0",
        stop_status == "0",
        "status=%r" % (stop_status,),
    )

    startup = head(log_path, 262144)
    report.check(
        "server logged the 30s shutdown checkpoint setup",
        re.search(r"shutdown checkpoint on \(deadline 30s", startup) is not None,
        "log head=%s" % brief(startup[:400]),
    )

    text = tail(log_path, 262144)
    summary = re.search(
        r"shutdown checkpoint: saved (\d+), skipped (\d+), failed (\d+)", text
    )
    report.check(
        "server logged a shutdown checkpoint summary",
        summary is not None,
        "log tail=%s" % brief(text[-400:]),
    )
    saved = int(summary.group(1)) if summary else 0
    failed = int(summary.group(3)) if summary else -1
    report.check(
        "shutdown checkpoint saved at least one slot",
        summary is not None and saved >= 1,
        "saved=%d summary=%r" % (saved, summary.group(0) if summary else None),
    )
    report.check(
        "shutdown checkpoint had no failures",
        summary is not None and failed == 0,
        "failed=%d summary=%r" % (failed, summary.group(0) if summary else None),
    )
    report.check(
        "shutdown checkpoint wrote a format-2 entry",
        re.search(r"reason=shutdown\b[^\n]*format=2", text) is not None,
        "log tail=%s" % brief(text[-400:]),
    )
    return report.summary()


def run_shutdown_restart(base, state_path, timeout):
    """Restart on the same archive; the prefix must come from disk."""
    report = Report()
    state = read_state(state_path)
    assistant_reply = state.get("assistant_reply") or ""
    prompt_a = long_prompt(PARAGRAPH_A)

    status, body = http(base, "GET", "/slots/archive", None, timeout)
    archive_body = as_obj(body)
    report.check(
        "restart GET /slots/archive returns 200",
        status == 200,
        "status=%s body=%s" % (status, brief(body)),
    )
    check_metadata(report, archive_body, state.get("archive_dir") or "")

    entries = entries_of(archive_body)
    report.check(
        "restart lists the shutdown checkpoint",
        status == 200 and bool(entries),
        "status=%s body=%s" % (status, brief(archive_body)),
    )
    # Prefer an entry the shutdown checkpoint wrote; fall back to any entry so
    # the reason check reports why the restart did not produce one.
    shutdown_entries = [e for e in entries if e.get("reason") == "shutdown"]
    pool = shutdown_entries or entries
    entry = max(pool, key=lambda e: as_int(e.get("n_tokens"))) if pool else None
    if entry is None:
        report.check("shutdown entry has an id", False, "no entries")
        report.check("shutdown entry has format 2", False, "no entries")
        report.check("shutdown entry n_tokens > 0", False, "no entries")
        report.check("shutdown entry bytes > 0", False, "no entries")
        report.check("shutdown entry reason is shutdown", False, "no entries")
    else:
        report.check(
            "shutdown entry has an id",
            isinstance(entry.get("id"), str) and bool(entry.get("id")),
            "entry=%s" % brief(entry),
        )
        report.check(
            "shutdown entry has format 2",
            as_int(entry.get("format")) == 2,
            "format=%r entry=%s" % (entry.get("format"), brief(entry)),
        )
        report.positive("shutdown entry n_tokens > 0", as_int(entry.get("n_tokens")))
        report.positive("shutdown entry bytes > 0", as_int(entry.get("bytes")))
        report.check(
            "shutdown entry reason is shutdown",
            entry.get("reason") == "shutdown",
            "reason=%r entry=%s" % (entry.get("reason"), brief(entry)),
        )
        print("NOTE shutdown entry: %s" % brief(entry), flush=True)

    continuation = [
        {"role": "user", "content": prompt_a},
        {"role": "assistant", "content": assistant_reply},
        {"role": "user", "content": "Continue with one short sentence"},
    ]
    status, body = chat(base, continuation, timeout)
    obj = as_obj(body)
    cached = cached_of(obj)
    report.check(
        "continuation of A returns 200 after the restart",
        status == 200,
        "status=%s body=%s" % (status, brief(obj)),
    )
    report.check(
        "continuation restores the shutdown checkpoint prefix",
        status == 200 and cached > 0,
        "cached_tokens=%d body=%s" % (cached, brief(obj)),
    )
    return report.summary()


def run_gc(base, archive_dir, log_path, timeout):
    """Two entries, one pinned; the 1s periodic GC must expire only the other."""
    report = Report()
    prompt_a = long_prompt(PARAGRAPH_A)
    prompt_b = long_prompt(PARAGRAPH_B)

    status, body = http(base, "GET", "/slots/archive", None, timeout)
    archive_body = as_obj(body)
    report.check(
        "gc: GET /slots/archive returns 200",
        status == 200,
        "status=%s body=%s" % (status, brief(body)),
    )
    check_metadata(report, archive_body, archive_dir)
    report.check(
        "gc: the fresh archive starts empty",
        status == 200 and not entries_of(body),
        "status=%s body=%s" % (status, brief(body)),
    )
    startup = head(log_path, 262144)
    report.check(
        "gc: server logged the 1s periodic GC timer",
        re.search(r"periodic gc every 1s", startup) is not None,
        "log head=%s" % brief(startup[:400]),
    )

    # Entry A: chat then an explicit synchronous checkpoint, pinned at once so
    # the 1s TTL cannot take it before the pin request lands.
    status, body = chat(base, [{"role": "user", "content": prompt_a}], timeout)
    reply_a = reply_of(as_obj(body))
    report.check(
        "gc: prompt A answered",
        status == 200 and bool(reply_a),
        "status=%s reply=%s" % (status, brief(reply_a)),
    )

    status, body = http(base, "POST", "/slots/0?action=checkpoint", {}, timeout)
    obj = as_obj(body)
    id_a = obj.get("id")
    report.check(
        "gc: explicit checkpoint A returns 200",
        status == 200,
        "status=%s body=%s" % (status, brief(obj)),
    )
    report.check(
        "gc: explicit checkpoint A returns an id",
        isinstance(id_a, str) and bool(id_a),
        "id=%r body=%s" % (id_a, brief(obj)),
    )
    report.positive("gc: explicit checkpoint A saved tokens", as_int(obj.get("n_saved")))

    pinned = False
    if isinstance(id_a, str) and id_a:
        path = "/slots/archive/" + urllib.parse.quote(id_a, safe="")
        status, body = http(base, "POST", path + "?action=pin", {}, timeout)
        obj = as_obj(body)
        report.check(
            "gc: pin of entry A returns 200",
            status == 200,
            "status=%s body=%s" % (status, brief(obj)),
        )
        report.check(
            "gc: pin reports pinned=true",
            obj.get("pinned") is True,
            "body=%s" % brief(obj),
        )
        pinned = status == 200 and obj.get("pinned") is True
    else:
        report.check("gc: entry A could be pinned", False, "no entry A id")

    status, body = http(base, "GET", "/slots/archive", None, timeout)
    archive_body = as_obj(body)
    listed = entry_by_id(archive_body, id_a)
    report.check(
        "gc: GET /slots/archive returns 200 after the pin",
        status == 200,
        "status=%s body=%s" % (status, brief(body)),
    )
    report.check(
        "gc: pinned entry A is listed",
        listed is not None,
        "id=%r body=%s" % (id_a, brief(archive_body)),
    )
    report.check(
        "gc: pinned entry A is marked pinned",
        listed is not None and listed.get("pinned") is True,
        "entry=%s" % brief(listed),
    )
    if not pinned:
        print("NOTE gc: entry A is not pinned; the survival check will fail", flush=True)

    # Entry B: a diverging chat plus an explicit checkpoint, deliberately left
    # unpinned. The divergence checkpoint only touches A, which is pinned.
    status, body = chat(base, [{"role": "user", "content": prompt_b}], timeout)
    reply_b = reply_of(as_obj(body))
    report.check(
        "gc: prompt B answered",
        status == 200 and bool(reply_b),
        "status=%s reply=%s" % (status, brief(reply_b)),
    )

    status, body = http(base, "POST", "/slots/0?action=checkpoint", {}, timeout)
    obj = as_obj(body)
    id_b = obj.get("id")
    id_b_ok = isinstance(id_b, str) and bool(id_b)
    report.check(
        "gc: explicit checkpoint B returns 200",
        status == 200,
        "status=%s body=%s" % (status, brief(obj)),
    )
    report.check(
        "gc: explicit checkpoint B returns a distinct id",
        id_b_ok and id_b != id_a,
        "id_a=%r id_b=%r body=%s" % (id_a, id_b, brief(obj)),
    )
    report.positive("gc: explicit checkpoint B saved tokens", as_int(obj.get("n_saved")))

    status, body = http(base, "GET", "/slots/archive", None, timeout)
    archive_body = as_obj(body)
    both = id_a in ids_of(archive_body) and id_b in ids_of(archive_body)
    report.check(
        "gc: both entries are listed before the TTL",
        status == 200 and both,
        "id_a=%r id_b=%r status=%s body=%s"
        % (id_a, id_b, status, brief(archive_body)),
    )

    # No POST /slots/archive/gc here: the background timer must do the work.
    time.sleep(3.5)
    deadline = time.monotonic() + 12.0
    last = "no response"
    while time.monotonic() < deadline:
        status, body = http(base, "GET", "/slots/archive", None, 10.0)
        last = "status=%s body=%s" % (status, brief(body))
        if status == 200:
            archive_body = as_obj(body)
            current = ids_of(archive_body)
            if id_b not in current and id_a in current:
                break
        time.sleep(0.25)

    archive_body = as_obj(body) if status == 200 else {}
    current = ids_of(archive_body)
    report.check(
        "gc: periodic GC dropped the unpinned TTL entry",
        status == 200 and id_b_ok and id_b not in current,
        "id_b=%r %s" % (id_b, last),
    )
    report.check(
        "gc: pinned entry A survived the periodic GC",
        status == 200 and id_a in current,
        "id_a=%r %s" % (id_a, last),
    )
    report.check(
        "gc: final GET /slots/archive returns 200",
        status == 200,
        last,
    )
    survivor = entry_by_id(archive_body, id_a)
    report.check(
        "gc: pinned entry A is still marked pinned",
        survivor is not None and survivor.get("pinned") is True,
        "entry=%s" % brief(survivor),
    )
    print(
        "NOTE gc: no POST /slots/archive/gc was sent; the 1s timer removed the entry",
        flush=True,
    )

    text = tail(log_path, 262144)
    lines = [line for line in text.splitlines() if "periodic gc:" in line]
    report.check(
        "gc: server log records periodic GC runs",
        bool(lines),
        "log tail=%s" % brief(text[-400:]),
    )
    expired = any(
        re.search(r"periodic gc: \d+ evicted, [1-9]\d* expired", line)
        for line in lines
    )
    report.check(
        "gc: periodic GC logged a TTL expiry",
        expired,
        "gc lines=%s" % brief(lines[-5:]),
    )
    return report.summary()


def head(path, limit=65536):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as file:
            return file.read(limit).rstrip()
    except OSError as err:
        return "could not read %s: %s" % (path, err)


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
        print("usage: check.py MODE ...", file=sys.stderr)
        return 2
    mode = argv[1]
    if mode == "wait":
        if len(argv) < 6:
            print("usage: check.py wait BASE_URL LOG PID TIMEOUT", file=sys.stderr)
            return 2
        return wait_ready(argv[2], argv[3], int(argv[4]), seconds(argv[5], 600.0))
    if mode == "prime":
        if len(argv) < 6:
            print(
                "usage: check.py prime BASE_URL STATE_FILE ARCHIVE_DIR TIMEOUT",
                file=sys.stderr,
            )
            return 2
        return run_prime(argv[2], argv[3], argv[4], seconds(argv[5], 600.0))
    if mode == "shutdown":
        if len(argv) < 5:
            print(
                "usage: check.py shutdown LOG STOP_OK STOP_STATUS",
                file=sys.stderr,
            )
            return 2
        return run_shutdown(argv[2], argv[3], argv[4])
    if mode == "shutdown-restart":
        if len(argv) < 5:
            print(
                "usage: check.py shutdown-restart BASE_URL STATE_FILE TIMEOUT",
                file=sys.stderr,
            )
            return 2
        return run_shutdown_restart(argv[2], argv[3], seconds(argv[4], 600.0))
    if mode == "gc":
        if len(argv) < 6:
            print(
                "usage: check.py gc BASE_URL ARCHIVE_DIR LOG TIMEOUT",
                file=sys.stderr,
            )
            return 2
        return run_gc(argv[2], argv[3], argv[4], seconds(argv[5], 600.0))
    print("unknown mode: %s" % mode, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
PY
}

# python_checks MODE ...: run a Python mode, show its checks and append them to
# the summary log. Returns the Python exit status through pipefail.
python_checks() {
  run_py "$@" | tee -a "$CHECKS"
}

FAILED=0

echo "test_slot_archive_shutdown: model=$MODEL"
echo "test_slot_archive_shutdown: binary=$BIN"
echo "test_slot_archive_shutdown: server dir=$SERVER_DIR"
echo "test_slot_archive_shutdown: port=$PORT"
echo "test_slot_archive_shutdown: shutdown archive dir=$ARCHIVE"
echo "test_slot_archive_shutdown: GC archive dir=$ARCHIVE_GC"
echo

echo "starting shutdown-checkpoint server on 127.0.0.1:$PORT"
start_server "$PORT" "$LOG1" shutdown
PID1=$SERVER_PID
if ! python_checks wait "http://127.0.0.1:$PORT" "$LOG1" "$PID1" "$LOAD_TIMEOUT"; then
  fail "shutdown-checkpoint server did not become ready"
fi

python_checks prime "http://127.0.0.1:$PORT" "$STATE" "$ARCHIVE" "$REQUEST_TIMEOUT" || FAILED=1

# SIGTERM after the chat response: the slot is idle and must be saved by the
# shutdown checkpoint before the process exits.
if stop_server "$PID1" "$STOP_TIMEOUT" "shutdown-checkpoint server"; then
  SHUT_OK=0
else
  SHUT_OK=1
fi
PID1=""
python_checks shutdown "$LOG1" "$SHUT_OK" "$STOP_STATUS" || FAILED=1

echo
echo "restarting the same server on 127.0.0.1:$PORT with the same archive"
start_server "$PORT" "$LOG2" shutdown
PID1=$SERVER_PID
if ! python_checks wait "http://127.0.0.1:$PORT" "$LOG2" "$PID1" "$LOAD_TIMEOUT"; then
  fail "restarted shutdown-checkpoint server did not become ready"
fi

python_checks shutdown-restart "http://127.0.0.1:$PORT" "$STATE" "$REQUEST_TIMEOUT" || FAILED=1
stop_and_report "$PID1" "restarted shutdown-checkpoint server" || true
PID1=""

echo
echo "starting periodic-GC server on 127.0.0.1:$PORT (TTL 1s, GC every 1s)"
start_server "$PORT" "$LOG3" gc
PID2=$SERVER_PID
if ! python_checks wait "http://127.0.0.1:$PORT" "$LOG3" "$PID2" "$LOAD_TIMEOUT"; then
  fail "periodic-GC server did not become ready"
fi

python_checks gc "http://127.0.0.1:$PORT" "$ARCHIVE_GC" "$LOG3" "$REQUEST_TIMEOUT" || FAILED=1
stop_and_report "$PID2" "periodic-GC server" || true
PID2=""

echo
TOTAL=$(grep -cE '^(PASS|FAIL) ' "$CHECKS" 2>/dev/null || true)
FAILED_CHECKS=$(grep -cE '^FAIL ' "$CHECKS" 2>/dev/null || true)
TOTAL=${TOTAL:-0}
FAILED_CHECKS=${FAILED_CHECKS:-0}
PASSED=$((TOTAL - FAILED_CHECKS))

if (( FAILED != 0 || FAILED_CHECKS != 0 )); then
  echo "test_slot_archive_shutdown: FAILED ($PASSED/$TOTAL checks passed)"
  exit 1
fi
echo "test_slot_archive_shutdown: PASS ($TOTAL/$TOTAL checks passed)"
exit 0
