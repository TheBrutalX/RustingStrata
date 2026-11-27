#!/usr/bin/env bash
# End-to-end test for the slot save/restore HTTP API in docs/SLOT-SAVE.md.
#
# It starts the server, primes one slot with a long prompt, discovers which
# slot the scheduler used, saves it, restores it into another slot, continues
# the conversation there, erases it, checks the error cases, then starts a
# second server without --slot-save-path to check the 501 response.
#
# Usage: tools/test_slot_save.sh MODEL BIN SERVER_DIR [PORT]
#
#   MODEL       GGUF model to serve.
#   BIN         rustingstrata binary (a path, or a name on PATH).
#   SERVER_DIR  directory the server runs in.
#   PORT        port of the save-enabled server (default 8080); the second,
#               no-flag server uses PORT + 1.
#
# Environment:
#   LOAD_TIMEOUT     seconds to wait for a model load (default 600).
#   REQUEST_TIMEOUT  seconds per HTTP test request (default 600).
#
# Requires python3 (stdlib urllib only), a CUDA GPU, and enough VRAM for
# --parallel 2 with a 64k q8 KV cache.

set -uo pipefail

usage() {
  cat <<'EOF'
Usage: tools/test_slot_save.sh MODEL BIN SERVER_DIR [PORT]

  MODEL       GGUF model to serve
  BIN         rustingstrata binary (path or name on PATH)
  SERVER_DIR  directory the server runs in
  PORT        save-enabled server port (default 8080); the no-flag
              server uses PORT + 1

Environment: LOAD_TIMEOUT (default 600), REQUEST_TIMEOUT (default 600)
EOF
}

fail() {
  echo "test_slot_save: $*" >&2
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

[[ "$PORT" =~ ^[0-9]+$ ]] || fail "PORT must be a number: $PORT"
(( PORT >= 1 && PORT <= 65534 )) || fail "PORT must be between 1 and 65534"
PORT2=$((PORT + 1))

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

WORK=$(mktemp -d "${TMPDIR:-/tmp}/strata-slot-save.XXXXXX") \
  || fail "could not create a temporary directory"
SAVE_DIR="$WORK/slots"
LOG1="$WORK/server-save.log"
LOG2="$WORK/server-noflag.log"
mkdir -p "$SAVE_DIR" || fail "could not create $SAVE_DIR"

PID1=""
PID2=""
SERVER_PID=""

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
  stop_server "${PID2:-}"
  rm -rf -- "$WORK"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

start_server() {
  local port=$1 log=$2 with_slots=$3
  if [[ "$with_slots" == "yes" ]]; then
    ( cd "$SERVER_DIR" && exec "$BIN" serve \
        --model "$MODEL" --gpu --backend cuda --parallel 2 --ctx 65536 \
        --kv q8 --slot-save-path "$SAVE_DIR" \
        --addr "127.0.0.1:$port" ) >"$log" 2>&1 &
  else
    ( cd "$SERVER_DIR" && exec "$BIN" serve \
        --model "$MODEL" --gpu --backend cuda --parallel 2 --ctx 65536 \
        --kv q8 --addr "127.0.0.1:$port" ) >"$log" 2>&1 &
  fi
  SERVER_PID=$!
}

# run_py MODE ...: stdlib-only HTTP checks and startup wait.
run_py() {
  python3 - "$@" <<'PY'
import json
import os
import sys
import time
import urllib.error
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


def brief(value, limit=180):
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    return text if len(text) <= limit else text[:limit] + "..."


def as_obj(body):
    return body if isinstance(body, dict) else {}


def as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


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

    def equal(self, name, got, want):
        self.check(name, got == want, "got %r, want %r" % (got, want))

    def positive(self, name, value):
        self.check(name, value > 0, "got %r" % (value,))

    def summary(self):
        print()
        if self.failed:
            print("%d of %d checks failed" % (self.failed, self.total))
            return 1
        print("all %d checks passed" % self.total)
        return 0


PARAGRAPH = (
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


def long_prompt():
    parts = []
    size = 0
    while size < 8000:  # roughly 2000 tokens
        parts.append(PARAGRAPH)
        size += len(PARAGRAPH)
    return "".join(parts) + "\n\nAnswer with one short sentence."


def expect_status(report, base, name, path, payload, want, timeout):
    status, body = http(base, "POST", path, payload, timeout)
    report.check(
        name,
        status == want,
        "status=%s body=%s" % (status, brief(body)),
    )


def run_full(base, timeout):
    report = Report()
    prompt = long_prompt()
    messages = [{"role": "user", "content": prompt}]

    # `id_slot` is the llama.cpp-compatible request field that pins a prompt
    # to a slot, so this one is known to be slot 0.
    status, body = http(
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
    reply = reply_of(as_obj(body))
    report.check(
        "long prompt answered",
        status == 200 and bool(reply),
        "status=%s reply=%s" % (status, brief(reply)),
    )

    # The scheduler picks the slot; ask it which one holds the conversation.
    status, body = http(base, "GET", "/slots", None, timeout)
    slots = as_obj(body).get("slots", [])
    used = [s for s in slots if isinstance(s, dict) and as_int(s.get("n_past")) > 0]
    report.check("GET /slots lists the used slot", status == 200 and bool(used),
                 "status=%s body=%s" % (status, brief(body)))
    if not used:
        return report.summary()
    save_slot = as_int(used[0].get("id"))
    restore_slot = 0 if save_slot != 0 else 1

    status, body = http(
        base, "POST", "/slots/%d?action=save" % save_slot, {"filename": "slot0.bin"}, timeout
    )
    obj = as_obj(body)
    n_saved = as_int(obj.get("n_saved"))
    report.check(
        "save returns 200",
        status == 200,
        "status=%s body=%s" % (status, brief(obj)),
    )
    report.equal("save reports id_slot", obj.get("id_slot"), save_slot)
    report.equal("save reports filename", obj.get("filename"), "slot0.bin")
    report.positive("save n_saved > 0", n_saved)
    report.positive("save n_written > 0", as_int(obj.get("n_written")))

    status, body = http(
        base, "POST", "/slots/%d?action=restore" % restore_slot, {"filename": "slot0.bin"}, timeout
    )
    obj = as_obj(body)
    n_restored = as_int(obj.get("n_restored"))
    report.check(
        "restore returns 200",
        status == 200,
        "status=%s body=%s" % (status, brief(obj)),
    )
    report.equal("restore reports id_slot", obj.get("id_slot"), restore_slot)
    report.equal("restore reports filename", obj.get("filename"), "slot0.bin")
    report.check(
        "restore n_restored == n_saved",
        n_restored == n_saved and n_restored > 0,
        "n_restored=%d n_saved=%d" % (n_restored, n_saved),
    )

    # Continue on slot 1, where slot0.bin was just restored.
    continuation = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": reply or "..."},
        {"role": "user", "content": "Continue with one more short sentence."},
    ]
    status, body = http(
        base,
        "POST",
        "/v1/chat/completions",
        {
            "messages": continuation,
            "max_tokens": 32,
            "temperature": 0,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout,
    )
    obj = as_obj(body)
    cached = cached_of(obj)
    report.check(
        "continuation returns 200",
        status == 200,
        "status=%s body=%s" % (status, brief(obj)),
    )
    report.check(
        "continuation cached_tokens > 0",
        cached > 0,
        "cached_tokens=%d" % cached,
    )

    # The continuation extended slot 1. Restore the file again so that erase
    # starts from exactly n_restored tokens, which the check below compares.
    status, body = http(
        base, "POST", "/slots/%d?action=restore" % restore_slot, {"filename": "slot0.bin"}, timeout
    )
    obj = as_obj(body)
    n_restored = as_int(obj.get("n_restored"))
    report.check(
        "second restore returns 200",
        status == 200,
        "status=%s body=%s" % (status, brief(obj)),
    )
    report.check(
        "second restore still reports n_saved tokens",
        n_restored == n_saved and n_restored > 0,
        "n_restored=%d n_saved=%d" % (n_restored, n_saved),
    )

    status, body = http(base, "POST", "/slots/%d?action=erase" % restore_slot, {}, timeout)
    obj = as_obj(body)
    n_erased = as_int(obj.get("n_erased"))
    report.check(
        "erase returns 200",
        status == 200,
        "status=%s body=%s" % (status, brief(obj)),
    )
    report.equal("erase reports id_slot", obj.get("id_slot"), restore_slot)
    report.check(
        "erase n_erased == n_restored",
        n_erased == n_restored and n_erased > 0,
        "n_erased=%d n_restored=%d" % (n_erased, n_restored),
    )

    expect_status(
        report,
        base,
        "save '../x' returns 400",
        "/slots/0?action=save",
        {"filename": "../x"},
        400,
        timeout,
    )
    expect_status(
        report,
        base,
        "save invalid slot 99 returns 400",
        "/slots/99?action=save",
        {"filename": "x.bin"},
        400,
        timeout,
    )
    expect_status(
        report,
        base,
        "restore missing file returns 400",
        "/slots/0?action=restore",
        {"filename": "missing.bin"},
        400,
        timeout,
    )

    return report.summary()


def run_noflag(base):
    report = Report()
    status, body = http(
        base, "POST", "/slots/0?action=save", {"filename": "slot0.bin"}
    )
    report.check(
        "save without --slot-save-path returns 501",
        status == 501,
        "status=%s body=%s" % (status, brief(body)),
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
    if mode == "full":
        timeout = seconds(argv[3], 600.0) if len(argv) > 3 else 600.0
        return run_full(base, timeout)
    if mode == "noflag":
        return run_noflag(base)
    print("unknown mode: %s" % mode, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
PY
}

FAILED=0

echo "test_slot_save: model=$MODEL"
echo "test_slot_save: binary=$BIN"
echo "test_slot_save: server dir=$SERVER_DIR"
echo "test_slot_save: ports=$PORT and $PORT2"
echo "test_slot_save: temporary save dir=$SAVE_DIR"
echo

echo "starting save-enabled server on 127.0.0.1:$PORT"
start_server "$PORT" "$LOG1" yes
PID1=$SERVER_PID
if ! run_py wait "http://127.0.0.1:$PORT" "$LOG1" "$PID1" "$LOAD_TIMEOUT"; then
  fail "save-enabled server did not become ready"
fi

run_py full "http://127.0.0.1:$PORT" "$REQUEST_TIMEOUT" || FAILED=1
stop_server "$PID1"
PID1=""

echo
echo "starting server without --slot-save-path on 127.0.0.1:$PORT2"
start_server "$PORT2" "$LOG2" no
PID2=$SERVER_PID
if ! run_py wait "http://127.0.0.1:$PORT2" "$LOG2" "$PID2" "$LOAD_TIMEOUT"; then
  fail "no-flag server did not become ready"
fi

run_py noflag "http://127.0.0.1:$PORT2" || FAILED=1
stop_server "$PID2"
PID2=""

exit "$FAILED"
