#!/usr/bin/env bash
# End-to-end test for the conversation archive in docs/SLOT-ARCHIVE.md.
#
# It starts a single-slot server with a small archive directory, chats with a
# long prompt A, then sends a different long prompt B. The divergence must
# checkpoint A before B is prefilled. It then returns to A, checks that the
# archived prefix is restored (cached_tokens), checkpoints the slot explicitly,
# deletes that checkpoint, runs a manual GC, restarts the server on the same
# archive directory, and checks that the divergence entry survived and still
# restores. It then starts a --parallel 2 server on a separate archive
# directory, sends A and B into different slots, sends a third conversation C
# that must evict one of them, checks the evicted state reached the archive,
# restarts the parallel server, and checks that the continuation of the
# evicted conversation restores its prefix from disk (cached_tokens). Finally,
# it starts a server without --slot-archive-dir and checks that the archive
# routes answer 501.
#
# Usage: tools/test_slot_archive.sh MODEL BIN SERVER_DIR [PORT]
#
#   MODEL       GGUF model to serve.
#   BIN         rustingstrata binary (a path, or a name on PATH).
#   SERVER_DIR  directory the server runs in.
#   PORT        port of the archive-enabled servers (default 8080); PORT + 1
#               is used by the server without --slot-archive-dir.
#
# Environment:
#   LOAD_TIMEOUT     seconds to wait for a model load (default 600).
#   REQUEST_TIMEOUT  seconds per HTTP test request (default 600).
#
# Requires python3 (stdlib urllib only), a CUDA GPU, and enough VRAM for
# --parallel 1 and --parallel 2 with a 64k q8 KV cache.

set -uo pipefail

usage() {
  cat <<'EOF'
Usage: tools/test_slot_archive.sh MODEL BIN SERVER_DIR [PORT]

  MODEL       GGUF model to serve
  BIN         rustingstrata binary (path or name on PATH)
  SERVER_DIR  directory the server runs in
  PORT        archive-enabled server port (default 8080), shared in turn by
              the single-slot, parallel and no-flag servers; PORT + 1 is
              used by the server without --slot-archive-dir

Environment: LOAD_TIMEOUT (default 600), REQUEST_TIMEOUT (default 600)
EOF
}

fail() {
  echo "test_slot_archive: $*" >&2
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

WORK=$(mktemp -d "${TMPDIR:-/tmp}/strata-slot-archive.XXXXXX") \
  || fail "could not create a temporary directory"
ARCHIVE="$WORK/archive"
ARCHIVE2="$WORK/archive-parallel"
STATE="$WORK/state.json"
STATE2="$WORK/parallel-state.json"
LOG1="$WORK/server-archive.log"
LOG2="$WORK/server-restart.log"
LOG3="$WORK/server-noflag.log"
LOG4="$WORK/server-parallel.log"
LOG5="$WORK/server-parallel-restart.log"

PID1=""
PID2=""
PID3=""
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
  stop_server "${PID3:-}"
  rm -rf -- "$WORK"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

start_server() {
  local port=$1 log=$2 mode=$3
  case "$mode" in
    archive)
      # --parallel 1 keeps one slot, so a different prompt must diverge from
      # the slot history and trigger a checkpoint. The small budget and the
      # low thresholds keep the test's few entries and checks meaningful.
      ( cd "$SERVER_DIR" && exec "$BIN" serve \
          --model "$MODEL" --gpu --backend cuda --parallel 1 --ctx 65536 \
          --kv q8 --slot-archive-dir "$ARCHIVE" --slot-archive-gb 5 \
          --slot-archive-min-tokens 64 --slot-archive-min-delta 1 \
          --addr "127.0.0.1:$port" ) >"$log" 2>&1 &
      ;;
    no)
      ( cd "$SERVER_DIR" && exec "$BIN" serve \
          --model "$MODEL" --gpu --backend cuda --parallel 1 --ctx 65536 \
          --kv q8 --addr "127.0.0.1:$port" ) >"$log" 2>&1 &
      ;;
    parallel)
      # Two slots make the scheduler keep A and B side by side; a third
      # conversation then has to evict one of them into ARCHIVE2.
      ( cd "$SERVER_DIR" && exec "$BIN" serve \
          --model "$MODEL" --gpu --backend cuda --parallel 2 --ctx 65536 \
          --kv q8 --slot-archive-dir "$ARCHIVE2" --slot-archive-gb 5 \
          --slot-archive-min-tokens 64 --slot-archive-min-delta 1 \
          --addr "127.0.0.1:$port" ) >"$log" 2>&1 &
      ;;
    *)
      fail "unknown server mode: $mode"
      ;;
  esac
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


# A and B must not share their first tokens: the second prompt has to diverge
# from the slot history so the scheduler checkpoints A before prefill.
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

PARAGRAPH_C = (
    "A printing house in the old quarter kept its metal type in numbered "
    "cases, and an apprentice learned the layout by touch before he set a "
    "single line. The foreman measured each page in picas, locked the forme "
    "with wooden quoins, and inked the letters with a leather ball. When a "
    "storm broke the skylight, the crew covered the presses with canvas and "
    "dried the paper beside the stove, because a wet sheet would smear every "
    "impression and waste a week of work. The last owner donated the cases to "
    "a museum, where visitors still trace the letters and smell the oil. "
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


def check_metadata(report, obj, archive_dir=None):
    directory = obj.get("dir")
    report.check(
        "archive reports a directory",
        isinstance(directory, str) and bool(directory),
        "dir=%r" % (directory,),
    )
    if archive_dir is not None:
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


def run_full(base, state_path, archive_dir, timeout):
    report = Report()
    prompt_a = long_prompt(PARAGRAPH_A)
    prompt_b = long_prompt(PARAGRAPH_B)
    assistant_reply = "..."

    status, body = chat(base, [{"role": "user", "content": prompt_a}], timeout)
    reply = reply_of(as_obj(body))
    assistant_reply = reply if reply else "..."
    report.check(
        "prompt A answered",
        status == 200 and bool(reply),
        "status=%s reply=%s" % (status, brief(reply)),
    )

    # The scheduler picks the slot; ask it which one holds the conversation.
    status, body = http(base, "GET", "/slots", None, timeout)
    slots = as_obj(body).get("slots", [])
    used = [s for s in slots if isinstance(s, dict) and as_int(s.get("n_past")) > 0]
    report.check(
        "GET /slots lists the used slot",
        status == 200 and bool(used),
        "status=%s body=%s" % (status, brief(body)),
    )
    if used:
        report.check(
            "--parallel 1 exposes slot 0",
            as_int(used[0].get("id")) == 0,
            "slot=%r" % (used[0].get("id"),),
        )

    # A different prompt cannot extend A's history in the only slot, so the
    # scheduler must checkpoint A before it resets the slot and prefills B.
    status, body = chat(base, [{"role": "user", "content": prompt_b}], timeout)
    report.check(
        "diverging prompt B answered",
        status == 200,
        "status=%s body=%s" % (status, brief(body)),
    )

    entry = None
    archive_body = {}
    last = "no response"
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        status, body = http(base, "GET", "/slots/archive", None, 5.0)
        last = "status=%s body=%s" % (status, brief(body))
        if status == 200:
            archive_body = as_obj(body)
            entries = entries_of(body)
            if entries:
                # Prefer a divergence row, but accept any entry: the point is
                # that A's state reached the archive before the prefill of B.
                divergent = [e for e in entries if e.get("reason") == "divergence"]
                pool = divergent or entries
                entry = max(pool, key=lambda e: as_int(e.get("n_tokens")))
                break
        time.sleep(0.25)

    report.check(
        "GET /slots/archive returns 200",
        last.startswith("status=200"),
        last,
    )
    check_metadata(report, archive_body, archive_dir)

    entries = entries_of(archive_body)
    report.check(
        "archive lists a checkpoint entry",
        bool(entries),
        last,
    )
    if entry is None and entries:
        entry = entries[0]
    if entry is not None:
        report.check(
            "checkpoint id is present",
            isinstance(entry.get("id"), str) and bool(entry.get("id")),
            "entry=%s" % brief(entry),
        )
        report.positive("checkpoint n_tokens > 0", as_int(entry.get("n_tokens")))
        report.positive("checkpoint bytes > 0", as_int(entry.get("bytes")))
        reason = entry.get("reason")
        report.check(
            "checkpoint reason is recorded",
            isinstance(reason, str) and bool(reason),
            "reason=%r entry=%s" % (reason, brief(entry)),
        )
        print("NOTE divergence entry: %s" % brief(entry), flush=True)

    # Persist what the restart phase needs. The assistant text must be the
    # same string in both runs, or the continuation prompt tokenizes
    # differently and cannot match A's archived history.
    write_state(
        state_path,
        {
            "divergence_id": entry.get("id") if entry is not None else None,
            "assistant_reply": assistant_reply,
            "archive_dir": archive_dir,
        },
    )

    continuation = [
        {"role": "user", "content": prompt_a},
        {"role": "assistant", "content": assistant_reply},
        {"role": "user", "content": "Continue with one short sentence"},
    ]
    status, body = chat(base, continuation, timeout)
    obj = as_obj(body)
    cached = cached_of(obj)
    report.check(
        "continuation of A returns 200",
        status == 200,
        "status=%s body=%s" % (status, brief(obj)),
    )
    report.check(
        "continuation restores the archived prefix",
        cached > 0,
        "cached_tokens=%d body=%s" % (cached, brief(obj)),
    )

    # Explicit checkpoint of the slot's current state (A plus both replies).
    status, body = http(
        base, "POST", "/slots/0?action=checkpoint", {}, timeout
    )
    obj = as_obj(body)
    checkpoint_id = obj.get("id")
    report.check(
        "explicit checkpoint returns 200",
        status == 200,
        "status=%s body=%s" % (status, brief(obj)),
    )
    report.check(
        "explicit checkpoint returns id",
        isinstance(checkpoint_id, str) and bool(checkpoint_id),
        "id=%r body=%s" % (checkpoint_id, brief(obj)),
    )
    report.positive("explicit checkpoint n_saved > 0", as_int(obj.get("n_saved")))
    report.positive("explicit checkpoint n_written > 0", as_int(obj.get("n_written")))
    timings = obj.get("timings")
    save_ms = timings.get("save_ms") if isinstance(timings, dict) else None
    report.check(
        "explicit checkpoint timings.save_ms >= 0",
        is_number(save_ms) and save_ms >= 0,
        "timings=%s" % brief(timings),
    )

    if isinstance(checkpoint_id, str) and checkpoint_id:
        path = "/slots/archive/" + urllib.parse.quote(checkpoint_id, safe="")
        status, body = http(base, "DELETE", path, None, timeout)
        obj = as_obj(body)
        report.check(
            "DELETE archive entry returns 200",
            status == 200,
            "status=%s body=%s" % (status, brief(obj)),
        )
        report.check(
            "DELETE archive entry reports deleted=true",
            obj.get("deleted") is True,
            "body=%s" % brief(obj),
        )
        report.positive(
            "DELETE archive entry freed_bytes > 0", as_int(obj.get("freed_bytes"))
        )

        status, body = http(base, "GET", "/slots/archive", None, timeout)
        listed = checkpoint_id in ids_of(body)
        report.check(
            "deleted entry is no longer listed",
            status == 200 and not listed,
            "status=%s listed=%s body=%s" % (status, listed, brief(body)),
        )

    status, body = http(
        base, "POST", "/slots/archive/gc", {}, timeout
    )
    obj = as_obj(body)
    report.check(
        "manual GC returns 200",
        status == 200,
        "status=%s body=%s" % (status, brief(obj)),
    )
    evicted = obj.get("evicted")
    freed = obj.get("freed_bytes")
    report.check(
        "manual GC returns numeric evicted",
        is_number(evicted) and evicted >= 0,
        "evicted=%r body=%s" % (evicted, brief(obj)),
    )
    report.check(
        "manual GC returns numeric freed_bytes",
        is_number(freed) and freed >= 0,
        "freed_bytes=%r body=%s" % (freed, brief(obj)),
    )

    return report.summary()


def run_restart(base, state_path, timeout):
    report = Report()
    state = read_state(state_path)
    divergence_id = state.get("divergence_id")
    assistant_reply = state.get("assistant_reply") or "..."
    prompt_a = long_prompt(PARAGRAPH_A)

    status, body = http(base, "GET", "/slots/archive", None, timeout)
    obj = as_obj(body)
    report.check(
        "restart GET /slots/archive returns 200",
        status == 200,
        "status=%s body=%s" % (status, brief(obj)),
    )
    check_metadata(report, obj, state.get("archive_dir"))

    entries = entries_of(obj)
    listed = divergence_id in ids_of(obj) if isinstance(divergence_id, str) else False
    report.check(
        "restart still lists the divergence entry",
        listed,
        "id=%r body=%s" % (divergence_id, brief(obj)),
    )
    if listed:
        entry = next((e for e in entries if e.get("id") == divergence_id), {})
        report.positive(
            "restart entry keeps n_tokens", as_int(entry.get("n_tokens"))
        )
        report.positive("restart entry keeps bytes", as_int(entry.get("bytes")))

    # No slot holds A after the restart: the reply can only come from the
    # archive. This checks persistence, not a shutdown checkpoint (v1 has no
    # shutdown trigger; the divergence entry was written before the stop).
    continuation = [
        {"role": "user", "content": prompt_a},
        {"role": "assistant", "content": assistant_reply},
        {"role": "user", "content": "Continue with one short sentence"},
    ]
    status, body = chat(base, continuation, timeout)
    obj = as_obj(body)
    cached = cached_of(obj)
    report.check(
        "restart continuation returns 200",
        status == 200,
        "status=%s body=%s" % (status, brief(obj)),
    )
    report.check(
        "restart continuation restores the archived prefix",
        cached > 0,
        "cached_tokens=%d body=%s" % (cached, brief(obj)),
    )
    return report.summary()


def slots_of(base, timeout):
    status, body = http(base, "GET", "/slots", None, timeout)
    slots = as_obj(body).get("slots")
    if not isinstance(slots, list):
        return status, body, []
    return status, body, [slot for slot in slots if isinstance(slot, dict)]


def used_map(slots):
    return {
        as_int(slot.get("id")): as_int(slot.get("n_past"))
        for slot in slots
        if as_int(slot.get("n_past")) > 0
    }


def run_parallel(base, state_path, archive_dir, timeout):
    report = Report()
    prompt_a = long_prompt(PARAGRAPH_A)
    prompt_b = long_prompt(PARAGRAPH_B)
    # C is deliberately longer than A and B, so its token count cannot be
    # mistaken for the evicted slot's previous count when the slots are read.
    prompt_c = long_prompt(PARAGRAPH_C, 9200)

    status, body = chat(base, [{"role": "user", "content": prompt_a}], timeout)
    reply_a = reply_of(as_obj(body))
    report.check(
        "parallel: prompt A answered",
        status == 200 and bool(reply_a),
        "status=%s reply=%s" % (status, brief(reply_a)),
    )
    status_a, body_a, slots_a = slots_of(base, timeout)
    used_a = used_map(slots_a)
    report.check(
        "parallel: GET /slots after A returns 200",
        status_a == 200,
        "status=%s body=%s" % (status_a, brief(body_a)),
    )
    report.check(
        "parallel: A occupies exactly one slot",
        len(used_a) == 1,
        "slots=%s" % brief(body_a),
    )
    a_slot = next(iter(used_a)) if len(used_a) == 1 else None
    a_n = used_a.get(a_slot)

    # B must be admitted to the other free slot: with two slots and no prior
    # state the scheduler has no reason to reuse A's, and if it does the
    # eviction test below is meaningless.
    status, body = chat(base, [{"role": "user", "content": prompt_b}], timeout)
    reply_b = reply_of(as_obj(body))
    report.check(
        "parallel: prompt B answered",
        status == 200 and bool(reply_b),
        "status=%s reply=%s" % (status, brief(reply_b)),
    )
    status_b, body_b, slots_b = slots_of(base, timeout)
    used_b = used_map(slots_b)
    report.check(
        "parallel: GET /slots after B returns 200",
        status_b == 200,
        "status=%s body=%s" % (status_b, brief(body_b)),
    )
    others = [slot for slot in used_b if slot != a_slot]
    b_slot = others[0] if len(others) == 1 else None
    placement_ok = (
        len(used_b) == 2
        and a_slot is not None
        and b_slot is not None
        and used_b.get(a_slot) == a_n
    )
    report.check(
        "parallel: A and B land in different slots",
        placement_ok,
        "A slot=%r n_past=%r; after B used=%s body=%s"
        % (a_slot, a_n, used_b, brief(body_b)),
    )
    report.check(
        "parallel: A's slot is untouched by B",
        a_slot is not None and used_b.get(a_slot) == a_n,
        "A slot=%r n_past=%r; after B used=%s"
        % (a_slot, a_n, used_b),
    )
    if not placement_ok:
        print(
            "NOTE parallel: A and B share a slot; cannot test eviction",
            flush=True,
        )
        return report.summary()

    # C extends neither history, so the scheduler must pick the least recently
    # used slot, checkpoint its conversation and replace it with C.
    status, body = chat(base, [{"role": "user", "content": prompt_c}], timeout)
    reply_c = reply_of(as_obj(body))
    report.check(
        "parallel: prompt C answered",
        status == 200 and bool(reply_c),
        "status=%s reply=%s" % (status, brief(reply_c)),
    )
    status_c, body_c, slots_c = slots_of(base, timeout)
    used_c = used_map(slots_c)
    report.check(
        "parallel: GET /slots after C returns 200",
        status_c == 200,
        "status=%s body=%s" % (status_c, brief(body_c)),
    )

    # The slot that changed n_past now holds C; the other still holds exactly
    # the token count A or B had after its turn. Compare per slot, so the test
    # does not assume whether the scheduler chose A's or B's slot.
    changed = [slot for slot in (a_slot, b_slot) if used_c.get(slot) != used_b.get(slot)]
    unchanged = [slot for slot in (a_slot, b_slot) if used_c.get(slot) == used_b.get(slot)]
    resident_slot = unchanged[0] if len(unchanged) == 1 else None
    c_slot = changed[0] if len(changed) == 1 else None
    eviction_ok = (
        len(used_c) == 2
        and resident_slot is not None
        and c_slot is not None
        and resident_slot != c_slot
        and used_c.get(c_slot, 0) > 0
    )
    report.check(
        "parallel: C evicts one conversation",
        eviction_ok,
        "after B=%s after C=%s changed=%s unchanged=%s body=%s"
        % (used_b, used_c, changed, unchanged, brief(body_c)),
    )
    if resident_slot == a_slot:
        resident_label, evicted_label = "A", "B"
    elif resident_slot == b_slot:
        resident_label, evicted_label = "B", "A"
    else:
        resident_label, evicted_label = None, None
    if not eviction_ok or evicted_label is None:
        print("NOTE parallel: could not identify the evicted conversation", flush=True)
        return report.summary()
    evicted_slot = a_slot if evicted_label == "A" else b_slot
    evicted_n = used_b.get(evicted_slot)
    print(
        "NOTE parallel: %s remains in slot %r (n_past=%r); %s was evicted from slot %r (n_past=%r)"
        % (resident_label, resident_slot, used_c.get(resident_slot),
           evicted_label, evicted_slot, evicted_n),
        flush=True,
    )

    entry = None
    archive_body = {}
    last = "no response"
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        status, body = http(base, "GET", "/slots/archive", None, 5.0)
        last = "status=%s body=%s" % (status, brief(body))
        if status == 200:
            archive_body = as_obj(body)
            entries = entries_of(body)
            if entries:
                # The evicted slot was checkpointed at exactly its last turn's
                # n_past, so the entry's token count identifies it among the
                # conversations that lived in the two slots.
                matching = [
                    e for e in entries if as_int(e.get("n_tokens")) == evicted_n
                ]
                entry = (
                    matching[0]
                    if matching
                    else max(entries, key=lambda e: as_int(e.get("n_tokens")))
                )
                break
        time.sleep(0.25)

    report.check(
        "parallel: GET /slots/archive returns 200",
        last.startswith("status=200"),
        last,
    )
    check_metadata(report, archive_body, archive_dir)

    entries = entries_of(archive_body)
    report.check(
        "parallel: archive lists the evicted entry",
        entry is not None and bool(entries),
        "evicted=%s n_past=%r %s" % (evicted_label, evicted_n, last),
    )
    if entry is None and entries:
        entry = entries[0]
    report.check(
        "parallel: archive entry matches the evicted conversation",
        entry is not None
        and evicted_n is not None
        and evicted_n > 0
        and as_int(entry.get("n_tokens")) == evicted_n,
        "evicted=%s n_past=%r entry=%s" % (evicted_label, evicted_n, brief(entry)),
    )
    if entry is not None:
        report.check(
            "parallel: archive entry has an id",
            isinstance(entry.get("id"), str) and bool(entry.get("id")),
            "entry=%s" % brief(entry),
        )
        reason = entry.get("reason")
        report.check(
            "parallel: archive entry records a reason",
            isinstance(reason, str) and bool(reason),
            "reason=%r entry=%s" % (reason, brief(entry)),
        )
        report.positive(
            "parallel: archive entry n_tokens > 0", as_int(entry.get("n_tokens"))
        )
        report.positive(
            "parallel: archive entry bytes > 0", as_int(entry.get("bytes"))
        )
        print("NOTE parallel evicted entry: %s" % brief(entry), flush=True)

    replies = {"A": reply_a, "B": reply_b}
    prompts = {"A": prompt_a, "B": prompt_b}
    # The restart phase must replay the exact strings the server tokenized.
    write_state(
        state_path,
        {
            "archive_dir": archive_dir,
            "entry_id": entry.get("id") if entry is not None else None,
            "entry_n_tokens": (
                as_int(entry.get("n_tokens")) if entry is not None else 0
            ),
            "evicted_label": evicted_label,
            "evicted_prompt": prompts.get(evicted_label, ""),
            "evicted_reply": replies.get(evicted_label) or "...",
            "evicted_n_past": evicted_n if evicted_n is not None else 0,
            "resident_label": resident_label,
        },
    )
    return report.summary()


def run_parallel_restart(base, state_path, timeout):
    report = Report()
    state = read_state(state_path)
    entry_id = state.get("entry_id")
    entry_n = as_int(state.get("entry_n_tokens"))
    evicted_label = state.get("evicted_label")
    prompt = state.get("evicted_prompt") or ""
    assistant_reply = state.get("evicted_reply") or "..."

    status, body = http(base, "GET", "/slots/archive", None, timeout)
    obj = as_obj(body)
    report.check(
        "parallel: restart GET /slots/archive returns 200",
        status == 200,
        "status=%s body=%s" % (status, brief(obj)),
    )
    check_metadata(report, obj, state.get("archive_dir"))

    entries = entries_of(obj)
    listed = (
        entry_id in ids_of(obj) if isinstance(entry_id, str) else False
    )
    report.check(
        "parallel: archive survives restart",
        listed,
        "id=%r body=%s" % (entry_id, brief(obj)),
    )
    if listed:
        entry = next((e for e in entries if e.get("id") == entry_id), {})
        report.positive(
            "parallel: restart entry keeps n_tokens", as_int(entry.get("n_tokens"))
        )
        report.check(
            "parallel: restart entry keeps the evicted token count",
            entry_n > 0 and as_int(entry.get("n_tokens")) == entry_n,
            "n_tokens=%r want=%r entry=%s"
            % (entry.get("n_tokens"), entry_n, brief(entry)),
        )
        report.positive(
            "parallel: restart entry keeps bytes", as_int(entry.get("bytes"))
        )

    # Same strings as the evicted turn: no slot holds the conversation after
    # the restart, so a cached prefix can only come from ARCHIVE2 on disk.
    continuation = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": assistant_reply},
        {"role": "user", "content": "Continue with one short sentence"},
    ]
    status, body = chat(base, continuation, timeout)
    obj = as_obj(body)
    cached = cached_of(obj)
    report.check(
        "parallel: restart continuation returns 200",
        status == 200,
        "status=%s body=%s" % (status, brief(obj)),
    )
    report.check(
        "parallel: restart restores the archived prefix",
        status == 200 and cached > 0,
        "evicted=%s cached_tokens=%d body=%s"
        % (evicted_label, cached, brief(obj)),
    )

    status, body, slots = slots_of(base, timeout)
    report.check(
        "parallel: GET /slots after restart returns 200",
        status == 200,
        "status=%s body=%s" % (status, brief(body)),
    )
    report.check(
        "parallel: /slots reports the expected number of slots",
        len(slots) == 2,
        "slots=%d body=%s" % (len(slots), brief(body)),
    )
    report.check(
        "parallel: /slots reports pinned as a boolean for every slot",
        bool(slots) and all(isinstance(slot.get("pinned"), bool) for slot in slots),
        "slots=%s" % brief(slots),
    )
    used = used_map(slots)
    report.check(
        "parallel: restored conversation occupies a slot",
        len(used) >= 1,
        "used=%s slots=%s" % (used, brief(slots)),
    )
    print("NOTE parallel slots after restart: %s" % brief(slots), flush=True)
    return report.summary()


def run_noflag(base):
    report = Report()
    status, body = http(base, "GET", "/slots/archive")
    report.check(
        "GET /slots/archive without --slot-archive-dir returns 501",
        status == 501,
        "status=%s body=%s" % (status, brief(body)),
    )
    status, body = http(base, "POST", "/slots/0?action=checkpoint", {})
    report.check(
        "action=checkpoint without --slot-archive-dir returns 501",
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
        if len(argv) < 6:
            print(
                "usage: check.py full BASE_URL STATE_FILE ARCHIVE_DIR TIMEOUT",
                file=sys.stderr,
            )
            return 2
        return run_full(base, argv[3], argv[4], seconds(argv[5], 600.0))
    if mode == "restart":
        if len(argv) < 5:
            print(
                "usage: check.py restart BASE_URL STATE_FILE TIMEOUT",
                file=sys.stderr,
            )
            return 2
        return run_restart(base, argv[3], seconds(argv[4], 600.0))
    if mode == "parallel":
        if len(argv) < 6:
            print(
                "usage: check.py parallel BASE_URL STATE_FILE ARCHIVE_DIR TIMEOUT",
                file=sys.stderr,
            )
            return 2
        return run_parallel(base, argv[3], argv[4], seconds(argv[5], 600.0))
    if mode == "parallel-restart":
        if len(argv) < 5:
            print(
                "usage: check.py parallel-restart BASE_URL STATE_FILE TIMEOUT",
                file=sys.stderr,
            )
            return 2
        return run_parallel_restart(base, argv[3], seconds(argv[4], 600.0))
    if mode == "noflag":
        return run_noflag(base)
    print("unknown mode: %s" % mode, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
PY
}

FAILED=0

echo "test_slot_archive: model=$MODEL"
echo "test_slot_archive: binary=$BIN"
echo "test_slot_archive: server dir=$SERVER_DIR"
echo "test_slot_archive: ports=$PORT and $PORT2"
echo "test_slot_archive: temporary archive dir=$ARCHIVE"
echo "test_slot_archive: temporary parallel archive dir=$ARCHIVE2"
echo

echo "starting archive server on 127.0.0.1:$PORT"
start_server "$PORT" "$LOG1" archive
PID1=$SERVER_PID
if ! run_py wait "http://127.0.0.1:$PORT" "$LOG1" "$PID1" "$LOAD_TIMEOUT"; then
  fail "archive server did not become ready"
fi

run_py full "http://127.0.0.1:$PORT" "$STATE" "$ARCHIVE" "$REQUEST_TIMEOUT" || FAILED=1
stop_server "$PID1"
PID1=""

echo
echo "restarting archive server on 127.0.0.1:$PORT with the same directory"
start_server "$PORT" "$LOG2" archive
PID1=$SERVER_PID
if ! run_py wait "http://127.0.0.1:$PORT" "$LOG2" "$PID1" "$LOAD_TIMEOUT"; then
  fail "restarted archive server did not become ready"
fi

run_py restart "http://127.0.0.1:$PORT" "$STATE" "$REQUEST_TIMEOUT" || FAILED=1
stop_server "$PID1"
PID1=""

echo
echo "starting --parallel 2 archive server on 127.0.0.1:$PORT"
start_server "$PORT" "$LOG4" parallel
PID3=$SERVER_PID
if ! run_py wait "http://127.0.0.1:$PORT" "$LOG4" "$PID3" "$LOAD_TIMEOUT"; then
  fail "parallel archive server did not become ready"
fi

run_py parallel "http://127.0.0.1:$PORT" "$STATE2" "$ARCHIVE2" "$REQUEST_TIMEOUT" || FAILED=1
stop_server "$PID3"
PID3=""

echo
echo "restarting --parallel 2 archive server on 127.0.0.1:$PORT with the same directory"
start_server "$PORT" "$LOG5" parallel
PID3=$SERVER_PID
if ! run_py wait "http://127.0.0.1:$PORT" "$LOG5" "$PID3" "$LOAD_TIMEOUT"; then
  fail "restarted parallel archive server did not become ready"
fi

run_py parallel-restart "http://127.0.0.1:$PORT" "$STATE2" "$REQUEST_TIMEOUT" || FAILED=1
stop_server "$PID3"
PID3=""

echo
echo "starting server without --slot-archive-dir on 127.0.0.1:$PORT2"
start_server "$PORT2" "$LOG3" no
PID2=$SERVER_PID
if ! run_py wait "http://127.0.0.1:$PORT2" "$LOG3" "$PID2" "$LOAD_TIMEOUT"; then
  fail "no-flag server did not become ready"
fi

run_py noflag "http://127.0.0.1:$PORT2" || FAILED=1
stop_server "$PID2"
PID2=""

exit "$FAILED"
