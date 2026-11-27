#!/usr/bin/env bash
# Fault-injection harness for the segmented conversation archive
# (docs/SLOT-ARCHIVE-PHASE2.md §12, "Correctness still to add").
#
# The harness builds one small v2 archive with two conversations that share a
# long system-prompt prefix, then exercises independent copies of that archive:
#
#   manifest    truncate manifest.json. v2 rows are not rebuilt in this wave, so
#               the entries must be dropped; the archive must still open and a
#               chat must be answered by the normal prefill path.
#   segments    truncate segments.json. The segment store rebuilds it from the
#               .seg files and the archive reconciles refcounts from the
#               surviving entries; a continuation must still restore its prefix.
#   segment     truncate one referenced .seg file. The segment store drops the
#               row, the archive drops every entry that needs it, and chat falls
#               back to prefill.
#   state       truncate one referenced STRSTATE file. The entry is dropped and
#               chat falls back to prefill.
#   missing     delete a .seg file while the server is down and verify the entry
#               is dropped at startup. A second copy deletes a .seg file while
#               the server is running: the restore must fail closed, the chat
#               must answer, and a later GC must evict the unusable entry.
#
# It also checks the lifecycle invariants that make a crash survivable:
#
#   * kill -9 during an explicit checkpoint (poll for a writer temp file, fall
#     back to a short delay), restart, and verify temp cleanup, a valid
#     manifest/segments.json, no dangling references and an answering chat;
#   * a small-budget run that creates entries, pins one, forces GC and checks
#     `used_bytes <= budget` with the pin still present;
#   * restart refcount survival: shared segments keep their files and their
#     refs match the manifest after a restart, and the survivor still restores
#     after the first entry is deleted and the process restarts again.
#
# The lease -> 409 path is exercised as a race probe. In this wave a v2 restore
# holds the archive mutex for the whole read/install, so DELETE normally blocks
# and then succeeds instead of observing the lease; only a v1 async restore
# leaves the lock and exposes the lease for a useful window. The probe accepts
# either answer and documents which one was observed. A deterministic 409 test
# would need a deliberately slow v2 reader (or the v1 async path); that is a
# live-run limitation, not a failure of the harness.
#
# Usage: tools/test_slot_archive_faults.sh MODEL BIN SERVER_DIR [PORT]
#
#   MODEL       GGUF model to serve.
#   BIN         rustingstrata binary (a path, or a name on PATH).
#   SERVER_DIR  directory the server runs in.
#   PORT        port shared by the scenario servers (default 8080).
#
# Environment:
#   LOAD_TIMEOUT       seconds to wait for a model load (default 600).
#   REQUEST_TIMEOUT    seconds per HTTP request (default 600).
#   FAULT_GC_TURNS     conversations for the small-budget GC case (default 6).
#   FAULT_GC_CHARS     approximate prompt size per GC conversation (default 72000).
#   FAULT_CRASH_CHARS  prompt size for the kill -9 checkpoint case (default 100000).
#
# Requires python3 (stdlib only), a CUDA GPU, and enough VRAM for
# --parallel 1 with a 64k q8 KV cache. Nothing here is run by `bash -n`; the
# script is syntax-checked only in the unit-test environment.

set -uo pipefail

usage() {
  cat <<'EOF'
Usage: tools/test_slot_archive_faults.sh MODEL BIN SERVER_DIR [PORT]

  MODEL       GGUF model to serve
  BIN         rustingstrata binary (path or name on PATH)
  SERVER_DIR  directory the server runs in
  PORT        scenario server port (default 8080)

Environment: LOAD_TIMEOUT (default 600), REQUEST_TIMEOUT (default 600),
             FAULT_GC_TURNS (default 6), FAULT_GC_CHARS (default 72000),
             FAULT_CRASH_CHARS (default 100000)
EOF
}

fail() {
  echo "test_slot_archive_faults: $*" >&2
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
GC_TURNS=${FAULT_GC_TURNS:-6}
GC_CHARS=${FAULT_GC_CHARS:-72000}
CRASH_CHARS=${FAULT_CRASH_CHARS:-100000}

[[ "$PORT" =~ ^[0-9]+$ ]] || fail "PORT must be a number: $PORT"
(( PORT >= 1 && PORT <= 65534 )) || fail "PORT must be between 1 and 65534"
[[ "$GC_TURNS" =~ ^[0-9]+$ ]] && (( GC_TURNS >= 3 )) || fail "FAULT_GC_TURNS must be >= 3"
[[ "$GC_CHARS" =~ ^[0-9]+$ ]] && (( GC_CHARS >= 8000 )) || fail "FAULT_GC_CHARS must be >= 8000"
[[ "$CRASH_CHARS" =~ ^[0-9]+$ ]] && (( CRASH_CHARS >= 8000 )) || fail "FAULT_CRASH_CHARS must be >= 8000"

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

WORK=$(mktemp -d "${TMPDIR:-/tmp}/strata-slot-faults.XXXXXX") \
  || fail "could not create a temporary directory"
BASE="$WORK/base"
PRISTINE="$WORK/pristine"
CASES="$WORK/cases"
STATE="$WORK/faults-state.json"
STATE_CRASH="$WORK/faults-crash-state.json"
RESULTS="$WORK/results.txt"
: >"$RESULTS"
export FAULTS_RESULTS="$RESULTS"

SERVER_PID=""
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

# start_server DIR LOG BUDGET_GB [EXTRA...]
start_server() {
  local dir=$1 log=$2 budget=$3
  shift 3
  ( cd "$SERVER_DIR" && exec "$BIN" serve \
      --model "$MODEL" --gpu --backend cuda --parallel 1 --ctx 65536 \
      --kv q8 --slot-archive-dir "$dir" --slot-archive-gb "$budget" \
      --slot-archive-min-tokens 64 --slot-archive-min-delta 1 \
      --slot-archive-versions 2 "$@" \
      --addr "127.0.0.1:$PORT" ) >"$log" 2>&1 &
  SERVER_PID=$!
}

# run_py MODE ...: stdlib-only HTTP/file checks and startup wait.
run_py() {
  python3 - "$@" <<'PY'
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
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


def prompt_tokens_of(body):
    return as_int(as_obj(as_obj(body).get("usage")).get("prompt_tokens"))


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
        else:
            print("all %d checks passed" % self.total)
        path = os.environ.get("FAULTS_RESULTS")
        if path:
            try:
                with open(path, "a", encoding="utf-8") as file:
                    file.write("%d %d\n" % (self.total, self.failed))
            except OSError:
                pass
        return 1 if self.failed else 0


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

USER_A = "Name the buffer each slot owns."
USER_B = "Say what a checkpoint writes to disk."
USER_C = "Say what a restore checks first."
USER_D = "Say what happens when a slot is reused for a different conversation."


def shared_system(size=12000):
    parts = []
    total = 0
    while total < size:
        parts.append(PARAGRAPH)
        total += len(PARAGRAPH)
    return (
        "\n\n".join(parts)
        + "\n\nAnswer every user message with one short sentence."
    )


def long_prompt(label, size):
    parts = [label + ". "]
    total = len(parts[0])
    while total < size:
        parts.append(PARAGRAPH)
        total += len(PARAGRAPH)
    return "".join(parts) + "\n\nAnswer with one short sentence."


def chat(base, messages, timeout, max_tokens=32):
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


def user_turn(system, user):
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def continuation(system, user, assistant_reply):
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant_reply},
        {"role": "user", "content": CONTINUE},
    ]


def entries_of(body):
    entries = as_obj(body).get("entries")
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def entry_segments(entry):
    segments = entry.get("segments") if isinstance(entry, dict) else None
    if not isinstance(segments, list):
        return []
    return [segment for segment in segments if isinstance(segment, dict)]


def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, ValueError):
        return None


def write_state(path, data):
    try:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file)
    except OSError as err:
        print("could not write %s: %s" % (path, err), file=sys.stderr)


def read_state(path):
    data = load_json(path)
    return data if isinstance(data, dict) else {}


def wait_entries(base, minimum, timeout):
    status = None
    body = {}
    entries = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status, body = http(base, "GET", "/slots/archive", None, 5.0)
        entries = entries_of(body)
        if status == 200 and len(entries) >= minimum:
            return status, body, entries
        time.sleep(0.25)
    return status, body, entries


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


def parse_archive(report, archive_dir, label="", min_entries=0):
    """Read the archive files and validate the v2 entry/segment graph.

    This reads what the server wrote, not the HTTP API. It returns a dict with
    the entries and segment rows, or None when the files are unusable.
    """
    prefix = label + ": " if label else ""
    manifest = load_json(os.path.join(archive_dir, "manifest.json"))
    segments = load_json(os.path.join(archive_dir, "segments.json"))
    ok = isinstance(manifest, dict) and isinstance(segments, dict)
    report.check(
        prefix + "manifest.json and segments.json are readable",
        ok,
        "dir=%s" % (archive_dir,),
    )
    if not ok:
        return None
    entries = manifest.get("entries")
    rows = segments.get("segments")
    if not isinstance(entries, list) or not isinstance(rows, list):
        report.check(prefix + "archive JSON shapes are valid", False)
        return None
    report.check(
        prefix + "manifest.json version is 1",
        as_int(manifest.get("version")) == 1,
        "version=%r" % (manifest.get("version"),),
    )
    report.check(
        prefix + "segments.json version is 1",
        as_int(segments.get("version")) == 1,
        "version=%r" % (segments.get("version"),),
    )
    entries = [entry for entry in entries if isinstance(entry, dict)]
    rows = [row for row in rows if isinstance(row, dict)]
    report.check(
        prefix + "manifest lists at least %d entries" % min_entries,
        len(entries) >= min_entries,
        "entries=%d" % len(entries),
    )

    by_id = {}
    for row in rows:
        row_id = row.get("id")
        if is_id(row_id):
            by_id[row_id] = row
    bad_rows = []
    for row in rows:
        row_id = row.get("id")
        parent = row.get("parent")
        if (
            not is_int(row.get("bytes"))
            or row.get("bytes") <= 0
            or not is_int(row.get("n_tokens"))
            or row.get("n_tokens") <= 0
            or not is_int(row.get("refs"))
            or row.get("refs") < 0
            or as_int(row.get("format")) != 1
            or (parent is not None and not is_id(parent))
        ):
            bad_rows.append(row_id)
    report.check(
        prefix + "segment rows have positive sizes, token counts and refs",
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
        prefix + "every segment row has its file on disk",
        not bad_files,
        "missing_or_size_mismatch=%s" % brief(bad_files[:8]),
    )

    ref_counts = {}
    bad_refs = []
    for entry in entries:
        if as_int(entry.get("format")) != 2:
            continue
        position = 0
        segment_list = entry_segments(entry)
        if not segment_list:
            bad_refs.append("entry %r has no segments" % (entry.get("id"),))
        for segment in segment_list:
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
        prefix + "entry segment refs are contiguous and resolve",
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
        prefix + "segment refcounts match the entry references",
        not mismatched,
        "mismatch=%s" % brief(mismatched[:6]),
    )
    return {
        "entries": entries,
        "by_id": by_id,
        "ref_counts": ref_counts,
        "shared": sorted(k for k, v in ref_counts.items() if v >= 2),
    }


def temp_files(root):
    found = []
    for current, _, files in os.walk(root):
        for name in files:
            if ".tmp" in name:
                found.append(os.path.join(current, name))
    return found


def log_has_panic(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as file:
            file.seek(0, os.SEEK_END)
            size = file.tell()
            file.seek(max(0, size - 262144))
            return "panicked at" in file.read()
    except OSError:
        return False


def read_slot_id(base, timeout):
    status, body = http(base, "GET", "/slots", None, timeout)
    slots = as_obj(body).get("slots", [])
    used = [
        as_int(slot.get("id"))
        for slot in slots
        if isinstance(slot, dict) and as_int(slot.get("n_past")) > 0
    ]
    return status, used[0] if used else None


def run_prepare(base, state_path, archive_dir, timeout):
    report = Report()
    system = shared_system()
    reply_a = reply_b = reply_c = ""
    for label, user in (("A", USER_A), ("B", USER_B), ("C", USER_C)):
        status, body = chat(base, user_turn(system, user), timeout)
        reply = reply_of(as_obj(body))
        report.check(
            "prepare: conversation %s answered" % label,
            status == 200 and bool(reply),
            "status=%s reply=%s" % (status, brief(reply)),
        )
        if label == "A":
            reply_a = reply or "..."
        elif label == "B":
            reply_b = reply or "..."
        else:
            reply_c = reply or "..."

    status, body, entries = wait_entries(base, 2, 30.0)
    report.check(
        "prepare: archive lists at least two entries",
        status == 200 and len(entries) >= 2,
        "status=%s entries=%d body=%s" % (status, len(entries), brief(body)),
    )
    check_metadata(report, as_obj(body), archive_dir)
    report.check(
        "prepare: every listed entry is v2 (format 2)",
        bool(entries) and all(as_int(e.get("format")) == 2 for e in entries),
        "formats=%r" % ([e.get("format") for e in entries],),
    )
    analysis = parse_archive(report, archive_dir, "prepare", 2)
    if analysis is None:
        return report.summary()
    ordered = sorted(
        analysis["entries"],
        key=lambda entry: (as_int(entry.get("created_ms")), str(entry.get("id"))),
    )
    first, second = ordered[0], ordered[1]
    first_id = first.get("id")
    second_id = second.get("id")
    shared = analysis["shared"]
    report.check(
        "prepare: the two entries share at least one segment",
        bool(shared),
        "shared=%s" % brief(shared),
    )
    report.check(
        "prepare: entries have ids",
        isinstance(first_id, str) and bool(first_id)
        and isinstance(second_id, str) and bool(second_id),
        "first=%r second=%r" % (first_id, second_id),
    )
    print(
        "NOTE prepare: first=%r n_tokens=%r second=%r n_tokens=%r shared=%d"
        % (first_id, first.get("n_tokens"), second_id, second.get("n_tokens"), len(shared)),
        flush=True,
    )
    write_state(
        state_path,
        {
            "archive_dir": archive_dir,
            "system": system,
            "user_a": USER_A,
            "user_b": USER_B,
            "reply_a": reply_a,
            "reply_b": reply_b,
            "reply_c": reply_c,
            "first_id": first_id,
            "second_id": second_id,
            "shared_ids": shared,
        },
    )
    return report.summary()


def first_entry_row(manifest, entry_id):
    for row in manifest.get("entries", []):
        if isinstance(row, dict) and row.get("id") == entry_id:
            return row
    return None


def run_corrupt(state_path, copy_dir, kind):
    report = Report()
    state = read_state(state_path)
    entry_id = state.get("first_id")
    report.check(
        "corrupt: baseline state has an entry id",
        isinstance(entry_id, str) and bool(entry_id),
        "id=%r" % (entry_id,),
    )
    if kind in ("manifest", "segments"):
        name = "manifest.json" if kind == "manifest" else "segments.json"
        path = os.path.join(copy_dir, name)
        try:
            size = os.path.getsize(path)
            with open(path, "r+b") as file:
                file.truncate(max(1, size // 2))
            changed = True
        except OSError as err:
            print("NOTE corrupt: %s: %s" % (path, err), flush=True)
            changed = False
        report.check(
            "corrupt: truncated %s" % name,
            changed,
            "path=%s" % path,
        )
        print("NOTE corrupt: %s is now truncated (%d bytes)" % (path, size // 2 if changed else 0), flush=True)
        return report.summary()

    manifest = load_json(os.path.join(copy_dir, "manifest.json"))
    row = first_entry_row(manifest, entry_id) if isinstance(manifest, dict) else None
    if row is None and isinstance(manifest, dict):
        rows = [r for r in manifest.get("entries", []) if isinstance(r, dict)]
        row = rows[0] if rows else None
    report.check(
        "corrupt: found the v2 entry row in the copy",
        row is not None,
        "copy=%s id=%r" % (copy_dir, entry_id),
    )
    if row is None:
        return report.summary()

    segment_ids = [
        segment.get("id")
        for segment in row.get("segments", [])
        if isinstance(segment, dict) and is_id(segment.get("id"))
    ]
    segment_path = None
    if segment_ids:
        segment_id = segment_ids[0]
        segment_path = os.path.join(
            copy_dir, "segments", segment_id[:2], segment_id + ".seg"
        )
    state_rel = row.get("state_file")
    state_path = (
        os.path.join(copy_dir, state_rel)
        if isinstance(state_rel, str) and state_rel
        else None
    )

    if kind == "segment":
        report.check(
            "corrupt: found the first segment file",
            isinstance(segment_path, str) and os.path.isfile(segment_path),
            "path=%r" % (segment_path,),
        )
        if segment_path and os.path.isfile(segment_path):
            with open(segment_path, "r+b") as file:
                file.truncate(20)
            print("NOTE corrupt: truncated %s to 20 bytes" % segment_path, flush=True)
    elif kind == "state":
        report.check(
            "corrupt: found the state file",
            isinstance(state_path, str) and os.path.isfile(state_path),
            "path=%r" % (state_path,),
        )
        if state_path and os.path.isfile(state_path):
            with open(state_path, "r+b") as file:
                file.truncate(16)
            print("NOTE corrupt: truncated %s to 16 bytes" % state_path, flush=True)
    elif kind == "missing":
        report.check(
            "corrupt: found the segment file to remove",
            isinstance(segment_path, str) and os.path.isfile(segment_path),
            "path=%r" % (segment_path,),
        )
        if segment_path and os.path.isfile(segment_path):
            os.unlink(segment_path)
            print("NOTE corrupt: removed %s" % segment_path, flush=True)
    else:
        report.check("corrupt: known kind", False, "kind=%r" % (kind,))
    return report.summary()


def run_verify(base, state_path, copy_dir, log_path, kind, timeout):
    report = Report()
    state = read_state(state_path)
    system = state.get("system") or ""
    user_a = state.get("user_a") or ""
    reply_a = state.get("reply_a") or "..."
    first_id = state.get("first_id")
    second_id = state.get("second_id")

    status, _ = http(base, "GET", "/health", None, 5.0)
    report.check("%s: server answers /health" % kind, status == 200, "status=%r" % (status,))

    status, body = http(base, "GET", "/slots/archive", None, timeout)
    report.check(
        "%s: GET /slots/archive returns 200" % kind,
        status == 200,
        "status=%s body=%s" % (status, brief(body)),
    )
    check_metadata(report, as_obj(body), copy_dir)
    listed = [entry.get("id") for entry in entries_of(body)]
    report.check(
        "%s: every surviving entry is v2 (format 2)" % kind,
        all(as_int(entry.get("format")) == 2 for entry in entries_of(body)),
        "entries=%s" % brief(entries_of(body)),
    )
    analysis = parse_archive(report, copy_dir, kind, 0)

    temps = temp_files(copy_dir)
    report.check(
        "%s: startup left no writer temp files" % kind,
        not temps,
        "temps=%s" % brief(temps[:8]),
    )

    if kind == "manifest":
        report.check(
            "manifest: unrebuildable v2 entries are dropped",
            isinstance(first_id, str) and first_id not in listed
            and (not isinstance(second_id, str) or second_id not in listed),
            "listed=%s first=%r second=%r" % (brief(listed), first_id, second_id),
        )
        print(
            "NOTE manifest: v2 rows are not rebuilt from STRSTATE in this wave; "
            "the entries were dropped and the chat falls back to prefill",
            flush=True,
        )
    elif kind == "segments":
        report.check(
            "segments: both entries survive the segment-manifest rebuild",
            first_id in listed and second_id in listed,
            "listed=%s first=%r second=%r" % (brief(listed), first_id, second_id),
        )
        if analysis is not None:
            report.check(
                "segments: the rebuilt archive still has shared segment refs",
                bool(analysis["shared"]),
                "shared=%s" % brief(analysis["shared"]),
            )
    else:
        report.check(
            "%s: the entry with the damaged file is dropped" % kind,
            isinstance(first_id, str) and first_id not in listed,
            "listed=%s first=%r" % (brief(listed), first_id),
        )

    status, body = chat(
        base, continuation(system, user_a, reply_a), timeout
    )
    obj = as_obj(body)
    report.check(
        "%s: continuation answers after the fault (fallback)" % kind,
        status == 200 and bool(reply_of(obj)),
        "status=%s body=%s" % (status, brief(obj)),
    )
    if kind == "segments":
        report.check(
            "segments: continuation restores the archived prefix",
            status == 200 and cached_of(obj) > 0,
            "cached_tokens=%d body=%s" % (cached_of(obj), brief(obj)),
        )
    else:
        print(
            "NOTE %s: continuation cached_tokens=%d (0 is the expected prefill fallback)"
            % (kind, cached_of(obj)),
            flush=True,
        )

    status, body = http(base, "POST", "/slots/archive/gc", {}, timeout)
    report.check(
        "%s: GC after the fault returns 200" % kind,
        status == 200,
        "status=%s body=%s" % (status, brief(body)),
    )
    report.check(
        "%s: server did not panic" % kind,
        not log_has_panic(log_path),
        "log=%s" % log_path,
    )
    return report.summary()


def run_live_missing(base, state_path, log_path, timeout):
    report = Report()
    state = read_state(state_path)
    system = state.get("system") or ""
    user_a = state.get("user_a") or ""
    reply_a = state.get("reply_a") or "..."
    first_id = state.get("first_id")
    archive_dir = state.get("archive_dir")

    manifest = load_json(os.path.join(archive_dir, "manifest.json"))
    row = first_entry_row(manifest, first_id) if isinstance(manifest, dict) else None
    report.check(
        "live missing: found the first entry's manifest row",
        row is not None,
        "id=%r" % (first_id,),
    )
    segment_ids = [
        segment.get("id")
        for segment in (row or {}).get("segments", [])
        if isinstance(segment, dict) and is_id(segment.get("id"))
    ]
    segment_path = None
    if segment_ids:
        segment_path = os.path.join(
            archive_dir, "segments", segment_ids[0][:2], segment_ids[0] + ".seg"
        )
    report.check(
        "live missing: found a referenced segment file",
        isinstance(segment_path, str) and os.path.isfile(segment_path),
        "path=%r" % (segment_path,),
    )
    if segment_path and os.path.isfile(segment_path):
        os.unlink(segment_path)
        print("NOTE live missing: removed %s while the server was running" % segment_path, flush=True)

    status, body = chat(base, continuation(system, user_a, reply_a), timeout)
    obj = as_obj(body)
    report.check(
        "live missing: restore falls back and the chat answers",
        status == 200 and bool(reply_of(obj)),
        "status=%s body=%s" % (status, brief(obj)),
    )
    report.check(
        "live missing: the failed restore did not reuse a prefix",
        status == 200 and cached_of(obj) == 0,
        "cached_tokens=%d body=%s" % (cached_of(obj), brief(obj)),
    )
    report.check(
        "live missing: the server is still alive after the failed restore",
        http(base, "GET", "/health", None, 5.0)[0] == 200,
    )

    status, body = http(base, "POST", "/slots/archive/gc", {}, timeout)
    obj = as_obj(body)
    report.check(
        "live missing: GC returns 200",
        status == 200,
        "status=%s body=%s" % (status, brief(obj)),
    )
    report.check(
        "live missing: GC evicts the unusable entry",
        status == 200 and as_int(obj.get("evicted")) >= 1,
        "evicted=%r body=%s" % (obj.get("evicted"), brief(obj)),
    )
    status, body = http(base, "GET", "/slots/archive", None, timeout)
    listed = [entry.get("id") for entry in entries_of(body)]
    report.check(
        "live missing: the evicted entry is no longer listed",
        status == 200 and first_id not in listed,
        "status=%s listed=%s body=%s" % (status, brief(listed), brief(body)),
    )
    report.check(
        "live missing: server did not panic",
        not log_has_panic(log_path),
        "log=%s" % log_path,
    )
    return report.summary()


def run_lease_probe(base, state_path, log_path, timeout):
    report = Report()
    state = read_state(state_path)
    system = state.get("system") or ""
    user_a = state.get("user_a") or ""
    reply_a = state.get("reply_a") or "..."
    first_id = state.get("first_id")

    status, body = http(base, "GET", "/slots/archive", None, timeout)
    report.check(
        "lease: GET /slots/archive returns 200 before the probe",
        status == 200,
        "status=%s body=%s" % (status, brief(body)),
    )
    entry = next(
        (item for item in entries_of(body) if item.get("id") == first_id), None
    )
    report.check(
        "lease: the target entry is present",
        entry is not None,
        "id=%r body=%s" % (first_id, brief(body)),
    )
    report.check(
        "lease: GET exposes a boolean restoring flag",
        entry is not None and isinstance(entry.get("restoring"), bool),
        "entry=%s" % brief(entry),
    )
    if entry is None:
        return report.summary()

    # v2 restores hold the archive mutex for the whole read/install, so this
    # is a race probe, not a deterministic 409. The restore runs in a worker
    # thread and DELETE is sent while it is (probably) reading.
    import threading

    result = {}

    def restore_worker():
        try:
            result["restore"] = chat(
                base, continuation(system, user_a, reply_a), timeout
            )
        except Exception as err:  # pragma: no cover - urllib already catches
            result["restore"] = (None, str(err))

    thread = threading.Thread(target=restore_worker)
    thread.start()
    time.sleep(0.05)
    path = "/slots/archive/" + urllib.parse.quote(first_id, safe="")
    delete_status, delete_body = http(base, "DELETE", path, None, timeout)
    thread.join(timeout)

    restore_status, restore_body = result.get("restore", (None, "restore did not run"))
    report.check(
        "lease: the concurrent restore still answers",
        restore_status == 200,
        "status=%s body=%s" % (restore_status, brief(restore_body)),
    )
    report.check(
        "lease: DELETE during restore answers 409 or is serialized safely",
        delete_status in (200, 409),
        "status=%s body=%s" % (delete_status, brief(delete_body)),
    )
    if delete_status == 409:
        print(
            "NOTE lease: observed the lease -> 409 path",
            flush=True,
        )
    else:
        print(
            "NOTE lease: DELETE answered %r without 409; the v2 reader holds the "
            "archive mutex, so the lease window is not deterministically reachable. "
            "The 409 path is exercised by v1 async restores and the core lease tests."
            % (delete_status,),
            flush=True,
        )
    report.check(
        "lease: server did not panic",
        not log_has_panic(log_path),
        "log=%s" % log_path,
    )
    return report.summary()


def run_crash_prepare(base, base_dir, state_path, chars, timeout):
    report = Report()
    prompt = long_prompt("Crash checkpoint survival probe", chars)
    status, body = chat(base, [{"role": "user", "content": prompt}], timeout)
    obj = as_obj(body)
    reply = reply_of(obj)
    report.check(
        "crash: large prompt answered before the checkpoint",
        status == 200 and bool(reply),
        "status=%s reply=%s" % (status, brief(reply)),
    )
    slot_status, slot = read_slot_id(base, timeout)
    report.check(
        "crash: a slot holds the conversation",
        slot_status == 200 and slot is not None,
        "status=%s slot=%r" % (slot_status, slot),
    )
    write_state(
        state_path,
        {
            "archive_dir": base_dir,
            "prompt": prompt,
            "reply": reply or "...",
            "slot": slot if slot is not None else 0,
        },
    )
    return report.summary()


def run_checkpoint(base, state_path, timeout):
    report = Report()
    state = read_state(state_path)
    slot = as_int(state.get("slot"))
    status, body = http(
        base, "POST", "/slots/%d?action=checkpoint" % slot, {}, timeout
    )
    obj = as_obj(body)
    report.check(
        "checkpoint: explicit checkpoint returns 200",
        status == 200,
        "status=%s body=%s" % (status, brief(obj)),
    )
    report.check(
        "checkpoint: explicit checkpoint returns an id",
        isinstance(obj.get("id"), str) and bool(obj.get("id")),
        "body=%s" % brief(obj),
    )
    return report.summary()


def on_disk_segments(root):
    found = set()
    base = os.path.join(root, "segments")
    for current, _, files in os.walk(base):
        for name in files:
            if name.endswith(".seg"):
                found.add(name[:-4])
    return found


def run_crash_verify(base, state_path, log_path, timeout):
    report = Report()
    state = read_state(state_path)
    prompt = state.get("prompt") or ""
    archive_dir = state.get("archive_dir")

    status, _ = http(base, "GET", "/health", None, 5.0)
    report.check("crash: restarted server answers /health", status == 200, "status=%r" % (status,))
    status, body = http(base, "GET", "/slots/archive", None, timeout)
    report.check(
        "crash: GET /slots/archive returns 200 after restart",
        status == 200,
        "status=%s body=%s" % (status, brief(body)),
    )
    check_metadata(report, as_obj(body), archive_dir)
    analysis = parse_archive(report, archive_dir, "crash", 0)

    temps = temp_files(archive_dir)
    report.check(
        "crash: startup removed every writer temp file",
        not temps,
        "temps=%s" % brief(temps[:8]),
    )
    if analysis is not None:
        disk = on_disk_segments(archive_dir)
        rows = set(analysis["by_id"])
        unadopted = sorted(disk - rows)
        missing = sorted(rows - disk)
        # A crash between the segment rename and the segments.json persist can
        # leave a valid file outside the manifest. It must be adoptable (not a
        # truncated/partial file) and must not be referenced by a visible entry;
        # a normal startup GC usually removes it, but the harness accepts a
        # valid leftover rather than declaring the archive corrupt.
        invalid_orphans = []
        for row_id in unadopted:
            path = os.path.join(
                archive_dir, "segments", row_id[:2], row_id + ".seg"
            )
            try:
                with open(path, "rb") as file:
                    magic = file.read(8)
                if magic != b"STRATSEG":
                    invalid_orphans.append(row_id)
            except OSError:
                invalid_orphans.append(row_id)
        report.check(
            "crash: every surviving orphan segment is a valid re-adoptable file",
            not invalid_orphans,
            "invalid=%s" % brief(invalid_orphans[:8]),
        )
        if unadopted:
            print(
                "NOTE crash: %d valid orphan segment file(s) survived; a "
                "segments.json rebuild will adopt them" % len(unadopted),
                flush=True,
            )
        report.check(
            "crash: every segment row still has its file",
            not missing,
            "missing=%s" % brief(missing[:8]),
        )

    status, body = chat(base, [{"role": "user", "content": prompt}], timeout)
    obj = as_obj(body)
    report.check(
        "crash: chat answers after the restart",
        status == 200 and bool(reply_of(obj)),
        "status=%s body=%s" % (status, brief(obj)),
    )
    print(
        "NOTE crash: post-restart chat cached_tokens=%d (an entry may or may not "
        "have been committed before SIGKILL)" % cached_of(obj),
        flush=True,
    )
    report.check(
        "crash: server did not panic",
        not log_has_panic(log_path),
        "log=%s" % log_path,
    )
    return report.summary()


def run_gc(base, archive_dir, turns, chars, timeout):
    report = Report()
    replies = []
    for index in range(turns):
        prompt = long_prompt("GC conversation %d" % index, chars)
        status, body = chat(base, [{"role": "user", "content": prompt}], timeout)
        obj = as_obj(body)
        report.check(
            "gc: conversation %d answered" % index,
            status == 200 and bool(reply_of(obj)),
            "status=%s body=%s" % (status, brief(obj)),
        )
        replies.append(reply_of(obj))

    status, body, entries = wait_entries(base, 3, 30.0)
    report.check(
        "gc: archive accumulated entries",
        status == 200 and len(entries) >= 3,
        "status=%s entries=%d body=%s" % (status, len(entries), brief(body)),
    )
    check_metadata(report, as_obj(body), archive_dir)
    before = as_obj(body)
    report.check(
        "gc: used_bytes is within the configured budget before GC",
        is_number(before.get("used_bytes"))
        and is_number(before.get("budget_bytes"))
        and before.get("used_bytes") <= before.get("budget_bytes"),
        "used=%r budget=%r" % (before.get("used_bytes"), before.get("budget_bytes")),
    )
    if not entries:
        return report.summary()

    ordered = sorted(
        entries,
        key=lambda entry: (as_int(entry.get("created_ms")), str(entry.get("id"))),
    )
    pinned = ordered[0]
    pinned_id = pinned.get("id")
    path = "/slots/archive/" + urllib.parse.quote(str(pinned_id), safe="")
    status, body = http(base, "POST", path + "?action=pin", {}, timeout)
    obj = as_obj(body)
    report.check(
        "gc: pinning the oldest entry returns 200",
        status == 200 and obj.get("pinned") is True,
        "status=%s body=%s" % (status, brief(obj)),
    )
    report.check(
        "gc: the pin is recorded before GC",
        obj.get("pinned") is True,
        "body=%s" % brief(obj),
    )

    status, body = http(base, "POST", "/slots/archive/gc", {}, timeout)
    gc_obj = as_obj(body)
    report.check(
        "gc: forced GC returns 200",
        status == 200,
        "status=%s body=%s" % (status, brief(gc_obj)),
    )
    report.check(
        "gc: forced GC reports numeric evicted and freed_bytes",
        is_number(gc_obj.get("evicted"))
        and gc_obj.get("evicted") >= 0
        and is_number(gc_obj.get("freed_bytes"))
        and gc_obj.get("freed_bytes") >= 0,
        "body=%s" % brief(gc_obj),
    )

    status, body = http(base, "GET", "/slots/archive", None, timeout)
    obj = as_obj(body)
    report.check(
        "gc: GET /slots/archive returns 200 after GC",
        status == 200,
        "status=%s body=%s" % (status, brief(obj)),
    )
    check_metadata(report, obj, archive_dir)
    used = obj.get("used_bytes")
    budget = obj.get("budget_bytes")
    report.check(
        "gc: used_bytes <= budget after forced GC",
        is_number(used) and is_number(budget) and used <= budget,
        "used=%r budget=%r" % (used, budget),
    )
    after = entries_of(obj)
    surviving = next(
        (entry for entry in after if entry.get("id") == pinned_id), None
    )
    report.check(
        "gc: the pinned entry survives GC",
        surviving is not None and surviving.get("pinned") is True,
        "pinned=%r entries=%s" % (pinned_id, brief(after)),
    )
    analysis = parse_archive(report, archive_dir, "gc", 1)
    if analysis is not None:
        report.check(
            "gc: refcounts still match after GC",
            not analysis["ref_counts"]
            or all(
                as_int(analysis["by_id"].get(row_id, {}).get("refs")) == count
                for row_id, count in analysis["ref_counts"].items()
            ),
            "refs=%s" % brief(analysis["ref_counts"]),
        )
    print(
        "NOTE gc: before=%s bytes, after=%s bytes, budget=%s bytes, evicted=%r"
        % (before.get("used_bytes"), used, budget, gc_obj.get("evicted")),
        flush=True,
    )
    return report.summary()


def run_refcount_before(base, state_path, log_path, timeout):
    report = Report()
    state = read_state(state_path)
    system = state.get("system") or ""
    user_a = state.get("user_a") or ""
    reply_a = state.get("reply_a") or "..."
    first_id = state.get("first_id")
    second_id = state.get("second_id")
    shared_ids = [
        row_id for row_id in state.get("shared_ids", []) if isinstance(row_id, str)
    ]
    archive_dir = state.get("archive_dir")

    status, body = http(base, "GET", "/slots/archive", None, timeout)
    report.check(
        "refcount: GET /slots/archive returns 200 after restart",
        status == 200,
        "status=%s body=%s" % (status, brief(body)),
    )
    listed = [entry.get("id") for entry in entries_of(body)]
    report.check(
        "refcount: both entries survive the restart",
        first_id in listed and second_id in listed,
        "listed=%s first=%r second=%r" % (brief(listed), first_id, second_id),
    )
    analysis = parse_archive(report, archive_dir, "refcount", 2)
    if analysis is not None:
        missing = [
            row_id
            for row_id in shared_ids
            if row_id not in analysis["by_id"]
            or as_int(analysis["by_id"][row_id].get("refs")) < 2
        ]
        report.check(
            "refcount: shared segments keep refs >= 2 after restart",
            bool(shared_ids) and not missing,
            "missing_or_low=%s shared=%s" % (brief(missing), brief(shared_ids)),
        )
        report.check(
            "refcount: shared segment files are present",
            all(
                os.path.isfile(
                    os.path.join(
                        archive_dir, "segments", row_id[:2], row_id + ".seg"
                    )
                )
                for row_id in shared_ids
            ),
            "shared=%s" % brief(shared_ids),
        )

    status, body = chat(base, continuation(system, user_a, reply_a), timeout)
    obj = as_obj(body)
    report.check(
        "refcount: continuation A restores after the restart",
        status == 200 and cached_of(obj) > 0,
        "status=%s cached=%d body=%s" % (status, cached_of(obj), brief(obj)),
    )

    status, body = http(
        base,
        "DELETE",
        "/slots/archive/" + urllib.parse.quote(str(first_id), safe=""),
        None,
        timeout,
    )
    obj = as_obj(body)
    report.check(
        "refcount: DELETE of the first entry returns 200",
        status == 200 and obj.get("deleted") is True,
        "status=%s body=%s" % (status, brief(obj)),
    )
    analysis = parse_archive(report, archive_dir, "refcount after delete", 1)
    if analysis is not None:
        remaining = {
            entry.get("id")
            for entry in analysis["entries"]
            if isinstance(entry.get("id"), str)
        }
        report.check(
            "refcount: only the second entry remains",
            remaining == {second_id},
            "remaining=%s second=%r" % (brief(sorted(remaining)), second_id),
        )
        lost = [
            row_id
            for row_id in shared_ids
            if row_id not in analysis["by_id"]
            or as_int(analysis["by_id"][row_id].get("refs")) <= 0
        ]
        report.check(
            "refcount: shared segments survive deleting the first entry",
            bool(shared_ids) and not lost,
            "lost=%s shared=%s" % (brief(lost), brief(shared_ids)),
        )
    report.check(
        "refcount: server did not panic",
        not log_has_panic(log_path),
        "log=%s" % log_path,
    )
    return report.summary()


def run_refcount_after(base, state_path, log_path, timeout):
    report = Report()
    state = read_state(state_path)
    system = state.get("system") or ""
    user_b = state.get("user_b") or ""
    reply_b = state.get("reply_b") or "..."
    first_id = state.get("first_id")
    second_id = state.get("second_id")
    shared_ids = [
        row_id for row_id in state.get("shared_ids", []) if isinstance(row_id, str)
    ]
    archive_dir = state.get("archive_dir")

    status, body = http(base, "GET", "/slots/archive", None, timeout)
    report.check(
        "refcount: GET /slots/archive returns 200 after the second restart",
        status == 200,
        "status=%s body=%s" % (status, brief(body)),
    )
    listed = [entry.get("id") for entry in entries_of(body)]
    report.check(
        "refcount: the deleted entry stays gone and the survivor remains",
        first_id not in listed and second_id in listed,
        "listed=%s first=%r second=%r" % (brief(listed), first_id, second_id),
    )
    analysis = parse_archive(report, archive_dir, "refcount second restart", 1)
    if analysis is not None:
        missing = [
            row_id
            for row_id in shared_ids
            if row_id not in analysis["by_id"]
            or as_int(analysis["by_id"][row_id].get("refs")) <= 0
        ]
        report.check(
            "refcount: the surviving entry's shared segments remain after restart",
            bool(shared_ids) and not missing,
            "missing=%s shared=%s" % (brief(missing), brief(shared_ids)),
        )
        matching = [
            row_id
            for row_id, count in analysis["ref_counts"].items()
            if as_int(analysis["by_id"].get(row_id, {}).get("refs")) != count
        ]
        report.check(
            "refcount: refs still match the manifest after restart",
            not matching,
            "mismatch=%s" % brief(matching),
        )

    status, body = chat(base, continuation(system, user_b, reply_b), timeout)
    obj = as_obj(body)
    report.check(
        "refcount: survivor B restores after the second restart",
        status == 200 and cached_of(obj) > 0,
        "status=%s cached=%d body=%s" % (status, cached_of(obj), brief(obj)),
    )
    report.check(
        "refcount: server did not panic",
        not log_has_panic(log_path),
        "log=%s" % log_path,
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
    if mode == "prepare":
        if len(argv) < 6:
            print("usage: check.py prepare BASE_URL STATE ARCHIVE TIMEOUT", file=sys.stderr)
            return 2
        return run_prepare(base, argv[3], argv[4], seconds(argv[5], 600.0))
    if mode == "corrupt":
        if len(argv) < 5:
            print("usage: check.py corrupt STATE COPY KIND", file=sys.stderr)
            return 2
        # No URL in this mode: argv[2] is the state file, argv[3] the copy.
        return run_corrupt(argv[2], argv[3], argv[4])
    if mode == "verify":
        if len(argv) < 7:
            print("usage: check.py verify BASE_URL STATE COPY LOG KIND TIMEOUT", file=sys.stderr)
            return 2
        return run_verify(base, argv[3], argv[4], argv[5], argv[6], seconds(argv[7], 600.0))
    if mode == "live-missing":
        if len(argv) < 6:
            print("usage: check.py live-missing BASE_URL STATE LOG TIMEOUT", file=sys.stderr)
            return 2
        return run_live_missing(base, argv[3], argv[4], seconds(argv[5], 600.0))
    if mode == "lease":
        if len(argv) < 6:
            print("usage: check.py lease BASE_URL STATE LOG TIMEOUT", file=sys.stderr)
            return 2
        return run_lease_probe(base, argv[3], argv[4], seconds(argv[5], 600.0))
    if mode == "crash-prepare":
        if len(argv) < 7:
            print("usage: check.py crash-prepare BASE_URL ARCHIVE STATE CHARS TIMEOUT", file=sys.stderr)
            return 2
        return run_crash_prepare(base, argv[3], argv[4], int(argv[5]), seconds(argv[6], 600.0))
    if mode == "checkpoint":
        if len(argv) < 5:
            print("usage: check.py checkpoint BASE_URL STATE TIMEOUT", file=sys.stderr)
            return 2
        return run_checkpoint(base, argv[3], seconds(argv[4], 600.0))
    if mode == "crash-verify":
        if len(argv) < 6:
            print("usage: check.py crash-verify BASE_URL STATE LOG TIMEOUT", file=sys.stderr)
            return 2
        return run_crash_verify(base, argv[3], argv[4], seconds(argv[5], 600.0))
    if mode == "gc":
        if len(argv) < 7:
            print("usage: check.py gc BASE_URL ARCHIVE TURNS CHARS TIMEOUT", file=sys.stderr)
            return 2
        return run_gc(base, argv[3], int(argv[4]), int(argv[5]), seconds(argv[6], 600.0))
    if mode == "refcount-before":
        if len(argv) < 6:
            print("usage: check.py refcount-before BASE_URL STATE LOG TIMEOUT", file=sys.stderr)
            return 2
        return run_refcount_before(base, argv[3], argv[4], seconds(argv[5], 600.0))
    if mode == "refcount-after":
        if len(argv) < 6:
            print("usage: check.py refcount-after BASE_URL STATE LOG TIMEOUT", file=sys.stderr)
            return 2
        return run_refcount_after(base, argv[3], argv[4], seconds(argv[5], 600.0))
    print("unknown mode: %s" % mode, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
PY
}

FAILED=0

echo "test_slot_archive_faults: model=$MODEL"
echo "test_slot_archive_faults: binary=$BIN"
echo "test_slot_archive_faults: server dir=$SERVER_DIR"
echo "test_slot_archive_faults: port=$PORT"
echo "test_slot_archive_faults: work dir=$WORK"
echo

echo "building the baseline archive on 127.0.0.1:$PORT"
start_server "$BASE" "$WORK/server-base.log" 4
PID1=$SERVER_PID
if ! run_py wait "http://127.0.0.1:$PORT" "$WORK/server-base.log" "$PID1" "$LOAD_TIMEOUT"; then
  fail "baseline server did not become ready"
fi
run_py prepare "http://127.0.0.1:$PORT" "$STATE" "$BASE" "$REQUEST_TIMEOUT" || FAILED=1
stop_server "$PID1"
PID1=""
cp -a "$BASE" "$PRISTINE" || fail "could not preserve the pristine baseline archive"

echo
echo "restarting the baseline archive once to check refcount survival"
start_server "$BASE" "$WORK/server-refcount.log" 4
PID1=$SERVER_PID
if ! run_py wait "http://127.0.0.1:$PORT" "$WORK/server-refcount.log" "$PID1" "$LOAD_TIMEOUT"; then
  fail "refcount server did not become ready"
fi
run_py refcount-before "http://127.0.0.1:$PORT" "$STATE" "$WORK/server-refcount.log" "$REQUEST_TIMEOUT" || FAILED=1
stop_server "$PID1"
PID1=""

echo
echo "restarting after deleting the first entry to check the survivor's refcounts"
start_server "$BASE" "$WORK/server-refcount2.log" 4
PID1=$SERVER_PID
if ! run_py wait "http://127.0.0.1:$PORT" "$WORK/server-refcount2.log" "$PID1" "$LOAD_TIMEOUT"; then
  fail "second refcount server did not become ready"
fi
run_py refcount-after "http://127.0.0.1:$PORT" "$STATE" "$WORK/server-refcount2.log" "$REQUEST_TIMEOUT" || FAILED=1
stop_server "$PID1"
PID1=""

mkdir -p "$CASES"

for kind in manifest segments segment state missing; do
  echo
  echo "fault case: $kind"
  copy="$CASES/$kind"
  cp -a "$PRISTINE" "$copy" || fail "could not copy the pristine archive to $copy"
  run_py corrupt "$STATE" "$copy" "$kind" || FAILED=1
  start_server "$copy" "$WORK/server-$kind.log" 4
  PID1=$SERVER_PID
  if ! run_py wait "http://127.0.0.1:$PORT" "$WORK/server-$kind.log" "$PID1" "$LOAD_TIMEOUT"; then
    fail "$kind server did not become ready"
  fi
  run_py verify "http://127.0.0.1:$PORT" "$STATE" "$copy" "$WORK/server-$kind.log" "$kind" "$REQUEST_TIMEOUT" || FAILED=1
  stop_server "$PID1"
  PID1=""
done

echo
echo "fault case: a segment deleted while the server is running"
copy="$CASES/live-missing"
cp -a "$PRISTINE" "$copy" || fail "could not copy the pristine archive to $copy"
# The live case needs the archive path in its state; the baseline copy has it.
python3 - "$STATE" "$copy" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as file:
    state = json.load(file)
state["archive_dir"] = sys.argv[2]
with open(sys.argv[1], "w", encoding="utf-8") as file:
    json.dump(state, file)
PY
start_server "$copy" "$WORK/server-live-missing.log" 4
PID1=$SERVER_PID
if ! run_py wait "http://127.0.0.1:$PORT" "$WORK/server-live-missing.log" "$PID1" "$LOAD_TIMEOUT"; then
  fail "live-missing server did not become ready"
fi
run_py live-missing "http://127.0.0.1:$PORT" "$STATE" "$WORK/server-live-missing.log" "$REQUEST_TIMEOUT" || FAILED=1
stop_server "$PID1"
PID1=""

echo
echo "lease probe: DELETE while a restore is in flight"
copy="$CASES/lease"
cp -a "$PRISTINE" "$copy" || fail "could not copy the pristine archive to $copy"
python3 - "$STATE" "$copy" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as file:
    state = json.load(file)
state["archive_dir"] = sys.argv[2]
with open(sys.argv[1], "w", encoding="utf-8") as file:
    json.dump(state, file)
PY
# Put the lease probe on a fresh copy of the baseline: the refcount phase has
# already deleted the first entry from $BASE.
start_server "$copy" "$WORK/server-lease.log" 4
PID1=$SERVER_PID
if ! run_py wait "http://127.0.0.1:$PORT" "$WORK/server-lease.log" "$PID1" "$LOAD_TIMEOUT"; then
  fail "lease server did not become ready"
fi
run_py lease "http://127.0.0.1:$PORT" "$STATE" "$WORK/server-lease.log" "$REQUEST_TIMEOUT" || FAILED=1
stop_server "$PID1"
PID1=""

echo
echo "crash case: SIGKILL during an explicit checkpoint"
CRASH="$WORK/crash"
cp -a "$PRISTINE" "$CRASH" || fail "could not copy the archive for the crash case"
# The crash copy is still the original two-entry archive; the state path just
# needs to point at it for the HTTP helper.
python3 - "$STATE" "$CRASH" "$STATE_CRASH" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as file:
    state = json.load(file)
state["archive_dir"] = sys.argv[2]
with open(sys.argv[3], "w", encoding="utf-8") as file:
    json.dump(state, file)
PY
start_server "$CRASH" "$WORK/server-crash1.log" 4
PID1=$SERVER_PID
if ! run_py wait "http://127.0.0.1:$PORT" "$WORK/server-crash1.log" "$PID1" "$LOAD_TIMEOUT"; then
  fail "crash server did not become ready"
fi
run_py crash-prepare "http://127.0.0.1:$PORT" "$CRASH" "$STATE_CRASH" "$CRASH_CHARS" "$REQUEST_TIMEOUT" || FAILED=1

# This client is expected to die with the server; its report is intentionally
# not counted in the aggregate (FAULTS_RESULTS=/dev/null), and its output goes
# to a log for the NOTE after the kill.
FAULTS_RESULTS=/dev/null run_py checkpoint \
  "http://127.0.0.1:$PORT" "$STATE_CRASH" "$REQUEST_TIMEOUT" \
  >"$WORK/checkpoint-client.log" 2>&1 &
CLIENT=$!
TEMP_SEEN=0
for ((i = 0; i < 400; i++)); do
  if find "$CRASH" -type f \( -name '*.tmp' -o -name '*.tmp.*' \) -print -quit 2>/dev/null | grep -q .; then
    TEMP_SEEN=1
    break
  fi
  sleep 0.01
done
if (( TEMP_SEEN == 1 )); then
  echo "NOTE crash: observed a writer temp file; sending SIGKILL"
else
  echo "NOTE crash: no temp file observed within 4s; sending SIGKILL anyway"
fi
kill -9 "$PID1" 2>/dev/null || true
wait "$PID1" 2>/dev/null || true
PID1=""
kill "$CLIENT" 2>/dev/null || true
wait "$CLIENT" 2>/dev/null || true

echo "restarting the crashed archive"
start_server "$CRASH" "$WORK/server-crash2.log" 4
PID1=$SERVER_PID
if ! run_py wait "http://127.0.0.1:$PORT" "$WORK/server-crash2.log" "$PID1" "$LOAD_TIMEOUT"; then
  fail "crashed archive did not restart"
fi
run_py crash-verify "http://127.0.0.1:$PORT" "$STATE_CRASH" "$WORK/server-crash2.log" "$REQUEST_TIMEOUT" || FAILED=1
stop_server "$PID1"
PID1=""

echo
echo "small-budget GC case: budget 2 GiB, pin cap 1 GiB"
GC_DIR="$WORK/gc"
start_server "$GC_DIR" "$WORK/server-gc.log" 2 --slot-archive-pin-gb 1
PID1=$SERVER_PID
if ! run_py wait "http://127.0.0.1:$PORT" "$WORK/server-gc.log" "$PID1" "$LOAD_TIMEOUT"; then
  fail "GC server did not become ready"
fi
run_py gc "http://127.0.0.1:$PORT" "$GC_DIR" "$GC_TURNS" "$GC_CHARS" "$REQUEST_TIMEOUT" || FAILED=1
stop_server "$PID1"
PID1=""

echo
if [[ -s "$RESULTS" ]]; then
  read -r TOTAL_CHECKS FAILED_CHECKS <<<"$(awk '{t += $1; f += $2} END {print t + 0, f + 0}' "$RESULTS")"
  echo "test_slot_archive_faults: $TOTAL_CHECKS checks, $FAILED_CHECKS failed"
  (( FAILED_CHECKS == 0 )) || FAILED=1
fi

exit "$FAILED"
