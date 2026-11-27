#!/usr/bin/env bash
# End-to-end test for cross-conversation KV segment sharing (docs/SLOT-ARCHIVE-PHASE2.md §2.3).
#
# It starts a single-slot server with a small archive directory and chats with
# three conversations that share a long system prompt S but have distinct short
# user messages A, B and C:
#
#   [S, A]  -> fills the slot
#   [S, B]  -> diverges at the user message, so A's slot is checkpointed
#   [S, C]  -> diverges again, so B's slot is checkpointed
#
# That must leave at least two v2 entries in the single slot, both reporting
# format 2 in GET /slots/archive. The test then reads the archive on disk and
# proves that the entries share content-addressed KV segments:
#
#   * at least one row in <ARCHIVE>/segments.json is referenced by both entries
#     (refs >= 2);
#   * the number of unique segment rows is smaller than the sum of the
#     `segments` array lengths of the v2 manifest entries, so the bytes are
#     shared instead of written twice;
#   * every manifest segment range is non-negative, contiguous, resolves to a
#     real segment file, and the row refcount equals the number of entries that
#     reference it (no mismatch, no lost or negative refs).
#
# It continues A and then B with an assistant reply and a "continue" turn. Both
# must answer 200 and report cached_tokens > 0, which is only possible after the
# archive restored the shared segments. The continuations themselves checkpoint
# the conversation that was in the slot (C, then A'), so those extra entries are
# removed before the delete check, leaving only the two original entries.
#
# It then DELETEs the first (older) entry. The shared segment rows must still be
# in segments.json with refs > 0, and the surviving second entry must still list
# them. A different conversation D is sent to take the slot away, then B is
# continued again: 200 with cached_tokens > 0 proves the surviving entry still
# restores from the shared segments after the other entry is gone.
#
# Finally the server is restarted on the same archive directory and B is
# continued once more. Nothing holds B in a slot after the restart, so a cached
# prefix can only come from the segments and the state file on disk. This
# checks that the segment references and the archive survive a restart.
#
# Usage: tools/test_slot_archive_dedup.sh MODEL BIN SERVER_DIR [PORT]
#
#   MODEL       GGUF model to serve.
#   BIN         rustingstrata binary (a path, or a name on PATH).
#   SERVER_DIR  directory the server runs in.
#   PORT        port of the archive-enabled server (default 8080).
#
# Environment:
#   LOAD_TIMEOUT     seconds to wait for a model load (default 600).
#   REQUEST_TIMEOUT  seconds per HTTP test request (default 600).
#
# Requires python3 (stdlib urllib only), a CUDA GPU, and enough VRAM for
# --parallel 1 with a 64k q8 KV cache. The archive must use the v2 segmented
# entry format: the test fails with the observed `format` values if the server
# falls back to v1 files. V1 archive entries remain readable by the server but
# they cannot share KV bytes, so this test is specifically about v2.
#
# Assumptions for a live run: the model's chat template renders the system
# message (the first request checks the prompt-token count) and is
# deterministic. A template that injects a date or other per-request text
# before the system prompt would tokenize differently and defeat the prefix
# sharing this test is looking for. All chats use the same owner (no API
# keys), the same model, and the same KV layout.

set -uo pipefail

usage() {
  cat <<'EOF'
Usage: tools/test_slot_archive_dedup.sh MODEL BIN SERVER_DIR [PORT]

  MODEL       GGUF model to serve
  BIN         rustingstrata binary (path or name on PATH)
  SERVER_DIR  directory the server runs in
  PORT        archive-enabled server port (default 8080)

Environment: LOAD_TIMEOUT (default 600), REQUEST_TIMEOUT (default 600)
EOF
}

fail() {
  echo "test_slot_archive_dedup: $*" >&2
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

WORK=$(mktemp -d "${TMPDIR:-/tmp}/strata-slot-dedup.XXXXXX") \
  || fail "could not create a temporary directory"
ARCHIVE="$WORK/archive"
STATE="$WORK/state.json"
LOG1="$WORK/server-archive.log"
LOG2="$WORK/server-restart.log"

PID1=""
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
  rm -rf -- "$WORK"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

start_server() {
  local port=$1 log=$2
  # A single slot forces a divergence checkpoint whenever the next
  # conversation replaces the slot history. The low thresholds keep both
  # entries in the archive, so their shared segments can be observed.
  ( cd "$SERVER_DIR" && exec "$BIN" serve \
      --model "$MODEL" --gpu --backend cuda --parallel 1 --ctx 65536 \
      --kv q8 --slot-archive-dir "$ARCHIVE" --slot-archive-gb 5 \
      --slot-archive-min-tokens 64 --slot-archive-min-delta 1 \
      --addr "127.0.0.1:$port" ) >"$log" 2>&1 &
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
            return 1
        print("all %d checks passed" % self.total)
        return 0


# The long system prompt every conversation shares. The paragraph is repeated
# to ~12k characters (roughly 3k tokens), so the first full 1024-token chunks of
# A and B are identical and must resolve to the same content-addressed segments.
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
CONTINUE = "Continue with one short sentence."


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


def check_metadata(report, obj, archive_dir):
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


def parse_archive(report, archive_dir, label="", min_entries=2):
    """Read manifest.json + segments.json and validate the v2 reference graph.

    Returns a dict with the manifest entries, the segment rows keyed by id, the
    per-entry reference counts and the shared ids, or None when the files are
    not readable. This reads the files the server wrote, not the HTTP API, so
    it checks the format and refcounts directly.
    """
    prefix = label + ": " if label else ""
    manifest = None
    segments = None
    last = "no files"
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        manifest = load_json(os.path.join(archive_dir, "manifest.json"))
        segments = load_json(os.path.join(archive_dir, "segments.json"))
        if isinstance(manifest, dict) and isinstance(segments, dict):
            mrows = manifest.get("entries")
            srows = segments.get("segments")
            last = "manifest entries=%r segment rows=%r" % (
                len(mrows) if isinstance(mrows, list) else mrows,
                len(srows) if isinstance(srows, list) else srows,
            )
            if (
                isinstance(mrows, list)
                and len(mrows) >= min_entries
                and isinstance(srows, list)
                and srows
            ):
                break
        else:
            last = "missing or invalid JSON"
        time.sleep(0.2)

    ok = isinstance(manifest, dict) and isinstance(segments, dict)
    report.check(
        prefix + "manifest.json and segments.json are readable",
        ok,
        "dir=%s %s" % (archive_dir, last),
    )
    if not ok:
        return None
    if not isinstance(manifest.get("entries"), list) or not isinstance(
        segments.get("segments"), list
    ):
        report.check(prefix + "archive JSON shapes are valid", False, last)
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

    entries = [entry for entry in manifest["entries"] if isinstance(entry, dict)]
    rows = [row for row in segments["segments"] if isinstance(row, dict)]
    report.check(
        prefix + "manifest lists at least %d entries" % min_entries,
        len(entries) >= min_entries,
        "entries=%d" % len(entries),
    )

    formats = sorted({str(entry.get("format")) for entry in entries})
    report.check(
        prefix + "archive entries are v2 segmented entries (format 2)",
        bool(entries) and all(as_int(entry.get("format")) == 2 for entry in entries),
        "expected format 2, got formats=%r" % (formats,),
    )

    ids = [row.get("id") for row in rows]
    string_ids = [value for value in ids if isinstance(value, str)]
    ids_ok = (
        bool(rows)
        and len(string_ids) == len(ids)
        and len(set(string_ids)) == len(string_ids)
        and all(is_id(value) for value in string_ids)
    )
    report.check(
        prefix + "segment ids are unique 16-hex ids",
        ids_ok,
        "rows=%d ids=%s" % (len(rows), brief(ids[:6])),
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
        segments_list = entry_segments(entry)
        if not segments_list:
            bad_refs.append("entry %r has no segments" % (entry.get("id"),))
        for segment in segments_list:
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

    sum_refs = sum(ref_counts.values())
    shared = sorted(
        row_id for row_id, count in ref_counts.items() if count >= 2
    )
    return {
        "entries": entries,
        "by_id": by_id,
        "ref_counts": ref_counts,
        "sum_refs": sum_refs,
        "unique": len(by_id),
        "shared": shared,
    }


def check_dedup(report, analysis, label=""):
    prefix = label + ": " if label else ""
    shared = analysis["shared"]
    refs_ok = (
        bool(shared)
        and all(
            as_int(analysis["by_id"].get(row_id, {}).get("refs")) >= 2
            for row_id in shared
        )
    )
    report.check(
        prefix + "a segment is shared by both entries",
        bool(shared),
        "shared=%s rows=%d" % (brief(shared), analysis["unique"]),
    )
    report.check(
        prefix + "shared segment rows report refs >= 2",
        refs_ok,
        "refs=%s"
        % brief(
            {
                row_id: analysis["by_id"].get(row_id, {}).get("refs")
                for row_id in shared[:4]
            }
        ),
    )
    report.check(
        prefix + "unique segment rows are fewer than per-entry segments",
        analysis["unique"] < analysis["sum_refs"],
        "rows=%d per_entry=%d" % (analysis["unique"], analysis["sum_refs"]),
    )
    print("NOTE shared segments: %s" % brief(shared, 400), flush=True)


def order_entries(entries):
    return sorted(
        entries,
        key=lambda entry: (as_int(entry.get("created_ms")), str(entry.get("id"))),
    )


def check_continuation(report, base, system, user, assistant_reply, label, timeout):
    status, body = chat(base, continuation(system, user, assistant_reply), timeout)
    obj = as_obj(body)
    cached = cached_of(obj)
    report.check(
        label + " returns 200",
        status == 200,
        "status=%s body=%s" % (status, brief(obj)),
    )
    report.check(
        label + " restores a cached prefix",
        status == 200 and cached > 0,
        "cached_tokens=%d body=%s" % (cached, brief(obj)),
    )
    return cached


def delete_entry(base, entry_id, timeout):
    path = "/slots/archive/" + urllib.parse.quote(entry_id, safe="")
    return http(base, "DELETE", path, None, timeout)


def run_full(base, state_path, archive_dir, timeout):
    report = Report()
    system = shared_system()
    report.check(
        "shared system prompt is long enough for several segments",
        len(system) >= 8000,
        "chars=%d" % len(system),
    )

    # A fills the only slot. The prompt-token count proves the chat template
    # rendered the system message instead of silently dropping it: without it
    # the prompt would be a few dozen tokens, not thousands.
    status, body = chat(base, user_turn(system, USER_A), timeout)
    obj = as_obj(body)
    reply_a = reply_of(obj)
    report.check(
        "conversation A answered",
        status == 200 and bool(reply_a),
        "status=%s reply=%s" % (status, brief(reply_a)),
    )
    prompt_tokens = prompt_tokens_of(obj)
    report.check(
        "the chat template includes the system message",
        prompt_tokens > 500,
        "prompt_tokens=%d (a dropped system prompt would be far smaller)"
        % prompt_tokens,
    )

    status, body = chat(base, user_turn(system, USER_B), timeout)
    obj = as_obj(body)
    reply_b = reply_of(obj)
    report.check(
        "diverging conversation B answered",
        status == 200 and bool(reply_b),
        "status=%s reply=%s" % (status, brief(reply_b)),
    )

    # B cannot extend A's history in the only slot, so A is checkpointed
    # before B is prefilled. C then checkpoints B the same way.
    status, body = chat(base, user_turn(system, USER_C), timeout)
    obj = as_obj(body)
    report.check(
        "diverging conversation C answered",
        status == 200,
        "status=%s body=%s" % (status, brief(obj)),
    )

    status, body, entries = wait_entries(base, 2, 30.0)
    report.check(
        "GET /slots/archive lists at least two entries",
        status == 200 and len(entries) >= 2,
        "status=%s entries=%d body=%s" % (status, len(entries), brief(body)),
    )
    check_metadata(report, as_obj(body), archive_dir)

    formats = [entry.get("format") for entry in entries]
    report.check(
        "every listed archive entry is v2 (format 2)",
        bool(entries) and all(as_int(value) == 2 for value in formats),
        "expected format 2, got formats=%r" % (formats,),
    )
    v2_api = [entry for entry in entries if as_int(entry.get("format")) == 2]
    report.check(
        "at least two v2 entries are listed",
        len(v2_api) >= 2,
        "entries=%d formats=%r" % (len(v2_api), formats),
    )
    bad_meta = [
        entry.get("id")
        for entry in v2_api
        if not (
            is_int(entry.get("n_tokens"))
            and entry.get("n_tokens") > 0
            and is_int(entry.get("bytes"))
            and entry.get("bytes") > 0
        )
    ]
    report.check(
        "v2 entries report numeric n_tokens and bytes",
        bool(v2_api) and not bad_meta,
        "bad=%s entries=%s" % (brief(bad_meta), brief(v2_api)),
    )

    analysis = parse_archive(report, archive_dir, "", 2)
    if analysis is None:
        return report.summary()
    check_dedup(report, analysis)

    # Chats run one after another in the single slot, so the older entry is A
    # (checkpointed when B arrived) and the newer one is B (checkpointed when C
    # arrived). The continuation below replays B's exact assistant reply for
    # the surviving entry.
    ordered = order_entries(analysis["entries"])
    if len(ordered) < 2:
        report.check("two original entries identified", False, "entries=%s" % brief(ordered))
        return report.summary()
    first = ordered[0]
    second = ordered[1]
    first_id = first.get("id")
    second_id = second.get("id")
    shared_ids = list(analysis["shared"])
    second_n_tokens = as_int(second.get("n_tokens"))
    print(
        "NOTE original entries: first id=%r n_tokens=%r; second id=%r n_tokens=%r"
        % (first_id, first.get("n_tokens"), second_id, second.get("n_tokens")),
        flush=True,
    )
    report.check(
        "the two original entries have ids",
        isinstance(first_id, str) and bool(first_id)
        and isinstance(second_id, str) and bool(second_id),
        "first=%r second=%r" % (first_id, second_id),
    )

    # Restore each conversation from its archive entry. The slot currently
    # holds C, so the cached prefix of A cannot come from the slot; after A is
    # continued the slot holds A', so B's cached prefix must come from B's
    # entry. Both entries use the same shared segments.
    cached_a = check_continuation(
        report, base, system, USER_A, reply_a, "continuation A", timeout
    )
    cached_b = check_continuation(
        report, base, system, USER_B, reply_b, "continuation B", timeout
    )
    print(
        "NOTE continuation cached tokens: A=%d (first entry n_tokens=%r), "
        "B=%d (second entry n_tokens=%r)"
        % (cached_a, first.get("n_tokens"), cached_b, second_n_tokens),
        flush=True,
    )

    # Each continuation checkpoints the conversation it replaced (C, then A').
    # Those extra entries would also reference the shared S segments, so remove
    # them before the delete check. Afterwards only the two original entries
    # refer to the shared segments, which makes the next assertion precise.
    status, body = http(base, "GET", "/slots/archive", None, timeout)
    live = entries_of(body)
    keep = {first_id, second_id}
    extras = [entry for entry in live if entry.get("id") not in keep]
    print(
        "NOTE deleting %d continuation checkpoint(s) before the delete check"
        % len(extras),
        flush=True,
    )
    delete_failures = []
    for extra in extras:
        extra_id = extra.get("id")
        if not isinstance(extra_id, str) or not extra_id:
            delete_failures.append((extra_id, "missing id", ""))
            continue
        status, body = delete_entry(base, extra_id, timeout)
        if status != 200:
            delete_failures.append((extra_id, status, brief(body)))
    report.check(
        "DELETE of continuation checkpoints returns 200",
        not delete_failures,
        "failures=%s" % brief(delete_failures),
    )

    cleaned = parse_archive(report, archive_dir, "after cleanup", 2)
    if cleaned is None:
        return report.summary()
    remaining = {
        entry.get("id")
        for entry in cleaned["entries"]
        if isinstance(entry.get("id"), str)
    }
    report.check(
        "only the two original entries remain before the delete",
        remaining == keep,
        "remaining=%s want=%s" % (brief(sorted(remaining)), brief(sorted(keep))),
    )

    # Delete the older entry. The shared segments must stay because the second
    # entry still references them.
    status, body = delete_entry(base, first_id, timeout)
    obj = as_obj(body)
    report.check(
        "DELETE of the first entry returns 200",
        status == 200,
        "status=%s body=%s" % (status, brief(obj)),
    )
    report.check(
        "DELETE of the first entry reports deleted=true",
        obj.get("deleted") is True,
        "body=%s" % brief(obj),
    )
    report.check(
        "DELETE of the first entry reports numeric freed_bytes",
        is_number(obj.get("freed_bytes")) and obj.get("freed_bytes") >= 0,
        "freed_bytes=%r body=%s" % (obj.get("freed_bytes"), brief(obj)),
    )

    after = parse_archive(report, archive_dir, "after deleting the first entry", 1)
    if after is None:
        return report.summary()
    after_entries = {
        entry.get("id")
        for entry in after["entries"]
        if isinstance(entry.get("id"), str)
    }
    report.check(
        "the first entry is gone from the manifest",
        first_id not in after_entries,
        "first=%r entries=%s" % (first_id, brief(sorted(after_entries))),
    )
    missing_rows = [
        row_id
        for row_id in shared_ids
        if row_id not in after["by_id"]
        or as_int(after["by_id"][row_id].get("refs")) <= 0
    ]
    report.check(
        "shared segments survive deleting the first entry",
        bool(shared_ids) and not missing_rows,
        "missing_or_zero_refs=%s shared=%s"
        % (brief(missing_rows), brief(shared_ids)),
    )
    second_entry = next(
        (
            entry
            for entry in after["entries"]
            if entry.get("id") == second_id
        ),
        None,
    )
    report.check(
        "the second entry survives deleting the first",
        second_entry is not None,
        "second=%r entries=%s" % (second_id, brief(sorted(after_entries))),
    )
    if second_entry is not None:
        referenced = {
            segment.get("id")
            for segment in entry_segments(second_entry)
            if isinstance(segment.get("id"), str)
        }
        unreferenced = [row_id for row_id in shared_ids if row_id not in referenced]
        report.check(
            "the second entry still references every shared segment",
            bool(shared_ids) and not unreferenced,
            "unreferenced=%s shared=%s"
            % (brief(unreferenced), brief(shared_ids)),
        )

    # Take the slot away from B with a different conversation, then continue B.
    # D is checkpointed when B arrives, so the cached prefix of B cannot come
    # from the slot: it must be restored from the surviving entry and its
    # shared segments.
    status, body = chat(base, user_turn(system, USER_D), timeout)
    report.check(
        "conversation D takes the slot",
        status == 200 and bool(reply_of(as_obj(body))),
        "status=%s body=%s" % (status, brief(body)),
    )
    check_continuation(
        report,
        base,
        system,
        USER_B,
        reply_b,
        "continuation B after the delete",
        timeout,
    )

    write_state(
        state_path,
        {
            "archive_dir": archive_dir,
            "system": system,
            "user_b": USER_B,
            "reply_b": reply_b,
            "shared_ids": shared_ids,
            "second_id": second_id,
            "second_n_tokens": second_n_tokens,
        },
    )
    return report.summary()


def run_restart(base, state_path, timeout):
    report = Report()
    state = read_state(state_path)
    system = state.get("system")
    user_b = state.get("user_b")
    reply_b = state.get("reply_b")
    shared_ids = [
        row_id for row_id in state.get("shared_ids", []) if isinstance(row_id, str)
    ]
    second_id = state.get("second_id")
    archive_dir = state.get("archive_dir")

    state_ok = (
        isinstance(system, str)
        and bool(system)
        and isinstance(user_b, str)
        and bool(user_b)
        and isinstance(reply_b, str)
    )
    report.check(
        "restart state holds the continuation prompt",
        state_ok,
        "state=%s" % brief(state, 300),
    )
    if not state_ok:
        return report.summary()

    status, body = http(base, "GET", "/slots/archive", None, timeout)
    obj = as_obj(body)
    report.check(
        "restart GET /slots/archive returns 200",
        status == 200,
        "status=%s body=%s" % (status, brief(obj)),
    )
    if archive_dir is not None:
        check_metadata(report, obj, archive_dir)

    entries = entries_of(obj)
    listed = next(
        (entry for entry in entries if entry.get("id") == second_id), None
    )
    report.check(
        "restart still lists the second entry",
        listed is not None,
        "second=%r entries=%s" % (second_id, brief(obj)),
    )
    if listed is not None:
        report.check(
            "restart entry is still v2 (format 2)",
            as_int(listed.get("format")) == 2,
            "format=%r entry=%s" % (listed.get("format"), brief(listed)),
        )
        report.positive(
            "restart entry keeps n_tokens", as_int(listed.get("n_tokens"))
        )
        report.positive("restart entry keeps bytes", as_int(listed.get("bytes")))

    archive = None
    if isinstance(archive_dir, str) and archive_dir:
        archive = parse_archive(report, archive_dir, "restart", 1)
    if archive is not None:
        rows = archive["by_id"]
        missing_rows = [
            row_id
            for row_id in shared_ids
            if row_id not in rows or as_int(rows[row_id].get("refs")) <= 0
        ]
        report.check(
            "shared segments survive the restart",
            bool(shared_ids) and not missing_rows,
            "missing_or_zero_refs=%s shared=%s"
            % (brief(missing_rows), brief(shared_ids)),
        )
        second_entry = next(
            (
                entry
                for entry in archive["entries"]
                if entry.get("id") == second_id
            ),
            None,
        )
        report.check(
            "the restarted archive keeps the second entry in the manifest",
            second_entry is not None,
            "second=%r entries=%s"
            % (
                second_id,
                brief(
                    sorted(
                        entry.get("id")
                        for entry in archive["entries"]
                        if isinstance(entry.get("id"), str)
                    )
                ),
            ),
        )
        if second_entry is not None:
            referenced = {
                segment.get("id")
                for segment in entry_segments(second_entry)
                if isinstance(segment.get("id"), str)
            }
            unreferenced = [
                row_id for row_id in shared_ids if row_id not in referenced
            ]
            report.check(
                "the restarted second entry still references the shared segments",
                bool(shared_ids) and not unreferenced,
                "unreferenced=%s shared=%s"
                % (brief(unreferenced), brief(shared_ids)),
            )

    # No slot holds B after the restart: a cached prefix can only come from the
    # segments and the state file on disk.
    check_continuation(
        report,
        base,
        system,
        user_b,
        reply_b,
        "restart continuation B",
        timeout,
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
    print("unknown mode: %s" % mode, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
PY
}

FAILED=0

echo "test_slot_archive_dedup: model=$MODEL"
echo "test_slot_archive_dedup: binary=$BIN"
echo "test_slot_archive_dedup: server dir=$SERVER_DIR"
echo "test_slot_archive_dedup: port=$PORT"
echo "test_slot_archive_dedup: temporary archive dir=$ARCHIVE"
echo

echo "starting archive server on 127.0.0.1:$PORT"
start_server "$PORT" "$LOG1"
PID1=$SERVER_PID
if ! run_py wait "http://127.0.0.1:$PORT" "$LOG1" "$PID1" "$LOAD_TIMEOUT"; then
  fail "archive server did not become ready"
fi

run_py full "http://127.0.0.1:$PORT" "$STATE" "$ARCHIVE" "$REQUEST_TIMEOUT" || FAILED=1
stop_server "$PID1"
PID1=""

echo
echo "restarting archive server on 127.0.0.1:$PORT with the same directory"
start_server "$PORT" "$LOG2"
PID1=$SERVER_PID
if ! run_py wait "http://127.0.0.1:$PORT" "$LOG2" "$PID1" "$LOAD_TIMEOUT"; then
  fail "restarted archive server did not become ready"
fi

run_py restart "http://127.0.0.1:$PORT" "$STATE" "$REQUEST_TIMEOUT" || FAILED=1
stop_server "$PID1"
PID1=""

exit "$FAILED"
