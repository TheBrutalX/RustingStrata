#!/usr/bin/env bash
# End-to-end test for speculative decoding with --parallel > 1
# (docs/PARALLEL-SPEC.md).
#
# It starts a no-spec server with --parallel 3, sends a deterministic
# repeated-text prompt, records the reply and the decode speed, then starts a
# server with --spec-type ngram-map-k4v on the same flags and sends the same
# request with the same seed. It requires a byte-identical greedy reply, a
# non-zero accepted-draft count and a decode speed no worse than the no-spec
# run (within a small tolerance for timer noise). On the spec server it also
# sends three concurrent requests, which must all return 200 with a non-empty
# reply, proving that the slots work independently.
#
# Usage: tools/test_parallel_spec.sh MODEL BIN SERVER_DIR [PORT]
#
#   MODEL       GGUF model to serve (one with enough VRAM for --parallel 3).
#   BIN         rustingstrata binary (a path, or a name on PATH).
#   SERVER_DIR  directory the server runs in.
#   PORT        port of the first server (default 8090); the spec server uses
#               PORT + 1.
#
# Environment:
#   LOAD_TIMEOUT     seconds to wait for a model load (default 600).
#   REQUEST_TIMEOUT  seconds per HTTP test request (default 600).
#   SPEC_SPEED_MIN   minimum spec/no-spec decode ratio (default 0.90).
#
# Requires python3 (stdlib urllib only), a CUDA GPU, and enough VRAM for
# --parallel 3 with a 32k q8 KV cache.

set -uo pipefail

usage() {
  cat <<'EOF'
Usage: tools/test_parallel_spec.sh MODEL BIN SERVER_DIR [PORT]

  MODEL       GGUF model to serve
  BIN         rustingstrata binary (path or name on PATH)
  SERVER_DIR  directory the server runs in
  PORT        first server port (default 8090); the spec server uses PORT + 1

Environment: LOAD_TIMEOUT (default 600), REQUEST_TIMEOUT (default 600),
             SPEC_SPEED_MIN (default 0.90)
EOF
}

fail() {
  echo "test_parallel_spec: $*" >&2
  exit 1
}

if (( $# < 3 || $# > 4 )); then
  usage >&2
  exit 2
fi

MODEL_ARG=$1
BIN_ARG=$2
SERVER_DIR_ARG=$3
PORT=${4:-8090}
LOAD_TIMEOUT=${LOAD_TIMEOUT:-600}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-600}
SPEC_SPEED_MIN=${SPEC_SPEED_MIN:-0.90}

[[ "$PORT" =~ ^[0-9]+$ ]] || fail "PORT must be a number: $PORT"
(( PORT >= 1 && PORT <= 65533 )) || fail "PORT must be between 1 and 65533"
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

[[ -f "$MODEL" ]] || fail "model not found: $MODEL"
[[ -n "$BIN" && -x "$BIN" ]] || fail "binary not found or not executable: $BIN_ARG"
[[ -d "$SERVER_DIR" ]] || fail "server dir not found: $SERVER_DIR"
command -v python3 >/dev/null 2>&1 || fail "python3 is required"

TMP=$(mktemp -d "${TMPDIR:-/tmp}/parallel-spec.XXXXXX") || fail "mktemp failed"
SERVER_PID=
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  rm -rf "$TMP"
}
trap cleanup EXIT

PASS=0
CHECKS=0
ok() {
  PASS=$((PASS + 1))
  echo "ok $PASS - $*"
}
check() {
  CHECKS=$((CHECKS + 1))
  if ! "$@"; then
    fail "check failed after $PASS passed: $*"
  fi
}

# start_server LOG SPEC_ARGS...: run `serve` in the server dir and wait for
# /health. SPEC_ARGS is empty for the no-spec run.
start_server() {
  local log=$1 port=$2
  shift 2
  ( cd "$SERVER_DIR" && exec "$BIN" serve \
      --model "$MODEL" --mmproj none --gpu --backend cuda \
      --parallel 3 --ctx 32768 --kv q8 \
      --addr "127.0.0.1:$port" "$@" ) >"$log" 2>&1 &
  SERVER_PID=$!
  local waited=0
  while (( waited < LOAD_TIMEOUT )); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      tail -40 "$log" >&2 || true
      fail "server on port $port exited during load"
    fi
    if python3 - "$port" <<'PY' >/dev/null 2>&1
import sys, urllib.request
try:
    with urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/health", timeout=2) as r:
        sys.exit(0 if r.status == 200 else 1)
except Exception:
    sys.exit(1)
PY
    then
      return 0
    fi
    sleep 1
    waited=$((waited + 1))
  done
  tail -40 "$log" >&2 || true
  fail "server on port $port did not become healthy in ${LOAD_TIMEOUT}s"
}

stop_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=
}

# send_request PORT MAX_TOKENS SEED PREFIX OUT: one /v1/chat/completions call.
# Writes the reply text to OUT and prints "status wall completion_tokens".
send_request() {
  python3 - "$1" "$2" "$3" "$4" "$REQUEST_TIMEOUT" "$5" <<'PY'
import json, sys, time, urllib.request
port, max_tokens, seed, prefix, timeout, out = sys.argv[1:]
phrase = "alpha beta gamma delta epsilon zeta eta theta. "
text = (
    f"{prefix}Repeat the following sequence exactly twelve times, then stop.\n\n"
    f"Sequence: {phrase * 3}\n\nOutput:\n"
)
body = json.dumps({
    "messages": [{"role": "user", "content": text}],
    "max_tokens": int(max_tokens), "temperature": 0, "stream": False,
    "seed": int(seed), "chat_template_kwargs": {"enable_thinking": False},
}).encode()
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
    headers={"Content-Type": "application/json"}, method="POST")
t = time.time()
try:
    with urllib.request.urlopen(req, timeout=float(timeout)) as r:
        d = json.load(r)
        status = r.status
except Exception as e:
    print(f"request failed: {e}", file=sys.stderr)
    sys.exit(1)
wall = time.time() - t
reply = d["choices"][0]["message"].get("content") or ""
usage = d.get("usage") or {}
if status != 200 or not reply:
    print(f"bad reply: status={status} empty={not reply}", file=sys.stderr)
    sys.exit(1)
with open(out, "w") as f:
    f.write(reply)
print(status, round(wall, 2), usage.get("completion_tokens", 0))
PY
}

# send_concurrent PORT OUT_DIR: three parallel requests; fails unless all are 200.
send_concurrent() {
  python3 - "$1" "$2" "$REQUEST_TIMEOUT" <<'PY'
import json, sys, threading, urllib.request
port, out_dir, timeout = sys.argv[1:]
phrase = "alpha beta gamma delta epsilon zeta eta theta. "
res = {}

def one(i):
    text = (
        f"Stream {i}. Repeat this sequence exactly eight times, then stop.\n\n"
        f"Sequence: {phrase * 3}\n\nOutput:\n"
    )
    body = json.dumps({
        "messages": [{"role": "user", "content": text}],
        "max_tokens": 32, "temperature": 0, "stream": False,
        "seed": 200 + i, "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=float(timeout)) as r:
            d = json.load(r)
            reply = d["choices"][0]["message"].get("content") or ""
            with open(f"{out_dir}/stream-{i}.txt", "w") as f:
                f.write(reply)
            res[i] = (r.status, len(reply))
    except Exception as e:
        res[i] = (None, str(e))

threads = [threading.Thread(target=one, args=(i,)) for i in range(3)]
[t.start() for t in threads]
[t.join() for t in threads]
bad = {i: v for i, v in res.items() if v[0] != 200 or not isinstance(v[1], int) or v[1] == 0}
for i in sorted(res):
    print(f"  stream {i}: {res[i]}")
sys.exit(1 if bad else 0)
PY
}

# log_number LOG REGEX: last decimal number matched in LOG, or empty.
log_number() {
  grep -oE "$2" "$1" 2>/dev/null | tail -1 | grep -oE '[0-9]+(\.[0-9]+)?' | tail -1
}

echo "model: $MODEL"
echo "binary: $BIN"

echo
echo "== no-spec server on port $PORT =="
start_server "$TMP/nospec.log" "$PORT"
NOSPEC=$(send_request "$PORT" 64 42 "" "$TMP/nospec-reply.txt") || fail "no-spec request failed"
read -r _ NOSPEC_WALL NOSPEC_TOKENS <<<"$NOSPEC"
stop_server
NOSPEC_SPEED=$(log_number "$TMP/nospec.log" 'decode [0-9.]+ tok/s|decode [0-9.]+')
[[ -n "$NOSPEC_SPEED" ]] || fail "no-spec log has no decode speed"
[[ "$NOSPEC_TOKENS" == 64 ]] || fail "no-spec returned $NOSPEC_TOKENS tokens, expected 64"
ok "no-spec reply: 64 tokens in ${NOSPEC_WALL}s, $NOSPEC_SPEED decode tok/s"
grep -E 'done:' "$TMP/nospec.log" | tail -1 | sed 's/^/  /'

SPEC_FLAGS=(
  --spec-type ngram-map-k4v --spec-draft-n-max 2
  --spec-ngram-size-n 16 --spec-ngram-size-m 24 --spec-ngram-min-hits 1
)

echo
echo "== spec server (ngram-map-k4v) on port $PORT2 =="
start_server "$TMP/spec.log" "$PORT2" "${SPEC_FLAGS[@]}"
SPEC=$(send_request "$PORT2" 64 42 "" "$TMP/spec-reply.txt") || fail "spec request failed"
read -r _ SPEC_WALL SPEC_TOKENS <<<"$SPEC"
[[ "$SPEC_TOKENS" == 64 ]] || fail "spec returned $SPEC_TOKENS tokens, expected 64"
ok "spec reply: 64 tokens in ${SPEC_WALL}s"

check cmp -s "$TMP/nospec-reply.txt" "$TMP/spec-reply.txt"
ok "greedy replies are byte-identical"

# The accepted-draft summary is printed when the model shuts down.
stop_server
DRAFT_LINE=$(grep 'speculative ngram drafts accepted' "$TMP/spec.log" | tail -1)
[[ -n "$DRAFT_LINE" ]] || fail "spec log has no accepted-draft summary"
if [[ "$DRAFT_LINE" =~ accepted\ ([0-9]+)/([0-9]+) ]]; then
  ACCEPTED=${BASH_REMATCH[1]}
  DRAFTED=${BASH_REMATCH[2]}
else
  fail "cannot parse draft summary: $DRAFT_LINE"
fi
(( DRAFTED > 0 )) || fail "no drafts were proposed"
(( ACCEPTED <= DRAFTED )) || fail "accepted $ACCEPTED exceeds drafted $DRAFTED"
(( ACCEPTED > 0 )) || fail "no drafts were accepted"
ok "drafts accepted $ACCEPTED/$DRAFTED"
sed 's/^/  /' <<<"$DRAFT_LINE"

SPEC_SPEED=$(log_number "$TMP/spec.log" 'decode [0-9.]+ tok/s|decode [0-9.]+')
[[ -n "$SPEC_SPEED" ]] || fail "spec log has no decode speed"
python3 - "$NOSPEC_SPEED" "$SPEC_SPEED" "$SPEC_SPEED_MIN" <<'PY' || fail "spec is slower than no-spec beyond tolerance"
import sys
base, spec, floor = map(float, sys.argv[1:])
ratio = spec / base
print(f"  decode: no-spec {base:.1f} tok/s, spec {spec:.1f} tok/s, ratio {ratio:.2f} (min {floor:.2f})")
sys.exit(0 if ratio >= floor else 1)
PY
ok "spec decode is not slower than no-spec (>= ${SPEC_SPEED_MIN}x)"

echo
echo "== three concurrent requests on a fresh spec server =="
start_server "$TMP/concurrent.log" "$PORT2" "${SPEC_FLAGS[@]}"
if ! send_concurrent "$PORT2" "$TMP"; then
  tail -30 "$TMP/concurrent.log" >&2 || true
  fail "concurrent requests failed"
fi
for f in "$TMP"/stream-*.txt; do
  [[ -s "$f" ]] || fail "empty concurrent reply: $f"
done
ok "3 concurrent spec requests returned 200 with replies"

stop_server

echo
echo "PASS: $PASS checks"
