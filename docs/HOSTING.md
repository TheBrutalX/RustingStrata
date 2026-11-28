# Hosting RustingStrata for other people

This page is for running `rustingstrata serve` as a service that other people or programs use over the network, for
example to sell API access to a model. It covers keys, TLS through a reverse proxy, a systemd unit, health checks,
the usage log for billing, and how many requests one server answers at once.

The [license](../LICENSE.md) allows commercial services built on RustingStrata: you can charge for access to a model that
RustingStrata serves. Keep the license and the credit to the original author. You may not sell RustingStrata itself. The model has
its own license (most Qwen models are Apache 2.0; check the model card).

## How much one server does

By default `rustingstrata serve` answers one chat request at a time. Others wait in a queue and run in arrival order.
The queue has no limit unless `--max-waiting N` sets one. Set one on a server that strangers can reach: every waiting
connection holds a socket and a thread. `--max-waiting` counts every open connection (being read,
generated or queued, health checks included), so with `--max-waiting 16` and one reply running 15 places are left
for the queue. Beyond it a connection gets
`503 Service Unavailable` with `Retry-After: 2` (`--max-per-key` keeps one customer from taking all the places). One
request decodes at
the speed in the README (about 113 tok/s for Qwen3.6-35B-A3B Q2_K_XL on an RTX 3060), so that is also the total for
all users of one server. Long prompts block the queue for their prefill time (about 2 s for 4K tokens on the 3060).

### Several requests at once: `--parallel`

`--parallel N` keeps N conversations in GPU memory and decodes them together: the requests that are decoding share
each forward pass, up to 4 at a time. A pass reads the attention, recurrent and shared-expert weights once for all
its requests; only the routed experts each request picks are read per request. On the RTX 3060 below, a pass takes
10.6 ms for one request, 15.3 ms for 2 and 25 ms for 4.

Measured on the RTX 3060 with Qwen3.6-35B-A3B Q2_K_XL, `--ctx 4096`, short prompts and 200-token replies, all
requests sent at once:

| `--parallel` | Requests | Total | Each request |
|---|---|---|---|
| 1 | 4 | 90 tok/s | one after the other |
| 2 | 2 | 139 tok/s | 69 tok/s |
| 4 | 4 | 164 tok/s | 41 tok/s |
| 4 | 8 | 163 tok/s | 41 tok/s, 4 wait |
| 8 | 8 | 149 tok/s | 19 tok/s |

A request alone on a `--parallel 4` server decodes at 105 tok/s (106 with `--parallel 1`).

The gain is smaller for a model whose experts mostly do not fit in VRAM, because the requests in a pass route to
different experts and the CPU computes the missing ones for each. Qwen3-Next-80B-A3B IQ1_S on the same 3060 (9199
of 24576 experts resident with 4 slots), 4 requests: 56.5 tok/s with `--parallel 1`, 77.5 tok/s with
`--parallel 4`.

What changes with `--parallel N`:

- **Memory.** `--ctx` is the context of each slot, so the KV cache and the recurrent state take N times the memory
  of one, and fewer experts fit in VRAM. The `plan:` line at start shows the split. With 8 slots of 4096 tokens the
  35B model kept 7008 of its 10240 experts resident. A long `--ctx` costs the most here, and `--kv q8` gets much of
  it back. `--parallel 4`, 4 short requests at once on the 3060:

  `--kv-unified` makes `--ctx` the total budget instead: each slot starts at `ctx / N`, and a slot that needs
  more takes the KV of idle slots (down to one token each), so a 1M budget is four 256k slots or one near-1M
  slot. The reclaim drops that slot's VRAM prefix cache and saved state. It is one GPU only, and the budget is
  reserved at load: a 1M q8 budget is 10.9 GB of KV with every expert resident on a 32 GB card. Unified mode
  prefills prompts one at a time but reuses a slot's saved prefix when the next prompt continues it (a diverging
  prompt clears the slot, with no host-memory parking).

On the V100 32 GB, `--parallel 4 --ctx 1048576 --kv-unified --kv q8` reserved 10,880 MB of KV with all 10,240
experts resident. Four 31k-token prompts completed 4/4 on those 256k slots. An 801,742-token prompt allocated an
801,792-token slot, prefilled at 368.6 tok/s and generated 4 tokens at 25.3 tok/s in 36m22s; the cap can reach
1,048,573 (one token per idle slot is reserved).

With `--chunk 16384` the same 1M/4-slot plan keeps all 10,240 experts resident (15,008 MB weights+state+scratch plus
15,150 MB of expert slots) and four 31k prompts finish in 45.8 s (2,760 tok/s each) instead of 47.8 s at 8192.
`--chunk 32768` only fits by leaving 223 experts out of VRAM (10,017 of 10,240 resident), so 16384 is the largest
fully resident chunk. Prompt token embeddings are dequantized in parallel on the CPU and reused by the MTP draft
layer, which removes the 65-80 ms stalls at the chunk boundaries.
  | `--ctx` | `--kv` | KV for 4 slots | Resident experts | Total |
  |---|---|---|---|---|
  | 16384 | f16 | 1280 MB | 6876 of 10240 | 160 tok/s |
  | 32768 | f16 | 2560 MB | 5567 | 114 tok/s |
  | 32768 | q8 | 1360 MB | 6779 | 154 tok/s |

  `--kv q8` is about 6% slower to decode and 14% slower to prefill at 32K of actual context, so it pays off when the
  memory it frees holds experts. Greedy output can differ from `--kv f16` at near-ties.
- **Queue.** When every slot is busy, requests wait. A free slot goes to the waiting request whose key holds the
  fewest slots, the oldest first among those, so one customer who sends 20 requests at once does not hold back
  another customer's one. Without `--api-keys` the queue is first come, first served.
- **More than 4 slots** do not raise the total, because one pass decodes at most 4 requests. They let more users
  stream at the same time instead of waiting in the queue, each at a lower speed.
- **Prefill.** Prompts are prefilled one at a time, in chunks (`--chunk`, from `rustingstrata calibrate`). After each
  chunk the requests that are decoding run for as long as the chunk took, so a long prompt and the streams share
  the GPU evenly: the prompt takes about twice as long to its first token, and each stream pauses for one chunk at a
  time. A smaller `--chunk` gives shorter pauses but a slower prefill. A 10K-token prompt that arrives while 3
  requests stream, on the 3060 with `--parallel 4`:

  | `--chunk` | Longest pause of a stream | First token of the long prompt |
  |---|---|---|
  | 4096 (calibrated) | 2.2 s | 10.2 s |
  | 1024 | 0.7 s | 13.9 s |
  | 512 | 0.4 s | 16.4 s |

  Prefilled in one piece, the same prompt took 6.5 s to its first token, and the three streams stopped for all of it.
- **Speculative decoding with parallel slots.** `--mtp` and `--spec-type` (including `ngram-map-k4v`) now work
  with `--parallel > 1`: one MTP/ngram state per slot, verified one slot at a time. A speculative step pauses the
  batched forward model-wide, so it helps one long conversation more than many simultaneous short ones. On the
  V100/Ornith trial a repeated-text prompt went from 120.2 to 155.1 tok/s with `--mtp 2`, 162.7 with
  `ngram-map-k4v`, and 176.5 with both, with identical greedy output. Measure your own mix. With several cards
  (`--device 0,1`), the slots work over the split layers too (see [Several GPUs](#several-gpus)).
- **Conversation cache.** Each slot keeps the state of its last conversation, and a request that continues a
  conversation goes to the slot that holds it. Other conversations are parked in host memory (`--cache-ram`) as
  without `--parallel`. Keys still only reuse their own conversations.
- **Same answers.** A request decodes the same tokens as it would alone on the same server. Greedy output can still
  change at near-ties from one run to the next, because the expert placement adapts to the routing it sees (as
  with `--parallel 1`).
- Without `--gpu` the option is ignored: the CPU reference runs one request at a time.

For a server with a few users at a time, `--parallel 4` gives the most total speed. Measure your own mix with your
prompts: long prompts and long contexts change the picture.

For more users, run one server per GPU and put a load balancer in front (see [Several GPUs](#several-gpus)).

## Quick start

On a Linux machine with an NVIDIA driver (or a Vulkan 1.3 driver):

```sh
# 1. The binary: a release binary, or build it on this machine (.cargo/config.toml builds for the local CPU).
cargo build --release
sudo install -m 755 target/release/rustingstrata /usr/local/bin/rustingstrata

# 2. A service user, a model, and the calibration for this machine (once per model and machine). It runs as the
#    service user, with the --ctx and --kv of the server, because it is kept in that user's ~/.cache/rusting_strata.
sudo useradd --system --create-home --home-dir /var/lib/strata strata
sudo usermod -aG video,render strata      # GPU device access on most distributions
M=/srv/models/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf
sudo -u strata rustingstrata calibrate --model $M --ctx 32768 --kv q8

# 3. Keys: one admin key, and a file of customer keys.
openssl rand -hex 24                            # run once per key
sudo mkdir -p /etc/strata
sudo tee /etc/strata/keys.txt >/dev/null <<'EOF'
# name     key
acme       3f9c0d6e5b...
beta-corp  a81e44f2c7...
EOF
sudo chown strata /etc/strata/keys.txt && sudo chmod 600 /etc/strata/keys.txt

# 4. Run it on the loopback address; the reverse proxy below is the only thing that faces the network.
#    (systemd below runs the same command as a service.)
sudo -u strata env STRATA_API_KEY=<admin key> rustingstrata serve --model $M --gpu --ctx 32768 --addr 127.0.0.1:8080 \
    --api-keys /etc/strata/keys.txt --usage-log /var/lib/strata/usage.jsonl --max-tokens 4096 --parallel 4 --kv q8 \
    --max-per-key 4
```

Clients then use the OpenAI address `https://api.example.com/v1` with their key, or the Anthropic address
`https://api.example.com` (`POST /v1/messages`). The request fields RustingStrata honors (sampling, `stop`, tools,
thinking) are listed under [Server API](USAGE.md#server-api); the usage of a streamed reply comes in its last
chunk.

## Keys

- **Admin key** (`--api-key`, or `STRATA_API_KEY` to keep it out of the process list). It can use every route: the
  chat APIs, the chat page's stats, loading another model (`/api/load`, from the models folder
  or by the path of any GGUF file the server can read; or a chat request whose `model` is another name of
  `GET /v1/models`) and downloading models (`/api/download`). Keep it for
  yourself.
- **Customer keys** (`--api-keys FILE`). One `name key` pair a line, `#` starts a comment (so a key cannot contain
  `#`). Keys must be at least 16 characters, names and keys must not repeat, and the admin key must not be among
  them. A customer key reaches only `POST /v1/chat/completions`,
  `POST /v1/messages`, `POST /v1/messages/count_tokens` and `GET /v1/models`; other routes answer `403`. It sees
  only the loaded model in `/v1/models`, and the `model` of its requests never loads another one. The name goes into
  the usage log.
- **Changing the key file.** The server reads the file again when it changes, and the next request uses the new
  keys; replies being generated go on. A file that cannot be read or does not parse keeps the old keys (the server
  log says why). A file with no keys removes all customer keys when there is an admin key; without one it is refused,
  as it is at start, because the server would be open. Write a new file and rename it over the old one, so that the server never reads it half
  written:

  ```sh
  sudo sh -c 'cp /etc/strata/keys.txt /etc/strata/keys.new && echo "gamma $(openssl rand -hex 24)" >> /etc/strata/keys.new \
      && chown strata /etc/strata/keys.new && chmod 600 /etc/strata/keys.new && mv /etc/strata/keys.new /etc/strata/keys.txt'
  ```

  A removed key gets `401` from its next request on. A key that stays keeps its cached conversations, even when its
  line moves. To cut off a customer who used up a prepaid budget, sum the usage log per key and remove the key.
- With customer keys and no admin key, nobody can use the admin routes.
- With no key at all the server is open to anyone who reaches it. That is meant for a local machine only.

When the server is reached through a LAN name while bound to `0.0.0.0` (for example `http://ai-server.lan:8080`),
add that name to the Host check: `--allowed-hosts ai-server.lan,ai-server.local` or
`STRATA_ALLOWED_HOSTS=ai-server.lan,ai-server.local`. Names match exactly and ignore case and the port; loopback
names and the bound address stay allowed. Without keys this check still stops DNS rebinding.

Clients send the key as `Authorization: Bearer <key>` (OpenAI; `Bearer` as written) or `x-api-key: <key>`
(Anthropic).

**Privacy between keys.** RustingStrata keeps recent conversation states in GPU and host memory (`--cache-ram`) so that a
request that continues a conversation prefills only the new part. Each key only reuses states that its own requests
made, so one customer cannot tell from `cached_tokens` or from the response time what another customer sent. All
requests under one key count as one user: if a customer serves many end users through one key, the customer's end
users can learn this about each other. RustingStrata does not write prompts or replies to disk; the usage log holds only
counts and times.

## Reverse proxy and TLS

RustingStrata speaks plain HTTP. Put a reverse proxy in front of it for TLS, and bind RustingStrata to `127.0.0.1`. When keys are
set, RustingStrata accepts any `Host` header, so the proxy can pass the public name through.

Streaming replies (`"stream": true`) must reach the client as they are written, so turn off response buffering, and
give the proxy a read timeout that is longer than your longest reply. RustingStrata sends nothing, not even the headers of a
stream, until the prompt is prefilled, so the timeout must also cover the wait in the queue plus the prefill of your
longest prompt. A proxy whose timeout you cannot raise (Cloudflare gives up after 100 s without a response) needs a
`--ctx` small enough that the longest prompt prefills in time, or a direct route that bypasses it.

**Caddy** (gets and renews the certificate itself):

```
api.example.com {
    reverse_proxy 127.0.0.1:8080 {
        flush_interval -1
        transport http {
            read_timeout 15m
        }
    }
}
```

**nginx:**

```nginx
server {
    listen 443 ssl;
    server_name api.example.com;
    ssl_certificate     /etc/letsencrypt/live/api.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.example.com/privkey.pem;

    client_max_body_size 8m;        # RustingStrata refuses bodies over 8 MiB
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_buffering off;        # stream tokens as they come
        proxy_read_timeout 900s;    # a long reply, or a long wait in the queue
        proxy_send_timeout 900s;
    }
}
```

To keep the chat page and the admin routes off the public address, serve only `/v1/` and `/health` there:

```nginx
    location /v1/    { proxy_pass http://127.0.0.1:8080; proxy_http_version 1.1; proxy_buffering off; proxy_read_timeout 900s; }
    location /health { proxy_pass http://127.0.0.1:8080; }
    location /       { return 404; }
```

Then reach the chat page through an SSH tunnel: `ssh -L 8080:127.0.0.1:8080 server`, and open
http://127.0.0.1:8080 with the admin key.

RustingStrata caps the requests each key has open itself (`--max-per-key`, see [Limits per request](#limits-per-request)).
A cap on requests per minute belongs in the proxy:

```nginx
# in http { }
limit_req_zone  $http_authorization zone=perkey_rate:10m rate=60r/m;

# in the location that passes /v1/ to RustingStrata
limit_req  zone=perkey_rate burst=20 nodelay;
limit_req_status  429;
```

Anthropic clients send the key as `x-api-key`, so key the zone on `$http_x_api_key` too if you serve them.

## systemd

`/etc/strata/env` (mode 600):

```
STRATA_API_KEY=<admin key>
```

`/etc/systemd/system/strata.service`:

```ini
[Unit]
Description=RustingStrata inference server
After=network-online.target
Wants=network-online.target

[Service]
User=strata
EnvironmentFile=/etc/strata/env
ExecStart=/usr/local/bin/rustingstrata serve --model /srv/models/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf --gpu --ctx 32768 \
    --addr 127.0.0.1:8080 --api-keys /etc/strata/keys.txt --usage-log /var/lib/strata/usage.jsonl --max-tokens 4096 \
    --parallel 4 --kv q8 --max-per-key 4
StateDirectory=strata
Restart=on-failure
RestartSec=5
# Accepted requests (queued ones too) finish before the server stops; that needs more than the default 90 s.
TimeoutStopSec=300

[Install]
WantedBy=multi-user.target
```

When the GPU device is lost (a driver reset, a GPU hang), no later request can succeed, so `serve` exits with
status 1 and `Restart=on-failure` starts it again.

With the user, keys and calibration of the quick start:

```sh
sudo systemctl daemon-reload && sudo systemctl enable --now strata
journalctl -u strata -f
```

Run `calibrate` as the service user, because the calibration file and the CUDA kernel cache live in that user's
`~/.cache/rusting_strata`. The first start after an update compiles the CUDA kernels (about 15 s); `/health` answers
`503` until the model has loaded.

In a container that folder is lost when the container exits, and the next one starts uncalibrated with only the
`not calibrated on this machine` line to say so. Set `XDG_CACHE_HOME` to a mounted path (the folder is then
`$XDG_CACHE_HOME/rusting_strata`) for both `calibrate` and `serve`, for example `-v /srv/strata-cache:/cache -e
XDG_CACHE_HOME=/cache`.

**Stopping and restarting.** On `SIGTERM` (`systemctl stop` or `restart`) the server answers new requests, `/health`
included, with `503` and `Retry-After: 5`, lets the requests it already accepted finish (the queued ones too), saves
the routing profile it learned, so that the next start places the experts as well, and then exits.
`TimeoutStopSec` must cover a full queue, or keep `--max-waiting` low. A load balancer
sends the new requests to the other servers meanwhile, so restarting one server at a time cuts no reply. `systemd`
sends `SIGKILL` after `TimeoutStopSec`; Ctrl+C still stops at once. On Windows there is no such drain.

### Slot state files

`--slot-save-path DIR` enables explicit per-slot state files: `GET /slots` and
`POST /slots/{id}?action=save|restore|erase`, as in [USAGE.md](USAGE.md#slot-state-files-save-and-restore). It is
off by default, and the POST routes answer `501` without it.

Create the directory once, owned by the service user and closed to everyone else, then add the flag to the unit:

```sh
sudo install -d -o strata -g strata -m 0700 /var/lib/strata/slots
```

```ini
ExecStart=/usr/local/bin/rustingstrata serve --model /srv/models/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf --gpu --ctx 32768 \
    --addr 127.0.0.1:8080 --api-keys /etc/strata/keys.txt --usage-log /var/lib/strata/usage.jsonl --max-tokens 4096 \
    --parallel 4 --kv q8 --max-per-key 4 --slot-save-path /var/lib/strata/slots
```

`serve` checks at startup that the path is a directory and exits with
`--slot-save-path /var/lib/strata/slots is not a directory` otherwise. `StateDirectory=strata` creates the parent
but not `slots`. The server writes the files itself (mode `0600` on Unix), so the directory must be writable by
`strata`; `0700` keeps other local users from listing or opening them.

**Disk.** A file is a full snapshot of one slot: KV rows, DeltaNet/MTP/PLE/QSA state and the token history, with no
compression and no deltas. Size scales with `--ctx`, `--kv` and how many slots you save (one file per saved slot;
with `--parallel N` and private KV per slot, each can hold a full `--ctx` worth of KV). A 1M-token q8 slot is about
10.9 GB with the 35B-A3B layout. A slot that is only partly used writes its `n_past` rows, so short conversations
are smaller. Keep free space for the new file beside the old one until the rename, prune old files yourself, and
note that a crashed or killed save leaves a `.<name>.<pid>.<nanos>.tmp` file in the directory (the previous target
stays intact), which is safe to delete.

**Use.** Save and restore are explicit admin actions; the server never snapshots a slot on its own. Both block the
model thread for the length of the copy, so treat them as maintenance operations, not something to run per request.
A redeploy that keeps a long conversation:

1. save the slot (`POST /slots/0?action=save`, body `{"filename":"conversation.slot"}`);
2. stop the service (the queue drains first);
3. start the new binary with the same model, `--kv`, YaRN/RoPE flags and `--device`/`--split`, and enough `--ctx` for
   the saved token count;
4. restore the file into a slot and send the continuation.

A file from another model, KV layout, YaRN setting or split is refused, so changing one of these means prefilling
again.

**Concurrency.** A save/restore/erase for a busy slot is deferred, not rejected: the request waits until that slot is
idle. With `--parallel N` the other slots keep their conversations and resume after the copy, but no forward runs
while the file streams, so every stream pauses for its length. `GET /slots` shows `is_processing`, `pinned` and
`n_past`, which helps find the slot worth saving.

**`--kv-unified`.** A restore grows the target slot and reclaims only inactive slots. If the growth would take a
non-empty slot, the restore answers `409`; `?force=1` lets it erase those idle victims. A restored slot is pinned
until a request uses it or `action=erase` clears it, so another slot's growth cannot take the restored KV. Use
`force=1` deliberately: it discards the victims' conversations.

**Security and backups.** The files are plaintext: the full token history (prompt, system prompt, tools and
generated tokens) and the KV/recurrent state. Treat a file as a copy of the conversation: keep the directory `0700`,
do not serve it over HTTP, and include it in (or deliberately exclude it from) backups and retention. With
`--api-keys` the routes are admin-only and a customer key gets `403`; without the flag the POST routes answer `501`,
while `GET /slots` still lists slot metadata.

**Troubleshooting.**

- `400`: bad slot id or action, invalid filename (separator, `.`/`..`, control character, over 255 bytes),
  missing/unreadable/truncated/corrupt file, a file from another model/KV/YaRN/split, or a save of an empty slot.
- `409`: a unified restore would erase a non-empty slot; retry with `?force=1` if that is intended.
- `501`: the server was started without `--slot-save-path`.
- `503`: no model is loaded.
- Startup: `--slot-save-path ... is not a directory`; create it, fix its ownership and mode, and restart.

### Conversation archive

`--slot-archive-dir DIR` enables the automatic archive from
[USAGE.md](USAGE.md#conversation-archive): a slot is checkpointed before its history diverges or before it is
reclaimed, and a later prompt that continues an archived history restores it into a free slot. It is off by
default and must be a different directory from `--slot-save-path`; the server refuses to start when the two resolve
to the same directory.

Create the directory once, owned by the service user, and add the archive flags to the unit:

```sh
sudo install -d -o strata -g strata -m 0700 /var/lib/strata/archive
```

```ini
ExecStart=/usr/local/bin/rustingstrata serve --model /srv/models/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf --gpu --ctx 65536 \
    --addr 127.0.0.1:8080 --api-keys /etc/strata/keys.txt --usage-log /var/lib/strata/usage.jsonl --max-tokens 4096 \
    --parallel 4 --kv q8 --max-per-key 4 --slot-save-path /var/lib/strata/slots \
    --slot-archive-dir /var/lib/strata/archive --slot-archive-gb 200 \
    --slot-archive-min-tokens 256 --slot-archive-min-delta 1024 \
    --slot-archive-ttl 30d --slot-archive-per-key-gb 20 --slot-archive-pin-gb 20 \
    --slot-archive-gc-interval 5m --slot-archive-on-shutdown --slot-archive-shutdown-timeout 30s
```

The server creates a missing directory `0700` and writes entries and temp files `0600`, so the service user needs
write access. `StateDirectory=strata` creates the parent but not `archive`. Keep the directory out of any path that
the web server or a reverse proxy serves.

**Disk and sizing.** `--slot-archive-gb` is GiB (1 GiB = 2^30 bytes), and the example above reserves 200 GiB. The
budget is logical, with 1 GiB kept free for the temp file and manifest. With q8 KV on the 35B-A3B layout, SLOT-SAVE
measures about 10.9 GB per 1M tokens: 8k tokens is about 90 MB, 32k about 350 MB, 100k about 1.1 GB, 256k about
2.8 GB and 1M about 10.9 GB. A checkpoint also carries a fixed ~66 MB recurrent/DeltaNet/MTP/PLE/QSA state per
slot, which is why a short 1.4k-token checkpoint was 81 MB, not a few MB. q4 halves the KV (~5.8 GB/1M) and q2
reduces it to ~3.9 GB/1M; the fixed state is unchanged. A 200 GiB budget is about 18M archived tokens at q8
(~34M at q4, ~52M at q2) before the fixed state per conversation.

Format-2 entries deduplicate the KV: the state file is per entry, while `segments.json` + `segments/` hold
1024-token content-addressed K/V segments shared by conversations with the same prefix. A growing conversation
writes only new segments; a shared system prompt/tools/document is charged once. Do not edit `segments.json` or
the segment files by hand. Leave filesystem free space above the budget: a write needs room for its temp file, the
manifest, the segment manifest and the reserve below the logical limit.

**Manifests.** `manifest.json` and `segments.json` are compacted snapshots; mutations go to the append-only
`manifest.log`/`segments.log` and `touch` updates are batched. A truncated last journal record is ignored, a
corrupt snapshot is rebuilt from the logs/files, and compaction is adaptive/GC/Drop-driven. Back up the snapshots,
both logs and `segments/` together; do not edit any of them by hand.

**Eviction and lineage.** A checkpoint that would exceed `budget - 1 GiB` first prunes unpinned ancestors beyond
`--slot-archive-versions`, then evicts entries with a foreign fingerprint, then the least recently used unpinned
entries. Pinned entries are never evicted. The periodic `--slot-archive-gc-interval` (default 5 min, 0 off) applies
TTL/quota/LRU in the background; `POST /slots/archive/gc` drops unpinned entries older than
`--slot-archive-ttl`, then rows whose file is missing, then applies the same lineage pruning and LRU eviction when
the archive is over the target. Lineage pruning is lazy, so a parent and its continuation coexist while there is
room. Keeping only the latest entry per lineage saves disk but removes rollback points: a longer child is not a
substitute for its shorter parent, so raise `--slot-archive-versions` if earlier states matter. Foreign
fingerprints are ordinary LRU candidates in manual GC, which has no current-fingerprint argument.

**Lifecycle: TTL, quota and pin.** TTL, per-key quota and the pin cap are off or conservative by default so a
phase-1 deployment keeps its behavior; on a shared server, tune all three:

- `--slot-archive-ttl 30d` is the recommended TTL for a shared server (`0` is off). The TTL is lazy:
  `POST /slots/archive/gc` applies it, and a checkpoint prefers expired entries when it needs room; there is no
  timer, so an entry can outlive its TTL on an idle server. Run GC periodically if expiry must be prompt, for
  example from a systemd timer or cron:

  ```sh
  curl -fsS -X POST -H 'Content-Type: application/json' -d '{}' \
      http://127.0.0.1:8080/slots/archive/gc
  ```
  With `--api-keys`, use an admin key for that request.
- `--slot-archive-per-key-gb` caps one owner's bytes. At a checkpoint the owner's own least recently used unpinned
  entries are evicted first; the quota step never touches another owner's entries, and the global budget is
  enforced afterwards. Owner 0 is a single bucket without API keys. 25% of the global budget or 20 GiB, whichever
  is smaller, is a sensible shared-server value. An entry larger than the quota is refused, so a checkpoint can be
  skipped; keep the quota above the largest expected conversation state.
- `--slot-archive-pin-gb` caps what an admin may pin; the default is 10% of `--slot-archive-gb` (20 GiB at 200),
  and `0` removes the cap. The pin cap is a safety valve, not a target: a pin that would exceed it answers `409`.
  Pinning is an admin-only `POST /slots/archive/{id}?action=pin|unpin`; the entry's `pinned` flag is persisted in
  the manifest, so a pin survives a restart and protects the entry from the TTL, the budget and lineage pruning.

**Checkpoint stalls.** Automatic format-2 saves/restores now run through a background writer/reader: the model
thread performs only the GPU copies (and H2D on restore), while segment/state files and the manifest are written
off-thread; `Sched::yield_between` services other slots' decode steps for up to 2 ms per copy. The archive mutex is
not held during the copies. `finish` still waits for the final commit, so the model thread may wait for that commit
at the end; explicit `action=save`/`action=checkpoint` remain synchronous by design. A 1 GB state costs 1-2 s of
disk at 0.5-1 GB/s; expect a shorter but still visible pause on very large states. Keep
`--slot-archive-min-tokens`/`--slot-archive-min-delta` high enough that tiny states/extensions are not written on
every turn.

**Segment collisions.** The v2 segment id is content-addressed by the token prefix, not by the KV bytes. If a
nondeterministic GPU run produces different KV for an already stored segment id, the store refuses to overwrite the
existing segment; the checkpoint falls back to a v1 full snapshot for that entry and logs the fallback. The entry
is restorable, but it does not share the v2 segments of the earlier entries.

**Restore behavior.** Automatic restore uses a free slot only and never takes a slot a request holds. It never
uses the `force=1` of the explicit route. If the chosen free slot holds another conversation, the server
checkpoints it before the restore overwrites it. If no slot is free the request queues as usual and then
prefills; if a unified hit would need a non-empty reclaim victim, the hit is skipped and the prompt is prefilled
normally. A corrupt, foreign or owner-mismatched entry is a cache miss, not a failed request. A restored slot is
pinned until the request is admitted or `action=erase` clears it, so a reclaim cannot take it between restore and
admission. The restored prefix is reported as `usage.prompt_tokens_details.cached_tokens`.

**Shutdown.** With `--slot-archive-on-shutdown` (off by default), a graceful stop drains accepted requests and
checkpoints every non-empty resident slot, most recently used first, within `--slot-archive-shutdown-timeout`
(default 30 s). A slot still running at the deadline is skipped and logged; the process does not hang past the
watchdog, and `SIGKILL`/crash cannot be covered. Without the flag, only entries checkpointed earlier survive, so
call `action=checkpoint` before a planned restart. Set systemd `TimeoutStopSec` above the shutdown deadline plus
the expert-profile save (the example above uses the default 300 s).

**Security and backups.** Entries are plaintext: the full token history (prompt, system prompt, tools and
generated tokens) and the KV/recurrent state. Format-2 entries keep the tokens/recurrent state in `STRSTATE`
files and the KV in `segments/`; `segments.json` is the segment index. Segments are shared globally inside the
archive directory while entries remain owner-scoped, so the same prefix can serve several owners without exposing
an entry they do not own; use a separate archive directory if even that sharing is unacceptable. Pinning does not
encrypt an entry; it only exempts it from the TTL, the budget and lineage pruning, so unpin an entry you no longer
need. Treat the directory as a copy of the conversations: `0700`, files `0600`, no HTTP access, and include
`manifest.json`, `segments.json`, `segments/` and any v1 `.slot` files in (or deliberately exclude them from)
backups and retention. A backup taken while the server is writing can contain orphan temp/segment files; restore a
consistent snapshot or stop the service. With `--api-keys` entries are owner-scoped and the archive routes are
admin-only:
`GET /slots/archive`, `DELETE /slots/archive/{id}`, `POST /slots/archive/{id}?action=pin|unpin`,
`POST /slots/archive/gc` and `POST /slots/{id}?action=checkpoint` all answer `403` for a customer key. Without
keys, owner 0 is shared by every client. A copied entry still restores only with the same model, `--kv`,
YaRN/RoPE flags and split, because the fingerprint is checked first.

**Troubleshooting.**

- `400`: an invalid slot id or action, or `action=checkpoint` on a slot that is busy, partial, an image prompt, or
  has no token history or fewer than `--slot-archive-min-tokens` (the empty-slot rule is the same as `action=save`).
  A corrupt or foreign archive entry is not a `400` for a chat; it falls back to prefill.
- `403`: a customer key on any archive route; those routes are admin-only.
- `409`: a pin that would exceed `--slot-archive-pin-gb`, or `DELETE` of an entry leased by an in-flight restore
  (retry after it completes). Automatic restore itself never returns it. An explicit
  `POST /slots/{id}?action=restore` through `--slot-save-path` keeps its existing `409` when `--kv-unified` would
  evict a non-empty slot; retry with `?force=1` only if that is intended.
- `404`: `pin`/`unpin` of an id that is no longer in the manifest, for example after a TTL GC.
- `segments.json`/segment files: a corrupt segment manifest is rebuilt from the `.seg` files on startup; an entry
  whose state file or segment chain is missing or invalid is dropped rather than restored. A lost `manifest.json`
  rebuilds v1 `.slot` entries only; v2 entries are not reconstructable from segments alone, so include
  `manifest.json`, `segments.json` and `segments/` in the same backup.
- `501`: the server was started without `--slot-archive-dir`; the archive routes are off.
- `503`: no model is loaded.
- Startup: the manifest is validated against the shard files. A missing file drops its entry, an orphan file is
  adopted when its header is valid (or deleted), and a corrupt `manifest.json` is moved aside and rebuilt. If the
  archive directory cannot be created or opened, the server exits with the path in the error; fix the
  ownership/mount and start it again. A read or write that fails later is logged and skipped (a lookup is a cache
  miss), and the chat keeps serving.

## Docker

The repository has a `Dockerfile` for the server. The image holds the binary, NVRTC for the CUDA kernels and the
Vulkan loader; the GPU driver comes from the host (for NVIDIA cards through the NVIDIA Container Toolkit).

```sh
docker build -t rustingstrata .
docker run -d --name strata --gpus all -p 127.0.0.1:8080:8080 \
  -v /srv/models:/models -v /srv/strata-cache:/cache rustingstrata
```

- The default command is `serve --gpu --addr 0.0.0.0:8080 --models-dir /models`: no model is loaded at the start
  (`/health` answers `503` until one is), and the chat page at http://127.0.0.1:8080 lists the files in `/models`
  and loads the one you pick. Arguments after the image name replace the command, for example
  `rustingstrata serve --gpu --addr 0.0.0.0:8080 --model /models/model.gguf` to load a model at the start.
- `/models` is the folder of GGUF files. `/cache` keeps the calibration, the compiled CUDA kernels and the expert
  placement; without that mount every new container compiles the kernels again and runs uncalibrated.
- Calibrate once per model and machine, with the same mounts:
  `docker run --rm --gpus all -v /srv/models:/models -v /srv/strata-cache:/cache rustingstrata calibrate --model /models/model.gguf`.
- The example publishes the port on the loopback address only. To reach the chat page through a LAN name, add
  `-e STRATA_ALLOWED_HOSTS=that-name`; set a key with `-e STRATA_API_KEY=...` before the port faces a network.
- For a Vulkan card (AMD, Intel) pass the render device in place of `--gpus all`: `--device /dev/dri`, and add
  `--backend vulkan` to the command. This path has not been run on an AMD or Intel card yet.

## Health checks

`GET /health` needs no key and is answered at once, also while a reply is generated or a model loads:

- `200 {"status": "ok", "waiting": N, "slots": S}` once a model is loaded. `waiting` counts the requests being read,
  generated or queued besides this one; `slots` is how many are generated at once (`--parallel`). A load balancer
  can send the next request to the server with the lowest `waiting / slots`.
- `200 {"status": "ok", "unloaded": true, "waiting": N}` when `--auto-unload` has freed the model: the next chat
  request loads it again, so the server stays in rotation. During that load `/health` answers `503 loading`, as for
  any load.
- `503 {"status": "loading"}` or `503 {"status": "no model"}` otherwise.
- `503` with an `error` body (no `status`) when `--max-waiting` connections are open, or while the server stops. A
  load balancer then treats a full server as down, which sends the next requests elsewhere.

Point the load balancer or your monitoring at it.

`GET /metrics` gives the same numbers and more in Prometheus text format: `strata_up`, `strata_uptime_seconds`, `strata_slots`,
`strata_busy`, `strata_waiting`, `strata_requests_total`, `strata_prompt_tokens_total`,
`strata_cached_prompt_tokens_total`, `strata_completion_tokens_total`, `strata_vram_used_bytes` /
`strata_vram_total_bytes` and, for MoE models, `strata_expert_hits_total` / `strata_expert_misses_total`
(these two restart at each model load). Like `/api/stats` it needs the admin key, and it is answered while a
reply is generated:

```yaml
scrape_configs:
  - job_name: strata
    authorization:
      credentials: sk-admin-...   # the --api-key value
    static_configs:
      - targets: ["127.0.0.1:8080"]
```

## Usage log and billing

With `--usage-log FILE`, every chat request that reaches the model adds one JSON line:

```json
{"at":1793917331,"key":"acme","model":"Qwen3.6-35B-A3B-UD-Q2_K_XL","prompt_tokens":1200,"cached_tokens":1100,"system_tokens":300,"completion_tokens":250,"prefill_ms":95.1,"decode_ms":2210.4,"finish":"stop"}
```

- `at`: Unix time in seconds when the request started.
- `key`: the key's name from the key file (`admin` for the admin key, empty on a server without keys).
- `prompt_tokens`: all prompt tokens. `cached_tokens` of them came from the conversation cache and were not prefilled
  again; many providers bill them at a lower price.
- `prefill_ms`: from when the model takes the request until its prompt is read (the wait in the queue, or for a
  slot under `--parallel`, is not included); `decode_ms`: the reply after that.
- `completion_tokens`: reply tokens. `finish` is `stop`, `length` (hit `max_tokens` or the context) or `tool_calls`.
- `finish` is `cut` when the client closed the connection before the end. `completion_tokens` then counts the
  tokens decoded so far: a stream has sent nearly all of them, a reply without `stream` has sent none. `error` is a
  request that failed on the server; it got a `500` or a stream that stopped.

Requests refused before the model (bad key, bad JSON, a prompt that fills the context, `503`, `429`) and requests
whose client left while they waited in the queue are not logged. A client that leaves once its prefill has started
is logged as `cut`, with all its prompt tokens. Tokens per key and day, without failed requests:

```sh
jq -r 'select(.finish != "error") | [(.at | strftime("%Y-%m-%d")), .key, .prompt_tokens - .cached_tokens, .cached_tokens, .completion_tokens] | @tsv' usage.jsonl \
  | awk '{k=$1" "$2; p[k]+=$3; c[k]+=$4; o[k]+=$5} END {for (k in p) print k, p[k], c[k], o[k]}' | sort
```

The file only grows; rotate it with logrotate (`copytruncate`, since RustingStrata keeps it open).

## Limits per request

- `--ctx` is the context length for prompt plus reply (4096 by default). Requests whose prompt fills it get `400`.
  KV memory grows with it: about 650 MB at 32K for the 35B model, half that with `--kv q8`.
- `--max-tokens` caps the reply length a request may ask for, and is the length when it asks for none. Without it a
  request may decode until the context is full, which holds the server for minutes.
- Bodies over 8 MiB are refused, a connection that sends nothing for 30 s is closed, and a body needs a
  `Content-Length` (a chunked upload gets `400`). nginx buffers the request by default, which covers both: a slow
  client then holds nginx, not a RustingStrata place.
- A missing or wrong key (`401`), a malformed request (`400`), `GET /v1/models` and the chat page are answered as
  soon as they are read, without a place in the queue, so a scan for keys does not delay paying customers.
- A client that disconnects stops its reply within a few tokens, so the next request in the queue starts. One that
  leaves while its request waits in the queue costs no prefill: the request is dropped when its turn comes (with
  `--parallel`, within a quarter second).
- With `--parallel N`, `--ctx` is per slot: N slots of 32K need N times the KV memory of one.
- `--max-per-key N` caps the requests one customer key may have open at once (generated or queued; a request counts once it is read); more get
  `429 Too Many Requests` with `Retry-After: 1` and the OpenAI error type `rate_limit_error`, which the OpenAI and
  Anthropic SDKs retry. The admin key is not counted. Set it below `--max-waiting`, so that one customer cannot fill
  the whole queue and leave the others with `503`.

## Several GPUs

**One server per GPU, more users.** Start one server per card on its own port and balance between them:

```sh
rustingstrata serve --model $M --gpu --device 0 --addr 127.0.0.1:8081 --threads 6 ...
rustingstrata serve --model $M --gpu --device 1 --addr 127.0.0.1:8082 --threads 6 ...
```

```nginx
# max_fails=0: a busy 503 must not mark the server down for 10 s (with both busy, nginx would answer 502).
upstream strata { least_conn; server 127.0.0.1:8081 max_fails=0; server 127.0.0.1:8082 max_fails=0; }
# In the location block: a 503 (busy or stopping) is sent before any work, so another server may take the request.
proxy_next_upstream error http_503 non_idempotent;
```

Give each server half of the CPU cores (`--threads`), because each one computes the experts that do not fit in its
VRAM on the CPU. Conversation caches are per server, so a conversation that moves to the other server is prefilled
again; hashing on the key (`hash $http_authorization consistent;` instead of `least_conn`) keeps each customer on one
server.

**One server over several GPUs, a bigger model.** `--device 0,1` splits the layers over the cards, so more experts
fit in VRAM. `--parallel` works here too: each card holds the KV cache of its own layers for every slot. The cards
take their layers in turn for each step, so they do not compute at the same time, and the gain over `--parallel 1`
comes from the batching, as on one card. A split server does not park conversations in host memory: each slot only
keeps its last conversation. For many users on cards that each fit the model, one server per card is faster.

## Checklist

- [ ] RustingStrata listens on `127.0.0.1` only; the firewall opens 443 (and 80 for the certificate), not 8080.
- [ ] An admin key, and customer keys in a file with mode 600.
- [ ] The proxy streams without buffering and has a long read timeout.
- [ ] `--max-tokens` and `--ctx` fit your price list and your GPU memory (times `--parallel`).
- [ ] `--parallel` set for the number of users you expect at once (4 for the most total speed).
- [ ] `--max-waiting` set (no limit by default) and `--max-per-key` below it, so one customer cannot fill the queue.
- [ ] `--usage-log` is on and rotated.
- [ ] `/health` (or `/metrics`) is monitored, and systemd restarts the server on failure.
- [ ] `rustingstrata calibrate` ran on this machine as the service user.
