#!/usr/bin/env bash
# tests/run-tests.sh — argv-level tests for the launch scripts.
#
#   tests/run-tests.sh                 # run everything
#   UPDATE_GOLDEN=1 tests/run-tests.sh # re-record tests/golden/*.argv
#
# No GPU, no Docker daemon and no model weights are needed: llama-server is
# replaced by tests/stub-llama-server, and the HuggingFace cache by a temp
# directory of empty files laid out the way `hf download` writes them. That
# temp path is rewritten to @HUB@ before any comparison, so golden files are
# the same on every machine.
set -uo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TMP="$(mktemp -d)"
# Helpers the tests start (the stub "container", fake endpoints) must not
# outlive a failed run: a leftover one holds its port and fails the next run.
cleanup() {
  [[ -f "$TMP/stub/server.pid" ]] && kill "$(cat "$TMP/stub/server.pid")" 2>/dev/null
  [[ -n "${FAKE_PID:-}" ]] && kill "$FAKE_PID" 2>/dev/null
  [[ -n "${HTTP_PID:-}" ]] && kill "$HTTP_PID" 2>/dev/null
  rm -rf "$TMP"
}
trap cleanup EXIT

# A clean, fixed environment: nothing from the caller's shell may leak into
# what the scripts compose.
unset HOST LLAMA_DOCKER EXTRA_ARGS DOCKER_ARGS API_KEY LLAMA_API_KEY \
      LLAMA_IMAGE DEPLOY_CONTAINER HF_HOME HF_CACHE HF_TOKEN LLAMA_MAINLINE
export HF_HUB_CACHE="$TMP/hub"
export SETTINGS_DIR="$TMP/settings"
export SKIP_PREFLIGHT=1
export PORT=18999
export LLAMA_SERVER="$ROOT/tests/stub-llama-server"
export STUB_ARGV_OUT="$TMP/argv"
export DEPLOY_POLL_SECONDS=1

source ./env.sh

# fake_hub <hub dir> — empty files for every table entry's weights, projector
# and drafter, in the hub cache layout hf_resolve reads.
fake_hub() {
  local hub="$1" key f
  for key in "${MODEL_KEYS[@]}"; do
    [[ "$(model_info "$key" fmt)" == gguf ]] || continue
    for f in "$(model_info "$key" repo)|$(model_info "$key" path)" \
             "$(model_info "$key" repo)|$(model_info "$key" mmproj)" \
             "$(model_info "$key" draftrepo)|$(model_info "$key" draft)"; do
      local repo="${f%%|*}" rel="${f#*|}"
      [[ -n "$repo" && -n "$rel" ]] || continue
      local dir="$hub/models--${repo//\//--}"
      mkdir -p "$dir/refs" "$(dirname "$dir/snapshots/abc/$rel")"
      echo abc > "$dir/refs/main"
      : > "$dir/snapshots/abc/$rel"
    done
  done
}
fake_hub "$HF_HUB_CACHE"

pass=0; fail=0
ok()  { pass=$((pass + 1)); }
bad() { fail=$((fail + 1)); echo "FAIL: $*" >&2; }

norm() { sed "s|$HF_HUB_CACHE|@HUB@|g; s|$TMP|@TMP@|g"; }

# argv_of <run.sh args...> — launch through the stub; ARGV holds the
# llama-server argv, one element per line, paths normalised. RC is run.sh's
# exit status and ERR its stderr.
ARGV=""; RC=0; ERR=""
argv_of() {
  rm -f "$STUB_ARGV_OUT"
  ./run.sh "$@" >"$TMP/out" 2>"$TMP/err"; RC=$?
  ERR="$(cat "$TMP/err")"
  ARGV=""; [[ -f "$STUB_ARGV_OUT" ]] && ARGV="$(norm < "$STUB_ARGV_OUT")"
}

has()   { grep -qxF -- "$2" <<<"$ARGV" && ok || bad "$1: argv lacks '$2'"; }
lacks() { grep -qxF -- "$2" <<<"$ARGV" && bad "$1: argv has '$2'" || ok; }
# pair <name> <flag> <value> — flag immediately followed by value.
pair() {
  grep -xF -A1 -- "$2" <<<"$ARGV" | grep -qxF -- "$3" && ok \
    || bad "$1: argv lacks '$2 $3'"
}
eq()       { [[ "$2" == "$3" ]] && ok || bad "$1: got '$2', want '$3'"; }
contains() { [[ "$2" == *"$3"* ]] && ok || bad "$1: '$3' not in output"; }
excludes() { [[ "$2" == *"$3"* ]] && bad "$1: '$3' in output" || ok; }
fails()    { local n="$1"; shift; "$@" >/dev/null 2>"$TMP/err" && bad "$n: exited 0" || ok; ERR="$(cat "$TMP/err")"; }
succeeds() { local n="$1"; shift; "$@" >/dev/null 2>"$TMP/err" && ok || bad "$n: exited non-zero: $(tail -n2 "$TMP/err")"; }

# golden <name> <run.sh args...> — host-mode argv must match the recording.
golden() {
  local name="$1" file="tests/golden/$1.argv"; shift
  argv_of "$@"
  if [[ "${UPDATE_GOLDEN:-0}" == 1 ]]; then
    printf '%s\n' "$ARGV" > "$file"; ok; return
  fi
  if [[ ! -f "$file" ]]; then bad "golden $name: no recording"; return; fi
  if diff -u "$file" <(printf '%s\n' "$ARGV") >"$TMP/diff"; then ok
  else bad "golden $name: argv changed"; cat "$TMP/diff" >&2; fi
}

# ── Host-mode regression: argv recorded before the docker backend existed ────
golden q8                 q8
golden q8-ctx-kv-nothink  q8 --ctx 65536 --kv q8_0 --no-think
golden 38-q8-effort       38-q8 --effort low
golden 38-q8-novision-par 38-q8 --no-vision --parallel 2 --cache-reuse 256
golden deckard            deckard
golden hauhau-mtp4        hauhau --mtp 4
golden g4-heretic         g4-heretic
golden g4-heretic-nothink g4-heretic --no-think

# ── Model table ──────────────────────────────────────────────────────────────
eq "engine: gguf entries default to llama" "$(model_info q8 engine)" llama
eq "engine: 38-nvfp4 is sglang"            "$(model_info 38-nvfp4 engine)" sglang
eq "engine: nvfp4-unsloth has none"        "$(model_info nvfp4-unsloth engine)" none
eq "think: default is switch"              "$(model_info q8 think)" switch
eq "alias: default is empty"               "$(model_info q8 alias)" ""
eq "flint-q4 repo"     "$(model_info flint-q4 repo)" unsloth/flint-GGUF
eq "flint-q4 file"     "$(model_info flint-q4 file)" flint-UD-Q4_K_XL.gguf
eq "flint-q4 engine"   "$(model_info flint-q4 engine)" llama
eq "flint-q4 minbuild" "$(model_info flint-q4 minbuild)" 11803
eq "flint-q4 think"    "$(model_info flint-q4 think)" auto
eq "flint-q4 alias"    "$(model_info flint-q4 alias)" flint
eq "flint-q4 temp"     "$(model_info flint-q4 temp)" 1.0
eq "flint-q4 mmproj"   "$(model_info flint-q4 mmproj)" ""
eq "flint-q4 args"     "$(model_info flint-q4 args)" "--no-mtp --parallel 8 --ctx 655360 --slot-ctx 131072"
eq "flint-nvfp4 repo"  "$(model_info flint-nvfp4 repo)" RecursiveSI/flint-nvfp4-gguf
eq "flint-nvfp4 file"  "$(model_info flint-nvfp4 file)" flint-nvfp4.gguf
eq "flint-nvfp4 think" "$(model_info flint-nvfp4 think)" auto
eq "LLAMA_IMAGE default"      "$LLAMA_IMAGE" ghcr.io/ggml-org/llama.cpp:server-cuda
eq "DEPLOY_CONTAINER default" "$DEPLOY_CONTAINER" llm-serve
contains "list marks the sglang entry"  "$(model_list | grep '^38-nvfp4 ')" "[sglang]"
contains "list shows flint-q4"          "$(model_list)" "flint-q4"
eq "inkling-q4 repo"     "$(model_info inkling-q4 repo)" unsloth/Inkling-2-Mini-GGUF
eq "inkling-q4 file"     "$(model_info inkling-q4 file)" Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file
eq "inkling-q4 path"     "$(model_info inkling-q4 path)" Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file
eq "inkling-q4 mmproj"   "$(model_info inkling-q4 mmproj)" mmproj-F16.gguf
eq "inkling-q4 engine"   "$(model_info inkling-q4 engine)" llama
eq "inkling-q4 minbuild" "$(model_info inkling-q4 minbuild)" 12611
eq "inkling-q4 think"    "$(model_info inkling-q4 think)" effort
eq "inkling-q4 sampling" "$(model_info inkling-q4 sampling)" plain
eq "inkling-q4 temp"     "$(model_info inkling-q4 temp)" 1.0
eq "inkling-q4 alias"    "$(model_info inkling-q4 alias)" inkling-2-mini
eq "inkling-q4 args"     "$(model_info inkling-q4 args)" "--no-mtp --parallel 8 --ctx 524288 --slot-ctx 131072 --ubatch 64"
eq "inkling-q8 args"     "$(model_info inkling-q8 args)" "--no-mtp --parallel 8 --ctx 393216 --slot-ctx 131072 --ubatch 64"
eq "inkling-q8 file"     "$(model_info inkling-q8 file)" Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf_file
eq "inkling-q8 minbuild" "$(model_info inkling-q8 minbuild)" 12611
eq "inkling-q8 alias"    "$(model_info inkling-q8 alias)" inkling-2-mini
contains "list shows inkling-q4"        "$(model_list)" "inkling-q4"

# ── run.sh flags ─────────────────────────────────────────────────────────────
argv_of flint-q4
eq    "flint-q4 launches"              "$RC" 0
pair  "flint-q4 alias from table"      --alias flint
has   "flint-q4 shared KV pool"        --kv-unified
pair  "flint-q4 per-slot cap"          --kv-unified-per-slot 131072
pair  "flint-q4 slots"                 --parallel 8
pair  "flint-q4 pool size"             -c 655360
has   "flint-q4 metrics on"            --metrics
lacks "think=auto passes no --reasoning"          --reasoning
lacks "think=auto passes no --reasoning-preserve" --reasoning-preserve
lacks "flint-q4 no MTP"                --spec-type
pair  "flint-q4 thinking-mode temp"    --temp 1.0
pair  "flint-q4 top-k"                 --top-k 20
pair  "flint-q4 top-p"                 --top-p 0.95
lacks "flint-q4 never --no-mmap"       --no-mmap
lacks "flint-q4 never --mlock"         --mlock

fails "think=auto rejects --no-think" ./run.sh flint-q4 --no-think --print
contains "…and says why" "$ERR" "no thinking switch"
fails "think=auto rejects --think"    ./run.sh flint-q4 --think --print
fails "think=auto rejects --effort"   ./run.sh flint-q4 --effort low --print

argv_of q8 --api-key s3cret
pair  "--api-key passes through" --api-key s3cret
argv_of q8
lacks "no --metrics for a single-slot entry" --metrics
lacks "no --alias when the table has none"   --alias
lacks "no --kv-unified without --slot-ctx"   --kv-unified
argv_of q8 --parallel 4
lacks "--parallel alone adds no --metrics" --metrics
argv_of q8 --parallel 4 --slot-ctx 8192
has   "--metrics follows --slot-ctx" --metrics

fails "--slot-ctx rejects a non-integer" ./run.sh q8 --slot-ctx lots --print

# --ubatch: the micro-batch. Unset passes nothing, so llama-server's own
# default stands and no existing entry's argv moves.
argv_of q8
lacks "no -ub unless asked"            -ub
argv_of q8 --ubatch 64
pair  "--ubatch passes -ub"            -ub 64
fails "--ubatch rejects a non-integer" ./run.sh q8 --ubatch small --print
fails "--ubatch rejects zero"          ./run.sh q8 --ubatch 0 --print
./run.sh q8 --ubatch 64 --save-defaults --print >/dev/null 2>&1
contains "save-defaults keeps --ubatch" "$(cat "$SETTINGS_DIR/q8.conf" 2>/dev/null)" "--ubatch 64"
rm -f "$SETTINGS_DIR/q8.conf"

# Inkling 2 Mini: the plain sampler, and a template that reasons at an effort
# level with no switch (think=effort).
argv_of inkling-q4
eq    "inkling-q4 launches"            "$RC" 0
pair  "inkling-q4 alias"               --alias inkling-2-mini
pair  "inkling-q4 temp"                --temp 1.0
pair  "inkling-q4 top-p"               --top-p 1.0
pair  "inkling-q4 top-k"               --top-k 0
pair  "inkling-q4 min-p"               --min-p 0.0
pair  "inkling-q4 slots"               --parallel 8
pair  "inkling-q4 per-slot cap"        --kv-unified-per-slot 131072
pair  "inkling-q4 pool"                -c 524288
pair  "inkling-q4 micro-batch"         -ub 64
has   "inkling-q4 vision on"           --mmproj
lacks "inkling-q4 no reasoning switch" --reasoning
lacks "inkling-q4 no preserve flag"    --reasoning-preserve
lacks "inkling-q4 no preserve-off flag" --no-reasoning-preserve
lacks "inkling-q4 effort unset by default" --chat-template-kwargs
lacks "inkling-q4 no MTP"              --spec-type
lacks "inkling-q4 never --no-mmap"     --no-mmap
lacks "inkling-q4 never --mlock"       --mlock
for e in none minimal low medium high xhigh max; do
  argv_of inkling-q4 --effort "$e"
  pair "inkling-q4 --effort $e" --chat-template-kwargs "{\"reasoning_effort\":\"$e\"}"
done
argv_of inkling-q4 --temp 0.7
pair  "plain sampler honours --temp" --temp 0.7
# A bad effort fails at launch, not as a 500 on the first request.
fails "effort mode rejects an unknown name" ./run.sh inkling-q4 --effort hi --print
contains "…lists the valid names"           "$ERR" "none, minimal, low, medium, high, xhigh, max"
fails "effort mode rejects a number"        ./run.sh inkling-q4 --effort 0.5 --print
# --no-think out of habit must not start a server that reasons anyway.
fails "effort mode rejects --no-think"      ./run.sh inkling-q4 --no-think --print
contains "…names the alternative"           "$ERR" "--effort none"
fails "effort mode rejects --think"         ./run.sh inkling-q4 --think --print
argv_of inkling-q4 --no-vision
lacks "inkling-q4 --no-vision drops the projector" --mmproj
mkdir -p "$TMP/noproj"; fake_hub "$TMP/noproj"
rm -f "$TMP/noproj/models--unsloth--Inkling-2-Mini-GGUF/snapshots/abc/mmproj-F16.gguf"
HF_HUB_CACHE="$TMP/noproj" argv_of inkling-q4
eq       "missing projector still launches" "$RC" 0
lacks    "…without --mmproj"                --mmproj
contains "…and warns"                       "$ERR" "serving text-only"
# Qwen3.8's own effort set is untouched.
fails "switch mode still rejects --effort high" ./run.sh 38-q8 --effort high --print
./run.sh inkling-q4 --effort low --save-defaults --print >/dev/null 2>&1
contains "save-defaults keeps --effort" "$(cat "$SETTINGS_DIR/inkling-q4.conf" 2>/dev/null)" "--effort low"
rm -f "$SETTINGS_DIR/inkling-q4.conf"
# The launch summary names the think mode actually in force.
argv_of inkling-q4
contains "summary: effort default"  "$(cat "$TMP/out")" "thinking=effort:template default (high)  vision="
argv_of inkling-q4 --effort low
contains "summary: effort level"    "$(cat "$TMP/out")" "thinking=effort:low  vision="
argv_of flint-q4
contains "summary: auto unchanged"  "$(cat "$TMP/out")" "thinking=auto  vision="
excludes "summary: auto has no effort" "$(cat "$TMP/out")" "effort:"
argv_of 38-q8 --effort low
contains "summary: switch unchanged" "$(cat "$TMP/out")" "thinking=on(preserve=on,effort=low)  vision="

out="$(./run.sh q8 --ctx 65536 --print 2>/dev/null | norm)"
contains "--print shows the host command" "$out" "tests/stub-llama-server -m @HUB@/models--unsloth--Qwen3.6-27B-MTP-GGUF/snapshots/abc/Qwen3.6-27B-Q8_0.gguf"
contains "--print carries the flags"      "$out" "-c 65536"
rm -f "$STUB_ARGV_OUT"; ./run.sh q8 --print >/dev/null 2>&1
[[ ! -e "$STUB_ARGV_OUT" ]] && ok || bad "--print launched the server"

# Review Focus 1: what --save-defaults may and may not persist.
./run.sh q8 --api-key s3cret --ctx 65536 --slot-ctx 32768 --save-defaults --print >/dev/null 2>&1
conf="$(cat "$SETTINGS_DIR/q8.conf" 2>/dev/null)"
contains "save-defaults keeps --slot-ctx" "$conf" "--slot-ctx 32768"
contains "save-defaults keeps --ctx"      "$conf" "--ctx 65536"
excludes "save-defaults drops the key flag"  "$conf" "api-key"
excludes "save-defaults drops the key value" "$conf" "s3cret"
excludes "save-defaults drops --print"       "$conf" "print"
rm -f "$SETTINGS_DIR/q8.conf"

# ── run.sh --docker ──────────────────────────────────────────────────────────
dprint() { ./run.sh "$@" --print 2>"$TMP/err" | norm; }
out="$(dprint flint-q4 --docker)"
contains "docker: foreground uses --rm"   "$out" "docker run --rm --name llm-serve"
contains "docker: all GPUs"               "$out" "--gpus all"
contains "docker: hub mounted at its own path, read-only" "$out" "-v @HUB@:@HUB@:ro"
contains "docker: publishes HOST:PORT"    "$out" "-p 0.0.0.0:18999:18999"
contains "docker: key label"              "$out" "--label llm.key=flint-q4"
contains "docker: alias label"            "$out" "--label llm.alias=flint"
contains "docker: image then the same llama-server args" "$out" \
  "ghcr.io/ggml-org/llama.cpp:server-cuda -m @HUB@/models--unsloth--flint-GGUF/snapshots/abc/flint-UD-Q4_K_XL.gguf"
contains "docker: server binds 0.0.0.0 inside" "$out" "--host 0.0.0.0 --port 18999"
contains "docker: healthcheck follows the port" "$out" "--health-cmd curl\\ -fsS\\ http://localhost:18999/health"
# The container is handed the mainline *.gguf_file path as it sits in the cache.
out="$(dprint inkling-q8 --docker)"
contains "docker: mainline file path" "$out" \
  "ghcr.io/ggml-org/llama.cpp:server-cuda -m @HUB@/models--unsloth--Inkling-2-Mini-GGUF/snapshots/abc/Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf_file"
contains "docker: inkling alias label" "$out" "--label llm.alias=inkling-2-mini"
contains "docker: load time is not counted as unhealthy" "$out" "--health-start-period 20m"
excludes "docker: no host binary in the command" "$out" "stub-llama-server"
excludes "docker: no -it when printing"   "$out" " -it "

out="$(dprint flint-q4 --docker --detach)"
contains "detach: background + restart policy" "$out" "docker run -d --restart unless-stopped "
contains "detach: same container name"         "$out" "--name llm-serve"
excludes "detach: no --rm"                "$out" "--rm"

out="$(HOST=127.0.0.1 dprint flint-q4 --docker)"
contains "HOST narrows the published address" "$out" "-p 127.0.0.1:18999:18999"
contains "…but not the bind inside the container" "$out" "--host 0.0.0.0 --port 18999"

out="$(LLAMA_DOCKER=1 dprint flint-q4)"
contains "LLAMA_DOCKER=1 selects the backend" "$out" "docker run --rm"

out="$(LLAMA_IMAGE=example/img:tag DEPLOY_CONTAINER=other DOCKER_ARGS='--cpus 4' dprint flint-q4 --docker)"
contains "LLAMA_IMAGE override"      "$out" "example/img:tag -m "
contains "DEPLOY_CONTAINER override" "$out" "--name other"
contains "DOCKER_ARGS before the image" "$out" "--cpus 4 example/img:tag"

out="$(LLAMA_API_KEY=s3cret dprint flint-q4 --docker)"
contains "LLAMA_API_KEY is forwarded by name" "$out" "-e LLAMA_API_KEY"
excludes "…and its value stays off the argv"  "$out" "s3cret"
out="$(dprint flint-q4 --docker)"
excludes "no -e LLAMA_API_KEY when unset" "$out" "LLAMA_API_KEY"

out="$(cd "$TMP" && "$ROOT/run.sh" flint-q4 --docker --slot-save-path slots --lora lora/a.gguf --lora lora/b.gguf --print 2>/dev/null | norm)"
contains "slot-save dir mounted read-write" "$out" "-v @TMP@/slots:@TMP@/slots "
contains "lora dir mounted once, read-only" "$out" "-v @TMP@/lora:@TMP@/lora:ro"
eq "…exactly once" "$(grep -o -- '-v @TMP@/lora:' <<<"$out" | wc -l)" 1
contains "lora passed by absolute path" "$out" "--lora @TMP@/lora/a.gguf"

# Review Focus 5: a space in the cache path stays one argument.
fake_hub "$TMP/hf cache/hub"
out="$(HF_HUB_CACHE="$TMP/hf cache/hub" ./run.sh flint-q4 --docker --print 2>/dev/null | sed "s|$TMP|@TMP@|g")"
contains "space in HF_HUB_CACHE survives quoting" "$out" '-v @TMP@/hf\ cache/hub:@TMP@/hf\ cache/hub:ro'

fails "relative HF_HUB_CACHE is refused in docker mode" env HF_HUB_CACHE=rel/hub ./run.sh flint-q4 --docker --print
fails "--detach without --docker is an error" ./run.sh q8 --detach --print
contains "…and says so" "$ERR" "--detach"

# Review Focus 1: backend flags are per-launch, never saved.
./run.sh flint-q4 --docker --detach --save-defaults --print >/dev/null 2>&1
conf="$(cat "$SETTINGS_DIR/flint-q4.conf" 2>/dev/null)"
contains "save-defaults still writes the model flags" "$conf" "--slot-ctx 131072"
excludes "save-defaults drops --docker" "$conf" "docker"
excludes "save-defaults drops --detach" "$conf" "detach"
rm -f "$SETTINGS_DIR/flint-q4.conf"

# Review Focus 3: a port held by something that is not our container fails
# before any `docker run`.
mkdir -p "$TMP/bin"
printf '#!/usr/bin/env bash\necho "$*" >> "%s"\n' "$TMP/docker.calls" > "$TMP/bin/docker"
chmod +x "$TMP/bin/docker"
python3 -I -m http.server 18998 --bind 127.0.0.1 >/dev/null 2>&1 &
HTTP_PID=$!
for _ in $(seq 50); do ss -ltn 2>/dev/null | grep -q ':18998 ' && break; sleep 0.1; done
: > "$TMP/docker.calls"
PATH="$TMP/bin:$PATH" PORT=18998 ./run.sh flint-q4 --docker >/dev/null 2>"$TMP/err"; rc=$?
kill "$HTTP_PID" 2>/dev/null; wait "$HTTP_PID" 2>/dev/null
eq "foreign port holder: exit 1" "$rc" 1
contains "foreign port holder: named" "$(cat "$TMP/err")" "port 18998 is already in use"
excludes "foreign port holder: no docker run" "$(cat "$TMP/docker.calls")" "run "

# Docker mode never touches a host llama-server.
: > "$TMP/docker.calls"; rm -f "$STUB_ARGV_OUT"
PATH="$TMP/bin:$PATH" ./run.sh flint-q4 --docker >/dev/null 2>&1
contains "docker mode runs docker" "$(cat "$TMP/docker.calls")" "run --rm --name llm-serve"
[[ ! -e "$STUB_ARGV_OUT" ]] && ok || bad "docker mode exec'd the host binary"

# ── preflight.sh in docker mode ──────────────────────────────────────────────
# A stub docker stands in for the daemon; what it answers is set per case.
cat > "$TMP/bin/docker" <<STUB
#!/usr/bin/env bash
echo "\$*" >> "$TMP/docker.calls"
case "\$*" in
  info*)            exit \${STUB_INFO_RC:-0} ;;
  ps*)              [[ -n "\${STUB_PS:-}" ]] && echo "\$STUB_PS" ;;
  port*)            [[ -n "\${STUB_PS:-}" ]] && echo "\$PORT/tcp -> 0.0.0.0:\$PORT" ;;
  inspect*)         echo flint-q4 ;;
  "image inspect"*) exit \${STUB_IMAGE_RC:-0} ;;
  *--list-devices*) echo "Available devices:"; echo "\${STUB_DEVICES-  CUDA0: Stub GPU (48000 MiB, 47000 MiB free)}" ;;
  *--version*)      echo "version: 1.0.0-dev (build \${STUB_BUILD:-12429}, commit abc)" ;;
esac
exit 0
STUB
pre() { PATH="$TMP/bin:$PATH" LLAMA_DOCKER=1 ./preflight.sh "$@" >"$TMP/out" 2>&1; PRC=$?; POUT="$(cat "$TMP/out")"; }
pre flint-q4
eq       "preflight docker: passes with image, GPU and build" "$PRC" 0
contains "preflight docker: reports the image"  "$POUT" "ghcr.io/ggml-org/llama.cpp:server-cuda (build 12429)"
excludes "preflight docker: no host binary check" "$POUT" "llama-server not found"
STUB_BUILD=11711 pre flint-q4
eq       "preflight docker: old image fails"    "$PRC" 1
contains "preflight docker: names the floor"    "$POUT" "needs >= b11803"
STUB_IMAGE_RC=1 pre flint-q4
eq       "preflight docker: missing image fails" "$PRC" 1
contains "preflight docker: says how to get it"  "$POUT" "docker pull ghcr.io/ggml-org/llama.cpp:server-cuda"
STUB_DEVICES="" pre flint-q4
eq       "preflight docker: no GPU in container fails" "$PRC" 1
contains "preflight docker: points at the toolkit"     "$POUT" "nvidia-container-toolkit"
STUB_INFO_RC=1 pre flint-q4
eq       "preflight docker: unreachable daemon fails"  "$PRC" 1
contains "preflight docker: says the daemon is the problem" "$POUT" "docker daemon"
LLAMA_SERVER=/nonexistent ./preflight.sh q8 >"$TMP/out" 2>&1
contains "preflight host mode still checks the binary" "$(cat "$TMP/out")" "llama-server not found"

# ── fetch-models.sh: finding the hf CLI ──────────────────────────────────────
# A PATH with no hf on it, an empty hub so the model counts as missing, and a
# python3 whose venv module is unavailable (Debian/Ubuntu without python3-venv).
mkdir -p "$TMP/nohf" "$TMP/emptyhub"
printf '#!/usr/bin/env bash\necho "$*" >> "%s"\nexit 1\n' "$TMP/python.calls" > "$TMP/nohf/python3"
chmod +x "$TMP/nohf/python3"
: > "$TMP/python.calls"
PATH="$TMP/nohf:/usr/bin:/bin" HF_HUB_CACHE="$TMP/emptyhub" HF_VENV="$TMP/venv" \
  ./fetch-models.sh flint-q4 >"$TMP/out" 2>"$TMP/err"; rc=$?
eq       "no hf and no venv module: exit 1"      "$rc" 1
contains "…names the package to install"         "$(cat "$TMP/err")" "python3-venv"
contains "…tried a venv"                         "$(cat "$TMP/python.calls")" "-m venv $TMP/venv"
excludes "…never pip install --user"             "$(cat "$TMP/python.calls")" "--user"

# An hf already inside the venv is used without creating anything.
mkdir -p "$TMP/venv2/bin"
printf '#!/usr/bin/env bash\necho "$*" >> "%s"\n' "$TMP/hf.calls" > "$TMP/venv2/bin/hf"
chmod +x "$TMP/venv2/bin/hf"
: > "$TMP/hf.calls"; : > "$TMP/python.calls"
PATH="$TMP/nohf:/usr/bin:/bin" HF_HUB_CACHE="$TMP/emptyhub" HF_VENV="$TMP/venv2" \
  ./fetch-models.sh flint-q4 >/dev/null 2>&1
contains "venv hf is used for the download" "$(cat "$TMP/hf.calls")" "download unsloth/flint-GGUF flint-UD-Q4_K_XL.gguf"
eq       "…without touching python"         "$(cat "$TMP/python.calls")" ""

# A file in a repo subdirectory (Inkling's mainline layout) reaches the hf CLI
# with that path intact, and the projector is fetched alongside it.
: > "$TMP/hf.calls"
PATH="$TMP/nohf:/usr/bin:/bin" HF_HUB_CACHE="$TMP/emptyhub" HF_VENV="$TMP/venv2" \
  ./fetch-models.sh inkling-q4 >/dev/null 2>&1
contains "fetch: mainline file by its subdirectory path" "$(cat "$TMP/hf.calls")" \
  "download unsloth/Inkling-2-Mini-GGUF Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file"
contains "fetch: projector too" "$(cat "$TMP/hf.calls")" "download unsloth/Inkling-2-Mini-GGUF mmproj-F16.gguf"

# ── deploy.sh ────────────────────────────────────────────────────────────────
fails "deploy: unknown key"            ./deploy.sh nope
contains "…says so"                    "$ERR" "unknown model"
fails "deploy: sglang entry"           ./deploy.sh 38-nvfp4
contains "…points at its own script"   "$ERR" "run-sglang.sh"
fails "deploy: entry with no engine"   ./deploy.sh nvfp4-unsloth
contains "…says it is not deployable"  "$ERR" "not deployable"
fails "deploy: no arguments"           ./deploy.sh

eq "key: API_KEY wins" "$(API_KEY=abc ./deploy.sh --print-key)" abc
k="$(./deploy.sh --print-key)"
(( ${#k} >= 32 )) && ok || bad "key: generated key is short (${#k})"
eq "key: file is mode 600" "$(stat -c %a "$SETTINGS_DIR/api-key" 2>/dev/null)" 600
eq "key: stable across runs" "$(./deploy.sh --print-key)" "$k"

# End to end against tests/stub-docker: no daemon, no GPU, a real health poll.
export STUB_DIR="$TMP/stub"; mkdir -p "$STUB_DIR" "$TMP/dbin"
ln -sf "$ROOT/tests/stub-docker" "$TMP/dbin/docker"
cat > "$TMP/dbin/nvidia-smi" <<'SMI'
#!/usr/bin/env bash
if [[ -n "${STUB_SMI_FAIL:-}" ]]; then
  echo "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver."; exit 9
fi
echo "Stub GPU, 46068, 580.95.05"
SMI
chmod +x "$TMP/dbin/nvidia-smi"
dep() {
  : > "$STUB_DIR/docker.calls"
  PATH="$TMP/dbin:$PATH" PORT=18997 ./deploy.sh "$@" >"$TMP/out" 2>"$TMP/err"; DRC=$?
  DOUT="$(cat "$TMP/out" "$TMP/err")"; DCALLS="$(norm < "$STUB_DIR/docker.calls")"
}
launch_call() { grep -- '--name llm-serve' <<<"$DCALLS" | grep -- ' -d ' ; }

: > "$STUB_DIR/image-absent"
dep flint-q4 --timeout 30
eq       "deploy: succeeds"                  "$DRC" 0
contains "deploy: pulls a missing image"     "$DCALLS" "pull ghcr.io/ggml-org/llama.cpp:server-cuda"
contains "deploy: launches detached"         "$(launch_call)" "run -d --restart unless-stopped "
contains "deploy: same model flags as run.sh" "$(launch_call)" "--parallel 8 --kv-unified --kv-unified-per-slot 131072"
contains "deploy: key forwarded by name"     "$(launch_call)" "-e LLAMA_API_KEY"
excludes "deploy: key value not on the argv" "$DCALLS" "$k"
contains "deploy: prints the endpoint"       "$DOUT" "http://localhost:18997/v1"
contains "deploy: prints the model id"       "$DOUT" "model id:  flint"
contains "deploy: says where the key is"     "$DOUT" "$SETTINGS_DIR/api-key"
excludes "deploy: does not print the key"    "$DOUT" "$k"

# Review Focus 4: redeploying replaces the container and waits on the new one.
dep flint-q4 --timeout 30
eq       "redeploy: succeeds"                "$DRC" 0
contains "redeploy: names what it replaced"  "$DOUT" "replacing container llm-serve (was serving flint-q4)"
excludes "redeploy: image present, no pull"  "$DCALLS" "pull "
rm_line="$(grep -n '^rm -f llm-serve' <<<"$DCALLS" | head -n1 | cut -d: -f1)"
run_line="$(grep -n -- ' -d --restart' <<<"$DCALLS" | head -n1 | cut -d: -f1)"
[[ -n "$rm_line" && -n "$run_line" ]] && (( rm_line < run_line )) && ok || bad "redeploy: old container not removed before the new launch"

dep flint-q4 --update --timeout 30
contains "--update forces a pull"            "$DCALLS" "pull ghcr.io/ggml-org/llama.cpp:server-cuda"

dep flint-q4 --no-auth --ctx 131072 --port 18996 --timeout 30
eq       "deploy: flags pass through"        "$DRC" 0
excludes "--no-auth: no key in the container" "$(launch_call)" "LLAMA_API_KEY"
contains "--no-auth: warned"                 "$DOUT" "unauthenticated"
contains "passthrough reaches llama-server"  "$(launch_call)" "-c 131072"
contains "health polled on the port docker reports" "$DOUT" "http://localhost:18996/v1"

PATH="$TMP/dbin:$PATH" ./deploy.sh status >"$TMP/out" 2>&1
contains "status: names the model"  "$(cat "$TMP/out")" "flint-q4"
contains "status: reports health"   "$(cat "$TMP/out")" "health: 200"
PATH="$TMP/dbin:$PATH" ./deploy.sh stop >"$TMP/out" 2>&1
contains "stop: removes the container" "$(cat "$STUB_DIR/docker.calls")" "rm -f llm-serve"
PATH="$TMP/dbin:$PATH" ./deploy.sh status >"$TMP/out" 2>&1
contains "status: nothing running" "$(cat "$TMP/out")" "not running"

# Review Focus 2: a container that dies while loading ends the wait at once.
start=$SECONDS
STUB_DIES=1 dep flint-q4 --timeout 120
eq       "dies: exit 1"              "$DRC" 1
contains "dies: shows the log tail"  "$DOUT" "out of memory"
(( SECONDS - start < 60 )) && ok || bad "dies: waited out the timeout ($(( SECONDS - start ))s)"
PATH="$TMP/dbin:$PATH" ./deploy.sh stop >/dev/null 2>&1

# Too-old image: one pull, then give up if it is still too old.
STUB_BUILD=11711 dep flint-q4 --timeout 30
eq       "old image, pull does not help: exit 1" "$DRC" 1
contains "…names the floor"                      "$DOUT" "11803"
eq       "…pulled exactly once" "$(grep -c '^pull ' <<<"$DCALLS")" 1
excludes "…never launched"                       "$DCALLS" " -d --restart"
rm -f "$STUB_DIR/pulled"
STUB_BUILD=11711 STUB_BUILD_AFTER_PULL=12000 dep flint-q4 --timeout 30
eq       "old image, pull fixes it: deploys"     "$DRC" 0
PATH="$TMP/dbin:$PATH" ./deploy.sh stop >/dev/null 2>&1

# The image a Flint deploy left behind (b12429) is too old for Inkling, and
# below b12611 that model answers with garbage rather than refusing to load.
rm -f "$STUB_DIR/pulled"
STUB_BUILD=12429 dep inkling-q4 --timeout 30
eq       "inkling, old image, pull does not help: exit 1" "$DRC" 1
contains "…names the floor"                      "$DOUT" "12611"
excludes "…never launched"                       "$DCALLS" " -d --restart"
rm -f "$STUB_DIR/pulled"
STUB_BUILD=12429 STUB_BUILD_AFTER_PULL=12778 dep inkling-q4 --timeout 30
eq       "inkling, old image, pull fixes it: deploys" "$DRC" 0
contains "deploy: prints the inkling model id"   "$DOUT" "model id:  inkling-2-mini"
PATH="$TMP/dbin:$PATH" ./deploy.sh stop >/dev/null 2>&1

STUB_NO_GPU=1 dep flint-q4 --timeout 30
eq       "no GPU in container: exit 1"           "$DRC" 1
contains "…points at the toolkit"                "$DOUT" "nvidia-container-toolkit"
excludes "…never launched"                       "$DCALLS" " -d --restart"

# ── Review fixes ─────────────────────────────────────────────────────────────
# --print is exactly one line and touches nothing.
eq "--print: one line on stdout" "$(./run.sh flint-q4 --docker --print 2>/dev/null | wc -l)" 1
( cd "$TMP" && "$ROOT/run.sh" flint-q4 --docker --slot-save-path printdir --print >/dev/null 2>&1 )
[[ ! -e "$TMP/printdir" ]] && ok || bad "--print created the slot-save directory"

out="$(dprint flint-q4 --docker --detach)"
contains "detach: container log is rotated" "$out" "--log-opt max-size=50m --log-opt max-file=3"

fails "run.sh refuses the sglang entry" ./run.sh 38-nvfp4
contains "…naming its script from the engine field" "$ERR" "./run-sglang.sh"
fails "run.sh refuses an entry with no engine" ./run.sh nvfp4-unsloth
contains "…naming vLLM" "$ERR" "vllm serve unsloth/Qwen3.6-27B-NVFP4"

# preflight, docker mode, redeploying over our own container: say it will be
# replaced instead of warning that the launch will fail.
python3 -I -m http.server 18995 --bind 127.0.0.1 >/dev/null 2>&1 &
HTTP_PID=$!
for _ in $(seq 50); do ss -ltn 2>/dev/null | grep -q ':18995 ' && break; sleep 0.1; done
STUB_PS=c0ffee PORT=18995 pre flint-q4
kill "$HTTP_PID" 2>/dev/null; wait "$HTTP_PID" 2>/dev/null
contains "preflight docker: existing container will be replaced" "$POUT" "will be replaced"
excludes "preflight docker: no false port alarm on redeploy"     "$POUT" "fail loudly"

# An API key file that holds nothing must never mean "no key".
printf '\n' > "$SETTINGS_DIR/api-key"; chmod 644 "$SETTINGS_DIR/api-key"
k2="$(./deploy.sh --print-key)"
(( ${#k2} >= 32 )) && ok || bad "key: blank key file was accepted as a key (${#k2} chars)"
eq "key: mode 600 enforced on an existing file" "$(stat -c %a "$SETTINGS_DIR/api-key")" 600
k="$k2"

# Flags are checked before anything is pulled or downloaded.
: > "$STUB_DIR/image-absent"
dep flint-q4 --ctxx 5
eq       "bad flag: exit 1"                    "$DRC" 1
contains "bad flag: named"                     "$DOUT" "unknown flag: --ctxx"
excludes "bad flag: nothing pulled first"      "$DCALLS" "pull "
dep flint-q4 --no-think
eq       "flag the model rejects: exit 1"      "$DRC" 1
excludes "…nothing pulled first"               "$DCALLS" "pull "
dep flint-q4 --help
eq       "--help after the key: exit 0"        "$DRC" 0
contains "…shows deploy usage"                 "$DOUT" "usage: ./deploy.sh"
eq       "…and touches docker not at all"      "$DCALLS" ""
rm -f "$STUB_DIR/image-absent"

# The key reaches the container's environment, and --print-key reports the key
# the running server actually has.
API_KEY=team-key dep flint-q4 --timeout 30
eq       "API_KEY deploy succeeds"             "$DRC" 0
contains "key is in the container environment" "$(cat "$STUB_DIR/env")" "LLAMA_API_KEY=team-key"
eq "--print-key: the running server's key wins" "$(PATH="$TMP/dbin:$PATH" ./deploy.sh --print-key)" team-key

# A proxy in the environment must not swallow the localhost health poll.
http_proxy=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 dep flint-q4 --timeout 20
eq       "health poll ignores http_proxy"      "$DRC" 0
PATH="$TMP/dbin:$PATH" ./deploy.sh stop >/dev/null 2>&1

# A container that exits at once (bad flag, unreadable weights): docker keeps
# restarting it and reports no port. Deploy must show the log and stop it.
STUB_CRASHLOOP=1 dep flint-q4 --timeout 60
eq       "crash loop: exit 1"                  "$DRC" 1
contains "crash loop: shows the log tail"      "$DOUT" "out of memory"
contains "crash loop: stops the container"     "$DCALLS" "stop llm-serve"
PATH="$TMP/dbin:$PATH" ./deploy.sh stop >/dev/null 2>&1

STUB_SMI_FAIL=1 dep flint-q4 --timeout 20
eq       "broken nvidia-smi: exit 1"           "$DRC" 1
contains "broken nvidia-smi: says so"          "$DOUT" "nvidia-smi failed"

# ── smoke.py against a fake endpoint ─────────────────────────────────────────
fake() { python3 -I tests/fake-openai.py "$1" "$2" & FAKE_PID=$!
         for _ in $(seq 50); do ss -ltn 2>/dev/null | grep -q ":$1 " && break; sleep 0.1; done; }
smoke() { python3 -I smoke.py --base-url "http://127.0.0.1:$1/v1" -n 3 --timeout 10 >"$TMP/out" 2>&1; SRC=$?; SOUT="$(cat "$TMP/out")"; }
fake 18991 ok
smoke 18991
eq       "smoke: healthy endpoint passes"      "$SRC" 0
contains "smoke: counts the streams"           "$SOUT" "3/3 finished"
http_proxy=http://127.0.0.1:9 smoke 18991
eq       "smoke: ignores http_proxy"           "$SRC" 0
kill "$FAKE_PID" 2>/dev/null; wait "$FAKE_PID" 2>/dev/null
fake 18992 stream-error
smoke 18992
eq       "smoke: an error event fails the stream" "$SRC" 1
contains "smoke: and shows the server's message"  "$SOUT" "slot unavailable"
kill "$FAKE_PID" 2>/dev/null; wait "$FAKE_PID" 2>/dev/null
fake 18993 empty-stream
smoke 18993
eq       "smoke: a stream with no tokens fails" "$SRC" 1
kill "$FAKE_PID" 2>/dev/null; wait "$FAKE_PID" 2>/dev/null
fake 18994 no-tool
smoke 18994
eq       "smoke: a missing tool call fails"    "$SRC" 1
contains "smoke: names the failed check"       "$SOUT" "tool call  FAIL"
kill "$FAKE_PID" 2>/dev/null; wait "$FAKE_PID" 2>/dev/null

# ── smoke.py ─────────────────────────────────────────────────────────────────
succeeds "smoke.py --help" python3 -I smoke.py --help
fails    "smoke.py against a dead endpoint" python3 -I smoke.py --base-url http://127.0.0.1:18990/v1 --model x -n 1 --timeout 3

echo
echo "passed: $pass  failed: $fail"
(( fail == 0 ))
