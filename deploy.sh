#!/usr/bin/env bash
# deploy.sh — fetch a model and serve it from a container, in one command.
#
#   ./deploy.sh flint-q4               # check, pull, download, serve, wait
#   ./deploy.sh flint-q4 --ctx 393216  # any run.sh flag passes through
#   ./deploy.sh status                 # what is running, and is it healthy
#   ./deploy.sh smoke                  # 8 parallel requests + a tool call
#   ./deploy.sh logs                   # follow the server log
#   ./deploy.sh stop                   # remove the container
#
# For a box that serves several people: the entries meant for this (see
# ./run.sh --list) default to several slots on a shared KV pool. Nothing is
# built or installed on the host — llama.cpp comes from $LLAMA_IMAGE — and
# nothing here uses sudo: a missing prerequisite is reported, not installed.
#
# One model at a time. The container is named $DEPLOY_CONTAINER; deploying
# again, with the same model or another, replaces it. It restarts with the
# docker daemon, so it comes back after a reboot.
#
# The endpoint requires an API key unless you pass --no-auth: the server
# listens on every interface ($HOST, default 0.0.0.0) and is meant to be
# reached by other machines. The key is $API_KEY if set, else the contents of
# settings/api-key, else a new random one written to that file (mode 600).
# `./deploy.sh --print-key` shows it.
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

KEY_FILE="$SETTINGS_DIR/api-key"
POLL_SECONDS="${DEPLOY_POLL_SECONDS:-5}"

usage() {
  cat <<USAGE
usage: ./deploy.sh <model> [run.sh flags] [--update] [--no-auth] [--timeout S]
       ./deploy.sh stop | status | logs | smoke [N] | --print-key

Fetches <model> and serves it from the llama.cpp container ($LLAMA_IMAGE),
detached, as container "$DEPLOY_CONTAINER". Models: ./run.sh --list

  --update      pull the image even if one is already present
  --no-auth     serve without an API key (anyone who can reach the port can
                use the model)
  --timeout S   seconds to wait for the server to become healthy
                (default 1200 — a large model on a slow disk takes minutes)
  anything else is passed to ./run.sh — see ./run.sh --help

  stop          remove the container
  status        container state, model, endpoint and /health
  logs          follow the server log
  smoke [N]     check the running server: one chat, one tool call, and N
                parallel streams (default 8), with tokens/s per stream
  --print-key   print the API key, creating it if there is none yet

env: API_KEY, HOST, PORT, LLAMA_IMAGE, DEPLOY_CONTAINER, HF_TOKEN, HF_HOME
USAGE
}

die() { echo "deploy: $*" >&2; exit 1; }
step() { echo; echo "== $*"; }

# api_key — print the key, creating settings/api-key on first use.
# A key file that holds only whitespace is treated as no key file: an empty key
# would start the server with no authentication at all while everything here
# went on reporting that it had one.
api_key() {
  local key=""
  if [[ -n "${API_KEY:-}" ]]; then echo "$API_KEY"; return; fi
  [[ -f "$KEY_FILE" ]] && key="$(tr -d '[:space:]' < "$KEY_FILE")"
  if [[ -z "$key" ]]; then
    mkdir -p "$SETTINGS_DIR"
    # umask in a subshell: a new file is never readable by anyone else, not
    # even for a moment. chmod covers a file that already existed.
    ( umask 077
      if command -v openssl >/dev/null 2>&1; then openssl rand -hex 24
      else head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n'; echo
      fi > "$KEY_FILE" )
    chmod 600 "$KEY_FILE"
    key="$(tr -d '[:space:]' < "$KEY_FILE")"
  fi
  echo "$key"
}

container_exists() {
  [[ -n "$(docker ps -aq --filter "name=^${DEPLOY_CONTAINER}$" 2>/dev/null)" ]]
}

# endpoint — http://host:port of the running container, as docker published
# it. Asked of docker rather than worked out from flags: --port and $HOST can
# both arrive through the run.sh passthrough, and docker is what knows.
endpoint() {
  local line addr host port
  line="$(docker port "$DEPLOY_CONTAINER" 2>/dev/null | head -n1)"
  [[ -n "$line" ]] || return 1
  addr="${line##* -> }"; host="${addr%:*}"; port="${addr##*:}"
  case "$host" in 0.0.0.0|"[::]"|"::") host=localhost ;; esac
  echo "http://$host:$port"
}

label() {
  docker inspect -f "{{ index .Config.Labels \"$1\" }}" "$DEPLOY_CONTAINER" 2>/dev/null || true
}

# The key the *running* server was started with — which is what a client must
# send, whatever $API_KEY or the key file say now. Empty for --no-auth.
running_key() {
  docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$DEPLOY_CONTAINER" 2>/dev/null \
    | sed -n 's/^LLAMA_API_KEY=//p' | head -n1
}

# --noproxy: the server is on this machine. A box that needs http_proxy to
# reach the internet (for the pull and the download) would otherwise send this
# request to the proxy, never see a 200, and give up on a healthy server.
health_code() {
  curl -s -o /dev/null -m 5 --noproxy '*' -w '%{http_code}' "$1/health" 2>/dev/null || true
}

# image_build — llama.cpp build number of the image (no GPU needed to ask).
# Both --version formats, as preflight.sh's build_of reads them.
image_build() {
  docker run --rm "$LLAMA_IMAGE" --version 2>&1 | sed -n \
    -e 's/^version: .*(build \([0-9]*\).*/\1/p' \
    -e 's/^version: \([0-9]*\).*/\1/p' | head -n1
}

cmd_status() {
  if ! container_exists; then
    echo "container: $DEPLOY_CONTAINER — not running"
    return 0
  fi
  local url; url="$(endpoint || true)"
  echo "container: $DEPLOY_CONTAINER — running=$(docker inspect -f '{{.State.Running}}' "$DEPLOY_CONTAINER" 2>/dev/null)"
  echo "model:     $(label llm.key)  (id: $(label llm.alias))"
  if [[ -n "$url" ]]; then
    echo "endpoint:  $url/v1"
    echo "health: $(health_code "$url")"
  else
    echo "endpoint:  none published"
  fi
  if [[ -n "$(running_key)" ]]; then echo "auth:      API key required (./deploy.sh --print-key)"
  else                               echo "auth:      none"; fi
}

cmd_smoke() {
  local n="${1:-8}" url
  [[ "$n" =~ ^[0-9]+$ ]] && (( n >= 1 )) || die "smoke takes a positive number of streams (got: $n)"
  container_exists || die "nothing is running — ./deploy.sh <model> first"
  url="$(endpoint)" || die "the container publishes no port"
  # The key goes through the environment, not the argv, so it does not show
  # up in the process list.
  SMOKE_API_KEY="$(running_key)" NO_PROXY='*' no_proxy='*' exec python3 -I ./smoke.py \
    --base-url "$url/v1" --model "$(label llm.alias)" -n "$n"
}

[[ $# -gt 0 ]] || { usage >&2; exit 1; }
case "$1" in
  -h|--help) usage; exit 0 ;;
  --print-key)
    # What clients must send is the key the running server was started with,
    # which is not the key file's if that deploy used $API_KEY.
    RUNNING_KEY=""
    if command -v docker >/dev/null 2>&1 && container_exists; then RUNNING_KEY="$(running_key)"; fi
    if [[ -n "$RUNNING_KEY" ]]; then echo "$RUNNING_KEY"; else api_key; fi
    exit 0 ;;
  stop)
    if container_exists; then docker rm -f "$DEPLOY_CONTAINER" >/dev/null; echo "stopped $DEPLOY_CONTAINER"
    else echo "$DEPLOY_CONTAINER is not running"; fi
    exit 0 ;;
  status) cmd_status; exit 0 ;;
  logs)   exec docker logs -f "$DEPLOY_CONTAINER" ;;
  smoke)  shift; cmd_smoke "$@" ;;
  -*) echo "the model key comes first: ./deploy.sh <model> [flags]" >&2; usage >&2; exit 1 ;;
esac

# ── Deploy ───────────────────────────────────────────────────────────────────
MODEL_KEY="$1"; shift
model_info "$MODEL_KEY" repo >/dev/null 2>&1 \
  || { echo "unknown model: $MODEL_KEY" >&2; echo "see ./run.sh --list" >&2; exit 1; }

ENGINE="$(model_info "$MODEL_KEY" engine)"
case "$ENGINE" in
  llama) ;;
  sglang) die "$MODEL_KEY is served by ./run-sglang.sh — deploy.sh does not deploy it." ;;
  none)   die "$MODEL_KEY is not deployable: no script in this repo serves it (./run.sh --list)." ;;
  *)      die "$MODEL_KEY uses the $ENGINE engine, which deploy.sh does not support yet." ;;
esac

UPDATE=0; AUTH=1; TIMEOUT=1200
PASS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --update)  UPDATE=1; shift ;;
    --no-auth) AUTH=0; shift ;;
    --timeout) TIMEOUT="${2:?--timeout needs a value}"; shift 2 ;;
    # These would fight the deploy itself; the rest belongs to run.sh.
    --docker|--detach) shift ;;
    --print|--dry-run|--save-defaults|--reset-defaults)
      die "$1 is a run.sh flag that does not deploy anything: ./run.sh $MODEL_KEY --docker $1" ;;
    --api-key) die "set the key with API_KEY=... (or settings/api-key), so it stays off the command line" ;;
    *) PASS+=("$1"); shift ;;
  esac
done
[[ "$TIMEOUT" =~ ^[0-9]+$ ]] && (( TIMEOUT >= 1 )) || die "--timeout must be a positive number of seconds (got: $TIMEOUT)"

# Have run.sh check the passed-through flags now, by composing the command
# without running it — a typo should not cost a 7 GB pull and a 25 GB download
# before it is noticed.
if ! FLAG_ERR="$(./run.sh "$MODEL_KEY" --docker --detach ${PASS[@]+"${PASS[@]}"} --print 2>&1 >/dev/null)"; then
  echo "$FLAG_ERR" | grep -v '^defaults for ' | head -n 3 >&2
  die "run.sh rejected those flags — see ./run.sh --help"
fi

step "1/6 prerequisites"
command -v docker >/dev/null 2>&1 \
  || die "docker is not installed. Install Docker Engine: https://docs.docker.com/engine/install/"
docker info >/dev/null 2>&1 \
  || die "the docker daemon is not reachable by this user. Start it, or add yourself to the docker group (then log in again)."
command -v curl >/dev/null 2>&1 || die "curl is not installed (needed to wait for the server)."
command -v nvidia-smi >/dev/null 2>&1 \
  || die "nvidia-smi not found — install the NVIDIA driver for this GPU first."
SMI_OUT="$(nvidia-smi --query-gpu=name,memory.total,driver_version \
    --format=csv,noheader,nounits 2>&1)" \
  || die "nvidia-smi failed — the NVIDIA driver is installed but not working: $(head -n1 <<<"$SMI_OUT")"
IFS=',' read -r GPU_NAME GPU_TOTAL GPU_DRIVER <<<"$(head -n1 <<<"$SMI_OUT")"
[[ -n "${GPU_DRIVER:-}" ]] || die "nvidia-smi reported no GPU: ${SMI_OUT:-(no output)}"
# The download needs the hf CLI, which fetch-models.sh can install given
# python3 — find out now, not after the image pull.
command -v hf >/dev/null 2>&1 || command -v huggingface-cli >/dev/null 2>&1 \
  || [[ -x "${HF_VENV:-$PWD/.venv}/bin/hf" ]] || command -v python3 >/dev/null 2>&1 \
  || die "neither the hf CLI nor python3 is installed (needed to download weights)."
echo "  GPU: ${GPU_NAME# } — ${GPU_TOTAL# } MiB, driver ${GPU_DRIVER# }"
echo "  docker: $(docker --version 2>/dev/null | head -n1)"

step "2/6 image $LLAMA_IMAGE"
if (( UPDATE )) || ! docker image inspect "$LLAMA_IMAGE" >/dev/null 2>&1; then
  docker pull "$LLAMA_IMAGE"
else
  echo "  present (./deploy.sh $MODEL_KEY --update pulls a newer one)"
fi

# Before anything else uses the image: if the GPU is not reachable from a
# container, that — not some later symptom of it — is what gets reported.
# A device line on stdout ("  CUDA0: <name> (...)") is the evidence, not any
# mention of CUDA: the backend library logs that it loaded even when the
# driver gave it no device, and the server would then quietly run on the CPU.
GPU_ERR="$(mktemp)"
if ! GPU_OUT="$(docker run --rm --gpus all "$LLAMA_IMAGE" --list-devices 2>"$GPU_ERR")" \
   || ! grep -Eq '^[[:space:]]*CUDA[0-9]+:' <<<"$GPU_OUT"; then
  { echo "$GPU_OUT"; cat "$GPU_ERR"; } | tail -n 6 >&2
  rm -f "$GPU_ERR"
  echo >&2
  echo "deploy: no CUDA device is visible inside a container." >&2
  echo "  - Install nvidia-container-toolkit and restart docker:" >&2
  echo "    https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html" >&2
  echo "  - If that is installed, the host driver (${GPU_DRIVER# }) may be too old for this" >&2
  echo "    CUDA 12 image; update the driver." >&2
  exit 1
fi
rm -f "$GPU_ERR"
echo "  GPU visible in container"

MIN_BUILD="$(model_info "$MODEL_KEY" minbuild)"
BUILD="$(image_build)"
if (( MIN_BUILD > 0 )) && (( ${BUILD:-0} < MIN_BUILD )); then
  echo "  build ${BUILD:-unknown} is older than $MODEL_KEY needs (b$MIN_BUILD) — pulling once"
  docker pull "$LLAMA_IMAGE"
  BUILD="$(image_build)"
  (( ${BUILD:-0} >= MIN_BUILD )) \
    || die "$LLAMA_IMAGE is build ${BUILD:-unknown}; $MODEL_KEY needs llama.cpp >= b$MIN_BUILD. Set LLAMA_IMAGE to a newer tag."
fi
echo "  llama.cpp build ${BUILD:-unknown}"

step "3/6 API key"
SERVER_KEY=""
if (( AUTH )); then
  SERVER_KEY="$(api_key)"
  [[ -n "$SERVER_KEY" ]] || die "the API key is empty (check $KEY_FILE) — refusing to start an open server. Use --no-auth if that is what you want."
  if [[ -n "${API_KEY:-}" ]]; then echo "  from \$API_KEY"
  else                             echo "  $KEY_FILE"; fi
else
  echo "  --no-auth: the endpoint will be unauthenticated — anyone who can reach"
  echo "  the server's port on this machine can use the model."
fi

step "4/6 weights"
./fetch-models.sh "$MODEL_KEY"

step "5/6 launch"
# The key is put in the environment for this one command only — the download
# above had no business seeing it. An inherited LLAMA_API_KEY is dropped for
# --no-auth, or run.sh would forward that one.
# ${PASS[@]+...}: an empty array is "unbound" to bash < 4.4 under set -u.
unset LLAMA_API_KEY
if (( AUTH )); then
  LLAMA_API_KEY="$SERVER_KEY" ./run.sh "$MODEL_KEY" --docker --detach ${PASS[@]+"${PASS[@]}"}
else
  ./run.sh "$MODEL_KEY" --docker --detach ${PASS[@]+"${PASS[@]}"}
fi

step "6/6 waiting for the server (up to ${TIMEOUT}s)"
give_up() {
  echo >&2
  echo "deploy: $1" >&2
  echo "---- last lines of: docker logs $DEPLOY_CONTAINER ----" >&2
  docker logs --tail 40 "$DEPLOY_CONTAINER" >&2 2>&1 || true
  echo "------------------------------------------------------" >&2
  # The restart policy would otherwise reload a model that cannot load, over
  # and over. Stopped, not removed: the log above stays readable.
  docker stop "$DEPLOY_CONTAINER" >/dev/null 2>&1 || true
  exit 1
}
# A server that exits at once (a flag this image does not know, weights it
# cannot read) is already being restarted by docker and has no port to report.
URL="$(endpoint)" || give_up "the server exited as soon as it started."
START=$SECONDS; LAST_NOTE=0
while :; do
  [[ "$(health_code "$URL")" == 200 ]] && break
  # RestartCount catches a crash loop that happens to be mid-restart, and so
  # "running", at the instant we look.
  read -r RUNNING RESTARTS < <(docker inspect \
      -f '{{.State.Running}} {{.RestartCount}}' "$DEPLOY_CONTAINER" 2>/dev/null || echo "false 0")
  if [[ "$RUNNING" != true ]] || (( ${RESTARTS:-0} > 0 )); then
    give_up "the server exited while loading."
  fi
  ELAPSED=$(( SECONDS - START ))
  (( ELAPSED >= TIMEOUT )) && give_up "not healthy after ${TIMEOUT}s (raise it with --timeout)."
  if (( ELAPSED - LAST_NOTE >= 30 )); then echo "  ...loading (${ELAPSED}s)"; LAST_NOTE=$ELAPSED; fi
  sleep "$POLL_SECONDS"
done

ALIAS="$(label llm.alias)"
echo
echo "Serving $MODEL_KEY — healthy after $(( SECONDS - START ))s"
echo "  endpoint:  $URL/v1   (OpenAI-compatible)"
echo "  model id:  ${ALIAS:-see $URL/v1/models}"
if (( AUTH )); then
  if [[ -n "${API_KEY:-}" ]]; then echo "  API key:   the value of \$API_KEY"
  else                             echo "  API key:   in $KEY_FILE   (./deploy.sh --print-key)"; fi
else
  echo "  API key:   none (--no-auth)"
fi
echo "  check:     ./deploy.sh smoke      logs: ./deploy.sh logs      stop: ./deploy.sh stop"
