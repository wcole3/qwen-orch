#!/usr/bin/env bash
# preflight.sh — verify everything needed to launch is in place.
# Usage: ./preflight.sh <model-key>   (e.g. nvfp4, q8, bf16)
# Exits non-zero on the first hard failure with an actionable message.
#
# LLAMA_DOCKER=1 (set by `run.sh --docker`) checks the container backend
# instead of a host build: the docker daemon, the image, its llama.cpp build
# number, and that the GPU is visible from inside a container. The model,
# VRAM and port checks are the same either way.
set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh

MODEL_KEY="${1:?usage: ./preflight.sh <model-key>}"
model_info "$MODEL_KEY" repo >/dev/null 2>&1 \
  || { echo "unknown model key: $MODEL_KEY" >&2; exit 1; }
MODEL_PATH="$(model_path "$MODEL_KEY" || true)"
NEED_VRAM="$(model_info "$MODEL_KEY" vram)"
MIN_BUILD="$(model_info "$MODEL_KEY" minbuild)"

fail=0
note() { printf '  %s %s\n' "$1" "$2"; }

echo "Preflight for model: $MODEL_KEY"

# build_of — llama.cpp build number from `llama-server --version` text on stdin.
# New format: "version: 0.1.0-dev (build 10448, commit ad1de39e0)" — build number is
# in the parens. Old format: "version: 10014 (00fa7cb28)" — build number is the leading
# int. Try new first; only the matching branch replaces the pattern space so old lines
# fall through untouched.
build_of() {
  sed -n \
    -e 's/^version: .*(build \([0-9]*\).*/\1/p' \
    -e 's/^version: \([0-9]*\).*/\1/p'
}

# 1. Binary — or, for the container backend, the image that carries it.
if [[ "${LLAMA_DOCKER:-0}" == 1 ]]; then
  if ! command -v docker >/dev/null 2>&1 || ! docker info >/dev/null 2>&1; then
    note "✗" "docker is missing, or the docker daemon is not reachable by this user."
    echo "    Install Docker and start it; add yourself to the docker group if"
    echo "    'docker info' only works with sudo."
    fail=1
  elif ! docker image inspect "$LLAMA_IMAGE" >/dev/null 2>&1; then
    note "✗" "image not pulled: $LLAMA_IMAGE"
    echo "    docker pull $LLAMA_IMAGE    (or ./deploy.sh $MODEL_KEY, which pulls it)"
    fail=1
  else
    # No --gpus here: reading the version needs no GPU, so a broken GPU setup
    # is reported by the device check below, as itself.
    build="$(docker run --rm "$LLAMA_IMAGE" --version 2>&1 | build_of)"
    note "✓" "image: $LLAMA_IMAGE (build ${build:-unknown})"
    if [[ "$MIN_BUILD" -gt 0 && "${build:-0}" -lt "$MIN_BUILD" ]]; then
      note "✗" "build ${build:-unknown} is too old for $MODEL_KEY (needs >= b$MIN_BUILD)."
      echo "    docker pull $LLAMA_IMAGE    (the tag floats; a fresh pull is newer)"
      fail=1
    fi
    # A device line on stdout ("  CUDA0: <name> (...)"), not any mention of CUDA:
    # the backend library logs that it loaded even when it found no device.
    if ! docker run --rm --gpus all "$LLAMA_IMAGE" --list-devices 2>/dev/null \
         | grep -Eq '^[[:space:]]*CUDA[0-9]+:'; then
      note "✗" "no CUDA device visible inside the container."
      echo "    Install nvidia-container-toolkit and restart docker, and check the"
      echo "    host driver with nvidia-smi — this image needs a CUDA 12 capable driver."
      fail=1
    fi
  fi
elif [[ -x "$LLAMA_SERVER" ]] && "$LLAMA_SERVER" --version >/dev/null 2>&1; then
  build="$("$LLAMA_SERVER" --version 2>&1 | build_of)"
  note "✓" "llama-server: $LLAMA_SERVER (build ${build:-unknown})"
  if [[ "$MIN_BUILD" -gt 0 && "${build:-0}" -lt "$MIN_BUILD" ]]; then
    note "✗" "build ${build:-unknown} is too old for $MODEL_KEY (needs >= b$MIN_BUILD)."
    echo "    Rebuild: see 'Building llama.cpp' in README.md."
    fail=1
  fi
  if ! "$LLAMA_SERVER" --list-devices 2>/dev/null | grep -q 'CUDA'; then
    note "✗" "no CUDA device visible to llama-server — build lacks the CUDA backend"
    echo "    or the GPU driver is unavailable. Rebuild with -DGGML_CUDA=ON (README.md)"
    echo "    and check that GGML_CUDA:BOOL=ON in build/CMakeCache.txt."
    fail=1
  fi
else
  note "✗" "llama-server not found/executable at: $LLAMA_SERVER"
  echo "    Build it (see README.md), or set LLAMA_MAINLINE / LLAMA_SERVER."
  fail=1
fi

# Redeploying over our own container is the normal case in docker mode, and
# that container is what holds the VRAM and the port right now. run.sh removes
# it before launching, so say that instead of warning about either.
REPLACING=""
if [[ "${LLAMA_DOCKER:-0}" == 1 ]] \
   && [[ -n "$(docker ps -aq --filter "name=^${DEPLOY_CONTAINER}$" 2>/dev/null)" ]]; then
  REPLACING="$(docker inspect -f '{{ index .Config.Labels "llm.key" }}' "$DEPLOY_CONTAINER" 2>/dev/null || true)"
  note "⚠" "container $DEPLOY_CONTAINER (serving ${REPLACING:-an unknown model}) will be replaced."
  REPLACING="${REPLACING:-unknown}"
fi

# 2. Model file
if [[ -f "$MODEL_PATH" ]]; then
  note "✓" "model: $MODEL_PATH ($(du -hL "$MODEL_PATH" | cut -f1))"
else
  note "✗" "model missing: $MODEL_PATH  — run ./fetch-models.sh $MODEL_KEY"
  fail=1
fi

# 2b. Vision projector (only for models that declare one). Not a hard failure:
# run.sh drops to text-only and says so.
MMPROJ="$(model_mmproj_path "$MODEL_KEY")"
if [[ -n "$MMPROJ" ]]; then
  if [[ -f "$MMPROJ" ]]; then
    note "✓" "vision projector: $MMPROJ ($(du -hL "$MMPROJ" | cut -f1))"
  else
    note "⚠" "vision projector missing: $MMPROJ — text-only until you run ./fetch-models.sh $MODEL_KEY"
  fi
fi

# 2c. Separate MTP drafter (only for models that declare one). Not a hard
# failure either: run.sh serves without MTP and says so.
DRAFT="$(model_draft_path "$MODEL_KEY")"
if [[ -n "$DRAFT" ]]; then
  if [[ -f "$DRAFT" ]]; then
    note "✓" "MTP drafter: $DRAFT ($(du -hL "$DRAFT" | cut -f1))"
  else
    note "⚠" "MTP drafter missing: $DRAFT — MTP off until you run ./fetch-models.sh $MODEL_KEY"
  fi
fi

# 3. GPU / VRAM
if command -v nvidia-smi >/dev/null 2>&1; then
  IFS=',' read -r name free total < <(nvidia-smi \
      --query-gpu=name,memory.free,memory.total --format=csv,noheader,nounits | head -n1)
  name="${name#"${name%%[![:space:]]*}"}"; free="${free//[[:space:]]/}"; total="${total//[[:space:]]/}"
  note "✓" "GPU: $name  (${free} MiB free / ${total} MiB)"
  if [[ "${free:-0}" -lt "$NEED_VRAM" && -n "$REPLACING" ]]; then
    note "⚠" "~${NEED_VRAM} MiB expected for $MODEL_KEY — free VRAM is below that until $DEPLOY_CONTAINER is replaced."
  elif [[ "${free:-0}" -lt "$NEED_VRAM" ]]; then
    note "⚠" "~${NEED_VRAM} MiB expected for $MODEL_KEY at its default settings — free VRAM is below that."
    echo "      Another llama-server may be running (run.sh will stop it), or use"
    echo "      a smaller --ctx / --kv q8_0."
  fi
else
  note "⚠" "nvidia-smi not found — cannot verify GPU/VRAM."
fi

# 4. Port. Advisory only — run.sh is what actually refuses to launch. If the
# holder is a llama-server from this build, run.sh stops it and carries on; if
# it is anything else, run.sh fails loudly rather than let llama-server die on
# a raw bind error. Name the holder here when ss can see it.
if command -v ss >/dev/null 2>&1; then
  PORT_ROWS="$(ss -ltnp 2>/dev/null | awk -v port=":$PORT" '$1 == "LISTEN" && $4 ~ (port "$") {print}')"
  if [[ -n "$PORT_ROWS" && -n "$REPLACING" ]] \
     && docker port "$DEPLOY_CONTAINER" 2>/dev/null | grep -q ":$PORT\$"; then
    note "✓" "port $PORT is published by $DEPLOY_CONTAINER, which will be replaced"
  elif [[ -n "$PORT_ROWS" ]]; then
    PORT_PID="$(printf '%s\n' "$PORT_ROWS" | grep -oE 'pid=[0-9]+' | head -n1 | cut -d= -f2)"
    PORT_NAME="$(printf '%s\n' "$PORT_ROWS" | grep -oE '"[^"]+"' | head -n1 | tr -d '"')"
    if [[ -n "$PORT_PID" ]]; then
      note "⚠" "port $PORT is in use by pid $PORT_PID (${PORT_NAME:-unknown}) — run.sh stops it if it is a llama-server from this build, else it fails loudly."
    else
      note "⚠" "port $PORT is in use by another user's process (pid hidden — needs root to see) — run.sh will fail loudly rather than launch."
    fi
  else
    note "✓" "port $PORT is free"
  fi
else
  note "⚠" "ss not found — cannot check port $PORT in advance; run.sh still refuses a foreign holder at launch."
fi

echo
if [[ "$fail" -ne 0 ]]; then
  echo "Preflight FAILED — fix the ✗ items above."
  exit 1
fi
echo "Preflight OK."
