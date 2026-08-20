#!/usr/bin/env bash
# run-sglang.sh — launch Qwen3.8-27B under SGLang in Docker, defaulted to the
# lmsys cookbook's verified RTX PRO 6000 / NVFP4 / EAGLE / low-latency cell.
#
#   ./run-sglang.sh                        # the verified defaults, as-is
#   ./run-sglang.sh --effort low           # shallower thinking, server-wide
#   ./run-sglang.sh --concurrency 8        # pin --max-running-requests
#   ./run-sglang.sh --quant fp8            # FP8 checkpoint instead of NVFP4
#   ./run-sglang.sh --avg-len 32768        # recompute --mamba-full-memory-ratio
#   ./run-sglang.sh --print                # show the docker argv, launch nothing
#
# This is the SGLang half of the repo: ./run.sh serves GGUF through llama.cpp,
# this serves the safetensors checkpoints through the prebuilt SGLang image.
# The container pulls its own weights into $HF_CACHE from the HuggingFace repo
# id — ./fetch-models.sh is for the llama.cpp path and is not needed here.
#
# Flag precedence (last one wins): CLI flags > settings/<key>.sglang.conf
# (written by --save-defaults) > the built-in defaults below.
#
# Defaults, all overridable:
#   image lmsysorg/sglang:qwen38-27b · NVFP4 W4A4 checkpoint · fp8_e4m3 KV ·
#   mem-fraction 0.85 · flashinfer attention · 2048-token prefill chunks ·
#   qwen3 reasoning parser + qwen3_coder tool parser · EAGLE/MTP 3/1/4 with
#   ReplaySSM spec · extra_buffer radix strategy · float32 GDN state ·
#   --mamba-full-memory-ratio 0.29 — CPU multimodal feature transport
#
# The one deviation from the cookbook panel is --mm-feature-transport cpu. Left
# unset, SGLang resolves it to cuda_ipc on any single-node CUDA box, and CUDA IPC
# is unsupported under WSL2: the scheduler raises cudaErrorInvalidResourceHandle
# on the first request carrying an image, which the startup warmup does, so the
# server SIGQUITs seconds after "Application startup complete". Reproduced with a
# bare 20-line torch script on this host, so it is the platform, not SGLang.
#
# EXTRA_ARGS="..." is appended raw to the `sglang serve` argv (last, so it wins)
# and DOCKER_ARGS="..." to the `docker run` argv. Both are word-split: values
# containing spaces are not supported.
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

MODEL_KEY=38-nvfp4          # the env.sh table entry this script serves
CONF="$SETTINGS_DIR/$MODEL_KEY.sglang.conf"

usage() {
  cat <<USAGE
usage: ./run-sglang.sh [flags]

Serves Qwen3.8-27B with SGLang in Docker ($SGLANG_IMAGE).

flags:
  --quant nvfp4|fp8|bf16  checkpoint precision (default nvfp4 — W4A4, ~16.5GB
                          of weights, the RTX PRO 6000 recommendation). fp8 is
                          ~28.5GB, bf16 ~54GB
  --effort low|medium|xhigh
                          reasoning_effort applied to every request that does
                          not set its own (default: unset, so the template's
                          own xhigh stands). Sent as
                          --default-chat-template-kwargs
  --no-preserve-thinking  drop prior-turn reasoning from the template
                          (default: the template's own preserve_thinking, on)
  --ctx N                 --context-length (default: unset = the checkpoint's
                          native 262144)
  --concurrency N         --max-running-requests. Leave unset and SGLang pins
                          it to 48 whenever speculative decoding is on
  --avg-len N             average request length (input + output) used to
                          recompute --mamba-full-memory-ratio for the current
                          radix strategy, spec setting, and dtypes
  --mamba-ratio R         set --mamba-full-memory-ratio directly (default 0.29,
                          the cookbook panel's pin; wins over --avg-len)
  --radix auto|extra_buffer|extra_buffer_lazy|no_buffer|off
                          --mamba-radix-cache-strategy (default extra_buffer,
                          S=5); lazy is S=4 at no accuracy cost, off passes
                          --disable-radix-cache (S=1)
  --ssm-dtype float32|bfloat16
                          --mamba-ssm-dtype (default float32 — the checkpoint's
                          declared precision, 153.9 MB per state slot; bfloat16
                          is 78.4 MB and is an accuracy gate, not a free win)
  --kv-dtype fp8_e4m3|auto|bfloat16
                          --kv-cache-dtype (default fp8_e4m3)
  --mem-fraction F        --mem-fraction-static (default 0.85)
  --prefill-chunk N       --chunked-prefill-size (default 2048 — keeps decode
                          inter-token latency smooth on hybrid GDN)
  --attention-backend B   --attention-backend (default flashinfer; trtllm_mha
                          is SM100-only and will not run on SM120)
  --spec eagle|none       speculative decoding (default eagle = the in-checkpoint
                          MTP head at 3 steps / topk 1 / 4 draft tokens)
  --no-replayssm          drop --enable-linear-replayssm-spec, moving the spec
                          verify intermediates back onto per-request state
                          slots (D goes 0 -> 4, so the balanced ratio rises)
  --mm-transport cpu|cuda_ipc
                          how image/video features cross from the processor
                          process to the scheduler (default cpu). SGLang would
                          otherwise auto-pick cuda_ipc on a single-node CUDA
                          box, and CUDA IPC does not work under WSL2 — the
                          scheduler dies on the first image with
                          "invalid resource handle", which includes the
                          startup warmup, so the server never comes up. cpu
                          also frees the 1 GiB IPC pool
  --served-name NAME      --served-model-name, the id harnesses must send
                          (default: unset, so the id is the full repo path)
  --api-key KEY           --api-key (default: unset, unauthenticated)
  --port P                host port to publish (default $SGLANG_PORT)
  --name NAME             container name (default $SGLANG_CONTAINER)
  --print                 print the docker argv and exit without launching
  --save-defaults         persist the effective flags to
                          settings/$MODEL_KEY.sglang.conf (used automatically
                          next time)
  --reset-defaults        delete settings/$MODEL_KEY.sglang.conf and exit

env:
  HF_TOKEN                passed into the container when set (gated repos)
  HF_CACHE                host HuggingFace cache to bind (default $HF_CACHE)
  SGLANG_IMAGE            image to run (default lmsysorg/sglang:qwen38-27b)
  SKIP_PREFLIGHT=1        skip the docker/GPU/port checks
  EXTRA_ARGS / DOCKER_ARGS  raw passthrough, appended last
USAGE
}

# ── Built-in defaults ────────────────────────────────────────────────────────
QUANT=nvfp4
EFFORT=""              # empty = don't send reasoning_effort at all
PRESERVE=""            # empty = leave the template's own default (on)
CTX=""
CONCURRENCY=""
AVG_LEN=""
MAMBA_RATIO=0.29
RADIX=extra_buffer
SSM_DTYPE=float32
KV_DTYPE=fp8_e4m3
MEM_FRACTION=0.85
PREFILL_CHUNK=2048
ATTN_BACKEND=flashinfer
SPEC=eagle
REPLAYSSM=1
MM_TRANSPORT=cpu
SERVED_NAME=""
API_KEY=""
PORT_BASE="$SGLANG_PORT"
SGPORT="$SGLANG_PORT"
NAME="$SGLANG_CONTAINER"
PRINT_ONLY=0
SAVE_DEFAULTS=0

# Saved settings are prepended to the CLI, so an explicit flag still wins.
if [[ -f "$CONF" ]]; then
  SAVED_LINE="$(head -n1 "$CONF")"
  if [[ -n "$SAVED_LINE" ]]; then
    read -r -a SAVED_FLAGS <<< "$SAVED_LINE"
    set -- "${SAVED_FLAGS[@]}" "$@"
    echo "defaults for $MODEL_KEY: $SAVED_LINE"
  fi
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --quant)   QUANT="${2:?--quant needs a value}"; shift 2 ;;
    --effort)  EFFORT="${2:?--effort needs a value}"; shift 2 ;;
    --no-preserve-thinking) PRESERVE=0; shift ;;
    --preserve-thinking)    PRESERVE=1; shift ;;
    --ctx)     CTX="${2:?--ctx needs a value}"; shift 2 ;;
    --concurrency) CONCURRENCY="${2:?--concurrency needs a value}"; shift 2 ;;
    --avg-len) AVG_LEN="${2:?--avg-len needs a value}"; MAMBA_RATIO=""; shift 2 ;;
    --mamba-ratio) MAMBA_RATIO="${2:?--mamba-ratio needs a value}"; AVG_LEN=""; shift 2 ;;
    --radix)   RADIX="${2:?--radix needs a value}"; shift 2 ;;
    --ssm-dtype) SSM_DTYPE="${2:?--ssm-dtype needs a value}"; shift 2 ;;
    --kv-dtype)  KV_DTYPE="${2:?--kv-dtype needs a value}"; shift 2 ;;
    --mem-fraction) MEM_FRACTION="${2:?--mem-fraction needs a value}"; shift 2 ;;
    --prefill-chunk) PREFILL_CHUNK="${2:?--prefill-chunk needs a value}"; shift 2 ;;
    --attention-backend) ATTN_BACKEND="${2:?--attention-backend needs a value}"; shift 2 ;;
    --spec)    SPEC="${2:?--spec needs a value}"; shift 2 ;;
    --no-replayssm) REPLAYSSM=0; shift ;;
    --replayssm)    REPLAYSSM=1; shift ;;
    --mm-transport) MM_TRANSPORT="${2:?--mm-transport needs a value}"; shift 2 ;;
    --served-name) SERVED_NAME="${2:?--served-name needs a value}"; shift 2 ;;
    --api-key) API_KEY="${2:?--api-key needs a value}"; shift 2 ;;
    --port)    SGPORT="${2:?--port needs a value}"; shift 2 ;;
    --name)    NAME="${2:?--name needs a value}"; shift 2 ;;
    --print|--dry-run) PRINT_ONLY=1; shift ;;
    --save-defaults) SAVE_DEFAULTS=1; shift ;;
    --reset-defaults)
      rm -f "$CONF"
      echo "reset: $MODEL_KEY back to the built-in SGLang defaults"
      exit 0 ;;
    -*) echo "unknown flag: $1" >&2; usage >&2; exit 1 ;;
    *)  echo "unexpected argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

# ── Validate ─────────────────────────────────────────────────────────────────
case "$QUANT" in
  nvfp4) MODEL_PATH=RadixArk/Qwen3.8-27B-NVFP4; WEIGHTS_GB=16.5 ;;
  fp8)   MODEL_PATH=Qwen/Qwen3.8-27B-FP8;       WEIGHTS_GB=28.5 ;;
  bf16)  MODEL_PATH=Qwen/Qwen3.8-27B;           WEIGHTS_GB=54 ;;
  *) echo "unknown --quant: $QUANT (use nvfp4, fp8, or bf16)" >&2; exit 1 ;;
esac

# The Qwen3.8 template raises on anything outside this set, and a bad value only
# surfaces as a 500 on the first request — reject it here instead.
if [[ -n "$EFFORT" ]]; then
  case "$EFFORT" in
    low|medium|xhigh) ;;
    high) echo "--effort high is not a Qwen3.8 value: use xhigh." >&2; exit 1 ;;
    none|off) echo "Qwen3.8 always reasons — there is no off. Use --effort low." >&2; exit 1 ;;
    *) echo "--effort must be low, medium, or xhigh (got: $EFFORT)" >&2; exit 1 ;;
  esac
fi

case "$RADIX" in
  auto|extra_buffer|extra_buffer_lazy|no_buffer|off) ;;
  *) echo "unknown --radix: $RADIX" >&2; exit 1 ;;
esac
case "$SSM_DTYPE" in float32|bfloat16) ;; *)
  echo "--ssm-dtype must be float32 or bfloat16 (got: $SSM_DTYPE)" >&2; exit 1 ;;
esac
case "$KV_DTYPE" in fp8_e4m3|auto|bfloat16) ;; *)
  echo "--kv-dtype must be fp8_e4m3, auto, or bfloat16 (got: $KV_DTYPE)" >&2; exit 1 ;;
esac
case "$SPEC" in eagle|none) ;; *)
  echo "--spec must be eagle or none (got: $SPEC)" >&2; exit 1 ;;
esac
case "$MM_TRANSPORT" in cpu|cuda_ipc) ;; *)
  echo "--mm-transport must be cpu or cuda_ipc (got: $MM_TRANSPORT)" >&2; exit 1 ;;
esac
if [[ -n "$AVG_LEN" && ! "$AVG_LEN" =~ ^[0-9]+$ ]]; then
  echo "--avg-len must be a positive integer (got: $AVG_LEN)" >&2; exit 1
fi
if [[ -n "$MAMBA_RATIO" && ! "$MAMBA_RATIO" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
  echo "--mamba-ratio must be a non-negative number (got: $MAMBA_RATIO)" >&2; exit 1
fi

SPEC_DRAFT_TOKENS=4

# ── --mamba-full-memory-ratio ────────────────────────────────────────────────
# Hybrid GDN splits post-weight memory into a worst-case GDN state pool (which
# sets the concurrency ceiling) and a paged attention KV pool, divided by this
# ratio. The balanced value is the per-request cost ratio:
#
#   ratio = (S + D) * state_bytes / (L * kv_bytes_per_token)
#
#   S — state slots per running request, set by the radix strategy
#   D — speculative verify intermediates: 0 with spec off, and 0 with
#       --enable-linear-replayssm-spec (the intermediates move to a fixed ring);
#       otherwise --speculative-num-draft-tokens
#   L — average total request length in tokens, input + output
#
# state_bytes / kv_bytes_per_token is the state slot priced in KV tokens:
# 4698 at fp32 state over fp8 KV, 2394 at bf16 state, doubling if KV is bf16.
# The default 0.29 is the cookbook panel's own pin, which corresponds to
# L ~= 81k at the defaults below.
case "$RADIX" in
  extra_buffer|auto)  S=5 ;;
  extra_buffer_lazy)  S=4 ;;
  no_buffer)          S=3 ;;
  off)                S=1 ;;
esac
if [[ "$SPEC" == none ]] || (( REPLAYSSM )); then D=0; else D=$SPEC_DRAFT_TOKENS; fi
if [[ "$SSM_DTYPE" == float32 ]]; then TOKEN_EQUIV=4698; else TOKEN_EQUIV=2394; fi
if [[ "$KV_DTYPE" == bfloat16 ]]; then TOKEN_EQUIV=$(( TOKEN_EQUIV / 2 )); fi

RATIO_NOTE=""
if [[ -n "$AVG_LEN" ]]; then
  MAMBA_RATIO="$(awk -v s="$S" -v d="$D" -v te="$TOKEN_EQUIV" -v l="$AVG_LEN" \
                     'BEGIN { printf "%.2f", (s + d) * te / l }')"
  RATIO_NOTE="computed for avg length $AVG_LEN (S=$S, D=$D, ${TOKEN_EQUIV} KV-tokens/slot)"
else
  IMPLIED_LEN="$(awk -v s="$S" -v d="$D" -v te="$TOKEN_EQUIV" -v r="$MAMBA_RATIO" \
                     'BEGIN { if (r > 0) printf "%d", (s + d) * te / r; else print 0 }')"
  RATIO_NOTE="implies an avg request length of ~${IMPLIED_LEN} tokens (S=$S, D=$D)"
fi

# ── Save defaults ────────────────────────────────────────────────────────────
# Reconstructed from the effective state, not the raw CLI, so the file carries
# everything non-built-in even when this launch only changed one flag.
if (( SAVE_DEFAULTS )); then
  SAVED=()
  [[ "$QUANT" != nvfp4 ]]            && SAVED+=(--quant "$QUANT")
  [[ -n "$EFFORT" ]]                 && SAVED+=(--effort "$EFFORT")
  [[ "$PRESERVE" == 0 ]]             && SAVED+=(--no-preserve-thinking)
  [[ -n "$CTX" ]]                    && SAVED+=(--ctx "$CTX")
  [[ -n "$CONCURRENCY" ]]            && SAVED+=(--concurrency "$CONCURRENCY")
  [[ "$MAMBA_RATIO" != 0.29 ]]       && SAVED+=(--mamba-ratio "$MAMBA_RATIO")
  [[ "$RADIX" != extra_buffer ]]     && SAVED+=(--radix "$RADIX")
  [[ "$SSM_DTYPE" != float32 ]]      && SAVED+=(--ssm-dtype "$SSM_DTYPE")
  [[ "$KV_DTYPE" != fp8_e4m3 ]]      && SAVED+=(--kv-dtype "$KV_DTYPE")
  [[ "$MEM_FRACTION" != 0.85 ]]      && SAVED+=(--mem-fraction "$MEM_FRACTION")
  [[ "$PREFILL_CHUNK" != 2048 ]]     && SAVED+=(--prefill-chunk "$PREFILL_CHUNK")
  [[ "$ATTN_BACKEND" != flashinfer ]] && SAVED+=(--attention-backend "$ATTN_BACKEND")
  [[ "$SPEC" != eagle ]]             && SAVED+=(--spec "$SPEC")
  (( REPLAYSSM == 0 ))               && SAVED+=(--no-replayssm)
  [[ "$MM_TRANSPORT" != cpu ]]       && SAVED+=(--mm-transport "$MM_TRANSPORT")
  [[ -n "$SERVED_NAME" ]]            && SAVED+=(--served-name "$SERVED_NAME")
  [[ "$SGPORT" != "$PORT_BASE" ]]    && SAVED+=(--port "$SGPORT")
  [[ "$NAME" != "$SGLANG_CONTAINER" ]] && SAVED+=(--name "$NAME")
  mkdir -p "$SETTINGS_DIR"
  printf '%s\n' "${SAVED[*]-}" > "$CONF"
  echo "saved defaults for $MODEL_KEY: ${SAVED[*]-(none)}"
fi

# ── Compose the sglang serve command ─────────────────────────────────────────
SERVE=(
  sglang serve
  --trust-remote-code
  --model-path "$MODEL_PATH"
  --kv-cache-dtype "$KV_DTYPE"
  --mem-fraction-static "$MEM_FRACTION"
  --attention-backend "$ATTN_BACKEND"
  --chunked-prefill-size "$PREFILL_CHUNK"
  --reasoning-parser qwen3
  --tool-call-parser qwen3_coder
  --mamba-full-memory-ratio "$MAMBA_RATIO"
  --mm-feature-transport "$MM_TRANSPORT"
  --host 0.0.0.0
  --port "$SGLANG_PORT"
)

if [[ "$SPEC" == eagle ]]; then
  SERVE+=(--speculative-algorithm EAGLE
          --speculative-num-steps 3
          --speculative-eagle-topk 1
          --speculative-num-draft-tokens "$SPEC_DRAFT_TOKENS")
  (( REPLAYSSM )) && SERVE+=(--enable-linear-replayssm-spec)
fi

if [[ "$RADIX" == off ]]; then
  SERVE+=(--disable-radix-cache)
else
  SERVE+=(--mamba-radix-cache-strategy "$RADIX")
fi

SERVE+=(--mamba-ssm-dtype "$SSM_DTYPE")

[[ -n "$CTX" ]]         && SERVE+=(--context-length "$CTX")
[[ -n "$CONCURRENCY" ]] && SERVE+=(--max-running-requests "$CONCURRENCY")
[[ -n "$SERVED_NAME" ]] && SERVE+=(--served-model-name "$SERVED_NAME")
[[ -n "$API_KEY" ]]     && SERVE+=(--api-key "$API_KEY")

# Thinking depth. SGLang has no dedicated reasoning_effort flag: the server-wide
# default is a chat-template kwarg, applied to every request that does not carry
# its own chat_template_kwargs / reasoning_effort (serving_chat.py promotes the
# key into request.reasoning_effort only when the request left it unset). The
# same map is where preserve_thinking lives.
TEMPLATE_KWARGS=()
[[ -n "$EFFORT" ]]   && TEMPLATE_KWARGS+=("\"reasoning_effort\":\"$EFFORT\"")
[[ "$PRESERVE" == 0 ]] && TEMPLATE_KWARGS+=("\"preserve_thinking\":false")
[[ "$PRESERVE" == 1 ]] && TEMPLATE_KWARGS+=("\"preserve_thinking\":true")
if (( ${#TEMPLATE_KWARGS[@]} )); then
  KWARGS_JSON="{$(IFS=,; echo "${TEMPLATE_KWARGS[*]}")}"
  SERVE+=(--default-chat-template-kwargs "$KWARGS_JSON")
fi

# Raw passthrough for anything this script doesn't wrap. Appended last so it
# overrides the composed flags. Word-split: no values containing spaces.
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  read -r -a EXTRA_ARGV <<< "$EXTRA_ARGS"
  SERVE+=("${EXTRA_ARGV[@]}")
fi

# ── Compose the docker run command ───────────────────────────────────────────
DOCKER=(
  docker run --rm --name "$NAME"
  --gpus all
  --shm-size 32g
  --ipc=host
  -p "$SGPORT:$SGLANG_PORT"
  -v "$HF_CACHE:/root/.cache/huggingface"
)
[[ -n "${HF_TOKEN:-}" ]] && DOCKER+=(--env "HF_TOKEN=$HF_TOKEN")
(( PRINT_ONLY )) || DOCKER+=(-it)
if [[ -n "${DOCKER_ARGS:-}" ]]; then
  read -r -a DOCKER_ARGV <<< "$DOCKER_ARGS"
  DOCKER+=("${DOCKER_ARGV[@]}")
fi
DOCKER+=("$SGLANG_IMAGE" "${SERVE[@]}")

if (( PRINT_ONLY )); then
  printf '%q' "${DOCKER[0]}"; printf ' %q' "${DOCKER[@]:1}"; echo
  exit 0
fi

# ── Preflight ────────────────────────────────────────────────────────────────
if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
  fail=0
  note() { printf '  %s %s\n' "$1" "$2"; }
  echo "Preflight for SGLang: $MODEL_KEY ($QUANT)"

  if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    note "OK" "docker daemon reachable"
  else
    note "!!" "docker is missing or the daemon is not reachable"
    fail=1
  fi

  if docker image inspect "$SGLANG_IMAGE" >/dev/null 2>&1; then
    note "OK" "image: $SGLANG_IMAGE"
  else
    note "!!" "image not pulled: $SGLANG_IMAGE"
    echo "     docker pull $SGLANG_IMAGE"
    fail=1
  fi

  if [[ -d "$HF_CACHE" ]]; then
    note "OK" "HF cache: $HF_CACHE"
  else
    note "--" "HF cache $HF_CACHE does not exist yet — creating it"
    mkdir -p "$HF_CACHE"
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    IFS=',' read -r gname gfree gtotal < <(nvidia-smi \
        --query-gpu=name,memory.free,memory.total --format=csv,noheader,nounits | head -n1)
    gname="${gname#"${gname%%[![:space:]]*}"}"
    gfree="${gfree//[[:space:]]/}"; gtotal="${gtotal//[[:space:]]/}"
    note "OK" "GPU: $gname  (${gfree} MiB free / ${gtotal} MiB)"
    NEED_MIB="$(awk -v w="$WEIGHTS_GB" -v f="$MEM_FRACTION" \
                    'BEGIN { printf "%d", w * 1024 / f }')"
    if [[ "${gfree:-0}" -lt "$NEED_MIB" ]]; then
      note "--" "~${NEED_MIB} MiB wanted for $QUANT weights at --mem-fraction-static $MEM_FRACTION"
      echo "     Stop the llama.cpp server first, or lower --mem-fraction."
    fi
  else
    note "--" "nvidia-smi not found — cannot verify GPU/VRAM"
  fi

  if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -q ":$SGPORT "; then
    note "--" "port $SGPORT is already in use"
  else
    note "OK" "port $SGPORT is free"
  fi

  echo
  if (( fail )); then
    echo "Preflight FAILED — fix the !! items above."
    exit 1
  fi
fi

# Replace a container this script left running. Matched on the exact name it
# owns, so an unrelated container is never touched.
if [[ -n "$(docker ps -aq --filter "name=^${NAME}$" 2>/dev/null)" ]]; then
  echo "removing existing container: $NAME"
  docker rm -f "$NAME" >/dev/null
fi

if [[ "$SPEC" == eagle && -z "$CONCURRENCY" ]]; then
  echo "note: speculative decoding is on and --concurrency is unset, so SGLang" >&2
  echo "      pins --max-running-requests to 48. Pass --concurrency N to size it." >&2
fi
if [[ "$SSM_DTYPE" != float32 ]] && (( REPLAYSSM )) && [[ "$SPEC" == eagle ]]; then
  echo "note: --enable-linear-replayssm-spec with a non-fp32 state dtype logs a" >&2
  echo "      state-drift warning at boot. Expected, not an error." >&2
fi

echo
echo "Launching [$MODEL_KEY/$QUANT]  $MODEL_PATH"
echo "  kv=$KV_DTYPE  state=$SSM_DTYPE  radix=$RADIX(S=$S)  spec=$SPEC$( (( REPLAYSSM )) && [[ "$SPEC" == eagle ]] && echo "+replayssm")(D=$D)"
echo "  mamba-ratio=$MAMBA_RATIO  # $RATIO_NOTE"
echo "  mm-transport=$MM_TRANSPORT"
echo "  thinking=$([[ -n "$EFFORT" ]] && echo "effort=$EFFORT" || echo "effort=template default (xhigh)")$([[ "$PRESERVE" == 0 ]] && echo ", preserve=off")"
echo "  endpoint=http://localhost:$SGPORT/v1  (Anthropic-compatible: http://localhost:$SGPORT)"
printf '  %q' "${DOCKER[@]}"; echo
echo
exec "${DOCKER[@]}"
