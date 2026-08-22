#!/usr/bin/env bash
# run.sh — launch llama-server with a model from the table, tuned for precise
# coding.
#
#   ./run.sh q8                        # Q8_0, all defaults
#   ./run.sh q8 --ctx 65536            # Q8_0 at 64k context (much smaller KV)
#   ./run.sh bf16 --kv q8_0 --mtp 4    # BF16, quantized KV, deeper MTP draft
#   ./run.sh 38-q8 --effort low        # Qwen3.8 + vision, shallower thinking
#   ./run.sh deckard                   # per-model defaults applied (see --list)
#   ./run.sh --list                    # models + download state + defaults
#
# The model key must be the first argument. Flag precedence (last one wins):
#   CLI flags  >  settings/<key>.conf (written by --save-defaults)
#              >  the table's args field (env.sh)  >  the built-ins below.
#
# Built-in defaults (all overridable by flag):
#   context 262144 (native max) · f16 KV cache · MTP speculative decoding on
#   (draft depth 2) · thinking mode on · preserve-thinking on · vision on where
#   the model has a projector, at the table's per-model image-token floor
#   (2048 on the Qwen3.8 entries) · coding sampling: top-p 0.95, top-k 20,
#   min-p 0.0, presence 0, repeat 1, and the table's per-model temp.
#
# EXTRA_ARGS="..." is appended raw to the llama-server argv (last, so it
# overrides the composed flags) — for YaRN, --mtmd-batch-max-tokens, sampler
# tweaks.
# Word-split: values containing spaces are not supported.
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

usage() {
  cat <<USAGE
usage: ./run.sh <model> [flags] | --list

models: see ./run.sh --list  (the key must be the first argument)

flags:
  --ctx N                 context length (default 262144 = native max;
                          ~1M possible via YaRN — see README)
  --kv f16|bf16|q8_0      KV cache type (default f16; q8_0 is near-lossless
                          and halves KV memory; q4_0 refused — corrupts
                          code/structured output at long context)
  --mtp N                 MTP speculative draft depth 1-6 (default 2);
                          also re-enables MTP after a --no-mtp default
  --no-mtp                disable MTP speculative decoding
  --think                 enable thinking mode (default)
  --no-think              disable thinking mode entirely (also switches to
                          Unsloth's non-thinking sampling: temp 0.7,
                          top-p 0.8, presence-penalty 1.5)
  --temp T                sampling temperature, overriding the per-mode default
                          (the table's temp field when thinking — 0.6 Qwen3.6,
                          1.0 Qwen3.8; 0.7 with --no-think)
  --effort low|medium|xhigh
                          reasoning_effort chat-template kwarg (Qwen3.8 only;
                          its template defaults to xhigh). Ignored with
                          --no-think, and by models whose template lacks it
  --preserve-thinking     keep prior-turn thinking in the template (default)
  --no-preserve-thinking  drop prior-turn thinking from the template
  --no-vision             skip --mmproj for a vision model — text-only, frees
                          the projector's VRAM (no effect on text-only models)
  --image-tokens MIN[:MAX] | auto
                          how many tokens one image is expanded to. MIN is a
                          floor, MAX an optional ceiling (llama-server
                          --image-min-tokens / --image-max-tokens). Default is
                          the table's per-model imgtok — 2048 on the Qwen3.8
                          entries, because llama.cpp warns at load that Qwen-VL
                          needs >= 1024 for grounding. 'auto' passes neither
                          flag, leaving llama.cpp's model-derived sizing.
                          Ignored by text-only models and under --no-vision
  --port P                listen port (default $PORT)
  --save-defaults         persist the effective flags of this launch to
                          settings/<model>.conf (used automatically next time)
  --reset-defaults        delete settings/<model>.conf and exit
USAGE
}

# ── Parse arguments ──────────────────────────────────────────────────────────
[[ $# -eq 0 ]] && { usage >&2; exit 1; }
case "$1" in
  -h|--help) usage; exit 0 ;;
  -l|--list) model_list; exit 0 ;;
  -*) echo "the model key must be the first argument: ./run.sh <model> [flags]" >&2
      usage >&2; exit 1 ;;
esac
MODEL_KEY="$1"; shift

model_info "$MODEL_KEY" path >/dev/null 2>&1 \
  || { echo "unknown model: $MODEL_KEY" >&2; usage >&2; exit 1; }
MODEL_PATH="$(model_info "$MODEL_KEY" path)"

if [[ "$(model_info "$MODEL_KEY" fmt)" != gguf ]]; then
  echo "$MODEL_KEY is a safetensors checkpoint — llama.cpp cannot load it." >&2
  if [[ "$MODEL_KEY" == 38-nvfp4 ]]; then
    echo "Serve it with SGLang instead:  ./run-sglang.sh" >&2
  else
    echo "Serve it with vLLM instead:  vllm serve $MODEL_PATH" >&2
  fi
  exit 1
fi

CTX=262144
KV=f16
MTP=2
MTP_ON=1
THINK=1
PRESERVE=1
TEMP=""             # empty = use the per-mode default below
EFFORT=""           # empty = don't pass reasoning_effort at all
VISION=1
PORT_BASE="$PORT"   # --save-defaults only persists --port if it changed
SAVE_DEFAULTS=0

MMPROJ="$(model_mmproj_path "$MODEL_KEY")"   # empty for text-only models
TEMP_THINK="$(model_info "$MODEL_KEY" temp)" # thinking-mode default for this model
IMGTOK_TABLE="$(model_info "$MODEL_KEY" imgtok)"  # "" = no floor for this model
IMGTOK="$IMGTOK_TABLE"                       # --image-tokens overrides it

# Per-model defaults: settings/<key>.conf beats the table's args field; CLI
# flags are appended after them and the parser is last-one-wins, so precedence
# is CLI > settings file > table args > the built-ins above.
DEFAULTS="$(model_args "$MODEL_KEY")"
if [[ -n "$DEFAULTS" ]]; then
  read -r -a DEFAULT_FLAGS <<< "$DEFAULTS"   # whitespace split, no glob expansion
  set -- "${DEFAULT_FLAGS[@]}" "$@"
  echo "defaults for $MODEL_KEY: $DEFAULTS"
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --ctx)  CTX="${2:?--ctx needs a value}"; shift 2 ;;
    --kv)   KV="${2:?--kv needs a value}"; shift 2 ;;
    --mtp)  MTP="${2:?--mtp needs a value}"; MTP_ON=1; shift 2 ;;
    --no-mtp) MTP_ON=0; shift ;;
    --think) THINK=1; shift ;;
    --no-think) THINK=0; shift ;;
    --temp) TEMP="${2:?--temp needs a value}"; shift 2 ;;
    --effort) EFFORT="${2:?--effort needs a value}"; shift 2 ;;
    --preserve-thinking) PRESERVE=1; shift ;;
    --no-preserve-thinking) PRESERVE=0; shift ;;
    --no-vision) VISION=0; shift ;;
    --image-tokens) IMGTOK="${2:?--image-tokens needs a value}"; shift 2 ;;
    --port) PORT="${2:?--port needs a value}"; shift 2 ;;
    --save-defaults) SAVE_DEFAULTS=1; shift ;;
    --reset-defaults)
      rm -f "$SETTINGS_DIR/$MODEL_KEY.conf"
      echo "reset: $MODEL_KEY back to table defaults ($(model_info "$MODEL_KEY" args))"
      exit 0 ;;
    -*) echo "unknown flag: $1" >&2; usage >&2; exit 1 ;;
    *)  echo "unexpected argument: $1 (the model key must be the first argument)" >&2
        exit 1 ;;
  esac
done

case "$KV" in
  f16|bf16|q8_0) ;;
  q4_0|q4_1|q5_0|q5_1|iq4_nl)
    echo "refusing --kv $KV: 4/5-bit KV cache measurably corrupts code and" >&2
    echo "structured output at long context (see README). Use f16 or q8_0." >&2
    exit 1 ;;
  *) echo "unknown --kv type: $KV (use f16, bf16, or q8_0)" >&2; exit 1 ;;
esac

if (( MTP < 1 || MTP > 16 )); then
  echo "--mtp must be 1-16 (1-6 is the useful range; default 2)" >&2; exit 1
fi

if [[ -n "$TEMP" && ! "$TEMP" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
  echo "--temp must be a non-negative number (got: $TEMP)" >&2; exit 1
fi

# --image-tokens MIN[:MAX], or `auto` to pass neither flag. Split here so the
# compose step below stays a straight append, and so a typo fails now rather
# than as a llama-server usage error after the model has finished loading.
IMG_MIN=""; IMG_MAX=""
if [[ "$IMGTOK" == auto ]]; then
  :                                          # llama.cpp reads both from the model
elif [[ -n "$IMGTOK" ]]; then
  if [[ "$IMGTOK" =~ ^([0-9]+)(:([0-9]+))?$ ]]; then
    IMG_MIN="${BASH_REMATCH[1]}"; IMG_MAX="${BASH_REMATCH[3]}"
    if (( IMG_MIN < 1 )); then
      echo "--image-tokens floor must be >= 1 (got: $IMG_MIN)" >&2; exit 1
    fi
    if [[ -n "$IMG_MAX" ]] && (( IMG_MAX < IMG_MIN )); then
      echo "--image-tokens ceiling $IMG_MAX is below its floor $IMG_MIN" >&2; exit 1
    fi
  else
    echo "--image-tokens must be N, N:M, or auto (got: $IMGTOK)" >&2; exit 1
  fi
fi

# The Qwen3.8 template raises on anything outside this set, and a bad value only
# surfaces as a 500 on the first request — reject it here instead.
if [[ -n "$EFFORT" ]]; then
  case "$EFFORT" in
    low|medium|xhigh) ;;
    none) echo "--effort none is not a template value: use --no-think." >&2; exit 1 ;;
    *) echo "--effort must be low, medium, or xhigh (got: $EFFORT)" >&2; exit 1 ;;
  esac
fi

# ── Save defaults ────────────────────────────────────────────────────────────
# Reconstructed from the effective state, NOT the raw CLI: the settings file
# REPLACES the table's args on the next launch, so it must carry everything
# non-built-in (e.g. a table --no-mtp must survive a --ctx-only save).
if (( SAVE_DEFAULTS )); then
  SAVED=()
  if (( CTX != 262144 ));    then SAVED+=(--ctx "$CTX"); fi
  if [[ "$KV" != f16 ]];     then SAVED+=(--kv "$KV"); fi
  if (( MTP_ON == 0 ));      then SAVED+=(--no-mtp)
  elif (( MTP != 2 ));       then SAVED+=(--mtp "$MTP"); fi
  if (( THINK == 0 ));       then SAVED+=(--no-think); fi
  if [[ -n "$TEMP" ]];       then SAVED+=(--temp "$TEMP"); fi
  if [[ -n "$EFFORT" ]];     then SAVED+=(--effort "$EFFORT"); fi
  if (( PRESERVE == 0 ));    then SAVED+=(--no-preserve-thinking); fi
  if (( VISION == 0 ));      then SAVED+=(--no-vision); fi
  # Compared against the table, not a built-in: the floor is per-model, and the
  # settings file replaces the table's args wholesale on the next launch.
  if [[ "$IMGTOK" != "$IMGTOK_TABLE" ]]; then SAVED+=(--image-tokens "${IMGTOK:-auto}"); fi
  if [[ "$PORT" != "$PORT_BASE" ]]; then SAVED+=(--port "$PORT"); fi
  mkdir -p "$SETTINGS_DIR"
  printf '%s\n' "${SAVED[*]-}" > "$SETTINGS_DIR/$MODEL_KEY.conf"
  echo "saved defaults for $MODEL_KEY: ${SAVED[*]-(none)}"
fi

# ── Compose the server command ───────────────────────────────────────────────
ARGS=(
  -m "$MODEL_PATH"
  -ngl 99
  --jinja
  -fa on
  -c "$CTX"
)

[[ "$KV" != f16 ]] && ARGS+=(-ctk "$KV" -ctv "$KV")

(( MTP_ON )) && ARGS+=(--spec-type draft-mtp --spec-draft-n-max "$MTP")

# Vision projector, for models whose table entry declares one. Missing weights
# are a warning, not a failure: text-only still serves.
if [[ -n "$MMPROJ" ]] && (( VISION )); then
  if [[ -f "$MMPROJ" ]]; then
    ARGS+=(--mmproj "$MMPROJ")
    # Only meaningful alongside a projector — llama-server ignores them
    # otherwise, but keeping them inside this branch keeps the argv honest.
    [[ -n "$IMG_MIN" ]] && ARGS+=(--image-min-tokens "$IMG_MIN")
    [[ -n "$IMG_MAX" ]] && ARGS+=(--image-max-tokens "$IMG_MAX")
  else
    echo "warning: vision projector missing ($MMPROJ) — serving text-only." >&2
    echo "         run ./fetch-models.sh $MODEL_KEY to get it." >&2
  fi
fi

# Thinking is driven by llama-server's own flags rather than raw template
# kwargs: setting enable_thinking through --chat-template-kwargs is deprecated,
# and --reasoning-preserve does more than the preserve_thinking key we used to
# pass — it sets preserve_thinking, clear_thinking and truncate_history_thinking
# together, so it covers templates that spell the idea differently. It is also
# the key llama-server looks for when deciding whether to warn about preserve
# support. reasoning_effort has no dedicated flag, so it stays a kwarg; all of
# these land in the same map server-side, so mixing the two forms is fine.
if (( THINK )); then
  # Unsloth-recommended sampling for coding, thinking mode. The temperature is
  # per-model (the table's temp field: 0.6 for Qwen3.6, 1.0 for Qwen3.8).
  ARGS+=(--temp "${TEMP:-$TEMP_THINK}" --top-p 0.95 --top-k 20 --min-p 0.0
         --presence-penalty 0.0 --repeat-penalty 1.0
         --reasoning on)
  if (( PRESERVE )); then ARGS+=(--reasoning-preserve)
  else                    ARGS+=(--no-reasoning-preserve); fi
  # Models whose template ignores reasoning_effort are unaffected by the key.
  [[ -n "$EFFORT" ]] && ARGS+=(--chat-template-kwargs "{\"reasoning_effort\":\"$EFFORT\"}")
else
  # Unsloth-recommended sampling for non-thinking mode. reasoning_effort is
  # dead here — the template only reads it when thinking is on.
  ARGS+=(--temp "${TEMP:-0.7}" --top-p 0.8 --top-k 20 --min-p 0.0
         --presence-penalty 1.5 --repeat-penalty 1.0
         --reasoning off)
fi

ARGS+=(--host "$HOST" --port "$PORT")

# Raw passthrough for anything run.sh doesn't wrap (--mmproj, YaRN, sampler
# tweaks). Appended last so it overrides the composed flags. Word-split:
# values containing spaces are not supported — edit run.sh for those.
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  read -r -a EXTRA_ARGV <<< "$EXTRA_ARGS"
  ARGS+=("${EXTRA_ARGV[@]}")
fi

# Preflight (binary/model/GPU/port). Skip with SKIP_PREFLIGHT=1.
if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
  ./preflight.sh "$MODEL_KEY"
fi

# Stop any llama-server already running from this build. Matched on each PID's
# real executable, not on its command line: `pkill -f "$LLAMA_SERVER"` also
# matches any *other* process that merely mentions the path — a shell that
# exported it, an editor, this script's own parent — and kills it.
stop_running_server() {
  local target pid exe stopped=0
  target="$(readlink -f "$LLAMA_SERVER" 2>/dev/null)" || return 0
  [[ -z "$target" ]] && return 0
  for pid in $(pgrep -x "${LLAMA_SERVER##*/}" 2>/dev/null); do
    exe="$(readlink -f "/proc/$pid/exe" 2>/dev/null)" || continue
    [[ "$exe" == "$target" ]] || continue
    kill "$pid" 2>/dev/null && stopped=1
  done
  (( stopped )) && { echo "stopped a running llama-server"; sleep 2; }
  return 0
}
stop_running_server

echo
VISION_STATE=n/a
if [[ -n "$MMPROJ" ]]; then
  if   (( ! VISION ));      then VISION_STATE=off
  elif [[ -f "$MMPROJ" ]];  then
    VISION_STATE=on
    if [[ -n "$IMG_MIN" ]]; then
      VISION_STATE="on(img>=$IMG_MIN${IMG_MAX:+,<=$IMG_MAX})"
    else
      VISION_STATE="on(img=auto)"
    fi
  else                           VISION_STATE="off(not downloaded)"; fi
fi
echo "Launching [$MODEL_KEY]  ctx=$CTX  kv=$KV  mtp=$([[ $MTP_ON == 1 ]] && echo "$MTP" || echo off)  thinking=$([[ $THINK == 1 ]] && echo "on(preserve=$([[ $PRESERVE == 1 ]] && echo on || echo off)${EFFORT:+,effort=$EFFORT})" || echo off)  vision=$VISION_STATE:"
printf '  %q' "$LLAMA_SERVER" "${ARGS[@]}"; echo
echo
exec "$LLAMA_SERVER" "${ARGS[@]}"
