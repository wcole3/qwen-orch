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
#   ./run.sh flint-q4 --docker         # same flags, served from the llama.cpp
#                                      # container instead of a host build
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
#   min-p 0.0, presence 0, repeat 1, and the table's per-model temp (Gemma 4
#   entries use Google's top-k 64 instead, in both thinking modes).
#
# --docker (or LLAMA_DOCKER=1) swaps only the last step: the argument list is
# composed exactly as for a host build, then handed to the official llama.cpp
# server image ($LLAMA_IMAGE) instead of $LLAMA_SERVER. The HuggingFace cache is
# mounted at the same path inside the container, so no path in that list needs
# rewriting. ./deploy.sh wraps this with the fetch and a health wait.
#
# EXTRA_ARGS="..." is appended raw to the llama-server argv (last, so it
# overrides the composed flags) — for YaRN, --mtmd-batch-max-tokens, sampler
# tweaks.
# Word-split: values containing spaces are not supported.
set -euo pipefail
# Remember where the caller stood before we cd: --slot-save-path and --lora take
# paths, and a relative one means "relative to my cwd", not to this script's dir.
CALLER_PWD="$PWD"
cd "$(dirname "$0")"
source ./env.sh

usage() {
  cat <<USAGE
usage: ./run.sh <model> [flags] | --list

models: see ./run.sh --list  (the key must be the first argument)

flags:
  --ctx N                 context length (positive integer; default 262144 = native max;
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
                          top-p 0.8, presence-penalty 1.5). Both are refused
                          by models with no switch: Flint decides per turn,
                          and Inkling 2 always reasons (use --effort)
  --temp T                sampling temperature, overriding the per-mode default
                          (the table's temp field when thinking — 0.6 Qwen3.6,
                          1.0 Qwen3.8; 0.7 with --no-think)
  --effort low|medium|xhigh
                          reasoning_effort chat-template kwarg (Qwen3.8; its
                          template defaults to xhigh). Ignored with
                          --no-think, and by models whose template lacks it.
                          Inkling 2 takes none|minimal|low|medium|high|xhigh|
                          max instead (default high); none lowers reasoning
                          but does not remove it
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
  --port P                listen port (positive integer 1-65535; default $PORT)
  --ngl N                 layers offloaded to GPU (default 99 = all)
  --api-key KEY           require this key from clients (llama-server
                          --api-key). Never written by --save-defaults
  --print                 print the launch command and exit — no preflight,
                          nothing stopped, nothing started

 container backend (the flags above compose the same either way):
  --docker                serve from the llama.cpp container (\$LLAMA_IMAGE)
                          rather than a host build; LLAMA_DOCKER=1 does the
                          same. The container is named \$DEPLOY_CONTAINER and
                          replaces any container already holding that name
  --detach                with --docker: run in the background and restart
                          with the docker daemon (--restart unless-stopped).
                          Without it the container is foreground and removed
                          on exit
                          DOCKER_ARGS="..." is appended raw to the docker run
                          argv; LLAMA_API_KEY, when set, is passed into the
                          container by name so the key stays off the argv

 server-tuning passthroughs (unset = llama-server's own default):
  --parallel N            server slots. Note each slot gets ctx/N tokens, so
                          --parallel 2 halves the per-slot context — unless
                          --slot-ctx is set
  --slot-ctx N            share one KV pool between the slots instead of
                          splitting it: --ctx becomes the pool every slot
                          draws on, and N caps what one slot may hold
                          (--kv-unified --kv-unified-per-slot N). The setup
                          for several people on one server; also turns on
                          the /metrics endpoint
  --ubatch N              micro-batch: how many prompt tokens one forward
                          pass processes (-ub; llama-server's default is
                          512). Only prompt processing is affected, not
                          decode speed. Worth lowering for Inkling 2, whose
                          compute buffer grows with --ctx x this value — at
                          512 it is three times the size of the KV cache
  --slot-save-path DIR    where slot KV state is saved (created if missing;
                          a relative path resolves against your cwd)
  --cache-reuse N         min chunk size reused from the prompt-prefix cache
  --lora PATH             load a LoRA adapter, repeatable. Implies
                          --lora-init-without-apply: adapters load inactive and
                          are attached per-request via the server's API
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

model_info "$MODEL_KEY" repo >/dev/null 2>&1 \
  || { echo "unknown model: $MODEL_KEY" >&2; usage >&2; exit 1; }
MODEL_PATH="$(model_path "$MODEL_KEY" || true)"   # preflight names it if missing

if [[ "$(model_info "$MODEL_KEY" fmt)" != gguf ]]; then
  echo "$MODEL_KEY is a safetensors checkpoint — llama.cpp cannot load it." >&2
  case "$(model_info "$MODEL_KEY" engine)" in
    sglang) echo "Serve it with SGLang instead:  ./run-sglang.sh" >&2 ;;
    *)      echo "Serve it with vLLM instead:  vllm serve $(model_info "$MODEL_KEY" repo)" >&2 ;;
  esac
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
NGL=99
# Server-tuning passthroughs. Empty = don't pass the flag at all, so
# llama-server's own default applies and this script stays opinion-free about
# slots, KV persistence and cache reuse.
PARALLEL=""
SLOT_CTX=""
UBATCH=""
SLOT_SAVE_PATH=""
CACHE_REUSE=""
LORAS=()
API_KEY_ARG=""      # never persisted — see the --save-defaults block
PRINT_ONLY=0
# Launch backend. Per-launch, like --print: none of these is ever written by
# --save-defaults, so a settings file means the same thing on either backend.
DOCKER=0; [[ "${LLAMA_DOCKER:-0}" == 1 ]] && DOCKER=1
DETACH=0

MMPROJ="$(model_mmproj_path "$MODEL_KEY")"   # empty for text-only models
DRAFT="$(model_draft_path "$MODEL_KEY")"     # empty unless MTP is a separate model
TEMP_THINK="$(model_info "$MODEL_KEY" temp)" # thinking-mode default for this model
SAMPLING="$(model_info "$MODEL_KEY" sampling)" # qwen | google sampler family
IMGTOK_TABLE="$(model_info "$MODEL_KEY" imgtok)"  # "" = no floor for this model
IMGTOK="$IMGTOK_TABLE"                       # --image-tokens overrides it
THINK_MODE="$(model_info "$MODEL_KEY" think)" # switch | auto (no switch to drive)
ALIAS="$(model_info "$MODEL_KEY" alias)"     # "" = llama-server's default id

# For models that decide per turn whether to reason there is nothing for these
# flags to act on; say so rather than silently ignoring them.
no_think_switch() {
  echo "$1 does not apply: $MODEL_KEY decides per turn whether to reason; there is no thinking switch." >&2
  exit 1
}

# Likewise for a template that always reasons at an effort level: an on/off
# flag would be silently meaningless, and --no-think in particular would start
# a server that reasons anyway.
no_effort_switch() {
  echo "$1 does not apply: $MODEL_KEY always reasons at a reasoning-effort level. Use --effort none for the least reasoning (it lowers reasoning; it does not remove it)." >&2
  exit 1
}

# Per-model defaults: settings/<key>.conf beats the table's args field; CLI
# flags are appended after them and the parser is last-one-wins, so precedence
# is CLI > settings file > table args > the built-ins above.
DEFAULTS="$(model_args "$MODEL_KEY")"
if [[ -n "$DEFAULTS" ]]; then
  read -r -a DEFAULT_FLAGS <<< "$DEFAULTS"   # whitespace split, no glob expansion
  # To stderr when printing, so --print's stdout is the command and nothing
  # else. (Looked for by hand: the flags are not parsed yet at this point.)
  _printing=0
  for _arg in "$@"; do
    [[ "$_arg" == --print || "$_arg" == --dry-run ]] && _printing=1
  done
  if (( _printing )); then echo "defaults for $MODEL_KEY: $DEFAULTS" >&2
  else                     echo "defaults for $MODEL_KEY: $DEFAULTS"; fi
  unset _printing _arg
  set -- "${DEFAULT_FLAGS[@]}" "$@"
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --ctx)  CTX="${2:?--ctx needs a value}"; shift 2 ;;
    --kv)   KV="${2:?--kv needs a value}"; shift 2 ;;
    --mtp)  MTP="${2:?--mtp needs a value}"; MTP_ON=1; shift 2 ;;
    --no-mtp) MTP_ON=0; shift ;;
    --think)    [[ "$THINK_MODE" == auto ]] && no_think_switch "$1"
                [[ "$THINK_MODE" == effort ]] && no_effort_switch "$1"; THINK=1; shift ;;
    --no-think) [[ "$THINK_MODE" == auto ]] && no_think_switch "$1"
                [[ "$THINK_MODE" == effort ]] && no_effort_switch "$1"; THINK=0; shift ;;
    --temp) TEMP="${2:?--temp needs a value}"; shift 2 ;;
    --effort)   [[ "$THINK_MODE" == auto ]] && no_think_switch "$1"
                EFFORT="${2:?--effort needs a value}"; shift 2 ;;
    --preserve-thinking) PRESERVE=1; shift ;;
    --no-preserve-thinking) PRESERVE=0; shift ;;
    --no-vision) VISION=0; shift ;;
    --image-tokens) IMGTOK="${2:?--image-tokens needs a value}"; shift 2 ;;
    --port) PORT="${2:?--port needs a value}"; shift 2 ;;
    --ngl) NGL="${2:?--ngl needs a value}"; shift 2 ;;
    --parallel) PARALLEL="${2:?--parallel needs a value}"; shift 2 ;;
    --slot-ctx) SLOT_CTX="${2:?--slot-ctx needs a value}"; shift 2 ;;
    --ubatch) UBATCH="${2:?--ubatch needs a value}"; shift 2 ;;
    --api-key) API_KEY_ARG="${2:?--api-key needs a value}"; shift 2 ;;
    --print|--dry-run) PRINT_ONLY=1; shift ;;
    --docker) DOCKER=1; shift ;;
    --detach) DETACH=1; shift ;;
    --slot-save-path) SLOT_SAVE_PATH="${2:?--slot-save-path needs a path}"; shift 2 ;;
    --cache-reuse) CACHE_REUSE="${2:?--cache-reuse needs a value}"; shift 2 ;;
    --lora) LORAS+=("${2:?--lora needs a path}"); shift 2 ;;
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
if ! [[ "$CTX" =~ ^[0-9]+$ ]] || (( CTX < 1 )); then
  echo "--ctx must be a positive integer (got: $CTX)" >&2
  exit 1
fi
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
  echo "--port must be a TCP port between 1 and 65535 (got: $PORT)" >&2
  exit 1
fi

if (( DETACH && ! DOCKER )); then
  echo "--detach only applies with --docker (a host llama-server runs in the foreground)." >&2
  exit 1
fi
# Docker bind mounts need absolute sources: a relative one is read as a named
# volume, which would mount an empty volume where the weights should be.
if (( DOCKER )) && [[ "$HF_HUB_CACHE" != /* ]]; then
  echo "HF_HUB_CACHE must be an absolute path to mount it into the container (got: $HF_HUB_CACHE)" >&2
  exit 1
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
if [[ -n "$EFFORT" && "$THINK_MODE" == effort ]]; then
  # Inkling 2's template takes these names (it also takes a number in
  # 0.0-0.99, which this script does not wrap — use EXTRA_ARGS for that).
  case "$EFFORT" in
    none|minimal|low|medium|high|xhigh|max) ;;
    *) echo "--effort must be one of none, minimal, low, medium, high, xhigh, max (got: $EFFORT)" >&2; exit 1 ;;
  esac
elif [[ -n "$EFFORT" ]]; then
  case "$EFFORT" in
    low|medium|xhigh) ;;
    none) echo "--effort none is not a template value: use --no-think." >&2; exit 1 ;;
    *) echo "--effort must be low, medium, or xhigh (got: $EFFORT)" >&2; exit 1 ;;
  esac
fi

# Numeric passthroughs: catch a typo here, not as a llama-server usage error
# thirty seconds into loading the weights.
for _pair in "ngl:$NGL" "parallel:$PARALLEL" "cache-reuse:$CACHE_REUSE" "slot-ctx:$SLOT_CTX" "ubatch:$UBATCH"; do
  _name="${_pair%%:*}"; _val="${_pair#*:}"
  if [[ -n "$_val" && ! "$_val" =~ ^[0-9]+$ ]]; then
    echo "--$_name must be a non-negative integer (got: $_val)" >&2; exit 1
  fi
done
unset _pair _name _val
if [[ -n "$SLOT_CTX" ]] && (( SLOT_CTX < 1 )); then
  echo "--slot-ctx must be a positive integer (got: $SLOT_CTX)" >&2; exit 1
fi
if [[ -n "$UBATCH" ]] && (( UBATCH < 1 )); then
  echo "--ubatch must be a positive integer (got: $UBATCH)" >&2; exit 1
fi

# A relative --slot-save-path is relative to the caller's cwd; this script has
# already cd'd to its own directory, so resolve it before anything (the save
# below, llama-server itself) can read it as scripts-dir-relative.
if [[ -n "$SLOT_SAVE_PATH" ]]; then
  case "$SLOT_SAVE_PATH" in
    /*) ;;
    *) SLOT_SAVE_PATH="$CALLER_PWD/$SLOT_SAVE_PATH" ;;
  esac
fi

# Same for --lora in docker mode only: the file has to be mounted, and a mount
# needs an absolute path that means the same thing inside the container. Host
# mode keeps passing --lora through untouched, as it always has.
if (( DOCKER )); then
  for _i in "${!LORAS[@]}"; do
    case "${LORAS[$_i]}" in
      /*) ;;
      *) LORAS[$_i]="$CALLER_PWD/${LORAS[$_i]}" ;;
    esac
  done
  unset _i
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
  if (( NGL != 99 ));        then SAVED+=(--ngl "$NGL"); fi
  if [[ -n "$PARALLEL" ]];   then SAVED+=(--parallel "$PARALLEL"); fi
  if [[ -n "$SLOT_CTX" ]];   then SAVED+=(--slot-ctx "$SLOT_CTX"); fi
  if [[ -n "$UBATCH" ]];     then SAVED+=(--ubatch "$UBATCH"); fi
  if [[ -n "$CACHE_REUSE" ]]; then SAVED+=(--cache-reuse "$CACHE_REUSE"); fi
  # Already absolute by here, so the saved line means the same thing next time
  # no matter which directory that launch is started from.
  if [[ -n "$SLOT_SAVE_PATH" ]]; then SAVED+=(--slot-save-path "$SLOT_SAVE_PATH"); fi
  for _lora in "${LORAS[@]-}"; do [[ -n "$_lora" ]] && SAVED+=(--lora "$_lora"); done
  mkdir -p "$SETTINGS_DIR"
  printf '%s\n' "${SAVED[*]-}" > "$SETTINGS_DIR/$MODEL_KEY.conf"
  echo "saved defaults for $MODEL_KEY: ${SAVED[*]-(none)}"
fi

# ── Compose the server command ───────────────────────────────────────────────
ARGS=(
  -m "$MODEL_PATH"
  -ngl "$NGL"
  --jinja
  -fa on
  -c "$CTX"
)

[[ -n "$PARALLEL" ]] && ARGS+=(--parallel "$PARALLEL")
# Shared pool: -c is then what all slots draw on together rather than a total
# split evenly between them, and the per-slot cap keeps one long conversation
# from running past the model's trained context. /metrics rides along because
# this is the several-users configuration, where load is worth watching.
[[ -n "$SLOT_CTX" ]] && ARGS+=(--kv-unified --kv-unified-per-slot "$SLOT_CTX" --metrics)
[[ -n "$UBATCH" ]] && ARGS+=(-ub "$UBATCH")
[[ -n "$ALIAS" ]] && ARGS+=(--alias "$ALIAS")
[[ -n "$CACHE_REUSE" ]] && ARGS+=(--cache-reuse "$CACHE_REUSE")
# llama-server does not create the slot directory itself — it fails at startup
# if it is missing.
if [[ -n "$SLOT_SAVE_PATH" ]]; then
  (( PRINT_ONLY )) || mkdir -p "$SLOT_SAVE_PATH"   # --print changes nothing on disk
  ARGS+=(--slot-save-path "$SLOT_SAVE_PATH")
fi

# Adapters load inactive: --lora-init-without-apply lets a client attach them
# per-request, so requests that need the base weights are unaffected.
if (( ${#LORAS[@]} )); then
  ARGS+=(--lora-init-without-apply)
  for _lora in "${LORAS[@]}"; do ARGS+=(--lora "$_lora"); done
fi

[[ "$KV" != f16 ]] && ARGS+=(-ctk "$KV" -ctv "$KV")

# A separate drafter (Gemma 4's assistant) rides in on -md; a missing one turns
# MTP off rather than failing, since draft-mtp without a head has nothing to run.
if (( MTP_ON )) && [[ -n "$DRAFT" ]]; then
  if [[ -f "$DRAFT" ]]; then
    ARGS+=(-md "$DRAFT")
  else
    echo "warning: MTP drafter missing ($DRAFT) — serving without MTP." >&2
    echo "         run ./fetch-models.sh $MODEL_KEY to get it." >&2
    MTP_ON=0
  fi
fi
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
if [[ "$SAMPLING" == google ]]; then
  # Google's Gemma 4 recommendation, which does not change with thinking mode.
  ARGS+=(--temp "${TEMP:-$TEMP_THINK}" --top-p 0.95 --top-k 64 --min-p 0.0
         --presence-penalty 0.0 --repeat-penalty 1.0)
elif [[ "$SAMPLING" == plain ]]; then
  # Thinking Machines' Inkling 2 recommendation: the temperature and nothing
  # else — no nucleus, no top-k (0 disables it), no penalties.
  ARGS+=(--temp "${TEMP:-$TEMP_THINK}" --top-p 1.0 --top-k 0 --min-p 0.0
         --presence-penalty 0.0 --repeat-penalty 1.0)
elif (( THINK )); then
  # Unsloth-recommended sampling for coding, thinking mode. The temperature is
  # per-model (the table's temp field: 0.6 for Qwen3.6, 1.0 for Qwen3.8).
  ARGS+=(--temp "${TEMP:-$TEMP_THINK}" --top-p 0.95 --top-k 20 --min-p 0.0
         --presence-penalty 0.0 --repeat-penalty 1.0)
else
  # Unsloth-recommended sampling for non-thinking mode.
  ARGS+=(--temp "${TEMP:-0.7}" --top-p 0.8 --top-k 20 --min-p 0.0
         --presence-penalty 1.5 --repeat-penalty 1.0)
fi

if [[ "$THINK_MODE" == auto ]]; then
  # Nothing to switch: the template has no enable_thinking, and forcing
  # --reasoning on would have the parser assume every turn opens inside a
  # reasoning block, which this kind of model does not do.
  :
elif [[ "$THINK_MODE" == effort ]]; then
  # The template writes a thinking-effort line on every request and has no
  # enable_thinking or preserve key, so the only thing to send is the level.
  # Unset leaves the template's own default (0.9, which it calls high). A
  # request that carries its own reasoning_effort overrides this.
  [[ -n "$EFFORT" ]] && ARGS+=(--chat-template-kwargs "{\"reasoning_effort\":\"$EFFORT\"}")
elif (( THINK )); then
  ARGS+=(--reasoning on)
  if (( PRESERVE )); then ARGS+=(--reasoning-preserve)
  else                    ARGS+=(--no-reasoning-preserve); fi
  # Models whose template ignores reasoning_effort are unaffected by the key.
  [[ -n "$EFFORT" ]] && ARGS+=(--chat-template-kwargs "{\"reasoning_effort\":\"$EFFORT\"}")
else
  # reasoning_effort is dead here — the template only reads it when thinking is on.
  ARGS+=(--reasoning off)
fi

[[ -n "$API_KEY_ARG" ]] && ARGS+=(--api-key "$API_KEY_ARG")

# In a container the server always binds every interface; $HOST decides which
# host address the port is published on instead (see the -p below).
if (( DOCKER )); then ARGS+=(--host 0.0.0.0 --port "$PORT")
else                  ARGS+=(--host "$HOST" --port "$PORT"); fi

# Raw passthrough for anything run.sh doesn't wrap (--mmproj, YaRN, sampler
# tweaks). Appended last so it overrides the composed flags. Word-split:
# values containing spaces are not supported — edit run.sh for those.
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  read -r -a EXTRA_ARGV <<< "$EXTRA_ARGS"
  ARGS+=("${EXTRA_ARGV[@]}")
fi

# ── Compose the launch command ───────────────────────────────────────────────
# Host backend: the binary and its arguments. Docker backend: the same
# arguments after the image name — the image's entrypoint is llama-server.
if (( DOCKER )); then
  LAUNCH=(docker run)
  if (( DETACH )); then
    LAUNCH+=(-d --restart unless-stopped
             # A server left running for months must not fill the disk with
             # its own log; docker's default json-file driver never rotates.
             --log-opt max-size=50m --log-opt max-file=3)
  else
    LAUNCH+=(--rm)
    # A TTY lets Ctrl-C reach the server. Not when printing (keep the printed
    # line the same wherever it is generated) or when there is no terminal.
    if (( ! PRINT_ONLY )) && [[ -t 0 && -t 1 ]]; then LAUNCH+=(-it); fi
  fi
  LAUNCH+=(--name "$DEPLOY_CONTAINER" --gpus all
           # What is being served, readable back with `docker inspect` — how a
           # later launch (and deploy.sh status/smoke) names the model in here.
           --label "llm.key=$MODEL_KEY" --label "llm.alias=$ALIAS"
           -p "$HOST:$PORT:$PORT"
           # The image's built-in healthcheck probes :8080; point it at the
           # port actually served, or `docker ps` reports a healthy server
           # as unhealthy.
           --health-cmd "curl -fsS http://localhost:$PORT/health"
           # /health answers 503 until the weights are in, which takes minutes
           # for a large model; don't count those against the container.
           --health-start-period 20m
           # The whole hub directory, not one snapshot: snapshot entries are
           # relative symlinks into blobs/, which must resolve inside too.
           -v "$HF_HUB_CACHE:$HF_HUB_CACHE:ro")
  [[ -n "$SLOT_SAVE_PATH" ]] && LAUNCH+=(-v "$SLOT_SAVE_PATH:$SLOT_SAVE_PATH")
  _seen_dirs=""
  for _lora in "${LORAS[@]-}"; do
    [[ -n "$_lora" ]] || continue
    _dir="$(dirname -- "$_lora")"
    case "$_seen_dirs" in *"|$_dir|"*) continue ;; esac
    _seen_dirs="$_seen_dirs|$_dir|"
    LAUNCH+=(-v "$_dir:$_dir:ro")
  done
  unset _seen_dirs _lora _dir
  # By name only: docker copies the value from this process's environment, so
  # the key never appears on a command line or in what --print shows.
  [[ -n "${LLAMA_API_KEY:-}" ]] && LAUNCH+=(-e LLAMA_API_KEY)
  if [[ -n "${DOCKER_ARGS:-}" ]]; then
    read -r -a DOCKER_ARGV <<< "$DOCKER_ARGS"
    LAUNCH+=("${DOCKER_ARGV[@]}")
  fi
  LAUNCH+=("$LLAMA_IMAGE" "${ARGS[@]}")
else
  LAUNCH=("$LLAMA_SERVER" "${ARGS[@]}")
fi

if (( PRINT_ONLY )); then
  printf '%q' "${LAUNCH[0]}"; printf ' %q' "${LAUNCH[@]:1}"; echo
  exit 0
fi

# Preflight (binary or image / model / GPU / port). Skip with SKIP_PREFLIGHT=1.
if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
  LLAMA_DOCKER="$DOCKER" ./preflight.sh "$MODEL_KEY"
fi

# port_holder — the raw `ss` LISTEN row(s) for $PORT, or empty if free. Scans
# ALL matching rows, not just the first: on a dual-stack or SO_REUSEPORT port
# whose first row has an empty Process field but a later row carries a pid,
# reporting "hidden" off the first row alone would be wrong. Matched by port
# only, not by host: a process on 0.0.0.0:$PORT still collides with a bind on
# $HOST:$PORT. No early `exit` in the awk action — awk must drain ss's stdout
# or ss takes SIGPIPE, and under `set -o pipefail` the command substitution
# then fails and `set -e` kills this script silently, on exactly the path whose
# job is to print the failure below.
port_holder() {
  command -v ss >/dev/null 2>&1 || return 0
  ss -ltnp 2>/dev/null | awk -v port=":$PORT" '$1 == "LISTEN" && $4 ~ (port "$") {print}' || true
}

# Stop any llama-server already running from this build. Matched on each PID's
# real executable, not on its command line: `pkill -f "$LLAMA_SERVER"` also
# matches any *other* process that merely mentions the path — a shell that
# exported it, an editor, this script's own parent — and kills it.
stop_running_server() {
  local target pid exe stopped=0 waited_ms=0 announced=0
  target="$(readlink -f "$LLAMA_SERVER" 2>/dev/null)" || return 0
  [[ -z "$target" ]] && return 0
  for pid in $(pgrep -x "${LLAMA_SERVER##*/}" 2>/dev/null); do
    exe="$(readlink -f "/proc/$pid/exe" 2>/dev/null)" || continue
    [[ "$exe" == "$target" ]] || continue
    kill "$pid" 2>/dev/null && stopped=1
  done
  (( stopped )) || return 0
  echo "stopped a running llama-server"
  # A server holding tens of GB of weights routinely needs longer than a fixed
  # couple of seconds to free its VRAM and unbind. Poll, so a slow-but-fine
  # shutdown is not misreported below as a foreign process squatting the port.
  while [[ -n "$(port_holder)" ]]; do
    (( waited_ms >= 30000 )) && break
    if (( ! announced )) && (( waited_ms >= 1000 )); then
      echo "waiting for it to release port $PORT..."
      announced=1
    fi
    sleep 0.5
    waited_ms=$(( waited_ms + 500 ))
  done
  return 0
}
# replace_container — the docker counterpart of stop_running_server. Matched on
# the exact name this script owns, so an unrelated container is never touched.
# One GPU serves one model, so whatever holds the name goes — and is named, so
# replacing a different model is never a silent surprise.
replace_container() {
  local was waited_ms=0
  [[ -n "$(docker ps -aq --filter "name=^${DEPLOY_CONTAINER}$" 2>/dev/null)" ]] || return 0
  was="$(docker inspect -f '{{ index .Config.Labels "llm.key" }}' "$DEPLOY_CONTAINER" 2>/dev/null || true)"
  echo "replacing container $DEPLOY_CONTAINER (was serving ${was:-an unknown model})"
  docker rm -f "$DEPLOY_CONTAINER" >/dev/null
  # docker-proxy can hold the published port for a moment after the container
  # is gone; wait it out rather than report our own leftovers as a conflict.
  while [[ -n "$(port_holder)" ]] && (( waited_ms < 30000 )); do
    sleep 0.5; waited_ms=$(( waited_ms + 500 ))
  done
  return 0
}

if (( DOCKER )); then replace_container
else                  stop_running_server; fi

# Anything still LISTENing on the port belongs to someone else. Say so here,
# instead of letting llama-server die a few lines down on a raw bind error
# after it has already started loading.
CONFLICT="$(port_holder)"
if [[ -n "$CONFLICT" ]]; then
  # ss prints users:(("name",pid=NNNN,fd=N)) when the process is visible (same
  # user, or root) and leaves the field empty otherwise. Take the first pid and
  # first quoted name found across all matching rows; only fall back to the
  # hidden-pid wording when no row yields a pid at all.
  CONFLICT_PID="$(printf '%s\n' "$CONFLICT" | grep -oE 'pid=[0-9]+' | head -n1 | cut -d= -f2)"
  CONFLICT_NAME="$(printf '%s\n' "$CONFLICT" | grep -oE '"[^"]+"' | head -n1 | tr -d '"')"
  echo "FAIL: port $PORT is already in use" >&2
  if [[ -n "$CONFLICT_PID" ]]; then
    echo "      held by pid $CONFLICT_PID (${CONFLICT_NAME:-unknown})" >&2
  else
    echo "      held by another user's process (pid hidden — needs root to see)" >&2
  fi
  echo "      Stop it, or serve elsewhere: ./run.sh $MODEL_KEY --port <other>" >&2
  exit 1
fi

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
case "$THINK_MODE" in
  auto)   THINK_STATE=auto ;;
  effort) THINK_STATE="effort:${EFFORT:-template default (high)}" ;;
  *) if (( THINK )); then
       THINK_STATE="on(preserve=$( (( PRESERVE )) && echo on || echo off)${EFFORT:+,effort=$EFFORT})"
     else THINK_STATE=off; fi ;;
esac
echo "Launching [$MODEL_KEY]  ctx=$CTX  kv=$KV  mtp=$([[ $MTP_ON == 1 ]] && echo "$MTP" || echo off)  thinking=$THINK_STATE  vision=$VISION_STATE:"
printf '  %q' "${LAUNCH[@]}"; echo
echo
exec "${LAUNCH[@]}"
