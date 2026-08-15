#!/usr/bin/env bash
# preflight.sh — verify everything needed to launch is in place.
# Usage: ./preflight.sh <model-key>   (e.g. nvfp4, q8, bf16)
# Exits non-zero on the first hard failure with an actionable message.
set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh

MODEL_KEY="${1:?usage: ./preflight.sh <model-key>}"
model_info "$MODEL_KEY" path >/dev/null 2>&1 \
  || { echo "unknown model key: $MODEL_KEY" >&2; exit 1; }
MODEL_PATH="$(model_info "$MODEL_KEY" path)"
NEED_VRAM="$(model_info "$MODEL_KEY" vram)"
MIN_BUILD="$(model_info "$MODEL_KEY" minbuild)"

fail=0
note() { printf '  %s %s\n' "$1" "$2"; }

echo "Preflight for model: $MODEL_KEY"

# 1. Binary
if [[ -x "$LLAMA_SERVER" ]] && "$LLAMA_SERVER" --version >/dev/null 2>&1; then
  build="$("$LLAMA_SERVER" --version 2>&1 | sed -n 's/^version: \([0-9]*\).*/\1/p')"
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

# 2. Model file
if [[ -f "$MODEL_PATH" ]]; then
  note "✓" "model: $MODEL_PATH ($(du -h "$MODEL_PATH" | cut -f1))"
else
  note "✗" "model missing: $MODEL_PATH  — run ./fetch-models.sh $MODEL_KEY"
  fail=1
fi

# 2b. Vision projector (only for models that declare one). Not a hard failure:
# run.sh drops to text-only and says so.
MMPROJ="$(model_mmproj_path "$MODEL_KEY")"
if [[ -n "$MMPROJ" ]]; then
  if [[ -f "$MMPROJ" ]]; then
    note "✓" "vision projector: $MMPROJ ($(du -h "$MMPROJ" | cut -f1))"
  else
    note "⚠" "vision projector missing: $MMPROJ — text-only until you run ./fetch-models.sh $MODEL_KEY"
  fi
fi

# 3. GPU / VRAM
if command -v nvidia-smi >/dev/null 2>&1; then
  IFS=',' read -r name free total < <(nvidia-smi \
      --query-gpu=name,memory.free,memory.total --format=csv,noheader,nounits | head -n1)
  name="${name#"${name%%[![:space:]]*}"}"; free="${free//[[:space:]]/}"; total="${total//[[:space:]]/}"
  note "✓" "GPU: $name  (${free} MiB free / ${total} MiB)"
  if [[ "${free:-0}" -lt "$NEED_VRAM" ]]; then
    note "⚠" "~${NEED_VRAM} MiB expected for $MODEL_KEY at its default settings — free VRAM is below that."
    echo "      Another llama-server may be running (run.sh will stop it), or use"
    echo "      a smaller --ctx / --kv q8_0."
  fi
else
  note "⚠" "nvidia-smi not found — cannot verify GPU/VRAM."
fi

# 4. Port
if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -q ":$PORT "; then
  note "⚠" "port $PORT is already in use — run.sh will pkill the existing llama-server first."
else
  note "✓" "port $PORT is free"
fi

echo
if [[ "$fail" -ne 0 ]]; then
  echo "Preflight FAILED — fix the ✗ items above."
  exit 1
fi
echo "Preflight OK."
