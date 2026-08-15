#!/usr/bin/env bash
# fetch-models.sh — download Qwen model weights from HuggingFace.
#
#   ./fetch-models.sh                # show the menu (with download status)
#   ./fetch-models.sh nvfp4          # fetch one model
#   ./fetch-models.sh q8 bf16        # fetch several
#   ./fetch-models.sh all            # fetch everything (~360 GiB)
#
# Files land under $MODELS_DIR (see env.sh). Already-present files are skipped,
# and the free space is checked against what the requested models actually need
# before anything is downloaded.
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

# free_gib — GiB available on the filesystem holding $MODELS_DIR.
free_gib() {
  local kb
  kb="$(df -Pk "$MODELS_DIR" 2>/dev/null | tail -1 | awk '{print $4}')" || return 1
  [[ -n "$kb" ]] || return 1
  echo "$(( kb / 1024 / 1024 ))"
}

menu() {
  echo "usage: ./fetch-models.sh <model>... | all"
  echo
  model_list
  echo
  echo "Downloads go to: $MODELS_DIR"
  echo "Free disk space: $(free_gib 2>/dev/null || echo '?') GiB"
  echo "Everything at once needs ~360 GiB; the disk column is per model."
}

if [[ $# -eq 0 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  menu; exit 0
fi

TARGETS=("$@")
[[ "${TARGETS[0]}" == "all" ]] && TARGETS=("${MODEL_KEYS[@]}")

for key in "${TARGETS[@]}"; do
  model_info "$key" repo >/dev/null 2>&1 || { echo "unknown model: $key" >&2; menu >&2; exit 1; }
done

mkdir -p "$MODELS_DIR"

# Space check. Counts only what is still missing, so re-running after a partial
# download does not demand room for the files already on disk.
NEEDED="$(model_disk_needed "${TARGETS[@]}")"
FREE="$(free_gib 2>/dev/null || echo "")"
if (( NEEDED == 0 )); then
  echo "Nothing to download — every requested model is already present."
elif [[ -z "$FREE" ]]; then
  echo "warning: could not read free space for $MODELS_DIR; need ~${NEEDED} GiB." >&2
else
  echo "Space: need ~${NEEDED} GiB, have ${FREE} GiB free at $MODELS_DIR"
  if (( FREE < NEEDED )); then
    echo >&2
    echo "Not enough free space: ${NEEDED} GiB required, ${FREE} GiB available." >&2
    echo "Free up space, set MODELS_DIR to a bigger filesystem, or fetch fewer" >&2
    echo "models — ./fetch-models.sh lists the per-model requirement." >&2
    exit 1
  fi
fi

# Resolve a Hugging Face downloader, installing the CLI if necessary.
if command -v hf >/dev/null 2>&1; then
  HF=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF=(huggingface-cli download)
else
  echo "huggingface_hub CLI not found; installing with pip --user ..."
  python3 -m pip install --user -U huggingface_hub
  hash -r
  if command -v hf >/dev/null 2>&1; then HF=(hf download)
  else HF=(huggingface-cli download); fi
fi

fetch() {
  local key="$1"
  local repo file path fmt dest mmproj mmproj_path
  repo="$(model_info "$key" repo)"
  file="$(model_info "$key" file)"
  path="$(model_info "$key" path)"
  fmt="$(model_info "$key" fmt)"
  mmproj="$(model_info "$key" mmproj)"
  mmproj_path="$(model_mmproj_path "$key")"

  case "$fmt" in
    safetensors) dest="$path" ;;                     # whole repo into its own dir
    *)           dest="$MODELS_DIR/${repo##*/}" ;;   # GGUFs into <repo-name>/
  esac
  mkdir -p "$dest"

  if model_downloaded "$key"; then
    echo "✓ already present: $path"
  else
    echo "↓ $repo :: $file  ($(model_info "$key" size))  ->  $dest"
    case "$file" in
      *'*'*) "${HF[@]}" "$repo" --include "$file" --local-dir "$dest" ;;
      *)     "${HF[@]}" "$repo" "$file"           --local-dir "$dest" ;;
    esac
  fi

  # Vision projector: separate download, and separately skippable — two keys can
  # share one repo (and one mmproj), so this runs even when the weights were
  # already present.
  if [[ -n "$mmproj" ]]; then
    if [[ -f "$mmproj_path" ]]; then
      echo "✓ already present: $mmproj_path"
    else
      echo "↓ $repo :: $mmproj  (vision projector)  ->  $MODELS_DIR/${repo##*/}"
      "${HF[@]}" "$repo" "$mmproj" --local-dir "$MODELS_DIR/${repo##*/}"
    fi
  fi
}

for key in "${TARGETS[@]}"; do
  fetch "$key"
done

echo
echo "Done. Verifying expected paths:"
for key in "${TARGETS[@]}"; do
  path="$(model_info "$key" path)"
  if model_downloaded "$key"; then
    printf '  ✓ %s\n' "$path"
  else
    printf '  ✗ MISSING %s\n' "$path"
  fi
  mmproj_path="$(model_mmproj_path "$key")"
  if [[ -n "$mmproj_path" ]]; then
    if [[ -f "$mmproj_path" ]]; then
      printf '  ✓ %s\n' "$mmproj_path"
    else
      printf '  ✗ MISSING %s\n' "$mmproj_path"
    fi
  fi
done
