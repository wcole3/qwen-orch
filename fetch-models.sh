#!/usr/bin/env bash
# fetch-models.sh — download Qwen model weights from HuggingFace.
#
#   ./fetch-models.sh                # show the menu (with download status)
#   ./fetch-models.sh nvfp4          # fetch one model
#   ./fetch-models.sh q8 bf16        # fetch several
#   ./fetch-models.sh all            # fetch everything (~360 GiB)
#
# Files land in the standard HuggingFace cache, $HF_HUB_CACHE (see env.sh).
# Already-present files are skipped, and the free space is checked against what
# the requested models actually need before anything is downloaded.
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

# free_gib — GiB available on the filesystem holding $HF_HUB_CACHE.
free_gib() {
  local kb
  kb="$(df -Pk "$HF_HUB_CACHE" 2>/dev/null | tail -1 | awk '{print $4}')" || return 1
  [[ -n "$kb" ]] || return 1
  echo "$(( kb / 1024 / 1024 ))"
}

menu() {
  echo "usage: ./fetch-models.sh <model>... | all"
  echo
  model_list
  echo
  echo "Downloads go to: $HF_HUB_CACHE"
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

mkdir -p "$HF_HUB_CACHE"

# Space check. Counts only what is still missing, so re-running after a partial
# download does not demand room for the files already on disk.
NEEDED="$(model_disk_needed "${TARGETS[@]}")"
FREE="$(free_gib 2>/dev/null || echo "")"
if (( NEEDED == 0 )); then
  echo "Nothing to download — every requested model is already present."
elif [[ -z "$FREE" ]]; then
  echo "warning: could not read free space for $HF_HUB_CACHE; need ~${NEEDED} GiB." >&2
else
  echo "Space: need ~${NEEDED} GiB, have ${FREE} GiB free at $HF_HUB_CACHE"
  if (( FREE < NEEDED )); then
    echo >&2
    echo "Not enough free space: ${NEEDED} GiB required, ${FREE} GiB available." >&2
    echo "Free up space, set HF_HOME to a bigger filesystem, or fetch fewer" >&2
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

# No --local-dir anywhere: hf then writes the standard cache layout under
# $HF_HUB_CACHE (exported by env.sh), which model_path resolves.
fetch() {
  local key="$1"
  local repo file path mmproj mmproj_path
  repo="$(model_info "$key" repo)"
  file="$(model_info "$key" file)"
  mmproj="$(model_info "$key" mmproj)"
  mmproj_path="$(model_mmproj_path "$key")"

  if path="$(model_path "$key")"; then
    echo "✓ already present: $path"
  else
    echo "↓ $repo :: $file  ($(model_info "$key" size))  ->  $HF_HUB_CACHE"
    case "$file" in
      *'*'*) "${HF[@]}" "$repo" --include "$file" >/dev/null ;;
      *)     "${HF[@]}" "$repo" "$file"           >/dev/null ;;
    esac
  fi

  # Vision projector: separate download, and separately skippable — two keys can
  # share one repo (and one mmproj), so this runs even when the weights were
  # already present.
  if [[ -n "$mmproj" ]]; then
    if [[ -f "$mmproj_path" ]]; then
      echo "✓ already present: $mmproj_path"
    else
      echo "↓ $repo :: $mmproj  (vision projector)  ->  $HF_HUB_CACHE"
      "${HF[@]}" "$repo" "$mmproj" >/dev/null
    fi
  fi

  # Separate MTP drafter, for models whose MTP head is its own model. Shared
  # across keys like the projector, so checked independently of the weights.
  local draft draft_path draftrepo
  draft="$(model_info "$key" draft)"
  if [[ -n "$draft" ]]; then
    draft_path="$(model_draft_path "$key")"
    draftrepo="$(model_info "$key" draftrepo)"
    if [[ -f "$draft_path" ]]; then
      echo "✓ already present: $draft_path"
    else
      echo "↓ $draftrepo :: $draft  (MTP drafter)  ->  $HF_HUB_CACHE"
      "${HF[@]}" "$draftrepo" "$draft" >/dev/null
    fi
  fi
}

for key in "${TARGETS[@]}"; do
  fetch "$key"
done

echo
echo "Done. Verifying expected paths:"
for key in "${TARGETS[@]}"; do
  if path="$(model_path "$key")"; then
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
  draft_path="$(model_draft_path "$key")"
  if [[ -n "$draft_path" ]]; then
    if [[ -f "$draft_path" ]]; then
      printf '  ✓ %s\n' "$draft_path"
    else
      printf '  ✗ MISSING %s\n' "$draft_path"
    fi
  fi
done
