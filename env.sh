#!/usr/bin/env bash
# env.sh — shared environment + model table for the Qwen local-serving scripts.
# Source this from the other scripts: `source "$(dirname "$0")/env.sh"`.
# Every value is override-friendly: export it before running, or edit the
# defaults below.

# ── Toolchain ────────────────────────────────────────────────────────────────
# LLAMA_MAINLINE is the directory *containing* your llama.cpp checkout. If your
# layout differs, point LLAMA_SERVER straight at the binary instead:
#   export LLAMA_SERVER=/opt/llama.cpp/build/bin/llama-server
# With neither set, the usual locations are probed in order, then $PATH.
LLAMA_MAINLINE="${LLAMA_MAINLINE:-$HOME/devspace}"

if [[ -z "${LLAMA_SERVER:-}" ]]; then
  for _cand in \
      "$LLAMA_MAINLINE/llama.cpp/build/bin/llama-server" \
      "$HOME/llama.cpp/build/bin/llama-server" \
      "$HOME/src/llama.cpp/build/bin/llama-server" \
      "$(command -v llama-server 2>/dev/null || true)"; do
    if [[ -n "$_cand" && -x "$_cand" ]]; then LLAMA_SERVER="$_cand"; break; fi
  done
  # Still nothing: keep the primary guess so preflight can name it when it fails.
  LLAMA_SERVER="${LLAMA_SERVER:-$LLAMA_MAINLINE/llama.cpp/build/bin/llama-server}"
  unset _cand
fi

# ── Models ───────────────────────────────────────────────────────────────────
# Weights live here. ~360 GiB to hold every entry in the table at once; each
# entry's own requirement is its `disk` field, and ./fetch-models.sh checks the
# free space before it downloads anything.
MODELS_DIR="${MODELS_DIR:-$HOME/models}"

# ── Per-model saved settings (written by `run.sh <key> ... --save-defaults`) ─
SETTINGS_DIR="${SETTINGS_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/settings}"

# ── Server ───────────────────────────────────────────────────────────────────
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8081}"

export LLAMA_MAINLINE LLAMA_SERVER MODELS_DIR SETTINGS_DIR HOST PORT

# ── Model table ──────────────────────────────────────────────────────────────
# model_info <key> <field>
#   fields: repo | file | mmproj | path | fmt | minbuild | size | disk | vram |
#           temp | args | desc
#   repo     — HuggingFace repo id
#   file     — in-repo file (or include-glob for multi-file entries)
#   mmproj   — in-repo multimodal projector file, or "" for text-only models.
#              Downloaded next to the weights; run.sh passes it as --mmproj.
#   temp     — thinking-mode sampling temperature (default 0.6, Unsloth's
#              Qwen3.6 coding recommendation; Qwen3.8 wants 1.0). Belongs here
#              rather than in args because it is mode-specific: --no-think
#              always uses 0.7, which every Qwen generation agrees on.
#   path     — local path run.sh loads (-m); a directory for safetensors entries
#   fmt      — gguf (llama.cpp-servable) | safetensors (vLLM/SGLang only)
#   minbuild — minimum llama.cpp build number required (0 = any)
#   size     — download size as HuggingFace reports it, in decimal GB
#   disk     — space the weights occupy once written, in GiB (what df/du report).
#              Lower than `size` for the same bytes — 29 GB is 27 GiB — so the
#              two columns disagreeing by ~7% is expected, not an error.
#              Both cover the weights only: a vision projector is ~1 GiB more,
#              counted separately because it downloads separately. See
#              model_disk_needed, which fetch-models.sh checks free space against.
#   vram     — expected VRAM in MiB at the model's default settings (args applied;
#              f16 KV and 262k ctx unless args say otherwise)
#   args     — default run.sh flags, applied before CLI flags (CLI wins).
#              Space-separated simple flags only — values must not contain
#              spaces (the string is word-split). settings/<key>.conf overrides.
# GGUF path invariant: path == $MODELS_DIR/<repo basename>/<file> — fetch-models.sh
# computes its download dest from repo, and its skip/verify checks from path.
MODEL_KEYS=(nvfp4 nvfp4-attn q8 bf16 hauhau huihui aeon deckard nvfp4-unsloth
            38-q8 38-bf16)

model_info() {
  local key="$1" field="$2"
  local repo file mmproj path fmt minbuild size disk vram temp args desc
  mmproj=""    # text-only unless an entry sets it
  temp=0.6     # Qwen3.6 thinking-mode default unless an entry overrides it
  case "$key" in
    q8)
      repo=unsloth/Qwen3.6-27B-MTP-GGUF
      file=Qwen3.6-27B-Q8_0.gguf
      path="$MODELS_DIR/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-Q8_0.gguf"
      fmt=gguf; minbuild=0
      size="29 GB"; disk=28; vram=50000
      args=""
      desc="Q8_0, MTP baked in — near-lossless 8-bit" ;;
    bf16)
      repo=unsloth/Qwen3.6-27B-MTP-GGUF
      file='BF16/*'
      path="$MODELS_DIR/Qwen3.6-27B-MTP-GGUF/BF16/Qwen3.6-27B-BF16-00001-of-00002.gguf"
      fmt=gguf; minbuild=0
      size="55 GB"; disk=51; vram=76000
      args=""
      desc="BF16, MTP baked in — full precision reference" ;;
    nvfp4)
      repo=CodeFault/Nvidia-Qwen3.6-27B-NVFP4-GGUF
      file=Nvidia-Qwen3.6-27B-NVFP4-A.gguf
      path="$MODELS_DIR/Nvidia-Qwen3.6-27B-NVFP4-GGUF/Nvidia-Qwen3.6-27B-NVFP4-A.gguf"
      fmt=gguf; minbuild=9775
      size="18 GB"; disk=17; vram=40000
      args=""
      desc="NVFP4 FFN+attn, MTP head BF16 — native Blackwell FP4 path" ;;
    nvfp4-attn)
      repo=CodeFault/Nvidia-Qwen3.6-27B-NVFP4-GGUF
      file=Nvidia-Qwen3.6-27B-NVFP4-BF16-Attn.gguf
      path="$MODELS_DIR/Nvidia-Qwen3.6-27B-NVFP4-GGUF/Nvidia-Qwen3.6-27B-NVFP4-BF16-Attn.gguf"
      fmt=gguf; minbuild=9775
      size="28 GB"; disk=27; vram=50000
      args=""
      desc="NVFP4 FFN, BF16 attention — quality/speed middle ground" ;;
    hauhau)
      repo=HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
      file=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf
      path="$MODELS_DIR/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf"
      fmt=gguf; minbuild=0
      size="44 GB"; disk=41; vram=51000
      args="--no-mtp"
      desc="Q8_K_P 35B-A3B MoE uncensored — no MTP head, MTP off" ;;
    huihui)
      repo=huihui-ai/Huihui-Qwen3.6-27B-abliterated-MTP-GGUF
      file=Huihui-Qwen3.6-27B-abliterated-ggml-model-Q8_0.gguf
      path="$MODELS_DIR/Huihui-Qwen3.6-27B-abliterated-MTP-GGUF/Huihui-Qwen3.6-27B-abliterated-ggml-model-Q8_0.gguf"
      fmt=gguf; minbuild=0
      size="29 GB"; disk=28; vram=50000
      args=""
      desc="Q8_0 27B abliterated, MTP baked in" ;;
    aeon)
      repo=theLittleStone/Qwen3.6-27B-AEON-Ultimate-Uncensored-MTP-i1-GGUF
      file=Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16-MTP-i1.Q8_0.gguf
      path="$MODELS_DIR/Qwen3.6-27B-AEON-Ultimate-Uncensored-MTP-i1-GGUF/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16-MTP-i1.Q8_0.gguf"
      fmt=gguf; minbuild=0
      size="29 GB"; disk=28; vram=50000
      args=""
      desc="Q8_0 27B AEON uncensored merge (imatrix), MTP baked in" ;;
    deckard)
      repo=DavidAU/Qwen3.6-40B-Claude-4.6-Opus-Deckard-Heretic-Uncensored-Thinking-NEO-CODE-Di-IMatrix-MAX-GGUF
      file=Qwen3.6-40B-Deck-Opus-NEO-CODE-HERE-2T-OT-HIGH-Q8_0.gguf
      path="$MODELS_DIR/Qwen3.6-40B-Claude-4.6-Opus-Deckard-Heretic-Uncensored-Thinking-NEO-CODE-Di-IMatrix-MAX-GGUF/Qwen3.6-40B-Deck-Opus-NEO-CODE-HERE-2T-OT-HIGH-Q8_0.gguf"
      fmt=gguf; minbuild=0
      size="43 GB"; disk=40; vram=57000
      args="--no-mtp --ctx 131072"
      desc="Q8_0 40B Deckard layer-stack — no MTP head, KV ~1.5x/token, 131k default ctx" ;;
    nvfp4-unsloth)
      repo=unsloth/Qwen3.6-27B-NVFP4
      file='*'
      path="$MODELS_DIR/Qwen3.6-27B-NVFP4"
      fmt=safetensors; minbuild=0
      size="23 GB"; disk=22; vram=0
      args=""
      desc="official unsloth NVFP4 (safetensors) — vLLM/SGLang only, NOT llama.cpp" ;;
    # ── Qwen3.8-27B ──────────────────────────────────────────────────────────
    # Different architecture from the 3.6 entries above: GGUF arch is `qwen35`
    # (hybrid 16 x [3 x Gated DeltaNet -> 1 x Gated Attention]), and it is a
    # native vision-language model, hence the mmproj. Only 16 of the 64 layers
    # hold a KV cache, so 262k context costs ~16 GiB of KV rather than the ~21
    # GiB a same-size dense model needs; the DeltaNet layers keep a constant
    # ~150 MiB of recurrent state regardless of context length.
    # Qwen recommends temp 1.0 (not 0.6) for this model in thinking mode.
    38-q8)
      repo=unsloth/Qwen3.8-27B-GGUF
      file=Qwen3.8-27B-Q8_0.gguf
      mmproj=mmproj-F16.gguf
      path="$MODELS_DIR/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf"
      fmt=gguf; minbuild=0
      size="29 GB"; disk=27; vram=51000; temp=1.0   # measured 51230 MiB at 262k/f16 + mmproj
      args=""
      desc="Qwen3.8 Q8_0 + vision, MTP baked in — near-lossless 8-bit" ;;
    38-bf16)
      repo=unsloth/Qwen3.8-27B-GGUF
      file='BF16/*'
      mmproj=mmproj-F16.gguf
      path="$MODELS_DIR/Qwen3.8-27B-GGUF/BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf"
      fmt=gguf; minbuild=0
      size="55 GB"; disk=51; vram=76000; temp=1.0   # extrapolated from 38-q8 by weight delta
      args=""
      desc="Qwen3.8 BF16 + vision, MTP baked in — full precision reference" ;;
    *) return 1 ;;
  esac
  case "$field" in
    repo)     echo "$repo" ;;
    file)     echo "$file" ;;
    mmproj)   echo "$mmproj" ;;
    path)     echo "$path" ;;
    fmt)      echo "$fmt" ;;
    minbuild) echo "$minbuild" ;;
    size)     echo "$size" ;;
    disk)     echo "$disk" ;;
    vram)     echo "$vram" ;;
    temp)     echo "$temp" ;;
    args)     echo "$args" ;;
    desc)     echo "$desc" ;;
    *) return 1 ;;
  esac
}

# model_args <key> — effective default flags for run.sh.
# settings/<key>.conf wins over the table's args field, even when the file is
# empty (an empty file means "no defaults"; run.sh --reset-defaults deletes it).
model_args() {
  local f="$SETTINGS_DIR/$1.conf"
  if [[ -f "$f" ]]; then head -n1 "$f"; else model_info "$1" args; fi
}

# model_mmproj_path <key> — local path of the vision projector, empty for
# text-only models. Mirrors the GGUF path invariant: the projector lives at the
# repo root, so it lands beside (not inside) a split-GGUF subdirectory, and both
# keys sharing a repo share one download.
model_mmproj_path() {
  local mmproj repo
  mmproj="$(model_info "$1" mmproj)" || return 1
  [[ -z "$mmproj" ]] && return 0
  repo="$(model_info "$1" repo)"
  echo "$MODELS_DIR/${repo##*/}/$mmproj"
}

# model_downloaded <key> — 0 if the model's weights are present on disk.
model_downloaded() {
  local path fmt
  path="$(model_info "$1" path)" || return 1
  fmt="$(model_info "$1" fmt)"
  if [[ "$fmt" == safetensors ]]; then
    compgen -G "$path/*.safetensors" >/dev/null
  else
    [[ -f "$path" ]]
  fi
}

# model_list — status table: download state, space needed, VRAM, defaults, desc.
# Column markers are ASCII: printf field widths count bytes, and multibyte
# glyphs (✓, em-dash) would skew every column after them.
# The two space columns answer two different questions: `download` is how many
# bytes come over the wire, `disk` is how much free space you need to hold them.
model_list() {
  local fmt_row='%-14s %-3s %-9s %-6s %-6s %-24s %s\n'
  printf "$fmt_row" "key" "dl" "download" "disk" "~VRAM" "defaults" "description"
  printf "$fmt_row" "---" "--" "--------" "----" "-----" "--------" "-----------"
  local key fmt mark args marker mmproj vram
  for key in "${MODEL_KEYS[@]}"; do
    fmt="$(model_info "$key" fmt)"
    mark="-"; model_downloaded "$key" && mark="dl"
    # vram=0 means "not measured" — the vLLM-only entries never run here.
    vram="$(model_info "$key" vram)"
    if (( vram > 0 )); then vram="$(( vram / 1000 ))GB"; else vram="-"; fi
    args="$(model_info "$key" args)"; marker=""
    if [[ -f "$SETTINGS_DIR/$key.conf" ]]; then
      args="$(head -n1 "$SETTINGS_DIR/$key.conf")"; marker=" [saved]"
    fi
    if [[ "$fmt" != gguf ]]; then
      marker="$marker [vLLM only]"
    fi
    mmproj="$(model_mmproj_path "$key")"
    if [[ -n "$mmproj" ]]; then
      if [[ -f "$mmproj" ]]; then marker="$marker [vision]"
      else                        marker="$marker [vision: not dl]"; fi
    fi
    printf "$fmt_row" "$key" "$mark" \
      "$(model_info "$key" size)" "$(model_info "$key" disk)GiB" \
      "$vram" "${args:--}${marker}" "$(model_info "$key" desc)"
  done
  echo
  echo "dl = already downloaded. disk = free space needed; ~VRAM is at each model's own defaults."
  echo "[saved] = settings/<key>.conf overrides table args."
  echo "[vision] = mmproj on disk, run.sh serves images/video; disable per-launch with --no-vision."
}

# model_disk_needed <key>... — GiB required for the models not yet downloaded.
# Already-present weights cost nothing further, so this is what fetch-models.sh
# must actually find free, not the sum of every requested model's `disk`.
model_disk_needed() {
  local key total=0 mmproj
  for key in "$@"; do
    model_downloaded "$key" || total=$(( total + $(model_info "$key" disk) ))
    mmproj="$(model_mmproj_path "$key")"
    # Projector is ~1 GiB and downloads separately from the weights.
    if [[ -n "$mmproj" && ! -f "$mmproj" ]]; then total=$(( total + 1 )); fi
  done
  echo "$total"
}
