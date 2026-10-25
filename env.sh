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
# Weights live in the standard HuggingFace cache, the same one transformers,
# vLLM and the SGLang container use, so a model fetched once serves them all.
# HF_CACHE is the HF_HOME equivalent (run-sglang.sh binds it into the
# container); HF_HUB_CACHE is its hub/ subdirectory, where the repos sit as
# models--<org>--<name>/snapshots/<commit>/. Both follow huggingface_hub's own
# precedence, so an HF_HOME or HF_HUB_CACHE already in your environment wins.
# ~360 GiB to hold every entry in the table at once; each entry's own
# requirement is its `disk` field, and ./fetch-models.sh checks the free space
# before it downloads anything.
HF_CACHE="${HF_CACHE:-${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}}"
HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_CACHE/hub}"

# ── Per-model saved settings (written by `run.sh <key> ... --save-defaults`) ─
SETTINGS_DIR="${SETTINGS_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/settings}"

# ── Server ───────────────────────────────────────────────────────────────────
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8081}"

# HF_HUB_CACHE is exported so `hf download` writes where model_path reads.
export LLAMA_MAINLINE LLAMA_SERVER HF_CACHE HF_HUB_CACHE SETTINGS_DIR HOST PORT

# ── SGLang (Docker) ─────────────────────────────────────────────────
# Used by run-sglang.sh only — the llama.cpp path above is unaffected. The
# image ships its own SGLang build, so nothing is installed on the host.
# SGLANG_PORT is the port *inside* the container; run-sglang.sh --port changes
# only what is published on the host, so the two can differ.
SGLANG_IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:qwen38-27b}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
SGLANG_CONTAINER="${SGLANG_CONTAINER:-qwen38-sglang}"

# The SGLang container resolves its weights from a HuggingFace repo id through
# $HF_CACHE (above) — the cache ./fetch-models.sh fills — so it pulls whatever is
# missing on first launch, and ./fetch-models.sh 38-nvfp4 is an optional pre-pull.
export SGLANG_IMAGE SGLANG_PORT SGLANG_CONTAINER

# ── Model table ──────────────────────────────────────────────────────────────
# model_info <key> <field>
#   fields: repo | file | mmproj | path | fmt | minbuild | size | disk | vram |
#           temp | sampling | imgtok | draftrepo | draft | args | desc
#   repo     — HuggingFace repo id
#   file     — in-repo file (or include-glob for multi-file entries)
#   mmproj   — in-repo multimodal projector file, or "" for text-only models.
#              Downloaded next to the weights; run.sh passes it as --mmproj.
#   draftrepo/draft — separate MTP drafter (repo id, in-repo GGUF) for models
#              whose MTP head ships as its own model rather than baked into the
#              weights, or "" for none. run.sh loads it with -md whenever MTP is
#              on; fetch-models.sh downloads it alongside the weights.
#   imgtok   — minimum tokens an image is expanded to (llama-server
#              --image-min-tokens), or "" to leave llama.cpp's model-derived
#              value alone. Vision entries only. llama.cpp warns at load that
#              Qwen-VL needs >= 1024 for grounding, so the Qwen3.8 entries pin
#              2048; run.sh --image-tokens overrides per launch.
#   temp     — thinking-mode sampling temperature (default 0.6, Unsloth's
#              Qwen3.6 coding recommendation; Qwen3.8 wants 1.0). Belongs here
#              rather than in args because it is mode-specific: --no-think
#              always uses 0.7, which every Qwen generation agrees on.
#   sampling — sampler family run.sh applies: qwen (default; top-k 20, and
#              Qwen's separate non-thinking sampler) or google (Gemma 4's
#              temp 1.0 / top-p 0.95 / top-k 64, the same in both modes).
#   path     — in-repo path run.sh loads (-m). Defaults to `file` for GGUF
#              entries, so only glob entries set it (the first split shard);
#              "" for safetensors entries, meaning the snapshot directory itself.
#              model_path resolves it to an absolute path in the HF cache.
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
# Keys sharing a repo share its cache entry, so a file both use (a vision
# projector) downloads once.
MODEL_KEYS=(nvfp4 nvfp4-attn q8 bf16 hauhau huihui aeon deckard nvfp4-unsloth
            38-q8 38-bf16 38-huihui 38-nvfp4 g4-heretic g4-huihui)

model_info() {
  local key="$1" field="$2"
  local repo file mmproj path fmt minbuild size disk vram temp sampling imgtok
  local draftrepo="" draft="" args desc
  path=""      # GGUF entries default to `file` below
  mmproj=""    # text-only unless an entry sets it
  imgtok=""    # no image-token floor unless a vision entry sets one
  temp=0.6     # Qwen3.6 thinking-mode default unless an entry overrides it
  sampling=qwen
  case "$key" in
    q8)
      repo=unsloth/Qwen3.6-27B-MTP-GGUF
      file=Qwen3.6-27B-Q8_0.gguf
      fmt=gguf; minbuild=0
      size="29 GB"; disk=28; vram=50000
      args=""
      desc="Q8_0, MTP baked in — near-lossless 8-bit" ;;
    bf16)
      repo=unsloth/Qwen3.6-27B-MTP-GGUF
      file='BF16/*'
      path=BF16/Qwen3.6-27B-BF16-00001-of-00002.gguf
      fmt=gguf; minbuild=0
      size="55 GB"; disk=51; vram=76000
      args=""
      desc="BF16, MTP baked in — full precision reference" ;;
    nvfp4)
      repo=CodeFault/Nvidia-Qwen3.6-27B-NVFP4-GGUF
      file=Nvidia-Qwen3.6-27B-NVFP4-A.gguf
      fmt=gguf; minbuild=9775
      size="18 GB"; disk=17; vram=40000
      args=""
      desc="NVFP4 FFN+attn, MTP head BF16 — native Blackwell FP4 path" ;;
    nvfp4-attn)
      repo=CodeFault/Nvidia-Qwen3.6-27B-NVFP4-GGUF
      file=Nvidia-Qwen3.6-27B-NVFP4-BF16-Attn.gguf
      fmt=gguf; minbuild=9775
      size="28 GB"; disk=27; vram=50000
      args=""
      desc="NVFP4 FFN, BF16 attention — quality/speed middle ground" ;;
    hauhau)
      repo=HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
      file=Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf
      fmt=gguf; minbuild=0
      size="44 GB"; disk=41; vram=51000
      args="--no-mtp"
      desc="Q8_K_P 35B-A3B MoE uncensored — no MTP head, MTP off" ;;
    huihui)
      repo=huihui-ai/Huihui-Qwen3.6-27B-abliterated-MTP-GGUF
      file=Huihui-Qwen3.6-27B-abliterated-ggml-model-Q8_0.gguf
      fmt=gguf; minbuild=0
      size="29 GB"; disk=28; vram=50000
      args=""
      desc="Q8_0 27B abliterated, MTP baked in" ;;
    aeon)
      repo=theLittleStone/Qwen3.6-27B-AEON-Ultimate-Uncensored-MTP-i1-GGUF
      file=Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16-MTP-i1.Q8_0.gguf
      fmt=gguf; minbuild=0
      size="29 GB"; disk=28; vram=50000
      args=""
      desc="Q8_0 27B AEON uncensored merge (imatrix), MTP baked in" ;;
    deckard)
      repo=DavidAU/Qwen3.6-40B-Claude-4.6-Opus-Deckard-Heretic-Uncensored-Thinking-NEO-CODE-Di-IMatrix-MAX-GGUF
      file=Qwen3.6-40B-Deck-Opus-NEO-CODE-HERE-2T-OT-HIGH-Q8_0.gguf
      fmt=gguf; minbuild=0
      size="43 GB"; disk=40; vram=57000
      args="--no-mtp --ctx 131072"
      desc="Q8_0 40B Deckard layer-stack — no MTP head, KV ~1.5x/token, 131k default ctx" ;;
    nvfp4-unsloth)
      repo=unsloth/Qwen3.6-27B-NVFP4
      file='*'
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
      fmt=gguf; minbuild=0
      size="29 GB"; disk=27; vram=51000; temp=1.0; imgtok=2048   # measured 51230 MiB at 262k/f16 + mmproj
      args=""
      desc="Qwen3.8 Q8_0 + vision, MTP baked in — near-lossless 8-bit" ;;
    38-bf16)
      repo=unsloth/Qwen3.8-27B-GGUF
      file='BF16/*'
      mmproj=mmproj-F16.gguf
      path=BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf
      fmt=gguf; minbuild=0
      size="55 GB"; disk=51; vram=76000; temp=1.0; imgtok=2048   # extrapolated from 38-q8 by weight delta
      args=""
      desc="Qwen3.8 BF16 + vision, MTP baked in — full precision reference" ;;
    # Same architecture and tensor set as 38-q8 (verified: arch qwen35, 866
    # tensors, nextn_predict_layers=1, blk.64.nextn.* present), so MTP stays on
    # and the 38-q8 VRAM measurement carries over unchanged. Unlike 38-q8 and
    # 38-bf16 this is its own repo directory, so its projector is a separate
    # 0.9 GB download rather than the shared one. A Q8_0_L also exists in the
    # repo (38.8 GB — the ablated tensors kept at BF16); switch `file` and
    # raise `size`/`disk`/`vram` to use it.
    38-huihui)
      repo=huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF
      file=Huihui-Qwen3.8-27B-abliterated-Q8_0.gguf
      mmproj=mmproj-model-bf16.gguf
      fmt=gguf; minbuild=0
      size="29 GB"; disk=27; vram=51000; temp=1.0; imgtok=2048
      args=""
      desc="Qwen3.8 Q8_0 + vision abliterated, MTP baked in" ;;
    # Served by ./run-sglang.sh, not ./run.sh: NVFP4 W4A4 is a safetensors
    # checkpoint that llama.cpp cannot load. The SGLang container reads it from
    # $HF_CACHE by repo id and pulls it on first launch if missing, so
    # `./fetch-models.sh 38-nvfp4` is optional — a pre-pull into the same cache.
    38-nvfp4)
      repo=RadixArk/Qwen3.8-27B-NVFP4
      file='*'
      fmt=safetensors; minbuild=0
      size="17 GB"; disk=16; vram=0; temp=1.0
      args=""
      desc="Qwen3.8 NVFP4 W4A4 + FP8 KV — SGLang docker, ./run-sglang.sh" ;;
    # ── Gemma 4 26B-A4B ──────────────────────────────────────────────────────
    # GGUF arch `gemma4`: MoE, 128 experts (8 active + 1 shared), ~4B active, so
    # it decodes far faster than the 27B dense entries. Hybrid attention —
    # sliding-window layers interleaved with global ones — keeps 256k KV small.
    # Native vision, hence the mmproj; its image budgets run 70-1120 tokens, so
    # imgtok stays empty rather than inheriting Qwen-VL's 2048 floor. No MTP
    # head in the weights: MTP comes from Google's separate drafter,
    # gemma-4-26B-A4B-it-assistant (arch gemma4-assistant), loaded with -md. It
    # reads the target's hidden state and shares its KV cache, so it costs
    # ~0.5 GiB and no extra KV.
    # It was trained against the stock model; the heretic/abliterated weights
    # differ slightly, so expect somewhat lower acceptance than stock.
    # Thinking is the template's enable_thinking, which --reasoning drives.
    # Both are Q4-class picks: other sizes sit in the same repos.
    g4-heretic)
      repo=llmfan46/gemma-4-26B-A4B-it-uncensored-heretic-GGUF
      file=gemma-4-26B-A4B-it-uncensored-heretic-Q4_K_M.gguf
      mmproj=gemma-4-26B-A4B-it-mmproj-BF16.gguf
      fmt=gguf; minbuild=0
      draftrepo=AtomicChat/gemma-4-26B-A4B-it-assistant-GGUF
      draft=gemma-4-26B-A4B-it-assistant.Q8_0.gguf
      size="17 GB"; disk=16; vram=25000; temp=1.0; sampling=google   # vram estimated, not measured
      args=""
      desc="Gemma 4 26B-A4B MoE Q4_K_M + vision, heretic uncensored, MTP drafter" ;;
    g4-huihui)
      repo=groxaxo/Huihui-gemma-4-26B-A4B-it-abliterated-GGUF
      file=gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
      mmproj=mmproj-F16.gguf
      fmt=gguf; minbuild=0
      draftrepo=AtomicChat/gemma-4-26B-A4B-it-assistant-GGUF
      draft=gemma-4-26B-A4B-it-assistant.Q8_0.gguf
      size="17 GB"; disk=16; vram=25000; temp=1.0; sampling=google   # vram estimated, not measured
      args=""
      desc="Gemma 4 26B-A4B MoE UD-Q4_K_M + vision, abliterated, MTP drafter" ;;
    *) return 1 ;;
  esac
  [[ -z "$path" && "$fmt" == gguf ]] && path="$file"
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
    sampling) echo "$sampling" ;;
    imgtok)   echo "$imgtok" ;;
    draftrepo) echo "$draftrepo" ;;
    draft)    echo "$draft" ;;
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

# hf_resolve <repo> <in-repo path> — absolute path of a file in the HF hub
# cache; an empty in-repo path means the snapshot directory, which counts as
# present once it holds any *.safetensors. Tries the snapshot refs/main names
# first, then every other snapshot newest first: fetching more files after
# upstream pushed a commit moves refs/main to a fresh snapshot that lacks the
# files fetched earlier, and those stay valid where they are.
# Not cached: prints where it would land and returns 1, so callers can name it.
hf_resolve() {
  local dir="$HF_HUB_CACHE/models--${1//\//--}" rel="$2" rev snap
  local -a snaps=()
  rev="$(cat "$dir/refs/main" 2>/dev/null || true)"
  [[ -n "$rev" ]] && snaps+=("$dir/snapshots/$rev")
  while IFS= read -r snap; do snaps+=("${snap%/}"); done \
    < <(ls -1dt "$dir"/snapshots/*/ 2>/dev/null)
  for snap in "${snaps[@]}"; do
    if [[ -n "$rel" ]]; then
      [[ -f "$snap/$rel" ]] && { echo "$snap/$rel"; return 0; }
    elif compgen -G "$snap/*.safetensors" >/dev/null; then
      echo "$snap"; return 0
    fi
  done
  echo "$dir/snapshots/${rev:-main}${rel:+/$rel}"
  return 1
}

# model_path <key> — absolute path run.sh loads, resolved in the HF cache.
# Returns 1 (still printing the expected path) when it is not downloaded.
model_path() {
  local repo path
  repo="$(model_info "$1" repo)" || return 1
  path="$(model_info "$1" path)"
  hf_resolve "$repo" "$path"
}

# model_mmproj_path <key> — absolute path of the vision projector, empty for
# text-only models. Keys sharing a repo share one download. Prints the expected
# path even when it is not downloaded yet; test it with -f.
model_mmproj_path() {
  local mmproj
  mmproj="$(model_info "$1" mmproj)" || return 1
  [[ -z "$mmproj" ]] && return 0
  hf_resolve "$(model_info "$1" repo)" "$mmproj" || true
}

# model_draft_path <key> — absolute path of the separate MTP drafter, empty for
# models without one. Like model_mmproj_path, prints the expected path even
# when it is not downloaded yet; test it with -f.
model_draft_path() {
  local draft
  draft="$(model_info "$1" draft)" || return 1
  [[ -z "$draft" ]] && return 0
  hf_resolve "$(model_info "$1" draftrepo)" "$draft" || true
}

# model_downloaded <key> — 0 if the model's weights are present in the cache.
model_downloaded() {
  model_path "$1" >/dev/null
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
  local key fmt mark args marker mmproj draft vram
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
      marker="$marker [vLLM/SGLang]"
    fi
    mmproj="$(model_mmproj_path "$key")"
    if [[ -n "$mmproj" ]]; then
      if [[ -f "$mmproj" ]]; then marker="$marker [vision]"
      else                        marker="$marker [vision: not dl]"; fi
    fi
    draft="$(model_draft_path "$key")"
    if [[ -n "$draft" && ! -f "$draft" ]]; then marker="$marker [drafter: not dl]"; fi
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
  local key total=0 mmproj draft
  for key in "$@"; do
    model_downloaded "$key" || total=$(( total + $(model_info "$key" disk) ))
    mmproj="$(model_mmproj_path "$key")"
    # Projector is ~1 GiB and downloads separately from the weights.
    if [[ -n "$mmproj" && ! -f "$mmproj" ]]; then total=$(( total + 1 )); fi
    # So is an MTP drafter (<1 GiB); rounded up like the projector.
    draft="$(model_draft_path "$key")"
    if [[ -n "$draft" && ! -f "$draft" ]]; then total=$(( total + 1 )); fi
  done
  echo "$total"
}
