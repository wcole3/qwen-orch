# Run Qwen locally with llama.cpp

Two scripts to download **Qwen3.6-27B** and **Qwen3.8-27B** (plus a few
community finetunes up to 40B) and serve them on your own GPU, behind an
OpenAI-compatible API, tuned for **precise coding**.

The point is that one command gets you a working server. Every model in the
table launches with settings that already match its own recommendations —
sampling, context length, speculative decoding, vision — instead of leaving you
to reconstruct them from a dozen model cards.

- **MTP speculative decoding** is baked into most of the GGUFs (~1.4–2.2×
  faster generation, identical output) — no separate draft model to manage.
- **Vision** works out of the box on Qwen3.8: the projector is downloaded and
  wired up for you.
- **Per-model defaults** live in one table, so `./run.sh <key>` is the whole
  command, and `--save-defaults` makes your own tweaks stick.

Qwen3.6 keys are unprefixed (`q8`, `bf16`, …); Qwen3.8 keys are prefixed `38-`.

## Quick start

```bash
./fetch-models.sh          # menu: every model, its size, and your free space
./fetch-models.sh q8       # Qwen3.6-27B Q8_0 — 29 GB down, 28 GiB on disk
./run.sh q8                # serve on http://0.0.0.0:8081 (OpenAI-compatible /v1)
```

```bash
./fetch-models.sh 38-q8    # Qwen3.8-27B + vision — 29 GB + 0.9 GB projector
./run.sh 38-q8             # serves text, images and video
./run.sh --list            # every model: disk, VRAM, defaults
```

Point any OpenAI-compatible client at `http://localhost:8081/v1`:

```bash
curl -s http://localhost:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Write a bash one-liner to count files by extension."}]}'
```

## Requirements

- **An NVIDIA GPU**, ideally with enough VRAM to hold the model (see
  [How much VRAM?](#how-much-vram)). The scripts always request full GPU
  offload (`-ngl 99`).
- **A CUDA build of llama.cpp** — see [Building llama.cpp](#building-llamacpp).
  Set `LLAMA_SERVER` if yours is not in a location [env.sh](env.sh) probes.
- **Python with `huggingface_hub`** — `fetch-models.sh` installs the `hf` CLI
  with `pip --user` if it is missing.
- **Disk space**, from 17 GiB for one small model to ~360 GiB for all of them.
  See [Disk space](#disk-space).

Developed on Linux against an RTX PRO 6000 Blackwell (96 GB), which fits every
entry in the table at full 262k context and gets the native FP4 path for the
NVFP4 quants. Nothing here is specific to that card — smaller GPUs work with a
smaller quant, a shorter context, or both.

## Models

Download is what comes over the wire; disk is the free space you need to hold
it. They differ because HuggingFace counts in decimal GB and `df` counts in
GiB — 29 GB of bytes occupies 27 GiB — so the ~7% gap is a unit change, not
overhead.

| key | download | disk | ~VRAM* | what it is |
|---|---|---|---|---|
| `nvfp4` | 18 GB | 17 GiB | 40 GB | [CodeFault NVFP4-A](https://huggingface.co/CodeFault/Nvidia-Qwen3.6-27B-NVFP4-GGUF): NVFP4 FFN+attention, MTP head BF16. Native Blackwell FP4 kernels — fastest. |
| `nvfp4-attn` | 28 GB | 27 GiB | 50 GB | Same repo, attention upcast to BF16 — quality/speed middle ground. |
| `q8` | 29 GB | 28 GiB | 50 GB | [unsloth MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF) `Q8_0` — near-lossless 8-bit. |
| `bf16` | 55 GB | 51 GiB | 76 GB | Same repo, BF16 split GGUF — full-precision reference. |
| `hauhau` | 44 GB | 41 GiB | 51 GB | [HauhauCS Aggressive](https://huggingface.co/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive) `Q8_K_P` — uncensored 35B-A3B MoE (256 experts). No MTP head → MTP off by default. |
| `huihui` | 29 GB | 28 GiB | 50 GB | [huihui-ai abliterated MTP-GGUF](https://huggingface.co/huihui-ai/Huihui-Qwen3.6-27B-abliterated-MTP-GGUF) `Q8_0` — 27B abliterated, official GGUF of [the safetensors original](https://huggingface.co/huihui-ai/Huihui-Qwen3.6-27B-abliterated), MTP intact. |
| `aeon` | 29 GB | 28 GiB | 50 GB | [theLittleStone MTP-i1 GGUF](https://huggingface.co/theLittleStone/Qwen3.6-27B-AEON-Ultimate-Uncensored-MTP-i1-GGUF) `Q8_0` — imatrix conversion of [AEON Ultimate Uncensored BF16](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16) with the MTP head kept. |
| `deckard` | 43 GB | 40 GiB | 57 GB† | [DavidAU NEO-CODE IMatrix-MAX GGUF](https://huggingface.co/DavidAU/Qwen3.6-40B-Claude-4.6-Opus-Deckard-Heretic-Uncensored-Thinking-NEO-CODE-Di-IMatrix-MAX-GGUF) `Q8_0` — his own GGUF of the [40B Deckard/Heretic finetune](https://huggingface.co/DavidAU/Qwen3.6-40B-Claude-4.6-Opus-Deckard-Heretic-Uncensored-Thinking), a layer-stacked 27B (96 layers vs 65). No MTP head; KV ≈1.5× per token → defaults to 131k ctx. |
| `nvfp4-unsloth` | 23 GB | 22 GiB | — | [official unsloth NVFP4](https://huggingface.co/unsloth/Qwen3.6-27B-NVFP4), **safetensors — vLLM/SGLang only**. `run.sh` refuses it; serve with `vllm serve ~/models/Qwen3.6-27B-NVFP4`. No unsloth NVFP4 *GGUF* exists, and safetensors→GGUF NVFP4 conversion is [known-flaky](https://github.com/ggml-org/llama.cpp/discussions/23627). |
| `38-q8` | 29 GB + 0.9 | 27 GiB + 1 | 51 GB‡ | **Qwen3.8-27B**, [unsloth GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) `Q8_0` + `mmproj-F16` — vision-language, MTP baked in. |
| `38-bf16` | 55 GB + 0.9 | 51 GiB + 1 | 76 GB | Same repo, BF16 split GGUF + the same projector — full-precision reference. |

\* at each model's *own* defaults (262k context, f16 KV unless its `args` say
otherwise). Smaller `--ctx` or `--kv q8_0` shrinks it a lot — see
[How much VRAM?](#how-much-vram).
† at its default 131k context; ~69 GB at 262k (`--ctx 262144` still fits).
‡ measured (51,230 MiB, projector loaded, 262k f16); `38-bf16` is extrapolated
from it by the weight delta.

The `+ 0.9` / `+ 1` on the Qwen3.8 rows is the shared vision projector. Both
keys live in one directory, so it is downloaded once and counted once.

### Qwen3.8 vs Qwen3.6

Qwen3.8-27B is a different architecture, not a point release — GGUF arch
`qwen35`, hidden layout `16 × (3 × Gated DeltaNet → 1 × Gated Attention)`,
64 layers, 24 Q heads / 4 KV heads at head-dim 256, and a **native vision
encoder** (images and video, 27-layer ViT, patch 16). Three things follow:

- **Vision is wired in.** The table entries carry an `mmproj` field;
  `fetch-models.sh` downloads it and `run.sh` passes `--mmproj` automatically.
  `--no-vision` serves text-only and frees the projector's memory.
- **Different sampling.** Qwen recommends **temp 1.0** in thinking mode (3.6
  wants 0.6), so both entries set `temp=1.0` in the table's `temp` field.
  Every other sampler value is unchanged.
- **`reasoning_effort`.** Its template accepts `low`, `medium`, `xhigh`
  (default) — exposed as `--effort`. There is no `none`; use `--no-think`.

Both keys share one repo directory, so the 0.9 GB projector is downloaded
once and reused.

The four finetune quants are provisional Q8-class picks; other sizes exist in
the same repos — edit the `file`/`path`/`size`/`disk`/`vram` fields in
[env.sh](env.sh) to switch.

## Disk space

Weights go to `$MODELS_DIR`, which defaults to `~/models`. Put them on another
filesystem by exporting it:

```bash
export MODELS_DIR=/mnt/big/models
```

Per-model requirements are the `disk` column above and in `./run.sh --list`.
**Holding every entry at once takes ~360 GiB.** Nothing forces you to: most
people want one or two.

`fetch-models.sh` adds up what is actually missing and refuses to start a
download that would not fit, so you find out before spending an hour on it:

```
$ ./fetch-models.sh 38-bf16
Space: need ~52 GiB, have 31 GiB free at /mnt/small/models

Not enough free space: 52 GiB required, 31 GiB available.
Free up space, set MODELS_DIR to a bigger filesystem, or fetch fewer
models — ./fetch-models.sh lists the per-model requirement.
```

Files already present cost nothing further, so re-running after an interrupted
download only asks for the remainder. Downloads resume rather than restart.

Two pairs of keys share a directory — `q8`/`bf16` and `38-q8`/`38-bf16` — so
fetching the second of a pair does not re-download anything shared.

## How much VRAM?

Roughly: **weights + KV cache + ~2–3 GB** of compute buffers, plus ~1 GB for
the vision projector on Qwen3.8.

The KV cache is the part you control, and on these models it is unusually
small. Qwen3.6-27B and Qwen3.8-27B are hybrids — 48 of their 64 layers are
Gated DeltaNet (linear attention, constant-size state, **no per-token KV**) and
only 16 are quadratic attention with 4 KV heads. So the cache is ~17 GB (3.6)
or ~16 GiB (3.8) at the full 262,144-token context, and it scales linearly:

| context | KV at f16 | KV at `--kv q8_0` |
|---|---|---|
| 262,144 | ~17 GB | ~9 GB |
| 131,072 | ~8.5 GB | ~4.5 GB |
| 65,536 | ~4.3 GB | ~2.2 GB |
| 32,768 | ~2.1 GB | ~1.1 GB |

So `q8` (28 GiB of weights) needs ~50 GB at its default 262k context, but only
~33 GB at 32k — and `nvfp4` (17 GiB) lands near ~22 GB at 32k. Shortening the
context is the cheapest lever; `--kv q8_0` is the next one and is
near-lossless.

```bash
./run.sh q8 --ctx 32768              # much smaller KV
./run.sh q8 --ctx 65536 --kv q8_0    # smaller still
```

If a model does not fit even then, use a smaller quant. The repos linked above
carry the full ladder (`Q4_K_M`, `IQ4_XS`, `Q5_K_M`, …) at roughly 4–20 GiB;
add one as a new entry in the [env.sh](env.sh) table — copy an existing block
and change `file`, `path`, `size`, `disk` and `vram`. llama.cpp will also spill
to system RAM rather than fail, at a large speed cost.

## Listing models

`./run.sh --list` (or `./fetch-models.sh` with no arguments) prints every key
with its download state (`dl` = weights on disk), download size, disk
requirement, expected VRAM, and the defaults it will launch with. `[saved]`
marks a `settings/<key>.conf` override, `[vision]` a model with its projector
downloaded.

## run.sh

```
./run.sh <model> [flags]      # the model key must be the first argument

--ctx N                 context length        (default 262144 = native max)
--kv f16|bf16|q8_0      KV cache type         (default f16)
--mtp N                 MTP draft depth 1-6   (default 2); re-enables MTP
                        after a --no-mtp default
--no-mtp                disable speculative decoding
--think                 enable thinking mode  (default)
--no-think              disable thinking mode entirely
--preserve-thinking     keep prior-turn thinking (default)
--no-preserve-thinking  drop prior-turn thinking from the chat template
--temp T                temperature override  (default 0.6 thinking / 0.7 not)
--effort L              reasoning_effort: low|medium|xhigh   (Qwen3.8 only)
--no-vision             skip --mmproj on a vision model, serve text-only
--port P                listen port           (default 8081)
--save-defaults         persist this launch's flags as the model's defaults
--reset-defaults        delete the saved defaults and exit
```

### Per-model defaults

Every model launches with one command — `./run.sh deckard` — because defaults
are layered, last one wins:

1. built-ins (262k ctx, f16 KV, MTP on at depth 2, thinking on)
2. the model's `args` field in the [env.sh](env.sh) table (e.g. `deckard`
   ships `--no-mtp --ctx 131072`)
3. `settings/<key>.conf` — written by `--save-defaults`, **replaces** the
   table args entirely when present
4. CLI flags

`--save-defaults` snapshots the *effective* settings (not just the flags you
typed), so a table default like `--no-mtp` survives a `--ctx`-only save:

```bash
./run.sh deckard --ctx 65536 --save-defaults   # settings/deckard.conf: --no-mtp --ctx 65536
./run.sh deckard                               # relaunches with exactly that
./run.sh deckard --reset-defaults              # back to the table args
```

`EXTRA_ARGS="..."` is appended raw to the llama-server argv, after everything
else (llama-server is last-flag-wins, so it overrides the composed flags).
Use it for YaRN or sampler experiments. Values containing spaces are not
supported (the string is word-split).

Baked-in sampling defaults follow the
[Unsloth Qwen3.6 recommendations](https://unsloth.ai/docs/models/qwen3.6) for
**coding in thinking mode**: `temp 0.6, top-p 0.95, top-k 20, min-p 0.0,
presence-penalty 0.0, repeat-penalty 1.0`. With `--no-think` the script
switches to the non-thinking recommendation (`temp 0.7, top-p 0.8,
presence-penalty 1.5`). Clients can still override any of these per-request.

[Qwen3.8's recommendations](https://unsloth.ai/docs/models/qwen3.8) match on
every value except temperature — **1.0** in thinking mode — so the thinking
temperature lives in the table's `temp` field per model rather than in `args`.
`--no-think` uses 0.7 for every model (all generations agree there), and
`--temp` overrides whichever per-mode default applies.

**Preserve thinking** (on by default) keeps prior-turn reasoning in the
context — Qwen3.6 was trained for this and it improves multi-turn coding. Turn
it off with `--no-preserve-thinking`.

Under the hood this is llama-server's `--reasoning on|off` and
`--reasoning-preserve` / `--no-reasoning-preserve`, not raw template kwargs:
setting `enable_thinking` through `--chat-template-kwargs` is deprecated, and
`--reasoning-preserve` sets `preserve_thinking`, `clear_thinking` **and**
`truncate_history_thinking` in one go, so it also covers templates that spell
the idea differently. It is likewise the key llama-server checks before
warning that a template's preserve support is going unused.

**Reasoning effort** (Qwen3.8) tunes how long the model thinks:
`--effort low|medium|xhigh`. It has no dedicated llama-server flag, so it is
the one thing still passed as `--chat-template-kwargs` — they share the same
map server-side, so the two forms mix cleanly. Unset means the template's own
default (`xhigh`). Passing it to a Qwen3.6 model is harmless — that template
never reads the key.

### Vision

`38-q8` and `38-bf16` declare an `mmproj` in the table, so `run.sh` adds
`--mmproj ~/models/Qwen3.8-27B-GGUF/mmproj-F16.gguf` and the OpenAI-compatible
endpoint accepts `image_url` content parts. `./run.sh --list` marks the state:
`[vision]` once the projector is on disk, `[vision: not dl]` before that.

A missing projector is a warning, not a failure — the server still starts,
text-only. `--no-vision` does the same deliberately.

llama.cpp warns at load that Qwen-VL models want **≥1024 image tokens** for
grounding tasks (bounding boxes, "point at the X"). Plain description and OCR
work fine at the default; if grounding output looks wrong, raise it:

```bash
EXTRA_ARGS="--image-min-tokens 1024" ./run.sh 38-q8
```

**MTP** uses the model's built-in draft head (`--spec-type draft-mtp`),
giving ~1.4–2.2x generation speedup at identical output quality. The optimal
`--mtp` depth is hardware-dependent — try 1–6 and watch tokens/sec.

⚠ **MTP on a model without the head is a hard failure**: llama-server aborts
at load with *"context type MTP requested but model doesn't contain MTP
layers"* — there is no graceful fallback. That is why `hauhau` and `deckard`
ship `--no-mtp` in their table args. To check a GGUF for the head:

```bash
# from your llama.cpp checkout
python3 gguf-py/gguf/scripts/gguf_dump.py --no-tensors <model.gguf> | grep -i nextn
```

`nextn_predict_layers ≥ 1` means the head exists — drop `--no-mtp` from the
model's `args` in env.sh (or just `./run.sh <key> --mtp 2 --save-defaults`).

## KV cache & context

[How much VRAM?](#how-much-vram) covers the sizes; this is the behaviour.

The 40B `deckard` stack has ~1.5× the layers of a 27B, hence its 131k default;
the `hauhau` MoE differs again — check `nvidia-smi` on first launch. Qwen3.8
also holds ~150 MiB of constant DeltaNet recurrent state that does not grow
with context.

- **Default is f16 at the full 262k context.** If it fits, there is no reason
  to trade precision for memory; shorten `--ctx` first if it does not.
- `--kv q8_0` is near-lossless and roughly halves the cache — the right lever
  when you want long context and cannot afford f16.
- `q4_0` is refused by `run.sh`: 4-bit KV measurably degrades code and
  structured output at long context (bad diffs, malformed JSON), even when
  perplexity looks fine.
- Context beyond 262k (up to ~1M) is possible via YaRN
  (`--rope-scaling yarn`), at some quality cost — not wired into `run.sh`;
  add the flags via `EXTRA_ARGS` if you want to experiment.

## Building llama.cpp

You need a CUDA build of `llama-server`. Some quants also need a minimum build
number (the `minbuild` field in the model table — e.g. NVFP4 GGUFs need
**build ≥ b9775**); preflight enforces it per model and tells you to rebuild.

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --config Release -j "$(nproc)" --target llama-server llama-cli
build/bin/llama-server --version
```

`CMAKE_CUDA_ARCHITECTURES=native` targets the card you build on. Pin it if you
need to — `120` for Blackwell (sm_120), `89` for Ada, `86` for Ampere.

Then tell the scripts where it is, unless it already sits somewhere
[env.sh](env.sh) probes (`$LLAMA_MAINLINE/llama.cpp`, `~/llama.cpp`,
`~/src/llama.cpp`, or `$PATH`):

```bash
export LLAMA_SERVER=/path/to/llama.cpp/build/bin/llama-server
```

CUDA note: **do not build with CUDA 13.2** — it has a known gibberish-output
bug. 13.3 is fine and preferred for Blackwell. If you have several toolkits
installed, pin the one you want:

```bash
cmake -B build -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120
```

## Files

- [env.sh](env.sh) — shared config + model table (override any var by exporting it)
- [fetch-models.sh](fetch-models.sh) — HuggingFace downloader (`./fetch-models.sh` for the menu)
- [run.sh](run.sh) — server launcher
- [preflight.sh](preflight.sh) — binary/build/model/VRAM/port checks (auto-run by run.sh; `SKIP_PREFLIGHT=1` to skip)
- `settings/` — per-model saved launch defaults (one line of run.sh flags per `<key>.conf`; written by `--save-defaults`, gitignored)

## Configuration

Every value in [env.sh](env.sh) is an override — export it before running, or
edit the default:

| variable | default | what it does |
|---|---|---|
| `MODELS_DIR` | `~/models` | where weights are downloaded and loaded from |
| `LLAMA_SERVER` | probed | path to the `llama-server` binary |
| `LLAMA_MAINLINE` | `~/devspace` | directory *containing* your `llama.cpp` checkout |
| `PORT` | `8081` | listen port (or `--port` per launch) |
| `HOST` | `0.0.0.0` | listen address; `127.0.0.1` to keep it local |
| `EXTRA_ARGS` | — | raw flags appended to the llama-server argv |
| `SKIP_PREFLIGHT` | `0` | `1` skips the pre-launch checks |

`HOST` defaults to `0.0.0.0`, so the server is reachable from your network and
llama.cpp sets permissive CORS with no API key. On an untrusted network, set
`HOST=127.0.0.1`.

## License

These scripts have no license file yet — add one before you rely on them.

The models are separate: every repo linked in the table above is currently
listed as **Apache-2.0** on HuggingFace, base models and community finetunes
alike. Licenses can change, so check the model card before you depend on one.
