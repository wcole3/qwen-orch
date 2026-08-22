# Run Qwen locally with llama.cpp or SGLang

A handful of scripts to download **Qwen3.6-27B** and **Qwen3.8-27B** (plus a
few community finetunes up to 40B) and serve them on your own GPU, behind an
OpenAI-compatible API, tuned for **precise coding**. GGUF goes through
llama.cpp; the NVFP4 safetensors checkpoint goes through SGLang in Docker.

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

```bash
docker pull lmsysorg/sglang:qwen38-27b
./run-sglang.sh            # Qwen3.8-27B NVFP4 under SGLang, http://0.0.0.0:30000/v1
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
| `38-huihui` | 29 GB + 0.9 | 27 GiB + 1 | 51 GB‡ | **Qwen3.8-27B abliterated**, [huihui-ai GGUF](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF) `Q8_0` + `mmproj-model-bf16` — official GGUF of [the safetensors original](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated); vision and MTP untouched by the ablation. Its own repo directory, so its projector is a second 0.9 GB download. A `Q8_0_L` (38.8 GB, ablated tensors kept at BF16) sits in the same repo. |
| `38-nvfp4` | 17 GB | —§ | — | [RadixArk NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4), **safetensors — SGLang only**. `run.sh` refuses it; serve with [`./run-sglang.sh`](#run-sglangsh-sglang-in-docker). W4A4 + FP8 projections, ~16.5 GB of weights, and the checkpoint declares FP8 KV. |

\* at each model's *own* defaults (262k context, f16 KV unless its `args` say
otherwise). Smaller `--ctx` or `--kv q8_0` shrinks it a lot — see
[How much VRAM?](#how-much-vram).
† at its default 131k context; ~69 GB at 262k (`--ctx 262144` still fits).
‡ measured on `38-q8` (51,230 MiB, projector loaded, 262k f16). `38-huihui`
carries the same number: identical architecture, identical 866-tensor set, and
a byte-for-byte identical 29.047 GB file size. `38-bf16` is extrapolated from
the measurement by the weight delta.

The `+ 0.9` / `+ 1` on the Qwen3.8 rows is the vision projector. `38-q8` and
`38-bf16` live in one directory, so theirs is downloaded once and counted once;
`38-huihui` is a different repo and carries its own copy.

§ `38-nvfp4` is pulled by the SGLang container into `~/.cache/huggingface`, not
`$MODELS_DIR`, so `./fetch-models.sh` is not part of its path.

### Qwen3.8 vs Qwen3.6

Qwen3.8-27B is a different architecture, not a point release — GGUF arch
`qwen35`, hidden layout `16 × (3 × Gated DeltaNet → 1 × Gated Attention)`,
64 layers, 24 Q heads / 4 KV heads at head-dim 256, and a **native vision
encoder** (images and video, 27-layer ViT, patch 16). Three things follow:

- **Vision is wired in.** The table entries carry an `mmproj` field;
  `fetch-models.sh` downloads it and `run.sh` passes `--mmproj` automatically.
  `--no-vision` serves text-only and frees the projector's memory.
- **Different sampling.** Qwen recommends **temp 1.0** in thinking mode (3.6
  wants 0.6), so every Qwen3.8 entry sets `temp=1.0` in the table's `temp` field.
  Every other sampler value is unchanged.
- **`reasoning_effort`.** Its template accepts `low`, `medium`, `xhigh`
  (default) — exposed as `--effort`. There is no `none`; use `--no-think`.

`38-q8` and `38-bf16` share one repo directory, so their 0.9 GB projector is
downloaded once and reused. `38-huihui` comes from huihui-ai's own repo, so it
downloads a second projector of its own.

The five finetune quants are provisional Q8-class picks; other sizes exist in
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
--image-tokens N[:M]    tokens per image: floor N, optional ceiling M
        | auto          (default 2048 on the Qwen3.8 keys; auto = let
                        llama.cpp size images from the model)
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

`38-q8`, `38-bf16` and `38-huihui` declare an `mmproj` in the table, so
`run.sh` adds `--mmproj` automatically — `~/models/Qwen3.8-27B-GGUF/mmproj-F16.gguf`
for the unsloth pair, `~/models/Huihui-Qwen3.8-27B-abliterated-GGUF/mmproj-model-bf16.gguf`
for `38-huihui` — and the OpenAI-compatible endpoint accepts `image_url`
content parts. `./run.sh --list` marks the state:
`[vision]` once the projector is on disk, `[vision: not dl]` before that.

A missing projector is a warning, not a failure — the server still starts,
text-only. `--no-vision` does the same deliberately.

Images reach the server three ways, all verified against `38-q8`: a
`data:image/png;base64,...` URI, a remote `https://` URL (llama-server
downloads it itself), and `file://` relative to `--media-path` if you pass that
through `EXTRA_ARGS`.

#### Image tokens

An image is expanded into tokens before the model sees it, and left alone
llama.cpp sizes that from the image's own resolution — roughly
**(width / 32) × (height / 32)**, which is patch size 16 times the projector's
spatial merge of 2. Measured on `38-q8`:

| image | native image tokens |
|---|---|
| 112×112 | 18 |
| 224×224 | 51 |
| 448×448 | 198 |
| 896×896 | 786 |
| 1344×756 | 1010 |

That is a problem at the small end, because llama.cpp warns at load:

```
Qwen-VL models require at minimum 1024 image tokens to function correctly
on grounding tasks
if you encounter problems with accuracy, try adding --image-min-tokens 1024
```

A 448×448 screenshot lands at 198 — five times under the floor Qwen-VL wants
for grounding work (bounding boxes, "point at the X"). So the Qwen3.8 entries
carry `imgtok=2048` in the table, and `run.sh` passes `--image-min-tokens 2048`
whenever a projector is loaded. Every image is padded up to at least 2048
tokens; anything already larger is untouched.

**It is not free.** The floor is a prefill cost — that same 448×448 image
measures **2,118** tokens with the floor on, against 198 without it, so it costs
about eleven times the prefill work and eleven times the context. (It overshoots
2048 because the image is rescaled up to a whole patch grid, not to the number
exactly.) Eight images at the floor is ~17k of your context window. Plain
description and OCR were fine at the native sizing; the floor buys grounding
accuracy.

Override per launch, or turn it off entirely:

```bash
./run.sh 38-q8 --image-tokens 1024        # floor only
./run.sh 38-q8 --image-tokens 1024:4096   # floor and ceiling — caps 4K screenshots
./run.sh 38-q8 --image-tokens auto        # neither flag; llama.cpp sizes from the model
./run.sh 38-q8 --image-tokens 4096 --save-defaults
```

A ceiling matters for large inputs: an unclamped 4K screenshot (3840×2160)
costs ~8,100 tokens on its own. The floor and ceiling map to llama-server's
`--image-min-tokens` and `--image-max-tokens`; `--mtmd-batch-max-tokens` is not
wrapped, so reach it through `EXTRA_ARGS`.

Because `imgtok` is its own table field rather than part of `args`, a
`settings/<key>.conf` written by `--save-defaults` cannot silently drop it —
saving an unrelated `--ctx` change leaves the model's floor intact.

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

## run-sglang.sh (SGLang in Docker)

`run.sh` serves GGUF through llama.cpp. `run-sglang.sh` is the other backend:
the **NVFP4 W4A4** safetensors checkpoint through lmsys' prebuilt SGLang image,
which is the only way to get the native FP4 path plus the in-checkpoint MTP
head, the `qwen3_coder` tool-call parser and an Anthropic-compatible
`/v1/messages` endpoint in one server.

Nothing is installed on the host — the image ships its own SGLang build:

```bash
docker pull lmsysorg/sglang:qwen38-27b
./run-sglang.sh                    # the verified defaults
./run-sglang.sh --print            # show the docker argv, launch nothing
./run-sglang.sh --effort low       # shallower thinking, server-wide
./run-sglang.sh --concurrency 8    # pin --max-running-requests
```

The defaults reproduce the lmsys cookbook's **verified** RTX PRO 6000 /
NVFP4 / EAGLE / low-latency / float32-state cell verbatim:

| default | value | why |
|---|---|---|
| checkpoint | `RadixArk/Qwen3.8-27B-NVFP4` | W4A4 + FP8 projections, ~16.5 GB of weights |
| KV cache | `fp8_e4m3` | the checkpoint declares `kv_cache_quant_algo: FP8` |
| attention | `flashinfer` | `trtllm_mha` is SM100-only — it will not run on Blackwell SM120 |
| prefill chunk | `2048` | decode stalls behind each prefill chunk on hybrid GDN; 8192 stalls it ~600 ms at a time |
| mem fraction | `0.85` | |
| spec decoding | `EAGLE` 3 / 1 / 4 + `--enable-linear-replayssm-spec` | the in-checkpoint MTP head; ReplaySSM moves the verify intermediates onto a fixed ring |
| radix strategy | `extra_buffer` | 5 GDN state slots per running request |
| GDN state dtype | `float32` | the checkpoint's declared precision, 153.9 MB per slot |
| parsers | `qwen3` + `qwen3_coder` | without them a harness gets tool calls as raw text |
| mm transport | `cpu` | the one deviation from the panel — see [WSL2](#wsl2-cuda-ipc) below |
| mamba ratio | `0.29` | see below |

Weights come from the HuggingFace repo id, not `$MODELS_DIR`, so the container
pulls them into `~/.cache/huggingface` on first launch. **`./fetch-models.sh
38-nvfp4` is not a prerequisite** — running it only puts a second copy under
`$MODELS_DIR` that nothing reads.

### WSL2: CUDA IPC

**If you are on WSL2, this is the one thing you must not leave on the default.**
`run-sglang.sh` already handles it; this is why.

Left unset, `--mm-feature-transport` auto-resolves to `cuda_ipc` on any
single-node CUDA box. Image and video features are then handed from the
multimodal processor to the scheduler through a CUDA IPC handle — and **CUDA IPC
does not work under WSL2's GPU paravirtualization.** Qwen3.8 is a
vision-language model, so SGLang's own startup warmup sends an image, and the
server dies seconds after reporting that it started:

```
[...] Failed to deserialize from cached pooled CUDA IPC handle (CUDA error: invalid resource handle
[...] torch.AcceleratorError: CUDA error: invalid resource handle
[...] SIGQUIT received. signum=None, frame=None. It usually means one child failed.
```

It is not an SGLang bug. A bare twenty-line PyTorch script that passes one CUDA
tensor between two processes fails on the same host with the identical error.

The fix is one flag, which the script passes by default:

```
--mm-feature-transport cpu
```

Features cross through CPU memory instead. Slightly more latency on image
requests, no effect on text, and it frees the 1 GiB the IPC pool would have
reserved on the GPU. On a native-Linux host you can take the faster path back
with `./run-sglang.sh --mm-transport cuda_ipc`.

`--dist-init-addr` looks like it should help — `determine_tensor_transport_mode()`
returns CPU transport for multi-node — but that value feeds a different code
path. `use_cuda_ipc` reads `server_args.mm_feature_transport`, so only this flag
changes the behaviour. Tested: setting `--dist-init-addr` alone still crashes.

### Thinking depth

Qwen3.8 always reasons; the depth is `reasoning_effort`, one of `low`,
`medium`, `xhigh` (the template's own default). SGLang has no dedicated flag
for it — the server-wide default is a chat-template kwarg:

```bash
./run-sglang.sh --effort low
#  -> --default-chat-template-kwargs '{"reasoning_effort":"low"}'
```

That applies to every request **that does not carry its own**
`chat_template_kwargs` or `reasoning_effort`; a client that sends one still
wins. Per request:

```bash
curl -s http://localhost:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"RadixArk/Qwen3.8-27B-NVFP4",
       "reasoning_effort":"low",
       "messages":[{"role":"user","content":"Reverse a linked list in Rust."}]}'
```

`--no-preserve-thinking` rides in the same map and drops prior-turn reasoning
from the template. The mechanism mirrors `run.sh --effort` on the llama.cpp
side, which passes the same key through `--chat-template-kwargs`.

### --mamba-full-memory-ratio

This is the one sizing flag that matters on hybrid GDN. Post-weight memory
splits into a worst-case-reserved **GDN state pool** (which sets the
concurrency ceiling) and a paged **attention KV pool**, divided by this ratio.
SGLang's own default (0.9) over-provisions KV and silently clamps concurrency.

```
ratio = (S + D) × state_bytes / (L × kv_bytes_per_token)
```

- `S` — state slots per running request: `extra_buffer` 5, `extra_buffer_lazy` 4,
  `no_buffer` 3, radix cache off 1.
- `D` — speculative verify intermediates: **0** with spec off *and* 0 with
  `--enable-linear-replayssm-spec`; otherwise `--speculative-num-draft-tokens`.
- `state_bytes / kv_bytes_per_token` — the state slot priced in KV tokens:
  **4698** at fp32 state over fp8 KV, **2394** at bf16 state.
- `L` — average total request length in tokens, input **+** output.

The built-in `0.29` is the cookbook panel's own pin, which at the defaults
(`S=5`, `D=0`) corresponds to `L` ≈ 81k. If your workload is shorter, let the
script recompute it for the flags actually in effect:

```bash
./run-sglang.sh --avg-len 32768                             # -> 0.72
./run-sglang.sh --avg-len 32768 --radix extra_buffer_lazy   # S=4 -> 0.57
./run-sglang.sh --mamba-ratio 1.5                           # or set it directly
```

Every launch prints the ratio it used and the average request length that
implies. After boot, check the `max_running_requests` line in the server log —
it should not be capped below your target concurrency.

### Concurrency

With speculative decoding on, SGLang pins `--max-running-requests` to **48**
whenever you leave it unset. The script warns and does not guess; pass
`--concurrency N` sized for what you actually run.

### Pointing a harness at it

The OpenAI endpoint is `http://localhost:30000/v1` and the `model` string must
equal `--model-path` (`RadixArk/Qwen3.8-27B-NVFP4`) unless you shorten it with
`--served-name`. SGLang also serves an Anthropic-compatible `/v1/messages` at
`http://localhost:30000`, which is what Claude Code wants:

```bash
export ANTHROPIC_BASE_URL=http://localhost:30000   # no /v1 suffix
export ANTHROPIC_AUTH_TOKEN=placeholder
```

`--api-key` is unset by default, so the server accepts unauthenticated
requests — set one if the port is reachable beyond localhost. Full OpenCode /
Pi / Claude Code / Hermes wiring is in the
[lmsys cookbook page](https://lmsysorg.mintlify.app/cookbook/autoregressive/Qwen/Qwen3.8-27B#3-agent-harnesses).

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
- [run.sh](run.sh) — server launcher (llama.cpp / GGUF)
- [run-sglang.sh](run-sglang.sh) — SGLang launcher (Docker / NVFP4 safetensors)
- [preflight.sh](preflight.sh) — binary/build/model/VRAM/port checks (auto-run by run.sh; `SKIP_PREFLIGHT=1` to skip)
- `settings/` — per-model saved launch defaults (one line of run.sh flags per `<key>.conf`, of run-sglang.sh flags per `<key>.sglang.conf`; written by `--save-defaults`, gitignored)

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
| `SGLANG_IMAGE` | `lmsysorg/sglang:qwen38-27b` | image `run-sglang.sh` runs |
| `SGLANG_PORT` | `30000` | port *inside* the container (`--port` changes the published one) |
| `SGLANG_CONTAINER` | `qwen38-sglang` | container name `run-sglang.sh` owns and replaces |
| `HF_CACHE` | `~/.cache/huggingface` | host cache bound into the SGLang container |
| `HF_TOKEN` | — | passed into the container when set (gated repos) |
| `DOCKER_ARGS` | — | raw flags appended to the `docker run` argv |

`HOST` defaults to `0.0.0.0`, so the server is reachable from your network and
llama.cpp sets permissive CORS with no API key. On an untrusted network, set
`HOST=127.0.0.1`.

## License

These scripts have no license file yet — add one before you rely on them.

The models are separate: every repo linked in the table above is currently
listed as **Apache-2.0** on HuggingFace, base models and community finetunes
alike. Licenses can change, so check the model card before you depend on one.
