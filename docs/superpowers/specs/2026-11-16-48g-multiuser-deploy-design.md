# Containerised multi-user deploy on a 48 GB GPU (Qwen3.8-27B, Flint)

Date: 2026-11-16. Status: awaiting approval.

## Intent

Serve two models from a remote box with one NVIDIA A40 (48 GB VRAM, Ampere) and
32 GB of system RAM, to about eight people running coding agents at the same
time:

- Qwen3.8-27B (already in the model table, at single-user settings sized for a
  96 GB card)
- Recursive's Flint (`RecursiveSI/flint`, not in the table yet)

The owner wants to clone this repo on that box and run one command that fetches
the weights and brings up a serving container, in the spirit of the existing
`run-sglang.sh`. Nothing is compiled on the host.

This differs from everything else in the repo, which is tuned for one person on
one workstation. Here concurrency matters more than single-stream speed or the
last bit of quantization fidelity.

## Goals

1. `./deploy.sh <key>` on a fresh clone takes the box from nothing to a healthy
   OpenAI-compatible endpoint: checks prerequisites, pulls the image, downloads
   the weights, starts the container, waits for health, prints the endpoint and
   the API key.
2. Defaults sized for eight concurrent coding agents on 48 GB.
3. Each model runs on the engine that suits it on this card (see Approaches),
   and the user-facing command is the same for both.
4. The existing single-user paths (`./run.sh <key>`, `./run-sglang.sh`) behave
   exactly as they do today.

## Non-goals

- Both models loaded at once. One model at a time; deploying another replaces
  the running container.
- Installing Docker, the NVIDIA driver, or nvidia-container-toolkit. The script
  detects what is missing and says what to install; it does not `sudo`.
- Automatic quant or context selection from detected VRAM.
- Per-user keys, quotas, or usage accounting. One shared key.
- A systemd unit, TLS, or a reverse proxy.
- Multi-GPU.
- Changing the defaults of existing table keys.
- Deploying the SGLang path with `deploy.sh`.

## Anti-goals

- A second copy of `run.sh`'s flag-composition logic.
- A path-translation layer between host and container paths.
- Any regression in `run.sh` without `--docker`, or in `run-sglang.sh`.
- A default that exposes an unauthenticated endpoint to the network.

## Constraints

- **GPU:** A40, 48 GB nominal; `nvidia-smi` reports roughly 46,068 MiB
  (~45 GiB). Ampere has no native FP4 or FP8 compute.
- **Host RAM:** 32 GB. Full GPU offload is mandatory. For llama.cpp, `--no-mmap`
  and `--mlock` must never be passed, and the host-RAM prompt cache
  (`--cache-ram`, default 8192 MiB) and context checkpoints have to stay well
  inside the budget. For vLLM, the 18 GiB INT4 checkpoint loads comfortably; the
  28.7 GiB FP8 checkpoint would not leave much room, one reason it is not used.
- **Driver on the A40 box is unknown.** CUDA 13 images need an R580 or newer
  driver. See section 5.
- **Cannot be tested on an A40 from the development box**, which is an RTX PRO
  6000 Blackwell (96 GB) under WSL2. See Verification.
- **Flint needs llama.cpp build >= 11803** (ggml-org/llama.cpp PR #30928).

### Facts the capacity numbers rest on

Sizes are from the HuggingFace API on 2026-11-16.

| Checkpoint | Size |
| --- | --- |
| `RedHatAI/Qwen3.8-27B-INT4` (safetensors, all files) | 19.45 GB / 18.12 GiB |
| `unsloth/Qwen3.8-27B-GGUF` `Qwen3.8-27B-UD-Q4_K_XL.gguf` | 17.66 GB / 16.45 GiB |
| `unsloth/Qwen3.8-27B-GGUF` `mmproj-F16.gguf` | 0.93 GB / 0.86 GiB |
| `unsloth/flint-GGUF` `flint-UD-Q4_K_XL.gguf` | 25.05 GB / 23.33 GiB |
| `RecursiveSI/flint-nvfp4-gguf` `flint-nvfp4.gguf` | 22.86 GB / 21.29 GiB |

- **Qwen3.8 KV cost.** Only 16 of 64 layers hold a KV cache: 4 KV heads of
  dimension 256, so 64 KiB per token at 16-bit and 32 KiB at 8-bit. This agrees
  with the ~16 GiB the repo measured for 262k tokens at f16.
- **Flint KV cost.** 28 KiB per token at 16-bit on the 14 full-attention layers.
  The 35 sliding-window layers keep only 513 tokens per sequence.
- **`RedHatAI/Qwen3.8-27B-INT4`** is W4A16 (GPTQ with AWQ smoothing,
  compressed-tensors `pack-quantized`). The vision tower, `lm_head` and
  embeddings are left unquantized, the MTP head is present
  (`mtp_num_hidden_layers: 1`), and an FP8 KV-cache scheme was calibrated in.
  Reported accuracy recovery against BF16 is 98.5-101% across six benchmarks.
- **vLLM on Ampere.** vLLM's quantization docs state that Turing and Ampere run
  weight-only kernels (Marlin) for W4A16 and for FP8 checkpoints. vLLM's
  Qwen3.8-27B recipe lists an INT4 (W4A16) variant but has no Ampere card in its
  verified hardware list.
- **llama-server concurrency.** Slots share one KV pool when `--kv-unified` is
  on; `--kv-unified-per-slot N` caps one slot's share. Continuous batching is on
  by default.
- **Flint** is text-only, 262,144-token context, has no MTP head and no thinking
  switch: the model decides per turn whether to reason. Recommended sampling is
  temperature 1.0, top-p 0.95, top-k 20. Recursive's llama.cpp example uses
  f16 KV.

### Capacity estimates on the A40 (unmeasured)

| Model | Weights | Left for KV | Shared pool | Per agent at 8 |
| --- | --- | --- | --- | --- |
| Qwen3.8 INT4, vLLM, FP8 KV | 18.1 GiB | ~21 GiB | ~680k tokens | ~85k |
| Qwen3.8 INT4, vLLM, 16-bit KV | 18.1 GiB | ~21 GiB | ~340k tokens | ~42k |
| Flint Q4, llama.cpp, f16 KV | 23.3 GiB | ~18-20 GiB | ~650-750k tokens | ~80-90k |

Pools are shared, so one agent can run long while others are short. Agents must
be configured with a context limit that fits; the README says so.

Speed is not estimated. Flint activates about 2.5B parameters per token and
suits this card with eight users. Qwen3.8 is a dense 27B model: eight agents
share one A40 and it will be usable, not fast.

## Approaches

**A. llama.cpp container for both models.** One image, certain to run on
Ampere. Chosen first, before the multi-user requirement was known. For a dense
27B model with eight concurrent agents, llama.cpp's batching is the wrong tool.

**B. Engine-native containers for both** (SGLang for Qwen3.8, vLLM for Flint).
Rejected: the SGLang Qwen3.8 cookbook verifies Blackwell and H200 only, and
vLLM's Flint kernels target Blackwell.

**C. Hybrid (chosen by the owner).** Qwen3.8 on vLLM with a 4-bit W4A16
checkpoint; Flint on llama.cpp. vLLM gives the dense model real request batching
and prefix caching. Flint is fast enough per token that llama.cpp's batching is
adequate, and llama.cpp is certain to run it on Ampere. Trade-off: two images,
a third launch script, and vLLM on an A40 for this model is plausible but
unverified. A llama.cpp Qwen entry is kept as the fallback for that risk.

**Ruled out: Flint under SGLang.** Recursive's kit requires >= 141 GB per GPU,
serves BF16 only, and states that quantized weights are not validated.

## Design

### 1. Model table

New field **`engine`**: `llama` (default for GGUF entries), `vllm`, `sglang`, or
`none`. Existing entries get the value that describes how they are served today
(`38-nvfp4` is `sglang`, `nvfp4-unsloth` is `none`); nothing else about them
changes. `run.sh`'s existing "this is a safetensors checkpoint" refusal reads
the field to name the right script.

New field **`alias`**: the model id clients send. Empty by default, which keeps
today's behaviour. Set on the new entries so a team has stable names.

New field **`think`**: `switch` (default) or `auto`. Flint has no thinking
switch. For `think=auto`, `run.sh` passes no `--reasoning on|off`, always uses
the thinking-mode sampler, and rejects `--think`, `--no-think` and `--effort`
with a one-line explanation.

New keys:

| Key | Engine | Checkpoint | Alias | Role |
| --- | --- | --- | --- | --- |
| `38-int4` | vllm | `RedHatAI/Qwen3.8-27B-INT4` | `qwen3.8-27b` | Qwen, primary |
| `38-q4-48g` | llama | `unsloth/Qwen3.8-27B-GGUF` `UD-Q4_K_XL` + `mmproj-F16.gguf` | `qwen3.8-27b` | Qwen, fallback if vLLM fails on the A40 |
| `flint-q4` | llama | `unsloth/flint-GGUF` `flint-UD-Q4_K_XL.gguf` | `flint` | Flint, primary |
| `flint-nvfp4` | llama | `RecursiveSI/flint-nvfp4-gguf` `flint-nvfp4.gguf` | `flint` | Flint, alternative |

- The llama entries carry multi-user defaults in `args`: eight slots, a shared
  pool, a per-slot cap of 131,072 tokens, MTP off. Starting values: pool
  655,360 tokens (8 x 81,920) for both. `38-q4-48g` adds `--kv q8_0`. The
  builder adjusts the pool after measuring, keeping at least 2 GiB of VRAM free.
- `flint-nvfp4` is Recursive's own quantization-aware 4-bit build, which
  benchmarks within about a point of BF16 and is slightly smaller than Q4.
  Recursive tested it on Blackwell only; whether llama.cpp's CUDA backend runs
  NVFP4 tensors on Ampere is unverified, and the description says so. If it
  works on the A40 it is the better Flint entry.
- Flint entries: `temp=1.0`, `minbuild=11803`, `think=auto`, no `mmproj`,
  `sampling=qwen` (its thinking-mode set is exactly Flint's recommendation).
  KV stays f16.
- `38-q4-48g` keeps `imgtok=2048` and `temp=1.0`, and shares its projector
  download with the other `unsloth/Qwen3.8-27B-GGUF` keys.
- `-48g` means "defaults sized for a 48 GB card". It is used where the weights
  also suit other uses.
- `size`, `disk` come from the file table; `vram` values are estimates and are
  marked as such until measured.

### 2. `run.sh`: docker backend and multi-user flags

`run.sh` keeps composing the `llama-server` argument list in one place. A new
launch backend changes only the last step.

- `--docker`, or `LLAMA_DOCKER=1`, selects the container backend. Without
  either, behaviour is unchanged. `run.sh` may be restructured to share the
  compose step; the requirement on host mode is that the same inputs produce
  the same `llama-server` argv as before.
- `--detach` (docker only) runs in the background with
  `--restart unless-stopped`. Without it, foreground with `--rm`.
- `--print` shows the full command for either backend and exits without
  preflight or launch.
- `--slot-ctx N` turns on the shared pool with a per-slot cap
  (`--kv-unified --kv-unified-per-slot N`). `--ctx` is then the pool size.
  Today's `--parallel` help text says each slot gets `ctx/N`; that remains true
  without `--slot-ctx`.
- `--api-key KEY` passes through to llama-server. Not persisted by
  `--save-defaults`.
- `--alias`, from the table's `alias` field when set.
- `--metrics` is passed for entries that set `--parallel` above 1.
- **Mounts use identical paths inside and outside the container**, so the
  composed argument list needs no rewriting: `$HF_HUB_CACHE` read-only at the
  same path (the whole hub directory, because snapshot files are relative
  symlinks into `blobs/`); `--slot-save-path` read-write; each `--lora` file's
  directory read-only. Every path the compose step emits (`-m`, `--mmproj`,
  `-md`) is under `$HF_HUB_CACHE`. A host path passed through `EXTRA_ARGS` is
  the caller's to mount with `DOCKER_ARGS`.
- Publishing: `-p "$HOST:$PORT:$PORT"`, llama-server inside on `0.0.0.0:$PORT`.
- `--gpus all`. `DOCKER_ARGS` is appended raw, as in `run-sglang.sh`.
- The container is named `$DEPLOY_CONTAINER` (default `llm-serve`). Launching
  removes an existing container of that exact name first, with a one-line notice
  naming the model it was serving. No confirmation prompt.
- The existing "stop a running llama-server from this build" logic applies to
  the host backend only.

`preflight.sh` in docker mode replaces the binary checks with: docker daemon
reachable; image present; build number at least the entry's `minbuild`; a CUDA
device visible inside the container. Model, projector, VRAM and port checks are
shared. The VRAM check stays a warning.

### 3. `run-vllm.sh`

Modelled on `run-sglang.sh`: same structure, same precedence rules
(CLI > `settings/<key>.vllm.conf` > built-ins), same `--print`,
`--save-defaults`, `EXTRA_ARGS` and `DOCKER_ARGS`. Takes the model key as its
first argument, so further vLLM entries can be added to the table later.

Built-in defaults for `38-int4`:

| Flag | vLLM argument | Default |
| --- | --- | --- |
| `--ctx N` | `--max-model-len` | 131072 (one request's cap, not the pool) |
| `--concurrency N` | `--max-num-seqs` | 16 |
| `--kv-dtype fp8\|auto` | `--kv-cache-dtype` | `fp8` |
| `--mem-fraction F` | `--gpu-memory-utilization` | 0.90 |
| `--spec mtp\|none` | `--speculative-config` | `none` |
| `--no-vision` | `--language-model-only` | vision on |
| `--effort`, `--no-preserve-thinking` | `--default-chat-template-kwargs` | template defaults |
| `--served-name` | `--served-model-name` | the table's `alias` |
| `--api-key` | `--api-key` | unset |
| `--port`, `--name`, `--detach` | docker | `$PORT`, `$DEPLOY_CONTAINER` |

Always passed: `--reasoning-parser qwen3`, `--enable-auto-tool-choice`, and a
tool-call parser (see "Left to the builder").

- Speculative decoding is off by default because vLLM's own recipe scopes it to
  low latency at small batch sizes; eight agents is not that.
- 16 concurrent sequences, not 8, because agents issue side requests. When the
  KV pool is full vLLM queues rather than fails.
- Docker: `--gpus all --ipc=host`, `$HF_CACHE` mounted at
  `/root/.cache/huggingface` (as `run-sglang.sh` does), `HF_TOKEN` passed when
  set, published as `$HOST:$PORT:8000`.
- Same container name and same host port as the llama.cpp backend. One name
  means either engine's deploy replaces the other; one port means clients keep
  their URL when the model changes.
- Preflight: docker, image, GPU visible in container, weights present, port.

### 4. `deploy.sh`

```
./deploy.sh <key> [engine flags]   # fetch + serve in a container
./deploy.sh stop                   # remove the container
./deploy.sh status                 # container state, model, /health
./deploy.sh logs                   # docker logs -f
./deploy.sh smoke [N]              # check the running endpoint (default N=8)
```

`./deploy.sh <key>`, stopping at the first hard failure with a message that says
what to install or run:

1. Look up the key's `engine`. `llama` and `vllm` are deployable; for anything
   else, say which script serves it and exit.
2. Check docker, `nvidia-smi`, and a way to get the `hf` CLI. Print the driver
   version and pick the image (section 5).
3. Pull the image if absent. `--update` forces a pull.
4. Confirm a CUDA device is visible inside the container. This precedes any
   other use of the image so that a missing nvidia-container-toolkit is reported
   as exactly that.
5. llama only: read the image's build number; if below the entry's `minbuild`,
   pull once and re-read; fail if still below.
6. Resolve the API key (below).
7. `./fetch-models.sh <key>`.
8. Launch detached through `run.sh --docker` or `run-vllm.sh`, passing flags
   through.
9. Poll `/health` until 200. The address comes from `docker port`, not from
   re-parsing flags; `127.0.0.1` is used when the bind address is `0.0.0.0`.
   Give up after a timeout (default 20 minutes; vLLM start-up plus a large file
   on a slow disk takes minutes) or as soon as the container has exited, and
   print the tail of its logs.
10. Print the endpoint, model id, API key location, and the stop, logs and
    smoke commands.

**API key.** A shared service on `0.0.0.0` must not be open by default. The key
is taken from `API_KEY` in the environment, else from `settings/api-key`
(`settings/` is already gitignored), else a random key is generated into that
file with mode 600 and its path is printed. `--no-auth` opts out explicitly.
`run.sh` and `run-vllm.sh` on their own keep today's behaviour: no key unless
`--api-key` is given.

**`smoke`.** A small Python standard-library script that sends one chat
completion, one tool call, and N parallel streaming completions to the running
endpoint, then reports pass or fail and tokens per second per stream. It is how
the owner confirms a deploy on the A40, and it is the same check used in
Verification.

### 5. Images and drivers

- llama.cpp: `ghcr.io/ggml-org/llama.cpp:server-cuda` (CUDA 12). Floating tag,
  guarded by the `minbuild` check. Its entrypoint is `llama-server`. Override
  with `LLAMA_IMAGE`.
- vLLM: pinned to v0.32.0, which is already on the development box and is the
  last release that publishes both CUDA variants. `vllm/vllm-openai:v0.32.0`
  (CUDA 13) when the host driver is R580 or newer, `v0.32.0-cu129` otherwise.
  v0.33.0 dropped the CUDA 12.9 images. Its entrypoint is `vllm serve`.
  Override with `VLLM_IMAGE`.
- If the driver is too old for the chosen image, the GPU check in step 4 fails
  and the message names the driver floor.

### 6. Downloader

- `fetch-models.sh` already handles multi-file safetensors entries
  (`file='*'`), so `38-int4` needs no new download logic.
- The `hf` CLI fallback chain becomes: `hf` on `PATH`, `huggingface-cli` on
  `PATH`, `.venv/bin/hf`, then create `.venv` (already gitignored) and install
  `huggingface_hub` into it. Today's `pip install --user` fails on
  distributions that mark the system Python as externally managed. If
  `python3 -m venv` is unavailable, fail and name the package to install.

### 7. README

A new "deploy.sh (multi-user, Docker)" section: the one-command quick start per
model; host prerequisites (NVIDIA driver and the floors above, Docker,
nvidia-container-toolkit, Python 3 with venv); the capacity table and the
instruction to set each agent's context limit inside it; what to try when
something fails on the A40 (Qwen: `--kv-dtype auto`, then `38-q4-48g`; Flint:
smaller `--ctx` pool); the API key behaviour; the 32 GB RAM note. A
"run-vllm.sh" section parallel to the SGLang one. Update the model table, the
"Files" list, the Requirements section, and the "~360 GiB for everything"
figures.

## Verification

On the development box (RTX PRO 6000, WSL2, Docker 29.9.3, vLLM v0.32.0 image
present):

1. `./run.sh g4-heretic --docker` with weights already on disk: mounts, symlink
   resolution, GPU visibility, port publishing.
2. For several existing keys and flag combinations, diff the host-mode
   `llama-server` argv before and after the change. No GPU needed.
3. `./run-sglang.sh --print` before and after: identical.
4. `./deploy.sh` end to end for each new key, then `./deploy.sh smoke 8`. For
   the Qwen keys add one image request.
5. **Emulate the A40's memory budget for vLLM.** vLLM budgets a fraction of the
   card, so `--mem-fraction 0.42` on the 96 GB card gives it roughly what 0.90
   gives it on an A40. Record the KV pool size vLLM reports at start-up, with
   `--kv-dtype fp8` and with `auto`, and replace the estimates above.
6. For llama.cpp keys, record `nvidia-smi` memory at the defaults and compare
   with 46,068 MiB; adjust the pool sizes. Record container RSS and confirm it
   stays well under 32 GB.

Steps 4-6 download about 85 GB of new weights (INT4 19.5, Qwen Q4 17.7 plus
projector 0.9, Flint Q4 25, Flint NVFP4 23); 240 GB is free. Ask the owner
before starting them.

What this cannot show: that vLLM's Marlin path, FP8 KV cache, and this model's
hybrid layers work on Ampere; that llama.cpp runs NVFP4 on Ampere; and real
speed on the A40. On the A40, by the owner: `./deploy.sh 38-int4` then
`./deploy.sh smoke`, and the same for `flint-q4`. Failures feed back into the
defaults.

## Decisions made by the designer, not the owner

The owner chose containers, the hybrid split, and stated the load (about eight
simultaneous coding agents). Everything below is a default they can overturn.

- `RedHatAI/Qwen3.8-27B-INT4` as the Qwen checkpoint: published accuracy
  figures, a calibrated FP8 KV scheme, MTP head and vision intact.
- Flint at Q4_K_XL rather than Q6, trading fidelity for about three times the
  shared context. NVFP4 offered as the alternative.
- One container name and one host port (`$PORT`, 8081) for both engines.
- API key on by default in `deploy.sh`, auto-generated if absent, one shared
  key.
- Speculative decoding and MTP off for the multi-user entries.
- Vision left on for Qwen.
- vLLM pinned to v0.32.0; llama.cpp on a floating tag with a build check.
- CUDA variant picked from the host driver version.
- Detached deploys use `--restart unless-stopped`.
- Replacing a running container prints a notice and does not ask.
- An out-of-VRAM estimate warns and does not block.
- Prerequisites are checked, never installed.
- A `smoke` subcommand ships with the repo.
- Stable model ids `qwen3.8-27b` and `flint`.

## Left to the builder

- The tool-call parser for vLLM. The RedHatAI card uses `qwen3_xml`; vLLM's
  recipe and this repo's SGLang script use `qwen3_coder`. Pick by running the
  tool-call smoke test against v0.32.0.
- Whether `--kv-cache-dtype fp8` is accepted and correct on the development box
  for this checkpoint. Ampere remains the owner's test.
- Whether llama-server's shared KV pool works with the `qwen35` hybrid
  architecture for `38-q4-48g`. If not, fall back to per-slot contexts and say
  so in the entry's description.
- For `think=auto`, whether to pass `--reasoning-preserve`, decided by reading
  the Flint chat template and checking that a multi-turn request with
  `reasoning_content` round-trips; and whether Flint tool calls parse with plain
  `--jinja`.
- `--cache-ram` and context-checkpoint settings for the llama multi-user
  entries, chosen from the measured RSS.
- Whether `--docker` is parsed before or after per-model defaults. It is not
  persisted by `--save-defaults`.
- How `deploy.sh` reads the llama image build number (from `--version` in a
  container, or an image label).
- Health-poll interval, flag names for timeouts, output wording.
