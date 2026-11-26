# Inkling 2 Mini on the A40 box, through `deploy.sh`

Date: 2026-11-26. Status: approved 2026-11-26. Size: a small change. The deploy
machinery from the Flint work is reused as it stands; this adds a model to it.

## Intent

Serve Thinking Machines' Inkling 2 Mini (26B total, 5.4B active, looped MoE,
text and image in) from the same remote box as Flint: one NVIDIA A40 (48 GB,
Ampere), 32 GB of host RAM, about eight people running coding agents at once.
The owner clones this repo there and runs one command, as with
`./deploy.sh flint-q4`.

## Goals

1. `./deploy.sh inkling-q4` on a fresh clone ends at a healthy, key-protected
   OpenAI-compatible endpoint serving Inkling 2 Mini.
2. Defaults sized for eight concurrent agents on 48 GB, with at least 2 GiB of
   the A40's 46,068 MiB left free.
3. Every existing key launches with the same argv as today.

## Non-goals

- SGLang or vLLM for this model. Neither runs it on Ampere (see Constraints).
- DSpark speculative decoding. It is merged in mainline llama.cpp
  (ggml-org/llama.cpp#32218), but Unsloth measured 1.03x at temperature 1.0
  with the mainline-format drafter, and 1.0 is the recommended temperature.
- Unsloth's llama.cpp fork (approach B below).
- Serving Inkling on the RTX PRO 6000 through `run-sglang.sh`, although the
  cookbook verifies that card.
- Anything `deploy.sh` does not already do: TLS, per-user keys, multi-GPU.

## Anti-goals

- A second copy of `run.sh`'s flag composition, or a model-specific branch in
  `deploy.sh`.
- Renaming or copying files inside the HuggingFace cache.
- A changed argv for any existing key (the golden files must still pass).

## Constraints

- **The linked SGLang cookbook cannot be used on this card.** SGLang's Inkling 2
  expert code supports SM90, SM100/103/107 and SM120 only; anything else raises
  `NVFP4 expert computation requires an SM100, SM103 or SM107 GPU` at load. The
  A40 is SM86. vLLM's recipe lists Blackwell, Hopper and DGX Spark only.
- **llama.cpp does run it on Ampere.** Unsloth benchmarks `UD-Q4_K_XL` on an
  A100 (45 tokens/s single-stream on mainline b12649).
- **Mainline llama.cpp reads only the `Mainline_Support/` files** of
  `unsloth/Inkling-2-Mini-GGUF`, which are named `*.gguf_file`, and needs build
  b12611 or later. Older builds load the model and give wrong output
  (perplexity about 130 instead of 7.65), so `minbuild` matters here more than
  it did for Flint. The root-layout files in that repo are for Unsloth's fork.
- **Sampling:** temperature 1.0, top-p 1.0, top-k 0, min-p 0.0. No existing
  sampler family matches.
- **Thinking:** the chat template always emits a thinking-effort line, taken
  from `reasoning_effort` (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`,
  `max`, or a number in 0.0-0.99; default 0.9, which is `high`). There is no
  on/off switch and no preserve switch: prior-turn `reasoning_content` is
  always rendered. The model card says `none` does not fully disable reasoning.
- **Host RAM is 32 GB**, so `--no-mmap` and `--mlock` stay forbidden, as for
  Flint.
- KV cost per token for this architecture is unknown. 20 of the 24 layers use
  a sliding window and the 12 looped layers keep a cache per pass.

## Approaches

**A. Mainline llama.cpp in the image Flint already uses (chosen).** New model
table entries pointing at the `Mainline_Support/` files; a sampler family and a
think mode added to `run.sh`. Nothing is built on the box and `deploy.sh` is
unchanged. Costs: mainline decode speed, no DSpark.

**B. Unsloth's llama.cpp fork in an image this repo builds.** About 1.5x
single-stream decode (A100: 67 against 45 tokens/s) and their faster DSpark
drafter. Costs: a Dockerfile and a build step in `deploy.sh`, a third party's
prebuilt binary pinned by hand, and a second file layout. The 1.5x is a
single-stream figure; nobody has measured it across eight parallel slots.

**C. A now, B later** as a second entry if eight-stream throughput disappoints.

The owner picked A. It keeps the Flint rule (official image, nothing built),
and aggregate throughput across eight agents matters more here than the speed
of one stream. `LLAMA_IMAGE` is already overridable, so B stays possible.

## What gets built

Decided by the owner: approach A; same box and load as Flint.

The rest are the builder's calls, listed so they can be rejected:

- **Two entries in `env.sh`.**
  - `inkling-q4`: `Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file`,
    17.6 GB. The default; Unsloth reports 98.0% top-token agreement with the
    reference.
  - `inkling-q8`: `Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf_file`,
    22.9 GB. The lossless reference (original NVFP4 experts, the rest BF16),
    for when fidelity matters more than KV room.
  - Both: repo `unsloth/Inkling-2-Mini-GGUF`, `mmproj=mmproj-F16.gguf`,
    `minbuild=12611`, `temp=1.0`, `alias=inkling-2-mini`, no MTP, f16 KV.
- **Vision on by default.** The projector is 0.64 GB and `--no-vision` already
  exists to drop it.
- **Sampler family `plain`** in `run.sh`: `--temp 1.0 --top-p 1.0 --top-k 0
  --min-p 0.0`, penalties off. Selected by the table's `sampling` field.
- **Think mode `effort`** in `run.sh`, selected by the table's `think` field:
  no `--reasoning` flag and no preserve flag are passed; `--effort` accepts the
  seven template names and is sent as the `reasoning_effort` template kwarg;
  unset leaves the template's own default. `--think` and `--no-think` are
  refused with a message naming `--effort none` and saying it lowers reasoning
  but does not remove it. Qwen3.8's `--effort` validation is untouched.
- **Slots and pool:** eight slots on a shared KV pool with a per-slot cap, as
  Flint. The starting point is Flint's `--parallel 8 --ctx 655360 --slot-ctx
  131072`; the committed numbers come from measurement.
- **Tests** in `tests/run-tests.sh`: table fields, the composed argv for both
  entries, the refused flags, and the docker argv carrying the `.gguf_file`
  path. Existing goldens unchanged.
- **README:** the two keys in the model table, a short Inkling section under
  the deploy heading (why not SGLang on this card, effort levels, the model
  card's advice to cap output tokens and agent turns because the model can loop
  in reasoning), and the "everything at once" disk figure.

## Verified live before the numbers are committed

On the RTX PRO 6000, as with Flint, and stated in the README as measured there
and not on an A40:

1. llama-server loads a path ending `.gguf_file` from the read-only cache
   mount. If it does not, stop and come back: the fallback touches the cache
   layout, which is an anti-goal.
2. `./deploy.sh inkling-q4` reaches healthy; `./deploy.sh smoke 8` passes chat,
   tool call and eight streams. Record tokens per second.
3. An image request succeeds.
4. `--effort none` and `--effort max` both return 200 and differ in reasoning
   length; a second turn replaying `reasoning_content` returns 200.
5. VRAM at the default pool, for both entries. Size the pool so the total stays
   under 44,000 MiB, and replace the estimated `vram` fields.
6. Container RSS; add a `--cache-ram` cap through the entry's args if it passes
   16 GiB.

The owner asked for cleanup after testing: remove what this work downloaded and
nothing else — `$HF_HUB_CACHE/models--unsloth--Inkling-2-Mini-GGUF` and the
`ghcr.io/ggml-org/llama.cpp:server-cuda` image (not present on the box before
this work). List each target before removing it, then confirm with
`./run.sh --list` and `docker images`.

## Left to the builder

Internal naming, test layout, README wording, and the exact pool and per-slot
figures once measured.
