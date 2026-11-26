# Inkling 2 Mini Deploy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `./deploy.sh inkling-q4` on a fresh clone fetches Inkling 2 Mini and serves it from the llama.cpp container, with defaults for eight concurrent coding agents on a 48 GB A40.

**Architecture:** Two new entries in the `env.sh` model table point at the mainline-layout GGUFs. `run.sh` gains one sampler family (`plain`) and one think mode (`effort`), both selected by table fields, so `deploy.sh`, `preflight.sh` and `fetch-models.sh` need no model-specific code.

**Tech Stack:** bash, Docker, `ghcr.io/ggml-org/llama.cpp:server-cuda`, the existing `tests/run-tests.sh` harness.

**Spec:** `docs/superpowers/specs/2026-11-26-inkling-2-mini-deploy-design.md`

## Global Constraints

- Every existing key produces the same argv as before: `tests/golden/*.argv` pass unchanged, and `UPDATE_GOLDEN=1` is never run.
- `deploy.sh`, `preflight.sh`, `fetch-models.sh` and `run-sglang.sh` are not modified, except the "~480 GiB" figure in `fetch-models.sh`.
- Repo `unsloth/Inkling-2-Mini-GGUF`. Files are the `Mainline_Support/*.gguf_file` ones. Nothing is renamed or copied in the HuggingFace cache.
- `minbuild` is 12611 for both entries.
- Sampling: `--temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0`, presence penalty 0.0, repeat penalty 1.0.
- Effort names: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`.
- Never pass `--no-mmap` or `--mlock`.
- Measured VRAM at the committed defaults stays under 44,000 MiB. Figures are marked "estimated" until Task 3 replaces them.
- No commits: the owner has not asked for any. Leave changes in the working tree.

## Review Focus

1. **An older image is already on the box** (Flint was verified on b12429). Below b12611 the model loads and answers with garbage, so `deploy.sh` must pull once and re-check before launching. Test in Task 1.
2. **A bad `--effort` value** (`--effort hi`, or a number such as `0.5`, which the template accepts but this script does not wrap) must fail at launch with the valid names, not as a 500 on the first request. Test in Task 2.
3. **`--no-think` out of habit.** It must fail and name `--effort none`, not start a server that reasons anyway. Test in Task 2.
4. **`--no-vision`, and a projector that was never downloaded.** Both serve text-only with no `--mmproj`; the second warns. Test in Task 2.
5. **A client that sends its own `reasoning_effort`** must override the server default for that request. Checked live in Task 3.

## File Structure

| File | Responsibility |
| --- | --- |
| `env.sh` (modify) | entries `inkling-q4`, `inkling-q8`; document `sampling=plain` and `think=effort` in the table comment |
| `run.sh` (modify) | `plain` sampler branch; `effort` think mode (flag handling, validation, kwarg, summary line, usage text) |
| `tests/run-tests.sh` (modify) | table, argv, refused-flag, docker-argv and old-image tests |
| `README.md`, `fetch-models.sh`, `env.sh` comment (modify) | model table rows, Inkling section, disk total |

---

### Task 1: Model table

**Files:**
- Modify: `env.sh` (the `MODEL_KEYS` array, the `model_info` case, the field comment above the table)
- Test: `tests/run-tests.sh` (the "Model table", "fetch-models.sh" and "deploy.sh" sections)

**Interfaces:**
- Produces: keys `inkling-q4`, `inkling-q8` in `MODEL_KEYS`; `model_info <key> sampling` returning `plain`; `model_info <key> think` returning `effort`.

- [ ] **Step 1: Failing tests**

```bash
eq "inkling-q4 repo"     "$(model_info inkling-q4 repo)" unsloth/Inkling-2-Mini-GGUF
eq "inkling-q4 file"     "$(model_info inkling-q4 file)" Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file
eq "inkling-q4 path"     "$(model_info inkling-q4 path)" Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file
eq "inkling-q4 mmproj"   "$(model_info inkling-q4 mmproj)" mmproj-F16.gguf
eq "inkling-q4 engine"   "$(model_info inkling-q4 engine)" llama
eq "inkling-q4 minbuild" "$(model_info inkling-q4 minbuild)" 12611
eq "inkling-q4 think"    "$(model_info inkling-q4 think)" effort
eq "inkling-q4 sampling" "$(model_info inkling-q4 sampling)" plain
eq "inkling-q4 temp"     "$(model_info inkling-q4 temp)" 1.0
eq "inkling-q4 alias"    "$(model_info inkling-q4 alias)" inkling-2-mini
eq "inkling-q4 args"     "$(model_info inkling-q4 args)" "--no-mtp --parallel 8 --ctx 655360 --slot-ctx 131072"
eq "inkling-q8 file"     "$(model_info inkling-q8 file)" Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf_file
eq "inkling-q8 minbuild" "$(model_info inkling-q8 minbuild)" 12611
eq "inkling-q8 alias"    "$(model_info inkling-q8 alias)" inkling-2-mini
contains "list shows inkling-q4" "$(model_list)" "inkling-q4"

# fetch: the in-repo subdirectory reaches the hf CLI intact (reuse the venv2 stub)
: > "$TMP/hf.calls"
PATH="$TMP/nohf:/usr/bin:/bin" HF_HUB_CACHE="$TMP/emptyhub" HF_VENV="$TMP/venv2" \
  ./fetch-models.sh inkling-q4 >/dev/null 2>&1
contains "fetch: mainline file by its subdirectory path" "$(cat "$TMP/hf.calls")" \
  "download unsloth/Inkling-2-Mini-GGUF Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file"
contains "fetch: projector too" "$(cat "$TMP/hf.calls")" "download unsloth/Inkling-2-Mini-GGUF mmproj-F16.gguf"

# Review Focus 1: Flint's image (b12429) is too old for Inkling
rm -f "$STUB_DIR/pulled"
STUB_BUILD=12429 dep inkling-q4 --timeout 30
eq       "inkling, old image, pull does not help: exit 1" "$DRC" 1
contains "…names the floor"      "$DOUT" "12611"
excludes "…never launched"       "$DCALLS" " -d --restart"
rm -f "$STUB_DIR/pulled"
STUB_BUILD=12429 STUB_BUILD_AFTER_PULL=12778 dep inkling-q4 --timeout 30
eq       "inkling, old image, pull fixes it: deploys" "$DRC" 0
contains "deploy: prints the inkling model id" "$DOUT" "model id:  inkling-2-mini"
PATH="$TMP/dbin:$PATH" ./deploy.sh stop >/dev/null 2>&1
```

- [ ] **Step 2:** Run `tests/run-tests.sh`. Expected: FAIL on `inkling-q4 repo`.
- [ ] **Step 3:** Implement the two entries after `flint-nvfp4`, with a comment block in the style of Flint's stating: 26B total and 5.4B active looped MoE; layers 6-17 run three passes; mainline llama.cpp reads only the `Mainline_Support/` layout and needs b12611 (ggml-org/llama.cpp#32704), below which output is wrong, not refused; SGLang and vLLM do not run it on Ampere; no MTP head; DSpark left out (1.03x at temperature 1.0). Values: `inkling-q4` size "17.6 GB", disk 17, vram 40000 (estimated); `inkling-q8` size "22.9 GB", disk 22, vram 45000 (estimated), same args. Both `fmt=gguf`. Descriptions name the quant, "+ vision", "8 slots, shared KV pool, ./deploy.sh"; q8's adds "lossless reference". Extend the field comment: `sampling` gains `plain` (temp from the table, no truncation: top-p 1.0, top-k 0), `think` gains `effort` (the template always reasons at a `reasoning_effort` level; no switch; `run.sh --effort` sets it and `--think`/`--no-think` are refused).
- [ ] **Step 4:** Run `tests/run-tests.sh`. Expected: PASS, goldens unchanged. (The deploy cases pass before Task 2 because `run.sh` treats an unknown think mode as `switch`; Task 2 changes what is launched, not whether.)

---

### Task 2: `run.sh` sampler and think mode

**Files:**
- Modify: `run.sh` (usage text, flag parsing near the `--think` cases, the effort validation block, the sampler block, the thinking block, the launch summary line)
- Test: `tests/run-tests.sh` (the "run.sh flags" and "run.sh --docker" sections)

**Interfaces:**
- Consumes: `model_info <key> sampling` = `plain`, `model_info <key> think` = `effort` (Task 1).
- Produces: for `think=effort` entries, `--effort <name>` emits `--chat-template-kwargs {"reasoning_effort":"<name>"}`; `--think` and `--no-think` exit 1.

- [ ] **Step 1: Failing tests**

```bash
argv_of inkling-q4
eq    "inkling-q4 launches"            "$RC" 0
pair  "inkling-q4 alias"               --alias inkling-2-mini
pair  "inkling-q4 temp"                --temp 1.0
pair  "inkling-q4 top-p"               --top-p 1.0
pair  "inkling-q4 top-k"               --top-k 0
pair  "inkling-q4 min-p"               --min-p 0.0
pair  "inkling-q4 slots"               --parallel 8
pair  "inkling-q4 per-slot cap"        --kv-unified-per-slot 131072
has   "inkling-q4 vision on"           --mmproj
lacks "inkling-q4 no reasoning switch" --reasoning
lacks "inkling-q4 no preserve flag"    --reasoning-preserve
lacks "inkling-q4 no preserve-off flag" --no-reasoning-preserve
lacks "inkling-q4 effort unset by default" --chat-template-kwargs
lacks "inkling-q4 no MTP"              --spec-type
lacks "inkling-q4 never --no-mmap"     --no-mmap
lacks "inkling-q4 never --mlock"       --mlock
for e in none minimal low medium high xhigh max; do
  argv_of inkling-q4 --effort "$e"
  pair "inkling-q4 --effort $e" --chat-template-kwargs "{\"reasoning_effort\":\"$e\"}"
done
argv_of inkling-q4 --temp 0.7
pair  "plain sampler honours --temp" --temp 0.7
# Review Focus 2
fails "effort mode rejects an unknown name" ./run.sh inkling-q4 --effort hi --print
contains "…lists the valid names"           "$ERR" "none, minimal, low, medium, high, xhigh, max"
fails "effort mode rejects a number"        ./run.sh inkling-q4 --effort 0.5 --print
# Review Focus 3
fails "effort mode rejects --no-think"      ./run.sh inkling-q4 --no-think --print
contains "…names the alternative"           "$ERR" "--effort none"
fails "effort mode rejects --think"         ./run.sh inkling-q4 --think --print
# Review Focus 4
argv_of inkling-q4 --no-vision
lacks "inkling-q4 --no-vision drops the projector" --mmproj
mkdir -p "$TMP/noproj"; fake_hub "$TMP/noproj"
rm -f "$TMP/noproj/models--unsloth--Inkling-2-Mini-GGUF/snapshots/abc/mmproj-F16.gguf"
HF_HUB_CACHE="$TMP/noproj" argv_of inkling-q4
eq       "missing projector still launches" "$RC" 0
lacks    "…without --mmproj"                --mmproj
contains "…and warns"                       "$ERR" "serving text-only"
# Qwen3.8's effort set is untouched
fails "switch mode still rejects --effort high" ./run.sh 38-q8 --effort high --print
# --save-defaults keeps the effort
./run.sh inkling-q4 --effort low --save-defaults --print >/dev/null 2>&1
contains "save-defaults keeps --effort" "$(cat "$SETTINGS_DIR/inkling-q4.conf" 2>/dev/null)" "--effort low"
rm -f "$SETTINGS_DIR/inkling-q4.conf"
# the container is handed the .gguf_file path as-is
out="$(dprint inkling-q8 --docker)"
contains "docker: mainline file path" "$out" \
  "ghcr.io/ggml-org/llama.cpp:server-cuda -m @HUB@/models--unsloth--Inkling-2-Mini-GGUF/snapshots/abc/Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf_file"
contains "docker: alias label" "$out" "--label llm.alias=inkling-2-mini"
```

- [ ] **Step 2:** Run `tests/run-tests.sh`. Expected: FAIL on `inkling-q4 top-p` (the qwen sampler gives 0.95).
- [ ] **Step 3:** Implement the `plain` sampler branch: `--temp "${TEMP:-$TEMP_THINK}" --top-p 1.0 --top-k 0 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0`, the same in every mode.
- [ ] **Step 4:** Implement `think=effort`. `--think` and `--no-think` exit 1 with: "`<flag>` does not apply: `<key>` always reasons at a reasoning-effort level. Use --effort none for the least reasoning (it lowers reasoning; it does not remove it)." `--effort` validates against the seven names and on failure prints "--effort must be one of none, minimal, low, medium, high, xhigh, max (got: `<value>`)". No `--reasoning` and no preserve flag is passed; the kwarg is passed only when `--effort` is set. `--preserve-thinking` and `--no-preserve-thinking` are accepted and ignored, as for `auto`. The existing `switch` validation (`low|medium|xhigh`) and the `auto` behaviour are unchanged. The launch summary prints `thinking=effort:<name>`, or `thinking=effort:template default (high)` when unset. Add the mode to the `--effort`, `--think` and `--no-think` lines of the usage text.
- [ ] **Step 5:** Run `tests/run-tests.sh`. Expected: PASS, goldens unchanged.

---

### Task 3: Live verification, tuning, cleanup

No new files. Changes are the `vram` and `args` values in `env.sh`, their mirror in the tests, and whatever the live run exposes. Runs on the RTX PRO 6000 (97,887 MiB); record baseline `nvidia-smi` memory first and report VRAM as the delta.

- [ ] **Step 1:** `./deploy.sh inkling-q4`. Expected: image pulled (not on the box), llama.cpp build >= 12611 printed, weights fetched, "Serving inkling-q4 — healthy". **If the server rejects the `.gguf_file` path, stop and report to the owner**: the fallback touches the cache layout, which the spec rules out.
- [ ] **Step 2:** `./deploy.sh smoke 8`. Expected: chat PASS, tool call PASS, 8/8 streams. Record per-stream and aggregate tokens per second.
- [ ] **Step 3:** One chat completion with an `image_url` content part (a small PNG as a data URL). Expected: 200 and a description of the image.
- [ ] **Step 4:** Effort. Same prompt with `"reasoning_effort":"none"` and `"reasoning_effort":"max"` in the request body. Expected: both 200, `reasoning_content` longer for `max`. This is Review Focus 5: the request value wins over the server default. Then redeploy with `--effort low` and confirm a request with no effort set reasons less than the default deploy did.
- [ ] **Step 5:** Multi-turn: a second turn that replays the first turn's `reasoning_content` and a tool result. Expected: 200 and a coherent answer.
- [ ] **Step 6:** Record VRAM delta and `docker stats --no-stream` RSS at the default pool. Fill one slot to its 131,072 cap with a long prompt and record VRAM again. Set `--ctx` (pool) so the total is at most 44,000 MiB including the projector; keep 8 slots and prefer keeping the 131,072 per-slot cap. Replace the estimated `vram`, update `args` and the matching test strings. If RSS exceeds 16 GiB, add a `--cache-ram` cap through the entry's args.
- [ ] **Step 7:** `./deploy.sh inkling-q8`, `./deploy.sh smoke 8`, record VRAM and tokens per second, size its pool the same way, update its entry.
- [ ] **Step 8:** `./deploy.sh stop`. Run `tests/run-tests.sh`. Expected: PASS.
- [ ] **Step 9: Cleanup the owner asked for.** Remove only what this plan downloaded: `$HF_HUB_CACHE/models--unsloth--Inkling-2-Mini-GGUF` and the image `ghcr.io/ggml-org/llama.cpp:server-cuda`. Print each target (path and size, image id) before removing it. Confirm with `./run.sh --list` (both Inkling keys not downloaded) and `docker images`.

---

### Task 4: README and disk totals

**Files:**
- Modify: `README.md`, `fetch-models.sh` (two "~480 GiB" mentions), `env.sh` (one "~480 GiB" comment)

- [ ] **Step 1:** Add `inkling-q4` and `inkling-q8` rows to the model table (download "17.6 GB + 0.6", disk "17 GiB + 1" and "22.9 GB + 0.6", "22 GiB + 1", measured VRAM with a footnote giving the MiB figures and pool size, as Flint's ¶ footnote does).
- [ ] **Step 2:** Add an "Inkling 2 Mini" subsection under "deploy.sh (a shared server in Docker)", after "One model, several people": quick start with `./deploy.sh inkling-q4`; why this is llama.cpp and not the SGLang cookbook on an A40 (SGLang and vLLM support Hopper and Blackwell only); the b12611 floor and that an older image gives wrong output; `--effort` names, the default (`high`), and that `none` lowers reasoning without removing it; vision on by default and `--no-vision`; the model card's advice to cap output tokens and agent turns because the model can loop in reasoning; throughput and VRAM from Task 3, stated as measured on an RTX PRO 6000 and not on an A40; `inkling-q8` as the lossless alternative.
- [ ] **Step 3:** Change "~480 GiB" to "~520 GiB" in `README.md` (two places), `fetch-models.sh` (two) and `env.sh` (one): 480 + 17 + 22 + 1 for the shared projector.
- [ ] **Step 4:** Run `tests/run-tests.sh`. Expected: PASS.
