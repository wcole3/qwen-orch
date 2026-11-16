# Flint Docker Deploy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `./deploy.sh flint-q4` on a fresh clone fetches Flint and serves it from a llama.cpp container, with defaults for eight concurrent coding agents on a 48 GB GPU.

**Architecture:** `run.sh` keeps the single copy of the `llama-server` flag composition and gains a docker launch backend that mounts the HuggingFace cache at the same path inside the container. `deploy.sh` is a thin wrapper: check, pull, fetch, launch detached, wait for health. New model-table fields (`engine`, `alias`, `think`) describe Flint without touching existing entries.

**Tech Stack:** bash, Docker, `ghcr.io/ggml-org/llama.cpp:server-cuda`, Python 3 standard library (smoke test only).

**Spec:** `docs/superpowers/specs/2026-11-16-48g-multiuser-deploy-design.md`

**Scope:** the Flint / llama.cpp half of the spec. Deferred to a later plan: `run-vllm.sh`, `38-int4`, `38-q4-48g`, and the vLLM driver/image selection. `deploy.sh` is written so a `vllm` engine can be added, and refuses it for now.

## Global Constraints

- Host mode (`./run.sh <key>` without `--docker`) must produce the same `llama-server` argv as before for the same inputs. `run-sglang.sh` is not modified.
- Flint needs llama.cpp build >= 11803.
- Image: `ghcr.io/ggml-org/llama.cpp:server-cuda`, overridable with `LLAMA_IMAGE`. Entrypoint is `llama-server`.
- Container name `$DEPLOY_CONTAINER`, default `llm-serve`. Host port `$PORT`, default 8081.
- Mounts use identical paths inside and outside the container. No path rewriting.
- Never pass `--no-mmap` or `--mlock`.
- `deploy.sh` never exposes an unauthenticated endpoint unless `--no-auth` is given.
- Prerequisites are checked, never installed. No `sudo`.
- VRAM figures for new entries are marked "estimated" until measured in Task 6.
- No commits in this plan: the owner has not asked for any. Leave changes in the working tree.

## Review Focus

1. **`--save-defaults` with `--api-key`, `--docker`, `--detach` or `--print`**: none of them may land in `settings/<key>.conf`. A key written there leaks. Test in Task 2 and Task 3.
2. **Container dies while loading** (out of memory, image too old for the architecture): `deploy.sh` must stop polling, print the log tail, and exit non-zero, not wait out the timeout. Checked live in Task 6.
3. **Port held by a non-container process in docker mode**: fail with the existing port-conflict message before `docker run`. Test in Task 3.
4. **Redeploying over a running container**: the old one is removed first and the health poll reads the new one, not a stale 200 from the old. Checked live in Task 6.
5. **`HF_HUB_CACHE` with a space in the path**: the printed and executed docker argv must keep it as one argument. Test in Task 3.

## File Structure

| File | Responsibility |
| --- | --- |
| `env.sh` (modify) | `LLAMA_IMAGE`, `DEPLOY_CONTAINER`; fields `engine`, `alias`, `think`; entries `flint-q4`, `flint-nvfp4` |
| `run.sh` (modify) | `think=auto`, `--alias`, `--slot-ctx`, `--api-key`, `--metrics`, `--print`; docker backend (`--docker`, `--detach`) |
| `preflight.sh` (modify) | docker-mode checks when `LLAMA_DOCKER=1` |
| `fetch-models.sh` (modify) | `hf` CLI resolution with `.venv` fallback |
| `deploy.sh` (create) | deploy, `stop`, `status`, `logs`, `smoke` |
| `smoke.py` (create) | endpoint check: chat, tool call, N parallel streams |
| `tests/run-tests.sh`, `tests/stub-llama-server`, `tests/golden/*.argv` (create) | argv tests, no GPU or docker needed |
| `README.md` (modify) | deploy section, table, files, requirements |

---

### Task 0: Test harness and host-mode baseline

**Files:**
- Create: `tests/run-tests.sh`, `tests/stub-llama-server`, `tests/golden/*.argv`

**Interfaces:**
- Produces: `tests/run-tests.sh` (exit 0 = all pass). Helpers `argv_of <run.sh args...>` (prints one argv element per line, with `HF_HUB_CACHE=/hf/hub`, an empty temp `SETTINGS_DIR`, `SKIP_PREFLIGHT=1`, `PORT=18999`, `LLAMA_SERVER=tests/stub-llama-server`), `assert_has <needle>`, `assert_lacks <needle>`, `assert_golden <name> <run.sh args...>`.

- [ ] **Step 1:** Write `tests/stub-llama-server`: prints each argument on its own line and exits 0.
- [ ] **Step 2:** Write `tests/run-tests.sh` with the helpers above. `UPDATE_GOLDEN=1` rewrites golden files instead of comparing.
- [ ] **Step 3:** Before touching `run.sh`, record goldens for: `q8`; `q8 --ctx 65536 --kv q8_0 --no-think`; `38-q8 --effort low`; `38-q8 --no-vision --parallel 2 --cache-reuse 256`; `deckard`; `hauhau --mtp 4`; `g4-heretic`; `g4-heretic --no-think`.

Run: `UPDATE_GOLDEN=1 tests/run-tests.sh`
- [ ] **Step 4:** Run `tests/run-tests.sh`. Expected: all golden comparisons pass against the unmodified script.

---

### Task 1: Model table

**Files:**
- Modify: `env.sh`
- Test: `tests/run-tests.sh`

**Interfaces:**
- Produces: `model_info <key> engine|alias|think`; keys `flint-q4`, `flint-nvfp4` in `MODEL_KEYS`; exported `LLAMA_IMAGE`, `DEPLOY_CONTAINER`.

- [ ] **Step 1: Failing tests**

```bash
[[ "$(model_info q8 engine)" == llama ]]
[[ "$(model_info 38-nvfp4 engine)" == sglang ]]
[[ "$(model_info nvfp4-unsloth engine)" == none ]]
[[ "$(model_info q8 think)" == switch && -z "$(model_info q8 alias)" ]]
[[ "$(model_info flint-q4 repo)" == unsloth/flint-GGUF ]]
[[ "$(model_info flint-q4 file)" == flint-UD-Q4_K_XL.gguf ]]
[[ "$(model_info flint-q4 minbuild)" == 11803 ]]
[[ "$(model_info flint-q4 think)" == auto && "$(model_info flint-q4 alias)" == flint ]]
[[ "$(model_info flint-q4 temp)" == 1.0 && -z "$(model_info flint-q4 mmproj)" ]]
[[ "$(model_info flint-q4 args)" == "--no-mtp --parallel 8 --ctx 655360 --slot-ctx 131072" ]]
[[ "$(model_info flint-nvfp4 repo)" == RecursiveSI/flint-nvfp4-gguf ]]
[[ "$(model_info flint-nvfp4 file)" == flint-nvfp4.gguf ]]
```

- [ ] **Step 2:** Run tests. Expected: FAIL on the first new assertion.
- [ ] **Step 3:** Implement. `engine` defaults to `llama` when `fmt=gguf`, else `none`; `38-nvfp4` sets `sglang`. `think` defaults to `switch`, `alias` to empty. `flint-q4`: size "25 GB", disk 24, vram 43000 (estimated). `flint-nvfp4`: size "23 GB", disk 22, vram 41000 (estimated), same args, description notes Recursive tested it on Blackwell only. Both `sampling=qwen`. `LLAMA_IMAGE` defaults to `ghcr.io/ggml-org/llama.cpp:server-cuda`, `DEPLOY_CONTAINER` to `llm-serve`. Document the three fields in the table comment. `model_list` marks non-llama engines from `engine` instead of the hardcoded `[vLLM/SGLang]`.
- [ ] **Step 4:** Run tests. Expected: PASS, goldens unchanged.

---

### Task 2: `run.sh` flags

**Files:**
- Modify: `run.sh`
- Test: `tests/run-tests.sh`

**Interfaces:**
- Consumes: `model_info <key> alias|think` (Task 1).
- Produces: flags `--slot-ctx N`, `--api-key KEY`, `--print`. `--print` writes the launch command as one `%q`-quoted line to stdout and exits 0 without preflight, stopping servers, or port checks.

- [ ] **Step 1: Failing tests** (via `argv_of`)

```bash
argv_of flint-q4
assert_has "--alias"; assert_has "flint"
assert_has "--kv-unified"; assert_has "--kv-unified-per-slot"; assert_has "131072"
assert_has "--parallel"; assert_has "655360"; assert_has "--metrics"
assert_lacks "--reasoning"; assert_lacks "--spec-type"
assert_has "--temp"; assert_has "1.0"; assert_has "--top-k"; assert_has "20"
# think=auto rejects the switch flags
! ./run.sh flint-q4 --no-think --print 2>/dev/null
! ./run.sh flint-q4 --effort low --print 2>/dev/null
argv_of q8 --api-key s3cret;  assert_has "--api-key"; assert_has "s3cret"
argv_of q8;                   assert_lacks "--metrics"; assert_lacks "--alias"
# Review Focus 1
./run.sh q8 --api-key s3cret --ctx 65536 --slot-ctx 32768 --save-defaults --print >/dev/null
grep -q -- "--slot-ctx 32768" "$SETTINGS_DIR/q8.conf"
! grep -q -e "api-key" -e "s3cret" -e "--print" "$SETTINGS_DIR/q8.conf"
```

- [ ] **Step 2:** Run tests. Expected: FAIL (`unknown flag: --slot-ctx`).
- [ ] **Step 3:** Implement. `--slot-ctx` must be a positive integer. For `think=auto`: no `--reasoning on|off`, no `--reasoning-preserve`, always the thinking-mode sampler; `--think`, `--no-think`, `--effort` exit 1 with "`<key>` decides per turn whether to reason; there is no thinking switch." `--metrics` only when `--parallel` is above 1. The launch summary line prints `thinking=auto` for these entries.
- [ ] **Step 4:** Run tests. Expected: PASS, goldens unchanged.

---

### Task 3: Docker backend and preflight

**Files:**
- Modify: `run.sh`, `preflight.sh`
- Test: `tests/run-tests.sh`

**Interfaces:**
- Consumes: `LLAMA_IMAGE`, `DEPLOY_CONTAINER` (Task 1); `--print` (Task 2).
- Produces: `--docker` (or `LLAMA_DOCKER=1`), `--detach`. Container label `llm.key=<key>` and `llm.alias=<alias>`. `preflight.sh` honours `LLAMA_DOCKER=1`.

Docker argv shape, in order: `docker run`, then `--rm` (foreground; plus `-it` when stdout is a TTY) or `-d --restart unless-stopped` (`--detach`), `--name $DEPLOY_CONTAINER`, `--gpus all`, `--label llm.key=...`, `--label llm.alias=...`, `-p $HOST:$PORT:$PORT`, `-v $HF_HUB_CACHE:$HF_HUB_CACHE:ro`, slot-save mount rw, one ro mount per distinct `--lora` directory, `-e LLAMA_API_KEY` when that variable is set in the environment, `$DOCKER_ARGS`, `$LLAMA_IMAGE`, then the composed llama-server arguments with `--host 0.0.0.0`.

- [ ] **Step 1: Failing tests** (via `./run.sh ... --docker --print`)

```bash
out="$(./run.sh flint-q4 --docker --print)"
[[ "$out" == docker\ run\ --rm* ]]
grep -q -- "--name llm-serve" <<<"$out"
grep -q -- "--gpus all" <<<"$out"
grep -q -- "-v /hf/hub:/hf/hub:ro" <<<"$out"
grep -q -- "-p 0.0.0.0:18999:18999" <<<"$out"
grep -q -- "--label llm.key=flint-q4" <<<"$out"
grep -q -- "ghcr.io/ggml-org/llama.cpp:server-cuda -m /hf/hub/models--unsloth--flint-GGUF/" <<<"$out"
grep -q -- "--host 0.0.0.0 --port 18999" <<<"$out"
out="$(./run.sh flint-q4 --docker --detach --print)"
grep -q -- "-d --restart unless-stopped" <<<"$out"; ! grep -q -- "--rm" <<<"$out"
out="$(HOST=127.0.0.1 ./run.sh flint-q4 --docker --print)"
grep -q -- "-p 127.0.0.1:18999:18999" <<<"$out"; grep -q -- "--host 0.0.0.0" <<<"$out"
# Review Focus 5: a space in the cache path stays one argument
out="$(HF_HUB_CACHE='/hf cache/hub' ./run.sh flint-q4 --docker --print)"
grep -q -- '-v /hf\\ cache/hub:/hf\\ cache/hub:ro' <<<"$out"
# Review Focus 1
./run.sh flint-q4 --docker --detach --save-defaults --print >/dev/null
! grep -q -e docker -e detach "$SETTINGS_DIR/flint-q4.conf"
# --detach without --docker is an error
! ./run.sh q8 --detach --print 2>/dev/null
# host mode unchanged: goldens still pass
```

Review Focus 3 (port held by a foreign process): start `python3 -m http.server 18998`, run `PORT=18998 ./run.sh flint-q4 --docker` with a stub `docker` first on `PATH` that records its calls; assert exit 1, "port 18998 is already in use" on stderr, and no `docker run` recorded.

- [ ] **Step 2:** Run tests. Expected: FAIL (`unknown flag: --docker`).
- [ ] **Step 3:** Implement the backend in `run.sh`. Before launch in docker mode: remove an existing container named `$DEPLOY_CONTAINER`, printing `replacing container llm-serve (was serving <llm.key label>)`, wait for the port to free, then run the existing port-holder check. Host-only code (`stop_running_server`) is skipped in docker mode.
- [ ] **Step 4:** Implement `preflight.sh` docker mode: docker daemon reachable; image present (else name `docker pull $LLAMA_IMAGE` and `./deploy.sh`); CUDA device visible via `docker run --rm --gpus all "$LLAMA_IMAGE" --list-devices`; build from `docker run --rm "$LLAMA_IMAGE" --version` at least `minbuild`. Model, projector, drafter, VRAM and port checks stay shared.
- [ ] **Step 5:** Run tests. Expected: PASS, goldens unchanged.

---

### Task 4: `hf` CLI bootstrap

**Files:**
- Modify: `fetch-models.sh`
- Test: `tests/run-tests.sh`

**Interfaces:**
- Produces: `fetch-models.sh` resolves the downloader in this order: `hf` on `PATH`; `huggingface-cli` on `PATH`; `.venv/bin/hf`; create `.venv` with `python3 -m venv` and `pip install huggingface_hub`. If venv creation fails: exit 1 naming `python3-venv`.

- [ ] **Step 1: Failing test:** with a `PATH` holding only coreutils plus a stub `python3` whose `-m venv` fails, `./fetch-models.sh flint-q4` exits 1 and stderr contains `python3-venv`. (Use a temp `HF_HUB_CACHE` so nothing counts as downloaded.)
- [ ] **Step 2:** Run. Expected: FAIL (today it attempts `pip install --user`).
- [ ] **Step 3:** Implement the chain.
- [ ] **Step 4:** Run tests. Expected: PASS.

---

### Task 5: `deploy.sh` and `smoke.py`

**Files:**
- Create: `deploy.sh`, `smoke.py`
- Test: `tests/run-tests.sh`

**Interfaces:**
- Consumes: `run.sh --docker --detach` (Task 3), `fetch-models.sh` (Task 4), container labels `llm.key`, `llm.alias`.
- Produces: `./deploy.sh <key> [run.sh flags] [--update] [--no-auth] [--timeout SECONDS]`, `./deploy.sh stop|status|logs|smoke [N]`. `smoke.py --base-url URL --model ID [--api-key KEY] [-n N]`, exit 0 on pass.

- [ ] **Step 1: Failing tests** (no docker needed)

```bash
! ./deploy.sh nope 2>err;               grep -q "unknown model" err
! ./deploy.sh 38-nvfp4 2>err;           grep -q "run-sglang.sh" err
! ./deploy.sh nvfp4-unsloth 2>err;      grep -q "not deployable" err
# API key resolution, exposed as: ./deploy.sh --print-key
[[ "$(API_KEY=abc ./deploy.sh --print-key)" == abc ]]
k="$(./deploy.sh --print-key)"; [[ ${#k} -ge 32 ]]
[[ "$(stat -c %a "$SETTINGS_DIR/api-key")" == 600 ]]
[[ "$(./deploy.sh --print-key)" == "$k" ]]        # stable across runs
python3 -I smoke.py --help >/dev/null
```

- [ ] **Step 2:** Run. Expected: FAIL (`deploy.sh: No such file`).
- [ ] **Step 3:** Implement `deploy.sh` following the spec's ten steps (step 5 of the spec is the `minbuild` pull-once-and-recheck). Only `engine=llama` deploys in this plan; `vllm` prints "not supported yet". The key reaches the container as the `LLAMA_API_KEY` environment variable, not on the argv. Health address comes from `docker port "$DEPLOY_CONTAINER"`; `0.0.0.0` is polled as `127.0.0.1`. Poll every 5 s; default timeout 1200 s; if `docker inspect -f '{{.State.Running}}'` turns false, print `docker logs --tail 40` and exit 1.
- [ ] **Step 4:** Implement `smoke.py` with the standard library only: one chat completion (pass: non-empty `content` or `reasoning_content`); one request with a `get_weather(city)` tool and `tool_choice: "required"` (pass: first tool call is named `get_weather` and its arguments parse as JSON); N parallel streaming completions of about 200 tokens with `stream_options.include_usage` (report completion tokens per second per stream and the aggregate; pass: all N finish without error).
- [ ] **Step 5:** Run tests. Expected: PASS.

---

### Task 6: Live verification, then cleanup

No new files. Changes here are tuning values in `env.sh` and whatever the live run exposes.

- [ ] **Step 1:** `./run.sh g4-heretic --docker --detach` (weights already on disk), then `curl -s localhost:8081/health`. Expected: `{"status":"ok"}`. Proves mounts, symlinks, GPU, port. Remove the container.
- [ ] **Step 2:** `./deploy.sh flint-q4`. Expected: ends with the endpoint and key location; exit 0.
- [ ] **Step 3:** `./deploy.sh smoke 8`. Expected: chat PASS, tool call PASS, 8/8 streams. Record tokens per second.
- [ ] **Step 4:** Multi-turn check: send a second turn that replays the first turn's `reasoning_content`. Expected: 200 and a coherent answer. Decide `--reasoning-preserve` for `think=auto` from this and from the template in the GGUF.
- [ ] **Step 5:** Record `nvidia-smi` memory and `docker stats --no-stream` RSS. Replace the estimated `vram` for `flint-q4`. If measured VRAM exceeds 44,000 MiB, lower the pool in `args` until at least 2 GiB of an A40's 46,068 MiB stays free. If RSS exceeds 16 GiB, add a `--cache-ram` cap through the entry's args path.
- [ ] **Step 6:** Review Focus 4: run `./deploy.sh flint-q4` again while it is serving. Expected: "replacing container" notice, healthy again, `docker ps` shows one container with a new ID.
- [ ] **Step 7:** Review Focus 2: `./deploy.sh flint-q4 --ctx 8000000`. Expected: container exits during load, log tail printed, exit 1, well before the timeout.
- [ ] **Step 8:** `./deploy.sh flint-nvfp4` and `./deploy.sh smoke 8`; record VRAM; update the entry.
- [ ] **Step 9:** `./deploy.sh stop`. Run `tests/run-tests.sh`. Expected: PASS.
- [ ] **Step 10: Cleanup the owner asked for.** Delete only what this plan downloaded: `$HF_HUB_CACHE/models--unsloth--flint-GGUF`, `$HF_HUB_CACHE/models--RecursiveSI--flint-nvfp4-gguf`, and `docker rmi ghcr.io/ggml-org/llama.cpp:server-cuda`. List each target before removing it. Confirm with `./run.sh --list` (both Flint keys back to not downloaded) and `docker images`.

---

### Task 7: README

**Files:**
- Modify: `README.md`

- [ ] **Step 1:** Add "deploy.sh (multi-user, Docker)" after the `run-sglang.sh` section: quick start for `flint-q4`; host prerequisites; measured capacity and throughput from Task 6, stated as measured on an RTX PRO 6000 and not on an A40; the instruction to set each agent's context limit inside the pool; API key behaviour; the 32 GB RAM note; what to try on failure (smaller `--ctx` pool, `flint-nvfp4`).
- [ ] **Step 2:** Add both Flint keys to the model table, `deploy.sh`, `smoke.py` and `tests/` to "Files", Docker to "Requirements", and correct the "~360 GiB for everything" figures (three places, including `fetch-models.sh` and `env.sh` comments).
- [ ] **Step 3:** Note that the Qwen3.8 multi-user path (vLLM) is designed but not built, with a pointer to the spec.
- [ ] **Step 4:** Run `tests/run-tests.sh`. Expected: PASS.
