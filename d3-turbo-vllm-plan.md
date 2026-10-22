# Plan: serve `vllm-sr/d3-turbo` on vLLM (CUDA + ROCm, extensible to XPU/CPU)

Status: plan only. Nothing here has been run. Written from `config.json`, `MODEL_MANIFEST.json`,
`decision_config.json`, `modeling_d3.py`, `d3_runtime.py`, `d3_format.py`, `d3_kernels.py` of the HF repo
(snapshot `a0f3e9e`), the vLLM v0.32.0 file listing, and the `semantic-router/src/model-runtime` README.
Items marked **[verify]** are assumptions about vLLM internals that must be checked against the vLLM source before
committing to the design.

## 1. What d3-turbo actually is (what vLLM must reproduce)

| Piece | Detail | vLLM today |
| --- | --- | --- |
| Backbone | `qwen3_5`: 64 layers, hidden 5120, 3:1 Gated-DeltaNet : full-attention (24 Q heads / 4 KV heads, head_dim 256, output gate), partial RoPE, mrope (interleaved) | `models/qwen3_5.py`, `qwen3_next.py` exist in v0.32.0, **[verify]** dense 27B variant + hybrid KV-cache manager |
| Vision | 27-layer ViT, hidden 1152, patch 16, merge 2, out 5120, no deepstack. Videos: 2 fps, max 32 frames | Qwen3-VL / Qwen3.5 multimodal processor exists **[verify]** |
| Output | **No lm_head, no sampling.** `readout.safetensors` = `[255, 5120]` FP32 matrix applied to final-norm hidden state at selected prompt positions, then softmax over the question's options (answer codes `A..Z, AA..`) | Not a stock task. Needs a custom pooler |
| Attention mode | `prefix_bidir`: *state* block attends bidirectionally inside itself; *question* block attends to the whole state + itself bidirectionally. Full-attention layers only; DeltaNet layers stay causal | Not supported. Needs non-causal attention per request |
| State-once | State is computed once per request; N question blocks (up to 255) run against cached state KV + DeltaNet recurrent/conv state | Closest analogue is prefix caching. DeltaNet-state caching is the hard part (see Risks) |
| Box / Track | Several readout positions per question (box-edge bins, confidence, presence), decoded in `d3_compose.py` (`decode`, `confidence`, `calibrate`, windows for tracks) | Pure Python, reusable outside vLLM |
| Numerics | BF16 backbone, FP32 readout and softmax. `d3_kernels.py` reproduces ROCm gfx942 eager rounding bit-for-bit (Triton, no FP fusion, FP64 rsqrt) | Not reproducible on CUDA or in vLLM kernels, so parity must be statistical |

Everything except the backbone forward pass (prompt rendering, answer codes, readout decode, Box/Track compose, validation,
System One API) is plain Python in the repo (`d3_format.py`, `d3_compose.py`, parts of `d3_runtime.py`, `d3_server.py`).
That is the main lever for scoping this down.

## 2. Target architecture

Split into a **frontend** that keeps all d3 semantics and a **backend** that is vLLM doing the heavy forward pass.

```
client ──System One API──> frontend (d3 plugin, pure Python)
                              │  render prompts, answer codes, state/question token split,
                              │  readout offsets, Box/Track compose, calibration, errors
                              ▼
                        vLLM engine (pooling runner)
                              │  D3ForDecision (out-of-tree model plugin)
                              │   • Qwen3.5 backbone from vLLM
                              │   • non-causal attention metadata per request
                              │   • D3Pooler: gather positions → FP32 @ readout.T → softmax
                              ▼
                        probabilities [positions × 255] per question
```

Two ways to ship the frontend. Pick after Phase 0:

- **A. Standalone vLLM plugin + thin server** (recommended first). `vllm.general_plugins` entry point registers
  `D3ForDecision`; a small FastAPI layer (adapted from `d3_server.py`) calls vLLM's `LLM`/`AsyncLLM` pooling API.
  Works on any vLLM build, so CUDA and ROCm come for free.
- **B. `vllm-srun` engine plugin.** The semantic-router model-runtime is entry-point based (`families`, `engines`,
  `accel`, `heads`, `scheduler` packages, plus `examples/third_party_plugin`). Add a `vllm` engine behind the existing
  d3 family so `vllm-sr serve vllm-sr/d3-turbo --engine --platform cuda|rocm` works. This is the upstream-friendly route
  and needs coordination with the maintainers.

## 3. Phases

### Phase 0: ground truth and feasibility (about 1 week)
1. Provision one CUDA box (H100/H200 80 GB class) and one ROCm box (MI300X/MI325X). BF16 weights are about 52 GB, so one 80 GB+ GPU or TP=2 on 48 GB cards.
2. Run the reference HF runtime (`AutoModel ... system_one`) on both. Confirm it matches the model card's published sample outputs. This is the oracle.
3. Build a **golden set**, serialized JSON of request → full answer probabilities:
   text choice/noul/score (varied option counts, long states), images, videos, Box, Track, images-in-options, long-context, permutation-average on and off.
   Include 255-option and near-limit-length cases.
4. Read vLLM v0.32 `qwen3_5.py`, the hybrid KV-cache manager, the attention backend interface, and the pooling runner. Answer:
   - Can a model run through `runner=pooling` while its backbone is a hybrid DeltaNet model? **[verify]**
   - Can per-request `causal=False` be expressed in FlashAttention / FlashInfer / ROCm backends (AITER, Triton)? **[verify]**
   - Does the pooler get the position indices of the prompt it needs, and can it return a `[k, 255]` tensor? **[verify]**
   - Does the multimodal processor path accept the `assets` placeholders (images inside questions and options)? **[verify]**
5. Decide A vs B. Exit criterion: written answers to the above and a go/no-go.

### Phase 1: text-only correctness, one forward per question (about 2–3 weeks)
Goal: choice / noul / score on text, correct, on CUDA and ROCm. No state-once yet.
1. **Model class `D3ForDecision`**: subclass vLLM's Qwen3.5 text model; drop the lm_head; load the 11 safetensors shards
   plus `readout.safetensors` into a `D3Pooler` (FP32 `[255, 5120]`, replicated across TP ranks).
2. **Bidirectional full-attention.** For each sequence, the state block `[0, s)` is non-causal over `[0, s)`; the question block `[s, n)` is
   non-causal over `[0, n)`. In one prefill-only forward this is two varlen attention calls per layer, exactly what `block_attention` does
   in `d3_runtime.py`: (state queries × state keys) and (question queries × all keys), both `causal=False`.
   - Carry `(state_end)` per sequence in custom attention metadata (a model-side `forward_context` extra).
   - Implement as an attention-backend wrapper first (call the backend twice on query slices), optimise later.
3. **DeltaNet layers stay causal**, so use vLLM's stock GDN path. Disable the sliding/chunk behaviours that assume decode.
4. **Prefill-only scheduling**: no decode, no KV retention across steps. Disable chunked prefill for these requests, since bidirectional attention needs the
   whole block in one step. Set `max_num_batched_tokens ≥ max prompt` (32,768 default, 131,072 for `long_context=True`).
5. **D3Pooler**: receives the final-norm hidden states, gathers the readout positions (last token, or the Box/Track offsets from
   `d3_compose.readout_offsets`), computes FP32 `hidden @ readout.T`, slices the first `n_options` logits per question and softmaxes. Returns probabilities.
6. **Frontend**: import `d3_format` (prompt render, answer codes) and the answer-assembly code from `d3_runtime` as a library. Do not re-implement the prompt format.
7. Tests: golden text cases. Gate: max abs probability diff ≤ 1e-2 and argmax agreement 100% on the text golden set (tolerances tuned in Phase 4).

### Phase 2: multimodal (about 2–3 weeks)
1. Wire the Qwen3.5 vision encoder through vLLM's multimodal registry; reproduce `linearize_patch_embed`, the 1.6 MP image cap and the
   video sampling (2 fps, ≤ 32 frames, 0.2 MP per frame) in the processor so tokens match the reference exactly.
2. mrope positions for image and video prompts; confirm vLLM's positions equal the HF ones used in `PromptEncoder`.
3. `assets` (images inside state, instructions or criteria): placeholders must land in the right block. Image tokens in a question block are
   seen only by that question. The frontend renders and passes per-question multimodal data.
4. **Box and Track**: multiple readout positions per question and the compose decode. Track windows (`window_plan`) become several
   requests per question. Port nothing; call `d3_compose`.
5. Gate: image / video / Box / Track golden sets. Box IoU@0.5 accuracy and tracking AUC within ±0.01 of the card's numbers on the public set.

### Phase 3: state-once (the 15× win) (about 3–4 weeks, highest risk)
Goal: N questions about one state cost about one state pass plus N small question passes.
- **Option 3a (preferred): vLLM prefix caching with a state boundary.** Submit all N questions with the same state prefix. The state
  KV for the 16 full-attention layers is cacheable because under `prefix_bidir` the state never sees the question, so its KV depends only on the state
  tokens. Cache the 48 DeltaNet layers' recurrent and conv state **at the state end**.
  - Blocker **[verify]**: vLLM's hybrid prefix caching snapshots Mamba/GDN state at block-aligned positions. The state end is arbitrary. Possible fix: pad the
    state block to the cache block size with a no-op token the reference runtime also uses (not allowed; changes numerics), or extend the cache manager to store a
    snapshot at an explicit boundary (a vLLM core change, upstream PR needed).
- **Option 3b: bespoke "decision batch" request type.** One engine request carries `(state, [question_1..N])`. The model runner runs the state pass once, keeps KV + GDN state in
  scratch buffers, then runs the question blocks as a right-padded batch (mirrors `StateOnce.questions`) and returns `[N, positions, 255]`. Avoids touching the prefix cache but
  needs a custom scheduler/runner hook. Closer to what d3's own engine does, more code, less upstreamable.
- Fallback if neither lands: ship Phase 1/2 (per-question forward). Correct, but slower than the card's d3-turbo numbers beyond about 4 questions.
- Gate: identical answers to the Phase 1 path, and p50 latency for 64 questions about one photo within 2× of the card's 867 ms on MI325X.

### Phase 4: numerics, quantization, hardening (about 2 weeks, partly parallel)
1. **Parity budget.** Bitwise equality with the reference is only promised on ROCm in the router images. For vLLM define: per-answer probability abs diff ≤ tolerance, argmax
   agreement ≥ 99.9% on the golden set, and the Jev Decision Index (official 0.3 kit, `decision_index run --engine`) within ±0.3 of 70.2 text / 72.9 vision.
   Write a `D3Engine`-style adapter for the kit so the same harness scores both paths.
2. Re-check FP32 readout and FP32 softmax survive vLLM's dtype handling (a BF16 pooler would silently change results).
3. Quantization: start from `machinez/d3-turbo-NVFP4` (CUDA Blackwell) and vLLM FP8 W8A8 for Hopper/MI3xx. Quantize the backbone only, keep the readout FP32; re-run the index. Accept
   only if the drop ≤ 0.5. Do not ship INT8/4-bit without that gate.
4. TP/PP: TP=2/4 for 80 GB-class and smaller cards; vision encoder TP-aware **[verify]**. The readout is replicated.
5. Safety limits copied from the reference: `max_length_exceeded` (never truncate), 255-question cap, image/video size limits, `long_context` flag.

### Phase 5: platform matrix and CI (continuous)
| Platform | Attention backend | GDN kernels | Notes |
| --- | --- | --- | --- |
| CUDA H100/H200 | FlashAttention 3 / FlashInfer | flash-linear-attention (Triton) | primary dev target |
| CUDA B200 | FlashInfer | FLA | NVFP4 path |
| ROCm MI300X/MI325X (gfx942) | AITER or Triton | FLA (Triton) | reference numerics were validated here; keep as the parity oracle |
| ROCm MI355X (gfx950) | AITER / Triton | FLA | **[verify]** FLA triton on gfx950 |
| CUDA via TensorRT-LLM | TRT-LLM attention (see section 6) | TRT-LLM Qwen3-Next/3.5 path | second backend, NVIDIA only |
| XPU / CPU | optional | PyTorch reference path | correctness only, 27B is impractical |
- CI: golden-set smoke test per platform on small hardware is impossible (52 GB weights), so run on nightly self-hosted GPUs; keep a tiny random-weight d3-shaped checkpoint (same architecture, 2 layers) for per-PR logic tests.
- Do **not** port `d3_kernels.py`: it is gfx942-specific rounding emulation, only needed for bit-exactness.

### Phase 6: upstreaming and docs (about 1 week)
- Package as `vllm-d3` (pip, entry point `vllm.general_plugins`), pinned vLLM range, README with a CUDA and a ROCm command.
- Open upstream issues/PRs for generic pieces: per-request non-causal attention metadata, pooler position gather, GDN state snapshot at arbitrary boundary.
- Coordinate with the vllm-sr maintainers on option B (engine plugin).

## 4. Key risks

1. **GDN state caching at an arbitrary boundary** (Phase 3). Could force a vLLM core change or the bespoke request type.
2. **Per-request non-causal attention** is not a standard backend feature **[verify]**. The two-call approach works but may be slow for 131k contexts. Fallback: dense block mask with SDPA/FlexAttention for long inputs.
3. **Chunked prefill incompatibility.** Bidirectional blocks need the whole block in one step; long contexts then need a large `max_num_batched_tokens` and lots of activation memory.
4. **Numerics.** Without ROCm bit-exactness, small probability shifts; the Box confidence calibration table (`box_calibration`) is tuned to the reference and may be sensitive.
5. **Multimodal parity.** Any token-count or position mismatch in images/video changes every answer; test against golden token ids, not only answers.
6. **License and provenance.** Apache-2.0. Check the model-runtime contributor and trademark rules before publishing a derived plugin named `d3`.

## 5. Suggested order and effort
Phase 0 → 1 → 2 → 4 (in parallel) → 3 → 5/6. First usable milestone (text, CUDA + ROCm, correct but unoptimised): about 4 weeks for one engineer.
Full multimodal with state-once: about 10–12 weeks, dominated by Phase 3.

## 6. TensorRT-LLM backend (NVIDIA-only, after vLLM passes the index)

Observed: `NVIDIA/TensorRT-LLM` `tensorrt_llm/_torch/models/` has `modeling_qwen3_5.py`, `modeling_qwen3_next.py`, `modeling_qwen3vl.py`
(latest release seen: v1.2.1, 2026-04-20; check that the Qwen3.5 file is in the release we pin). Internals below are **[verify]**.

**Path:** TensorRT-LLM PyTorch backend with a `D3ForDecision` model class. Do not use compiled engines (`trtllm-build`, raw TensorRT):
the hybrid DeltaNet layers, bidirectional attention and position-wise readout would need custom plugins and lose the flexibility.

**Design:** same frontend and same golden set as vLLM; only the backend changes, behind one interface
(`run(prepared) -> probabilities per question/position`).

| Item | Plan | Open question |
| --- | --- | --- |
| Readout | Small FP32 module after the final norm; softmax over the question's options | Can the model return hidden states at arbitrary positions, or run the readout inside the model? |
| Bidirectional attention | Two non-causal calls per layer (state × state, question × all keys), as in Phase 1 | Does the paged-KV varlen kernel accept non-causal masks? Else a custom kernel or a dense-mask fallback |
| State-once | Bespoke "decision batch" request (state pass once, N question blocks against saved KV + DeltaNet state) | TRT-LLM's own recurrent-state manager must snapshot DeltaNet state at the exact state end. Likely a core change |
| Quantization | FP8 (Hopper) / NVFP4 (Blackwell) on the backbone only; readout stays FP32 | Each recipe must hold the Jev Decision Index drop to ≤ 0.5. Start from `machinez/d3-turbo-NVFP4` |
| Speed | CUDA graphs and fused kernels mostly help the small question passes | No measurements yet; benchmark against the vLLM path on the same GPU |
| Serving | Optional Triton Inference Server wrapper | |

Costs: NVIDIA only (ROCm stays on vLLM, so two backends to maintain); numerics differ from the ROCm reference, so parity is statistical.

**Phase T0 (about 1 week, parallel to vLLM Phase 1):** read `modeling_qwen3_5.py` and the attention backends; answer the four open questions above; go/no-go.
**Phase T1:** text-only, one forward per question, FP8, golden-set parity. **Phase T2:** multimodal, Box, Track. **Phase T3:** state-once and NVFP4.

## 7. First concrete steps (this week)
1. Clone `vllm-project/vllm` at v0.32.0 and `vllm-project/semantic-router`; read `qwen3_5.py`, the GDN layer, the attention backend interface, the pooling runner and `src/model-runtime/vllm_srun/{families,engines}`.
2. Stand up the reference runtime on one GPU and generate the golden set.
3. Write a 100-line spike: register a `D3ForDecision` that reuses the Qwen3.5 model, bypass attention modifications, and return the last-token FP32 readout. Compare to the golden text answers. The gap size tells us how much the bidirectional attention matters and whether the pooling plumbing works.
