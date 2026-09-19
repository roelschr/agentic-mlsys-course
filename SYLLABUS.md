# Ten-week from-scratch MLSys curriculum

## Contract and pacing

**Every week: 8h guided study, 2h implementation, 2h Socratic self-grill = 12h.**
Total: **80h study + 20h coding + 20h review = 120h**. No additional mandatory
homework, provisioning project, dataset collection, or full-paper reading is
hidden outside these allocations. Setup and test execution count toward coding.

**Reading includes listening and watching.** Follow each week's study resources
in order: build a mental model with a video, visual explanation, or readable
chapter before consulting the original equations. The minutes below are total
study allowances, including pauses, notes, and the **Check** question; they are
not just playback times. Stop at the named sections, not the end of a website,
playlist, or paper. At the time limit, record the blocker and resume that week
in the next available study slot rather than adding homework.

For videos, the linked text adaptation is a **substitute** within the same
allowance, not a second assignment. Use audio for motivation and verbal recap;
attention diagrams, equations, and shape annotations need screen or desk time.
Checks are short predictions or explanations, not extra coding tasks or worked
solutions. The later implementation and review blocks deepen these first checks.
New explanatory links and named sections were checked on **2026-09-19**; video
ranges are stated explicitly. External resources may use different notation or
frameworks: study the assigned explanations, while the course's stub contracts
remain authoritative for implementation conventions.

**Local hardware track:** the verified WSL2 + RTX 3090 setup uses the
[README's uv workflow and device policy](README.md#verified-local-profile-wsl2--rtx-3090).
Use CUDA for the already-budgeted Week 3 memory and Week 8 transfer experiments;
keep numerical/gradient gates on CPU and Week 7 on two local CPU/Gloo ranks.
Hardware availability adds no components or study hours. With uv, prefix the
test commands below with `uv run --locked --extra dev`.

Weeks 1–8 contain exactly two components each. Each component must remain **under
150 nonblank, noncomment implementation lines**, excluding supplied docstrings
and tests. A function plus its small helper/backward methods counts as one
component. Existing PyTorch tensor operations and autograd are allowed; wrapping
the operation under study is not. Custom backward is mandatory only for Week 2
RMSNorm and the communication semantics required in Week 7. Other derivatives
may use ordinary autograd. No Triton kernel is required within this budget.

The two capstone weeks use the same allocation, with all four coding hours devoted
to one integrated project. Four small glue components replace the isolated
drills. They reuse earlier work and each stays under the same 150-line limit.

### Tracking

| Week | Theme | Reading | Coding | Review | Gate |
|---|---|---:|---:|---:|---|
| 1 | Optimization and precision | 8h | 2h | 2h | [ ] optimizer state + overflow |
| 2 | Normalization and positions | 8h | 2h | 2h | [ ] backward + rotation |
| 3 | Attention and cache memory | 8h | 2h | 2h | [ ] grouped causal/cache parity |
| 4 | Transformer FFNs and routing | 8h | 2h | 2h | [ ] gradients + capacity |
| 5 | Post-training objectives | 8h | 2h | 2h | [ ] masking + DPO derivatives |
| 6 | Contrastive retrieval | 8h | 2h | 2h | [ ] normalization + mining |
| 7 | Tensor and pipeline parallelism | 8h | 2h | 2h | [ ] two-rank forward/backward |
| 8 | Partitioning and inference | 8h | 2h | 2h | [ ] bytes + corrected distribution |
| 9 | Capstone: decoder + SFT | 8h | 2h | 2h | [ ] tiny corpus overfit |
| 10 | Capstone: DPO + serving | 8h | 2h | 2h | [ ] preference + cached decode |

Tests live in `tests/test_weekNN.py`. Formula symbols use real arithmetic unless
otherwise stated. Unless specified, reductions are means over valid examples,
tests use deterministic CPU float64 inputs, and `gradcheck` uses finite
differences. Constructor/API contracts and formulas are in the unsolved stubs.

### Reusable review format (2h, already budgeted)

- 30m: derive one equation and its shapes aloud, without the paper.
- 45m: run that week's pytest suite with your mentor; diagnose numerical, memory,
  and edge-case outcomes. Predict a failure before trying a perturbation.
- 30m: answer the four interview questions below, emphasizing alternatives and
  failure modes rather than reciting API names.
- 15m: write a compact evidence log: command, result, bytes/dtypes, misconception,
  and next question. A required gate needs passing tests **and** an explanation.

## Week 1: Optimization & mixed precision mechanics

**Time:** guided study 8h; implementation 2h; self-grill 2h.

### Core concepts

1. Optimizer state and the distinction between momentum, adaptive scaling, and
   regularization. With one common momentum convention,
   `v_t = μ v_(t−1) + g_t`, `θ_t = θ_(t−1) − η v_t`.
   Adam uses `m_t = β₁m_(t−1)+(1−β₁)g_t`,
   `v_t = β₂v_(t−1)+(1−β₂)g_t²` and bias-corrected `m̂_t, v̂_t`.
   AdamW updates `θ_t = (1−ηλ)θ_(t−1) − η m̂_t/(sqrt(v̂_t)+ε)`.
   Adding `λθ` to Adam's gradient instead changes both moment histories; an
   isotropic shrinkage is no longer equivalent after coordinatewise scaling.
2. Global norm clipping across parameters, missing gradients, per-parameter step
   counters, and overflow as an atomic *skip*, not a zero-gradient update.
3. FP16 (5 exponent, 10 fraction bits) vs. BF16 (8 exponent, 7 fraction bits).
   Scaling cannot recover information already lost in a forward cast. Unscale
   before clipping; reject nonfinite gradients before touching parameters/state.

### Reading & guided study (8h total)

- 35m: [Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w)
  — Grant Sanderson / 3Blue1Brown, visual video; **00:00–20:33**, with pauses and
  notes. Prerequisite: derivatives and vectors. Emphasize 03:01–06:55 on cost
  functions, 06:55–11:18 on gradient descent, and the 12:19–13:01 recap. Purpose:
  ground optimizer state in the geometry of repeated updates.
  **Check:** why can a locally downhill direction still produce slow zigzagging?
- 65m: [Why Momentum Really Works](https://distill.pub/2017/momentum/) — Gabriel
  Goh, interactive visual essay. Prerequisite: the gradient-descent video. Read
  “First Steps: Gradient Descent,” including “Decomposing the Error” and “Choosing
  A Step-size,” then “The Dynamics of Momentum” through “Optimal parameters”;
  skip the later examples and lower-bound proof. Purpose: distinguish acceleration
  from smoothing and expose momentum-induced oscillation.
  **Check:** can momentum create an oscillating loss on a smooth convex quadratic?
- 70m: [The Ultra-Scale Playbook: precision](https://huggingface.co/spaces/nanotron/ultrascale-playbook)
  — Hugging Face, illustrated systems guide. Prerequisite: exponent and fraction
  bits. Read “Mixed Precision Training” and “FP16 and BF16 training”; stop before
  FP8. Purpose: connect range, rounding, loss scaling, and higher-precision state.
  **Check:** what information loss cannot be repaired by increasing the loss scale?
- 70m: [Automatic Mixed Precision examples](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html)
  — PyTorch Contributors, official documentation. Prerequisite: the precision
  guide. Read “Typical Mixed Precision Training,” “Working with Unscaled Gradients,”
  and “Gradient clipping,” then inspect the parameter and return contracts for
  [`clip_grad_norm_`](https://docs.pytorch.org/docs/2.14/generated/torch.nn.utils.clip_grad_norm_.html).
  Purpose: establish unscaled global-norm and skipped-step semantics.
  **Check:** why does clipping scaled gradients silently change the requested threshold?
- 55m: [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
  — Kingma & Ba, primary-source excerpt. Prerequisite: momentum and exponential
  averages. Read §2, §§2.1–2.2, and Algorithm 1 only. Purpose: identify the two
  histories, time index, and initialization bias.
  **Check:** why are both uncorrected moment estimates biased toward zero?
- 55m: [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
  — Loshchilov & Hutter, primary-source excerpt. Prerequisite: Adam's state.
  Read §2, focusing on Propositions 1–2 and Algorithms 1–2. Purpose: locate where
  L2 regularization and isotropic decay diverge under adaptive scaling.
  **Check:** which operation destroys their SGD equivalence for Adam?
- 45m: [PyTorch AdamW](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html)
  — PyTorch Contributors, official API reference. Prerequisite: the two paper
  excerpts. Read the displayed algorithm, parameter definitions, `state_dict`,
  and `zero_grad`; skip hooks and performance modes. Purpose: connect equations
  to serialization, skipped gradients, and `grad=None`.
  **Check:** why is a missing gradient behaviorally different from a tensor of zeros?
- 85m: Active synthesis — learner-created state and precision ledger. Prerequisite:
  all preceding resources. Create a one-page transition table for finite, zero,
  missing, and nonfinite gradients in a multi-parameter step; separately compare
  values that expose FP16 overflow and BF16 rounding. Purpose: integrate clipping,
  scaling, state, and atomic rejection without writing implementation code.
  **Check:** after overflow, which parameter, moment, and counter values must remain identical?

### Target implementation drill (2h)

- `AdamW(torch.optim.Optimizer)`: dense real parameters, FP32 or FP64 state matching
  parameter dtype, independent parameter steps, no AMSGrad/foreach/fused mode.
  FP32 master parameters are the mixed-precision integration choice; implementing
  a master-copy optimizer is outside this component's scope.
- `unscale_clip_(grads, scale, max_norm)`: return `(preclip_norm, found_inf)`;
  finite lists are unscaled/clipped in place with FP64 norm accumulation. On any
  nonfinite input, return `(inf, True)` and leave **all** gradients untouched.
  Empty lists yield `(0, False)`; positive scale and nonnegative norm cap required.

Budget: 10m contract review, 40m AdamW, 25m precision utility, 45m tests.
SGD and Adam are derivation/reference comparisons, not additional implementations.

### PyTest verification target

`python -m pytest tests/test_week01.py -q`: multi-step parameter and moment parity
with `torch.optim.AdamW` (`foreach=False`), skipped gradients, decoupled decay on
zero gradient, state serialization, scale invariance, exact global clipping,
atomic overflow rejection, empty gradients, and FP16/BF16 range demonstrations.
No optimizer `gradcheck`: optimizer state transitions are checked directly.

### The “Grill Yourself” screen

1. When is L2 regularization equivalent to weight decay for SGD? Which step of
   the equivalence fails for Adam, even if both use the same scalar λ?
2. Why must an overflow skip moments and step counters as well as parameters?
   What happens if a parameter has `grad=None` for three steps?
3. Why can BF16 represent a value FP16 overflows on yet represent nearby values
   less accurately? Where should norm accumulation and optimizer state live?
4. What changes if you clip scaled gradients or clip each parameter separately?
   Does lowering the loss scale always repair a NaN in the forward pass?

**Checkpoint:** [ ] Explain the update equation and show passing state/precision
tests. Record one FP16 overflow and one BF16 rounding observation.

## Week 2: Normalization & positional encodings

**Time:** guided study 8h; implementation 2h; self-grill 2h.

### Core concepts

1. BatchNorm normalizes per channel over batch/spatial axes and tracks running
   statistics; its training biased variance and running-stat variance conventions
   differ. LayerNorm normalizes a token's feature axes and uses
   `(x−μ)/sqrt(var+ε)`. RMSNorm omits centering:
   `y = γ ⊙ x / sqrt(mean(x²)+ε)`. Derive which reductions appear in each VJP.
2. For RMSNorm, `r=(mean(x²)+ε)^(−1/2)` and upstream `u`,
   `dx = r(γ⊙u) − x r³ mean(x⊙γ⊙u)`; `dγ` sums `u⊙x⊙r` over leading axes.
   Epsilon placement and FP32 accumulation matter at tiny variance.
3. Absolute sinusoidal encodings add a position vector. RoPE rotates adjacent
   feature pairs by `pω_j`, `ω_j=base^(−2j/D)`; in complex form,
   `(x_(2j)+i x_(2j+1)) exp(i pω_j)`. Its dot product depends on relative position.

### Reading & guided study (8h total)

- 35m: [RoPE: Understanding Rotary Positional Embeddings in transformers](https://www.youtube.com/watch?v=jlGf2qieSk0)
  — Hugging Face, visual video. Watch **00:40–04:51**, **06:45–08:15**,
  **08:15–10:58**, **10:58–15:08**, and **15:08–17:30**, with pauses and notes.
  Prerequisite: embeddings and dot products. Purpose: see position as paired-axis
  rotation before reading its derivation.
  **Check:** which angle remains after a dot product at positions `p` and `q`?
- 60m: [Dive into Deep Learning: residual connection and layer normalization](https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html#residual-connection-and-layer-normalization)
  — Zhang, Lipton, Li & Smola, textbook explanation. Prerequisite: means,
  variances, and tensor axes. Read §11.7.3's explanation and normalization example;
  skip implementation listings. Purpose: make batch coupling and feature-axis
  normalization concrete. **Check:** for `[B,T,D]`, which reductions couple
  examples, tokens, or only features?
- 45m: [The Illustrated Transformer: positions](https://jalammar.github.io/illustrated-transformer/)
  — Jay Alammar, visual article. Prerequisite: embeddings. Read “Representing The
  Order of The Sequence Using Positional Encoding” and “The Residuals.” Purpose:
  establish additive sinusoidal positions and post-norm as a baseline for contrast.
  **Check:** what ambiguity remains if identical token embeddings have no position signal?
- 65m: [PyTorch normalization modules](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BatchNorm1d.html)
  — PyTorch Contributors, official API references. Prerequisite: the textbook
  comparison. Read the definitions, epsilon/variance statements, parameters, and
  shapes for `BatchNorm1d`, [`LayerNorm`](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LayerNorm.html),
  and [`RMSNorm`](https://docs.pytorch.org/docs/2.14/generated/torch.nn.RMSNorm.html);
  skip examples and methods. Purpose: compare axes, affine state, and train/eval behavior.
  **Check:** which module changes its source of statistics between training and evaluation?
- 60m: [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
  — Zhang & Sennrich, primary-source excerpt. Prerequisite: the normalization
  comparisons. Read §4, §§4.1–4.2, and stop before §5. Purpose: isolate the effect
  of removing centering and inspect the shared RMS term in gradients.
  **Check:** is RMSNorm invariant to adding the same constant to every feature?
- 70m: [RoFormer](https://arxiv.org/abs/2104.09864) — Su et al., primary-source
  excerpt. Prerequisite: the RoPE video and 2D rotation matrices. Read §3.1 and
  §§3.2.1–3.2.2; stop before §3.3. Purpose: derive the relative-position identity
  and adjacent-pair convention. **Check:** why does orthogonality preserve pair norm?
- 55m: [Numerical gradient checking](https://docs.pytorch.org/docs/2.14/autograd.html#torch.autograd.gradcheck)
  — PyTorch Contributors, official documentation. Prerequisite: chain rule and
  custom autograd functions. Read “Function” and “Numerical gradient checking,”
  focusing on `gradcheck`. Purpose: understand double precision, differentiable
  inputs, and finite-difference limitations.
  **Check:** why can a correct backward fail a finite-difference check in low precision?
- 90m: Active synthesis — learner-created normalization and rotation derivation.
  Prerequisite: all preceding resources. Label every reduction axis for BatchNorm,
  LayerNorm, and last-axis RMSNorm on `[B,T,D]`; derive the RMSNorm VJP with one
  upstream vector; draw a RoPE pair at `p` and `q` and analyze a nonzero cache
  offset. Purpose: join shape reasoning, shared gradients, epsilon, and offsets.
  **Check:** which invariant would detect rotating cached keys a second time?

### Target implementation drill (2h)

- `RMSNormFunction`: custom `torch.autograd.Function` forward/backward, last-axis
  normalization, learnable scale, no bias. Keep float64 for gradcheck; use at least
  FP32 reductions for FP16/BF16 inputs, return the input dtype.
- `apply_rope(x, positions, base)`: adjacent pairs, `x[B,H,T,D]`, positions `[T]`,
  even `D`; support nonzero cache offsets and negative positions for inversion.

Budget: 10m contract review, 50m backward, 25m RoPE, 35m testing. BatchNorm and
LayerNorm backwards are whiteboard comparisons, not two more coding exercises.

### PyTest verification target

`python -m pytest tests/test_week02.py -q`: PyTorch RMSNorm forward parity,
float64 `gradcheck` including scale, zeros/tiny inputs, noncontiguous input,
RoPE norm preservation, inverse/zero rotation, relative-position identity,
and RoPE input `gradcheck`. Odd head width must be rejected.

### The “Grill Yourself” screen

1. Why can BatchNorm behave badly for autoregressive inference or tiny batches?
   Which normalization statistics couple tokens or examples?
2. Which term disappears from the LayerNorm VJP when centering is removed?
   Is RMSNorm invariant to a constant feature shift? To rescaling with ε > 0?
3. Derive `R(p)ᵀR(q)=R(q−p)`. Why does rotating cached keys again corrupt decode?
4. How does ε change the gradient near the zero vector? What fails if a low
   precision square/reduction overflows before taking the reciprocal root?

**Checkpoint:** [ ] Derive the VJP and explain both RoPE position conventions.

## Week 3: Attention mechanics & memory profiling

**Time:** guided study 8h; implementation 2h; self-grill 2h.

### Core concepts

1. `softmax(QKᵀ/sqrt(d)+mask)V` with query heads `Hq` and KV heads `Hkv`.
   MHA has `Hkv=Hq`; MQA has `Hkv=1`; GQA has `1 < Hkv < Hq` with contiguous
   groups of `Hq/Hkv` queries. GQA reduces KV storage, not the number of Q heads.
2. Static KV allocation per layer is `2 B Hkv T_max d s` bytes; across `L`
   layers multiply by `L`. Distinguish allocated capacity from valid sequence
   length, and logical tensor storage from temporary attention scores.
3. FlashAttention trades recomputation and tiling for fewer HBM transfers;
   it is exact attention modulo floating-point ordering, not sparse attention.
   For a new score tile `S`, running maximum `m`, normalizer `ℓ`, and unnormalized
   value numerator `z`: `m′=max(m,max S)`,
   `ℓ′=exp(m−m′)ℓ+Σ exp(S−m′)`,
   `z′=exp(m−m′)z+Σ exp(S−m′)v`; final output `z′/ℓ′`.
   Fully masked rows need an explicit zero-output convention.

### Reading & guided study (8h total)

- 45m: [Attention in transformers, step-by-step](https://www.youtube.com/watch?v=eMlx5fFNoYc)
  — Grant Sanderson / 3Blue1Brown, visual video; **00:00–26:09**, plus pauses and
  notes. Prerequisite: matrix multiplication and softmax. Alternatively read the
  [text adaptation](https://www.3blue1brown.com/lessons/attention/), “Attention”
  through the multi-head explanation. Purpose: give Q, K, V, heads, and masks a
  visual meaning. **Check:** which score-matrix axis normalizes for each query?
- 65m: [All About Transformer Inference](https://jax-ml.github.io/scaling-book/inference/)
  — Austin et al. / Google DeepMind, illustrated systems chapter. Prerequisite:
  the attention video. Read “The Basics of Transformer Inference,” “What about
  memory?,” and the grouped multi-query attention passage under “Tricks for
  Improving Generation Throughput and Latency.” Purpose: distinguish prefill,
  generation, cache storage, and bandwidth.
  **Check:** what work and storage are reused when one token extends a prefix?
- 55m: [PyTorch scaled dot-product attention](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)
  — PyTorch Contributors, official API documentation. Prerequisite: the attention
  model. Read mask semantics, GQA constraints, parameters, and shape legend; then
  compare `key_padding_mask`, `attn_mask`, and `is_causal` in
  [`MultiheadAttention.forward`](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MultiheadAttention.html).
  Purpose: expose differing boolean-mask conventions and rectangular shapes.
  **Check:** what does `True` mean in each of the two mask APIs?
- 60m: [GQA](https://arxiv.org/abs/2305.13245) — Ainslie et al., primary-source
  excerpt. Prerequisite: Q/K/V heads and the cache introduction. Read §2.2 and
  Figure 2; use the abstract only for context. Purpose: make MHA, MQA, and GQA
  sharing precise while retaining all query heads.
  **Check:** if `Hkv` is halved while `Hq` is unchanged, which persistent bytes halve?
- 75m: [FlashAttention](https://arxiv.org/abs/2205.14135) — Dao et al.,
  primary-source excerpt. Prerequisite: stable softmax and memory hierarchy.
  Read §§2.1–2.2, §3.1, and the opening IO comparison in §3.2; use Algorithm 1
  only to identify the running maximum and normalizer. Purpose: separate exact
  attention from its IO schedule. **Check:** why is it exact without materializing
  the full score matrix in HBM?
- 65m: [CUDA memory management](https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-memory-management)
  — PyTorch Contributors, official documentation. Prerequisite: tensor byte
  accounting. Read “Asynchronous execution” and “Memory management,” including
  `memory_allocated`, `max_memory_allocated`, `memory_reserved`, and synchronization.
  Purpose: distinguish live allocation, reservation, logical bytes, and peaks.
  **Check:** why may reserved memory remain high after live tensors are released?
- 55m: Active synthesis — offset masks and online softmax. Prerequisite: attention,
  masks, and FlashAttention. Draw allowed positions for a two-query chunk beginning
  at offset three against five cached keys; analyze a fully masked row; manually
  merge two score tiles while tracking only maximum, normalizer, and value numerator.
  Purpose: connect offset masking, zero-output policy, and stable online softmax.
  **Check:** what must be rescaled when a later tile raises the running maximum?
- 60m: Active synthesis — memory ledger. Prerequisite: the inference and CUDA
  readings. Build symbolic and numeric ledgers separating cache capacity,
  valid-prefix bytes, score temporaries, live allocation, and reservation; include
  storage identity before and after append/reset. Purpose: prevent logical byte
  formulas from being confused with allocator measurements. **Check:** which quantities can
  change after append, and which cache-storage identity must not?

### Target implementation drill (2h)

- `gqa(q,k,v,allowed=None)`: stable, explicit attention; shapes
  `[B,Hq,Tq,d]`, `[B,Hkv,Tk,d]`. Boolean mask `[Tq,Tk]` uses **True = allowed**.
  No mask means bidirectional attention. Fully masked rows return zero without
  NaNs in forward/backward. Do not use SDPA in the learner implementation.
- `StaticKVCache`: preallocate K/V `[B,Hkv,capacity,d]` once; append detached
  inference tensors, expose valid-prefix views, reset length without reallocating.
  Overflow and incompatible shapes must be rejected without mutation.

Budget: 10m contract review, 40m attention, 25m cache, 35m tests, 10m memory
evidence. Use the byte formula on CPU; with CUDA, record synchronized peak
allocation instead. Never add time or require a fused kernel.

### PyTest verification target

`python -m pytest tests/test_week03.py -q`: MHA/MQA/GQA parity with PyTorch SDPA
(KV repetition allowed in the test reference), causal and fully masked rows,
float64 `gradcheck`, future-token isolation, chunked cached vs. full-prefix
outputs at nonzero offsets, unchanged cache storage pointer and exact bytes,
reset and overflow atomicity. Optional CUDA test measures allocation around a
cache append. No performance comparison is a pass/fail gate.

### The “Grill Yourself” screen

1. Derive KV bytes and attention-score bytes for prefill and one-token decode.
   Why does GQA improve decode bandwidth without equally shrinking all FLOPs?
2. Why is a `[Tq,Tk]` lower-triangular mask wrong for an offset query chunk?
3. Show that the online softmax merge preserves normalization. What goes wrong
   with a running sum of `exp(scores)` without a rescaled maximum?
4. Why can an allocation profiler report a larger peak than the logical cache?
   How do live allocation, reservation, and profiler synchronization differ?

**Checkpoint:** [ ] Cached/full parity and a byte ledger separating capacity,
valid data, score temporaries, and allocator overhead.

## Week 4: Modern transformer blocks & routing

**Time:** guided study 8h; implementation 2h; self-grill 2h.

### Core concepts

1. `SwiGLU(x) = (SiLU(xWg) ⊙ xWu)Wd`, `SiLU(z)=z sigmoid(z)`.
   Compare three projection matrices with a two-matrix GELU FFN; equal parameter
   budgets imply a different hidden width (approximately 2/3 of the GELU width).
2. Router logits → full softmax probabilities → deterministic top-k selection.
   Normalize selected weights to sum to one *before* capacity dropping.
3. Capacity and auxiliary loss are distinct: `C=ceil(capacity_factor·N·k/E)`;
   define selected assignment fraction `f_e=count_e/(Nk)` and mean full-softmax
   probability `P_e`. Use `L_aux=E Σ_e f_e P_e`, with stop-gradient through `f`.
   For k=1 this is Switch's convention; for k>1 it is this course's explicit
   extension. Count assignments before dropping; uniform routing has loss 1.

### Reading & guided study (8h total)

- 40m: [How might LLMs store facts](https://www.youtube.com/watch?v=9-Jl0dxWQs8)
  — Grant Sanderson / 3Blue1Brown, visual video; **00:00–22:42**, plus pauses and
  notes. Prerequisite: linear maps and Week 3 attention. Alternatively read the
  [text adaptation](https://www.3blue1brown.com/lessons/mlp/), “Where are facts in
  LLMs stored?” through “Superposition.” Purpose: establish the tokenwise MLP
  before gates and experts; the video's ReLU example is illustrative.
  **Check:** which operation mixes tokens, and which operates on each token independently?
- 50m: [PyTorch SiLU](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.silu.html)
  — PyTorch Contributors, official API references. Prerequisite: tensor axes and
  sigmoid. Read the definitions and parameter/return contracts for `silu`,
  [`softmax`](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.softmax.html),
  and [`topk`](https://docs.pytorch.org/docs/2.14/generated/torch.topk.html).
  Purpose: connect the router pipeline to precise tensor operations.
  **Check:** which course tie-breaking requirement is not guaranteed by `torch.topk`?
- 70m: [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
  — Shazeer, primary-source excerpt. Prerequisite: the MLP video and SiLU.
  Read §§1–2 and §3.1. Purpose: derive SwiGLU's three projections and the
  parameter-matched hidden-width relationship.
  **Check:** how do two-projection and three-projection FFN parameter counts compare?
- 80m: [Mixture of Experts Explained](https://huggingface.co/blog/moe)
  — Sanseviero et al. / Hugging Face, illustrated tutorial. Prerequisite: tokenwise
  FFNs. Read “What is a Mixture of Experts (MoE)?,” “What is Sparsity?,” “Load
  balancing tokens for MoEs,” “Switch Transformers,” and “Capacity Factor and
  communication costs.” Purpose: build routing, overload, and sparse-compute intuition.
  **Check:** why can expert count grow without proportional per-token compute?
- 110m: [Switch Transformers](https://arxiv.org/abs/2101.03961) — Fedus et al.,
  primary-source excerpt. Prerequisite: the MoE tutorial. Read §§2.1–2.3,
  concentrating on equations (1)–(6) and Figures 2–3. Purpose: ground capacity
  and balancing in the original top-1 formulation.
  **Check:** which auxiliary-loss factor is differentiable, and why does balanced
  mean routing not guarantee zero drops?
- 70m: Active synthesis — parameter-budget comparison. Prerequisite: the MLP,
  SwiGLU, and MoE resources. Build one page comparing GELU FFN, SwiGLU, dense FFN,
  and sparse MoE by projection count, active parameters, total parameters, and
  token mixing. Purpose: separate architecture size from active compute.
  **Check:** can every width or compute relationship be justified by one equation?
- 60m: Active synthesis — routing-convention worksheet. Prerequisite: capacity
  and auxiliary-loss equations. Hand-trace a small top-`k` table using full softmax,
  selected-weight normalization, deterministic order, pre-drop counts, capacity,
  and no post-drop renormalization. Purpose: reconcile Switch top-1 with the
  course's explicit top-`k` extension. **Check:** exactly which assignments survive?

### Target implementation drill (2h)

- `SwiGLU(d_model, hidden)`: bias-free `gate`, `up`, `down` linear modules, whose
  names are part of the test contract. No residual or complete block yet.
- `route_topk(logits,k,capacity_factor)`: return `indices[N,k]`, selected
  `weights[N,k]`, `keep[N,k]`, scalar auxiliary loss. Lower expert ID wins ties;
  process assignments in token-major, then slot-major order for capacity. Dropped
  weights become zero without renormalizing surviving weights. No expert dispatch,
  distributed all-to-all, or MoE model implementation this week.

Budget: 10m contract review, 20m SwiGLU, 45m router, 45m tests/capacity examples.

### PyTest verification target

`python -m pytest tests/test_week04.py -q`: SwiGLU constant-weight example and
input `gradcheck`, module parameter registration, top-k normalization without
drops, ties/capacity/drop-order fixture, uniform auxiliary baseline, overloaded
auxiliary penalty, finite router gradients, and invalid k/capacity rejection.
Selection is discrete: no `gradcheck` across a tie or top-k boundary.

### The “Grill Yourself” screen

1. What is `d SiLU(z)/dz`, and why can a gate attenuate a channel without setting
   it exactly to zero? How do you parameter-match SwiGLU and GELU FFNs?
2. Why can every token choosing the same expert remain locally attractive to
   the router? Which terms of the balancing loss carry gradients?
3. How do changing k and capacity factor affect compute, drops, and communication?
   Is a low auxiliary loss sufficient to guarantee no dropped assignments?
4. What changes if retained routing weights are renormalized after dropping?
   How would token order then affect outputs and reproducibility?

**Checkpoint:** [ ] Explain the routing conventions and predict an overloaded
expert's exact retained assignments before running the fixture.

## Week 5: Post-training & alignment foundations

**Time:** guided study 8h; implementation 2h; self-grill 2h.

### Core concepts

1. SFT is next-token conditional maximum likelihood:
   `L=−Σ m_(b,t) log π(y_(b,t)|x_≤t) / Σ m_(b,t)`.
   Shift logits/labels once, exclude prompt/padding from the loss, but preserve
   prompt context. Token-mean training NLL differs from sequence-summed DPO scores.
2. RLHF/PPO uses actor, frozen reference, reward model, and learned value critic;
   rollouts, KL penalties, advantage estimation, and the clipped surrogate add
   different memory/lifecycle costs. The critic estimates return, not preference.
3. From KL-regularized reward maximization,
   `π*(y|x) ∝ π_ref(y|x) exp(r(x,y)/β)`.
   Substitute `r=β log(π*/π_ref)+β log Z(x)` into Bradley–Terry preferences;
   the prompt partition function cancels. DPO becomes
   `−log σ(β[(logπ_w−logπ_l)−(logπref_w−logπref_l)])`.
   This is binary cross-entropy on a **pairwise log-ratio margin**, not ordinary
   independent cross-entropy on two chosen/rejected labels. β is the reference
   regularization parameter and also scales this margin.

### Reading & guided study (8h total)

- 45m: [Post-Training and RLHF Overview](https://www.youtube.com/watch?v=o6l6tJQgUg4)
  — Nathan Lambert, lecture video; **00:00–23:51**, plus pauses and notes.
  Prerequisite: next-token likelihood. Purpose: map SFT, preference data, reward
  models, PPO, and direct alignment before formal objectives.
  **Check:** what distinct supervision is available at each training stage?
- 65m: [RLHF Book: Training Overview](https://rlhfbook.com/c/03-training-overview)
  — Nathan Lambert, readable chapter. Prerequisite: the overview video. Read
  “Fine-Tuning and Regularization,” “Optimization Tools,” and “InstructGPT:
  Foundational RLHF Tools,” including pipeline figures. Purpose: identify model
  roles, KL regularization, and stage dependencies.
  **Check:** which stages consume demonstrations, comparisons, and unlabeled prompts?
- 60m: [RLHF Book: Instruction Fine-Tuning](https://rlhfbook.com/c/04-instruction-tuning)
  — Nathan Lambert, practical chapter. Prerequisite: autoregressive cross-entropy.
  Read “Chat Templates and the Structure of Instructions” and “Implementation
  Details,” especially prompt and multi-turn masking; skip code. Purpose: connect
  SFT to response-only token-mean NLL.
  **Check:** why does excluding prompt tokens from loss not remove their context?
- 70m: [Proximal Policy Optimization](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
  — OpenAI Spinning Up, algorithm explainer. Prerequisite: policy probabilities
  and expected returns. Read “Background,” “Quick Facts,” “Key Equations,” and
  “Pseudocode”; skip framework details. Purpose: understand behavior ratios,
  clipping, actor, and value function before RLHF's additional models.
  **Check:** for each advantage sign, where does clipping remove further incentive?
- 55m: [InstructGPT](https://arxiv.org/abs/2203.02155) — Ouyang et al.,
  primary-source excerpt. Prerequisite: the PPO overview. Study Figure 2, §3.1,
  and the SFT/RM/RL paragraphs in §3.5. Purpose: locate actor, frozen reference,
  reward model, and critic in a concrete pipeline.
  **Check:** is PPO's ratio denominator necessarily the frozen KL reference policy?
- 65m: [RLHF Book: Direct-Alignment Algorithms](https://rlhfbook.com/c/08-direct-alignment)
  — Nathan Lambert, mathematical tutorial. Prerequisite: KL regularization and
  sequence log probabilities. Read “Direct Preference Optimization,” “How DPO
  Works,” and the derivation through “Deriving DPO Objectives for BT Models.”
  Purpose: introduce the optimal-policy substitution before the original paper.
  **Check:** why must both responses share a prompt for the partition term to cancel?
- 75m: [DPO](https://arxiv.org/abs/2305.18290) — Rafailov et al., primary-source
  excerpt. Prerequisite: the mathematical tutorial. Read §§3–4, focusing on
  equations (3)–(7); use §5.1 only to clarify reward equivalence. Purpose: connect
  Bradley–Terry preferences to pairwise policy/reference classification.
  **Check:** why is this not independent cross-entropy on two response labels?
- 45m: Active synthesis — objective and lifecycle worksheet. Prerequisite: all
  preceding resources. Draw an SFT/PPO/DPO model-lifecycle table, then rederive
  the DPO pairwise logit while annotating prompt masking, response scores, sequence
  sums, reference detachment, and `β`. Purpose: reconcile mathematical objectives
  with their data, reduction, and memory contracts. **Check:** where is token averaging
  appropriate, and where is sequence summation required?

### Target implementation drill (2h)

- `masked_nll(logits,targets,mask)`: logits `[B,T,V]`, valid integer targets,
  boolean mask `[B,T]`; inputs are **already shifted**. All-masked batches raise
  `ValueError` rather than silently dividing by zero.
- `dpo_loss(policy_chosen,policy_rejected,ref_chosen,ref_rejected,beta)`:
  inputs `[B]` are response-only **summed** sequence log probabilities. Return
  mean stable logistic loss. Reference arguments must never receive gradients.
  No policy model, reward model, rollout loop, or PPO implementation this week.

Budget: 10m contract review, 25m masked loss, 35m DPO, 50m tests and saturation
analysis. Sequence scoring is capstone integration work, not a third weekly stub.

### PyTest verification target

`python -m pytest tests/test_week05.py -q`: masked loss parity with PyTorch CE on
valid positions, zero gradients at masked positions, all-masked rejection,
shift-contract fixture, DPO equality baseline `log(2)`, stable extreme margins,
chosen/rejected gradient signs, reference detachment, reference-margin shift
invariance, and float64 policy-score `gradcheck`.

### The “Grill Yourself” screen

1. Why is excluding prompt tokens from the loss different from hiding the prompt
   in the attention mask? How can double-shifting labels look deceptively valid?
2. Which four model roles are live during PPO training? Which are frozen, which
   need optimizer states, and why doesn't DPO need a critic?
3. Derive the DPO loss from the optimal KL-regularized policy. What assumption
   lets the partition terms cancel, and why must the two responses share a prompt?
4. Why do summed and averaged response log probabilities yield different DPO
   objectives? What do β, unequal lengths, and saturated margins do to gradients?

**Checkpoint:** [ ] Derive the pairwise logit and explain exactly which tokens
contribute to each objective.

## Week 6: Contrastive representation & retrieval foundations

**Time:** guided study 8h; implementation 2h; self-grill 2h.

### Core concepts

1. For paired normalized embeddings, `s_ij=(a_i·b_j)/τ` and
   `L_a=−mean_i log(exp(s_ii)/Σ_j exp(s_ij))`; symmetric InfoNCE averages
   the row and column classification losses. Positive alignment competes with
   discrimination against in-batch negatives.
2. L2 normalization separates angular similarity from norm inflation;
   `dL/ds` and the `1/τ` factor control gradient concentration. Small τ can
   emphasize a few hard pairs while stressing precision.
3. Hard-negative mining ranks nonmatching candidates, but semantic false
   negatives and stale embeddings can damage retrieval. A mined index is
   nondifferentiable selection, not a differentiable softmax distribution.

### Reading & guided study (8h total)

- 45m: [Intro to Dense Vectors for NLP and Vision](https://www.youtube.com/watch?v=bVZJ_O_-0RE)
  — James Briggs, visual video; **00:00–33:22**, plus pauses and notes.
  Prerequisite: vectors and dot products. Purpose: build geometric intuition for
  dense embeddings and similarity retrieval.
  **Check:** what information must encoders preserve for nearest-neighbor retrieval?
- 80m: [Contrastive Representation Learning](https://lilianweng.github.io/posts/2021-05-31-contrastive/)
  — Lilian Weng, illustrated tutorial. Prerequisite: softmax cross-entropy. Read
  “InfoNCE,” “Common Setup,” “Large Batch Size,” “Hard Negative Mining,” and
  “CLIP.” Purpose: connect positives, negatives, normalization, temperature, and
  false negatives. **Check:** why can a high-scoring negative be useful or harmful?
- 55m: [CLIP: Connecting text and images](https://openai.com/index/clip/)
  — OpenAI, visual research explainer. Prerequisite: the contrastive tutorial.
  Read “Approach,” “Key takeaways,” and “Limitations.” Purpose: see a paired-modality
  retrieval problem before formalizing its symmetric loss.
  **Check:** in an `N×N` similarity matrix, what do diagonal and off-diagonal entries mean?
- 55m: [Retrieve & Re-Rank](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)
  — Sentence Transformers maintainers, practical documentation. Prerequisite:
  dense similarity. Read “Retrieve & Re-Rank Pipeline,” “Retrieval: Bi-Encoder,”
  and “Re-Ranker: Cross-Encoder”; skip scripts. Purpose: distinguish cached
  embedding retrieval from expensive joint scoring.
  **Check:** why can a bi-encoder cache candidates while a cross-encoder cannot?
- 70m: [CPC](https://arxiv.org/abs/1807.03748) — van den Oord et al.,
  primary-source excerpt. Prerequisite: InfoNCE intuition. Read §§2.1–2.3.
  Purpose: identify positive-index classification and the sampling assumptions
  behind the mutual-information interpretation.
  **Check:** which assumption is strained by a semantically equivalent negative?
- 80m: [CLIP](https://arxiv.org/abs/2103.00020) — Radford et al., primary-source
  excerpt. Prerequisite: CPC and the CLIP overview. Read §2.3, Figure 3, and §2.5.
  Purpose: identify both classification directions, normalized similarities,
  temperature, and large-batch negatives.
  **Check:** why does a row-only loss differ from symmetric CLIP?
- 50m: Active synthesis — loss-matrix derivation. Prerequisite: all objective
  readings. Draw a small paired matrix, mark positives, write row and column
  targets, and trace normalization and temperature. Purpose: connect paper notation
  to the course objective. **Check:** which transformations leave the loss invariant?
- 45m: Active synthesis — retrieval failure cases. Prerequisite: the mining and
  retrieve/rerank readings. Construct a valid hard negative, a semantic false
  negative, and a stale-index example; classify each failure as labels, embeddings,
  or freshness. Purpose: separate useful ranking pressure from data and systems
  failures. **Check:** why is the selected index not a differentiable distribution?

### Target implementation drill (2h)

- `symmetric_infonce(a,b,temperature)`: `[N,D]`, normalize internally with fixed
  ε=1e−12, diagonal positives, stable row/column losses, positive temperature.
- `hard_negative_indices(scores,positive_indices,k)`: `[N,M]` scores, one positive
  index per row; exclude it, return descending top-k candidate indices with
  lower candidate ID breaking ties. `0≤k<M`; k=0 returns `[N,0]`.

Budget: 10m contract review, 35m loss, 30m miner, 45m testing and failure analysis.
Do not build a vector database, dataloader, or retrieval service this week.

### PyTest verification target

`python -m pytest tests/test_week06.py -q`: public CE reference parity,
joint permutation and positive rescaling invariance, paired-vs-shuffled loss,
input `gradcheck`, low-temperature finiteness, zero-vector behavior, positive
exclusion, exact ties, rectangular candidate sets, k=0 and invalid k/temperature.

### The “Grill Yourself” screen

1. Derive the gradient of the InfoNCE logit for its positive and negatives.
   Why does lowering τ not simply multiply every embedding gradient uniformly?
2. Why can unnormalized embeddings reduce loss by increasing their norms?
   What geometry and gradient component does normalization remove?
3. How does increasing the number of negatives change the classification problem
   and the mutual-information bound? When are more negatives harmful?
4. Why can the highest-scoring negative be a bad training example? How would
   duplicates, stale representations, and distributed negative gathering matter?

**Checkpoint:** [ ] Predict the miner fixture and explain a plausible false
negative even though its test labels declare it negative.

## Week 7: Distributed parallelism primitives (TP & PP)

**Time:** guided study 8h; implementation 2h; self-grill 2h.

### Core concepts

1. SUM AllReduce replicates the sum; ReduceScatter partitions the reduced tensor;
   AllGather concatenates shards. AllReduce can be decomposed into
   ReduceScatter + AllGather, but backward scaling follows the *logical tensor
   ownership*, not the name of a collective.
2. Megatron TP partitions output features of a column-parallel linear and input
   features of a row-parallel linear. For `W[out,in]`, column local weight is
   `[out/p,in]`, row local weight `[out,in/p]`. Column input gradients require a
   SUM AllReduce; row forward partial outputs require a SUM AllReduce. The
   replicated downstream logical loss must not be multiplied by world size.
3. Pipeline microbatches trade bubbles for activation residency. For balanced
   stages with equal forward/backward cost and no interleaving, the rough bubble
   fraction is `(p−1)/(m+p−1)` for p stages and m microbatches. 1F1B interleaves
   forward/backward after warmup to reduce saved activations relative to GPipe;
   it does not by itself eliminate fill/drain bubbles. TP×PP×DP is 3D parallelism.

### Reading & guided study (8h total)

- 110m: [CS336 Lecture 7: Parallelism 1](https://www.youtube.com/watch?v=l1RJcDjzK8M)
  — Stanford Online, lecture video; **00:00–1:24:42** in full, with 25m for pauses,
  diagrams, and notes. Prerequisite: transformer forward/backward and matrix
  multiplication. Purpose: establish a broad parallelism model before notation
  and APIs. **Check:** classify data, tensor, and stage parallelism by what crosses ranks.
- 55m: [The Ultra-Scale Playbook: collectives](https://nanotron-ultrascale-playbook.static.hf.space)
  — Tazi et al. / Hugging Face, illustrated systems guide. Prerequisite: rank and
  world-size terminology. Read Appendix A0's “Reduce & AllReduce,” “Gather &
  AllGather,” “Scatter & ReduceScatter,” and “A quick focus on Ring AllReduce.”
  Purpose: visualize replicated results versus owned shards.
  **Check:** for each collective, does every rank receive a replica, one shard, or no result?
- 45m: [PyTorch distributed communication package](https://docs.pytorch.org/docs/2.14/distributed.html)
  — PyTorch Contributors, official API documentation. Prerequisite: the collective
  diagrams. Inspect “Backends” and the entries for `all_reduce`,
  `all_gather_into_tensor`, and `reduce_scatter_tensor`. Purpose: connect ownership
  diagrams to shapes, semantics, and backend support.
  **Check:** for world size `p`, what shape and ownership does each operation return?
- 75m: [The Ultra-Scale Playbook: tensor parallelism](https://nanotron-ultrascale-playbook.static.hf.space)
  — Tazi et al., illustrated systems guide. Prerequisite: matrix partitioning and
  collectives. Read “Tensor Parallelism” and “Tensor Parallelism in a Transformer
  Block,” annotating row/column ownership. Purpose: derive why two partitioned
  linears avoid an intermediate synchronization.
  **Check:** under `W[out,in]`, what local shape and communication belongs to each linear?
- 55m: [Megatron-LM](https://arxiv.org/abs/1909.08053) — Shoeybi et al.,
  primary-source excerpt. Prerequisite: the TP diagrams. Read §3 and Figure 3 only.
  Purpose: validate communication placement against the original design.
  **Check:** why may GeLU remain local after the first partitioned GEMM?
- 65m: [The Ultra-Scale Playbook: pipeline schedules](https://nanotron-ultrascale-playbook.static.hf.space)
  — Tazi et al., visual systems guide. Prerequisite: forward/backward dependencies.
  Read “Pipeline Parallelism,” “Splitting layers on various nodes - All forward,
  all backward,” “One-forward-one-backward and LLama 3.1 schemes,” and “Interleaving
  stages”; stop before “Zero Bubble and DualPipe.” Purpose: compare bubbles,
  activation residency, and communication. **Check:** what does 1F1B change and not change?
- 45m: [Efficient Large-Scale Language Model Training](https://arxiv.org/abs/2104.04473)
  — Narayanan et al., primary-source excerpt. Prerequisite: TP and PP explanations.
  Read §§2.2–2.3 and Figures 2–5. Purpose: ground schedules and TP×PP×DP groups in
  the primary source. **Check:** in TP=2, PP=2, DP=2, what does each group own?
- 30m: Active synthesis — rank map and schedule. Prerequisite: all preceding
  resources. Spend 15m drawing TP=2, PP=2, DP=2 rank groups and 15m drawing a
  two-stage, four-microbatch 1F1B timeline. Purpose: consolidate collective,
  tensor, and stage ownership. **Check:** where are parameters, activations,
  gradients, and the logical loss replicated or sharded?

### Target implementation drill (2h)

- `ColumnParallelLinear(in_features,out_features,group=None)`: bias-free, replicated
  input, local output shard. Expose `.weight` and correct custom communication
  autograd semantics. Require divisible dimensions.
- `RowParallelLinear(in_features,out_features,group=None)`: bias-free, already
  input-sharded, replicated output, local `.weight`. Compose the two around a
  local activation using the supplied two-rank Gloo test harness.

Budget: 10m contract review, 35m column, 35m row, 40m distributed tests. The
guided-study block already contains the required schedule and ownership drawings;
no runtime scheduler or cluster provisioning is assigned.

### PyTest verification target

`python -m pytest tests/test_week07.py -q`: two local CPU ranks; compare full
forward outputs, local weight gradients, replicated input gradients, and local
SGD-updated shards against a single-process dense reference. Use nonidentical
rank shards and nonlinear activation. Each rank differentiates the same logical
loss; explicitly catch a world-size factor error. Verify collective algebra with
SUM AllReduce plus shard/gather; the test emulates ReduceScatter as reduce+slice
on Gloo to avoid backend-specific support requirements. Subprocess timeout is
bounded, so mismatched collectives fail rather than wait indefinitely.

### The “Grill Yourself” screen

1. For each linear, which dimension is sharded, and where is communication
   necessary in forward and backward? Why is blindly differentiating an
   AllReduce implementation likely to double-count a replicated loss?
2. Derive communication volume for SUM AllReduce vs. ReduceScatter + AllGather.
   Which collective returns a replica and which returns one owned shard?
3. Draw warmup, steady-state, and drain for 1F1B. How do unequal stage times or
   too few microbatches invalidate a simple bubble estimate?
4. In TP=2, PP=2, DP=2, assign the eight ranks to groups. Which parameters,
   activations, gradients, and optimizer states are replicated across each group?

**Checkpoint:** [ ] Two-rank gradient parity plus a labeled 1F1B diagram and
eight-rank 3D ownership sketch. Both sketches are within guided-study time.

## Week 8: Memory partitioning & inference optimization

**Time:** guided study 8h; implementation 2h; self-grill 2h.

### Core concepts

1. For P parameters, DP degree d, BF16 working parameters/gradients, FP32 master
   parameters, and FP32 Adam m/v, the idealized persistent per-rank bytes are:
   ZeRO-0 `16P`; ZeRO-1 `4P+12P/d`; ZeRO-2 `2P+14P/d`; ZeRO-3 `16P/d`.
   Master weights belong to the optimizer partition in this accounting.
   Activations, buckets, padding, gathered layers, caches, and allocator slack
   are **excluded**; actual peak ZeRO-3 memory is larger than its persistent state.
2. Exact speculative decoding proposes x from draft q, accepts with probability
   `min(1,p(x)/q(x))`, and on rejection samples from
   `r(v)=(p(v)−q(v))_+ / Σ_u(p(u)−q(u))_+`.
   Both distributions must describe the same prefix and sampling policy. For a
   multi-token draft, stop at first rejection; if all are accepted, draw a bonus
   target token. Only the one-step distribution primitive is implemented here.
3. Decode kernels often spend time moving weights/KV data; prefill is more
   compute-dense. Host pinning enables asynchronous DMA but is not sufficient
   for overlap: streams, dependencies, lifetimes, and useful concurrent work
   matter. Avoid `.item()`/synchronization in a measured hot path.

### Reading & guided study (8h total)

- 65m: [The Ultra-Scale Playbook: ZeRO](https://nanotron-ultrascale-playbook.static.hf.space)
  — Tazi et al. / Hugging Face, illustrated systems guide. Prerequisite: Week 7
  ownership and Adam state. Read “ZeRO,” “Memory usage revisited,” and the
  ZeRO-1/2/3 partitioning sections. Purpose: picture which persistent state each
  stage partitions. **Check:** at each stage, what remains replicated or partitioned?
- 50m: [DeepSpeed ZeRO tutorial](https://www.deepspeed.ai/tutorials/zero/)
  — Microsoft DeepSpeed Team, practical documentation. Prerequisite: the ZeRO
  diagrams. Read “ZeRO Overview,” “Enabling ZeRO Optimization,” both GPT-2 training
  examples, and only the introduction to ZeRO-Infinity; stop before CPU/NVMe
  offload details. Purpose: connect stage semantics to configuration without
  expanding scope. **Check:** which stage first partitions gradients and parameters?
- 75m: [ZeRO](https://arxiv.org/abs/1910.02054) — Rajbhandari et al., primary-source
  excerpt. Prerequisite: mixed-precision dtype sizes and collective semantics.
  Read §3.1, §4.1, §§5.1–5.4, §§7.1–7.2, Figure 1, and Table 1. Purpose: derive
  persistent-state savings and communication tradeoffs.
  **Check:** which source assumption differs from the course's byte ledger?
- 25m: [Faster LLMs: Accelerate Inference with Speculative Decoding](https://www.youtube.com/watch?v=VkWlLSTdHs8)
  — IBM Technology, explainer video; **00:00–09:39** in full, with pauses and a
  retrieval check. Prerequisite: autoregressive decoding. Purpose: introduce
  draft-and-verify before the probability argument.
  **Check:** if proposal token two is rejected, which proposed prefix may survive?
- 55m: [All About Transformer Inference](https://jax-ml.github.io/scaling-book/inference/)
  — Austin et al. / Google DeepMind, visual systems guide. Prerequisite: prefill,
  decode, and arithmetic intensity. Read “What do we actually want to optimize?,”
  “Linear operations: what bottlenecks us?,” “What about attention?,” and
  “Appendix D: Speculative Sampling.” Purpose: explain why extra parallel FLOPs
  can reduce bandwidth-bound latency. **Check:** how can more work require fewer
  model-weight and cache reads per accepted token?
- 70m: [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
  — Leviathan et al., primary-source excerpt. Prerequisite: conditional categorical
  distributions. Read §§2.1–2.3, §§3.1 and 3.3, and Appendix A.1. Purpose: supply
  the exact correction absent from greedy explanations.
  **Check:** what accepted and corrected mass must sum to target distribution `p`?
- 85m: [PyTorch pin-memory and non-blocking transfer tutorial](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)
  — Vincent Moens / PyTorch, illustrated practical tutorial. Prerequisite:
  host/device memory and CUDA streams. Read “Background,” both memory sections,
  “Asynchronous vs. Synchronous Operations,” “Other copy directions,” and
  “Practical recommendations.” Purpose: distinguish host asynchrony from actual
  copy/compute overlap. **Check:** what three conditions enable overlap, and which
  copy direction requires synchronization before host access?
- 55m: Active synthesis — memory, sampling, and transfer ledgers. Prerequisite:
  all preceding resources. Derive the stage 0–3 persistent-state ledger, draw
  accepted and corrected probability mass, and draw a pinned double-buffer
  timeline with events and lifetimes. Purpose: integrate persistent memory,
  exact sampling, and transfer critical paths. **Check:** identify excluded peak memory,
  recovered target mass, and the dependency preventing unsafe buffer reuse.

### Target implementation drill (2h)

- `zero_bytes(P,world_size,stage)`: idealized equal sharding, require `P%d=0`,
  return integer component bytes for `parameters`, `gradients`, `master`, `m`,
  `v`, and `total`. State assumptions stay explicit in the return contract.
- `speculative_distribution(p,q)`: normalized nonnegative finite `[V]` tensors;
  return vector acceptance probabilities and the normalized rejection residual.
  Set acceptance to 1 where q=0 (that event is never proposed). If p=q, return
  p as the unused residual distribution, avoiding a divide-by-zero artifact.

Budget: 10m contract review, 30m accounting, 35m distribution stub, 35m tests,
10m transfer observation. The guided-study block already contains the memory
derivation and stream timeline. If CUDA is available, record one synchronized
small-buffer observation; no offload engine, serving loop, or custom kernel is assigned.

### PyTest verification target

`python -m pytest tests/test_week08.py -q`: exact stage-by-stage component byte
counts, d=1 equivalence, sharding monotonicity and invalid input checks; fixed
acceptance/residual examples, p=q, disjoint supports, q(x)=0, and exact enumerated
output law `q*a + (1−Σq*a)*r = p` on many seeded distributions. Enumeration is
stronger and less flaky here than a Monte Carlo acceptance test.

### The “Grill Yourself” screen

1. Derive 16 bytes per parameter from the stated dtypes. How does the formula
   change with FP32 gradients or no FP32 master copy? Why isn't ZeRO-3 peak 16P/d?
2. Prove that accepted mass plus rejection-corrected mass equals p. Why does
   sampling p directly after rejection generally bias the result?
3. What happens at q(x)=0 or p=q? Why can applying different top-p filters to
   draft and verifier invalidate a naive acceptance implementation?
4. Why can `non_blocking=True` fail to improve throughput? Which synchronization
   and tensor-lifetime mistakes turn an apparently overlapped copy into a stall?

**Checkpoint:** [ ] Persistent-state ledger, exact target-distribution recovery,
and a stream timeline identifying the critical path and required events.

## Weeks 9–10: One capstone integration project

### Scope locked before starting

Build a **tiny, runnable decoder mini-library** using prior components. A CPU
reference configuration is vocabulary 32, width 32, two blocks, four query heads,
two KV heads, head width 8, SwiGLU width 64, context ≤32, batch 4, dropout 0,
FP32 parameters. Tests may use still smaller configurations for fast execution.
Use pre-norm residual blocks, final RMSNorm, token embeddings and an untied LM
head. Reuse `RMSNormFunction`, `apply_rope`, `gqa`, `SwiGLU`, `AdamW`,
`masked_nll`, `dpo_loss`, and `StaticKVCache`.

No new infrastructure layer is required: fixed integer token tensors stand in
for a tokenizer and corpus. A deterministic repeating-token corpus is the SFT
gate; a few fixed prompt/chosen/rejected tuples are the preference set. Single
process, single device. MoE, TP/PP, retrieval, ZeRO, and speculative generation
remain isolated verified exercises: forcing all of them into this capstone would
break the four-hour integration budget. Their memory and scheduling lessons inform
the final systems explanation.

Four unsolved components live in `capstone/project.py`:

1. `TinyDecoder`: block construction and causal forward, optional per-layer
   static caches with absolute position offsets. Under 150 implementation lines.
2. `fit_sft`: tiny fixed-batch custom-AdamW runner; under 150 lines.
3. `fit_dpo`: frozen-reference paired-scoring runner; under 150 lines, including
   any response-score helper. Sum only response-token log probabilities.
4. `CacheEngine`: allocate per-layer cache once, prefill then one-token updates,
   reset between requests, greedy generation; under 150 lines.

The supplied tests define input/output contracts. They exercise the mini-library
directly, so no CLI, web API, orchestration stack, logging service, or deployment
manifest is needed for “runnable.” `python -m pytest tests/test_capstone.py -q`
is the end-to-end entry point, with no dataset download or manual checkpoint prep.

## Week 9: Capstone — decoder assembly & SFT

**Time:** guided study 8h; capstone implementation 2h; self-grill 2h.

### Core concepts

1. Pre-norm residual shape/gradient paths, RoPE at each attention layer, and the
   relationship between query width and smaller KV projection width.
2. Teacher forcing, next-token shifting, causal isolation, and reproducible tiny
   overfitting as an integration diagnostic rather than a benchmark.
3. Parameter/moment/activation/cache ownership in the assembled model. Every
   trainable scale and projection must appear in the optimizer's parameter set.

### Reading & guided study (8h total)

- 60m: [Transformers, the tech behind LLMs](https://www.youtube.com/watch?v=wjZofJX0v4M)
  — Grant Sanderson / 3Blue1Brown, visual video; **00:00–27:14** in full, with
  pauses and a token-to-logits sketch. Alternatively read the
  [text adaptation](https://www.3blue1brown.com/lessons/gpt/), “What is a GPT model?”
  through “And That's The Overall Structure.” Prerequisite: Weeks 1–4. Purpose:
  reconnect isolated mechanisms to the full computation.
  **Check:** for logits `[B,T,V]`, what does each axis index?
- 60m: [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
  — Jay Alammar, illustrated article. Prerequisite: the decoder overview. Read
  “A High-Level Look,” “Bringing The Tensors Into The Picture,” “The Residuals,”
  “The Decoder Side,” “The Final Linear and Softmax Layer,” and “Recap Of Training.”
  Purpose: trace residual, causal, and output paths while contrasting its
  encoder-decoder post-norm design with this decoder-only pre-norm capstone.
  **Check:** which pictured decoder sublayer disappears without an encoder?
- 55m: [PyTorch Modules](https://docs.pytorch.org/docs/2.14/notes/modules.html)
  — PyTorch Contributors, official documentation. Prerequisite: Week 1 state and
  basic `nn.Module`. Read “A Simple Custom Module,” “Modules as Building Blocks,”
  “Neural Network Training with Modules,” and “Module State.” Purpose: establish
  why nested projections and scales must register for optimization and reload.
  **Check:** which state and optimization observations expose an unregistered layer?
- 55m: [RLHF Book: Instruction Fine-Tuning](https://rlhfbook.com/c/04-instruction-tuning)
  — Nathan Lambert, readable technical chapter. Prerequisite: Week 5 loss and
  masking. Read the opening, “Best Practices for Instruction Tuning,” and only
  the prompt masking, multi-turn masking, and same-loss bullets under “Implementation
  Details”; skip code. Purpose: connect teacher forcing to response-only masking
  while contrasting this capstone's all-token toy corpus.
  **Check:** how can a token provide context without contributing directly to loss?
- 50m: [LLaMA](https://arxiv.org/abs/2302.13971) — Touvron et al., primary-source
  excerpt. Prerequisite: the architecture overview. Read §2.2 and its pre-norm,
  SwiGLU, and rotary-embedding subsections; stop before §2.3. Purpose: ground
  modern decoder choices while noting the capstone's additional GQA choice.
  **Check:** which choices alter ownership, and which alter operation placement?
- 120m: Active synthesis — architecture and ownership rehearsal. Prerequisite:
  preceding resources and Weeks 1–4 notes. Produce a bounded two-page token-to-logits
  graph, one-block shape ledger, full-versus-offset position ledger, and training/
  inference ownership table for parameters, gradients, moments, activations, and
  cache. Finish with a five-minute oral trace. Purpose: integrate Weeks 1–4 into
  the exact capstone architecture. **Check:** during cached inference,
  which values are reused and which are newly produced?
- 80m: Active synthesis — SFT failure-design rehearsal. Prerequisite: the architecture
  artifact and Weeks 1, 3, and 5. For future leakage, double shifting, unregistered
  parameters, dead gradients, reload divergence, and Q/KV mismatch, record one
  symptom, one invariant, and distinguishing evidence. No code or pseudocode.
  Purpose: tie masking and state behavior to the tiny-overfit integration gate.
  **Check:** which defect can still reduce loss while invalidating autoregression?

### Integration deliverable (2h)

Budget: 10m contract review, 70m `TinyDecoder`, 25m `fit_sft`, 15m tests/overfit
and reload evidence. The guided-study block supplies the shape ledger. Use ordinary
PyTorch `state_dict` APIs; no checkpoint manager component is required.

- Wire `[B,T] → [B,T,V]` logits with per-layer causal attention.
- Reuse custom AdamW and masked SFT loss. `fit_sft` returns initial loss plus one
  post-update evaluation per step, so progress is unambiguous.
- Overfit one small repeating-token batch for at most 40 steps in the supplied
  test. Stop on numerical failure; do not add a dataset or launch a long run.
- Save/reload model and optimizer state locally if demonstrating resume; model
  reload equivalence is in the mandatory integration test and optimizer state
  persistence is already a Week 1 gate.

### PyTest verification target

`python -m pytest tests/test_capstone.py -q -m week9`: logits shapes,
causal invariance, registered finite nonzero gradients, deterministic state reload,
actual reuse of the drill modules, and SFT loss decreasing by at least 30% on
the tiny repeated corpus. The gate is deliberately broad; it is not a throughput
or generalization benchmark.

### The “Grill Yourself” screen

1. Trace all projection and residual shapes for one block. How do four Q heads
   share two KV heads, and where is RoPE applied?
2. How would omitting a parameter from registration manifest in loss curves,
   state dictionaries, and optimizer updates?
3. Why can a shifted-label or causal-mask bug still reduce training loss?
   Which test distinguishes learning from future-token leakage?
4. Separate parameter, gradient, moment, activation, and cache bytes. Which of
   these allocations exists during training vs. inference vs. both?

**Checkpoint:** [ ] Week 9 gates pass, deterministic toy loss trace recorded,
and model/optimizer/activation/cache ownership explained.

## Week 10: Capstone — mini-DPO & static-cache serving

**Time:** guided study 8h; capstone implementation 2h; self-grill 2h.

### Core concepts

1. Clone the post-SFT model as a frozen reference. Score policy/reference on the
   same prompt-response tokens and masks. Only policy weights are optimized.
2. Incremental decoding rotates new keys at their absolute positions and stores
   them once; cached attention spans the complete valid prefix. A chunk beginning
   at offset s allows key index j for query i iff `j ≤ s+i`.
3. A serving path has request-local cache length, explicit capacity, and a reset
   contract. Count prefill separately from time per output token; synchronize
   before timing on accelerators. Performance remains observational.

### Reading & guided study (8h total)

- 70m: [Direct Preference Optimization (DPO) and Friends](https://www.youtube.com/watch?v=6g6b4gvO-y0)
  — Nathan Lambert, lecture video. Watch **00:00–36:10** and **40:33–42:44**;
  skip the implementation chapter at 36:11–40:32. Prerequisite: Week 5 DPO and
  the Week 9 SFT checkpoint. Purpose: rebuild policy/reference log-ratio intuition
  and limitations without turning the lecture into an implementation recipe.
  **Check:** why does policy/reference parameter aliasing erase the comparison?
- 55m: [RLHF Book: Direct-Alignment Algorithms](https://rlhfbook.com/c/08-direct-alignment)
  — Nathan Lambert, explanatory chapter. Prerequisite: the lecture. Read “Direct-
  Alignment Algorithms,” “Direct Preference Optimization,” “How DPO Works,” and
  “Numerical Concerns, Weaknesses, and Alternatives”; skip derivation and code.
  Purpose: interpret relative preference shifts and limits of toy loss evidence.
  **Check:** can DPO loss fall while both response probabilities fall?
- 70m: [All About Transformer Inference](https://jax-ml.github.io/scaling-book/inference/)
  — Austin et al., illustrated systems chapter. Prerequisite: Week 3 cache and
  Week 8 bandwidth. Read “The Basics of Transformer Inference,” “What do we
  actually want to optimize?,” “What about attention?,” and “What about memory?”
  Purpose: separate prefill, decode, KV residency, latency, and throughput.
  **Check:** why may avoiding recomputed K/V yield a smaller latency improvement?
- 40m: [Cache strategies](https://huggingface.co/docs/transformers/kv_cache)
  — Hugging Face Contributors, framework documentation used for concepts only.
  Prerequisite: the inference chapter and Week 3 `StaticKVCache`. Read the opening
  comparison table and “Fixed-size cache”; skip all code. The course's raw-PyTorch
  contract remains authoritative. Purpose: contrast fixed allocation, valid length,
  stable shapes, and wasted capacity. **Check:** what grows while allocation stays fixed?
- 45m: [DPO](https://arxiv.org/abs/2305.18290) — Rafailov et al., primary-source
  revisit. Prerequisite: the lecture and chapter. Read §4, equations (4)–(7), and
  “What does the DPO update do?”; stop before §5. Purpose: confirm reparameterization,
  partition cancellation, and reference assumptions.
  **Check:** why must both responses share the same prompt?
- 30m: [GQA](https://arxiv.org/abs/2305.13245) — Ainslie et al., primary-source
  revisit. Prerequisite: the inference-cost explanation. Read the introduction,
  §2.2, and Figure 2 only. Purpose: connect KV-head count to cache storage and
  decode bandwidth. **Check:** if only KV-head count is halved, which tensors shrink?
- 100m: Active synthesis — DPO lifecycle rehearsal. Prerequisite: the DPO resources
  and Week 9 checkpoint. Draw post-SFT cloning, policy/reference ownership,
  prompt/response spans, masks, score reductions, gradients, optimizer state, and
  evidence; add three aliasing or masking failure cases. No code or pseudocode.
  Purpose: integrate state ownership, response scoring, and frozen-reference evidence.
  **Check:** what evidence proves the reference stayed frozen rather than net unchanged?
- 70m: Active synthesis — serving state and evidence rehearsal. Prerequisite:
  inference and cache resources. Trace two requests through prefill, offset chunk,
  single-token decode, reset, and capacity overflow; record positions, lengths,
  allowed keys, storage identity, bytes, and synchronization. Purpose: integrate
  positional, cache, capacity, and measurement invariants. **Check:** for a
  chunk at offset `s`, which keys may its first query attend to?

### Integration deliverable (2h)

Budget: 45m `fit_dpo`, 45m `CacheEngine`, 20m tests, 10m timing/bytes and final
evidence. Preserve a frozen post-SFT reference and use 10–20 tiny
DPO steps; the test fixture uses 12. Fixed preference pairs may reuse toy token
IDs, but chosen and rejected continuations must share each prompt.

- Return before/after DPO loss and per-pair policy log-probability margins;
  assert the reference state remains bitwise unchanged and has no gradients.
- Greedy generation prefills once, then feeds only the newest token for cached
  steps. Return prompt concatenated with generated tokens. Requests reset cache
  lengths without reallocating storage; no EOS/tokenizer protocol is required.
- Compare cached and uncached logits and complete greedy sequences. Report cache
  logical bytes and observed CPU/CUDA timing with dimensions, dtype, and device.
  Do not infer bandwidth from an unsynchronized wall-clock measurement.

### PyTest verification target

`python -m pytest tests/test_capstone.py -q`, then `python -m pytest -q`:
DPO reduces its toy loss and increases mean preference margin, the frozen
reference receives no gradient/update, response-only scoring is invariant to
masked trailing padding, policy changes, cache/full logits agree at multiple
offsets, greedy sequences agree, reset and overflow obey contracts, and a single
SFT→reference-clone→DPO→cached-serve smoke test uses the same trained policy.
The integration reuse gate observes the actual custom primitives during a
training/backward and cached-inference path; disconnected duplicate modules
do not satisfy it.

### The “Grill Yourself” screen

1. Why is computing reference log probabilities with the current policy a
   different objective? How can parameter aliasing silently destroy the reference?
2. Trace one chosen response token's contribution to sequence log probability.
   Where do shifting, prompt masking, padding, and sequence summation occur?
3. Why must cache positions be absolute across calls? Which two bugs can make
   single-token decode pass while multi-token offset chunks fail?
4. If latency is poor despite fewer attention FLOPs, how would you distinguish
   Python launch overhead, KV bandwidth, host transfers, and synchronization?

**Final checkpoint:** [ ] All mandatory gates pass; produce a one-page note with
the SFT/DPO loss changes, chosen/rejected margin change, cache bytes, cached/full
maximum absolute logit error, environment, and three limitations. State clearly
that a toy preference gain verifies loss wiring, not real-world alignment.
