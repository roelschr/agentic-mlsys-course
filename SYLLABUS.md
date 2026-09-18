# Ten-week from-scratch MLSys curriculum

## Contract and pacing

**Every week: 3h reading, 6h implementation, 2h Socratic self-grill = 11h.**
Total: **30h reading + 60h coding + 20h review = 110h**. No additional mandatory
homework, provisioning project, dataset collection, or full-paper reading is
hidden outside these allocations. Setup and test execution count toward coding.

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

The two capstone weeks use the same allocation, with all 12 coding hours devoted
to one integrated project. Four small glue components replace the isolated
drills. They reuse earlier work and each stays under the same 150-line limit.

### Tracking

| Week | Theme | Reading | Coding | Review | Gate |
|---|---|---:|---:|---:|---|
| 1 | Optimization and precision | 3h | 6h | 2h | [ ] optimizer state + overflow |
| 2 | Normalization and positions | 3h | 6h | 2h | [ ] backward + rotation |
| 3 | Attention and cache memory | 3h | 6h | 2h | [ ] grouped causal/cache parity |
| 4 | Transformer FFNs and routing | 3h | 6h | 2h | [ ] gradients + capacity |
| 5 | Post-training objectives | 3h | 6h | 2h | [ ] masking + DPO derivatives |
| 6 | Contrastive retrieval | 3h | 6h | 2h | [ ] normalization + mining |
| 7 | Tensor and pipeline parallelism | 3h | 6h | 2h | [ ] two-rank forward/backward |
| 8 | Partitioning and inference | 3h | 6h | 2h | [ ] bytes + corrected distribution |
| 9 | Capstone: decoder + SFT | 3h | 6h | 2h | [ ] tiny corpus overfit |
| 10 | Capstone: DPO + serving | 3h | 6h | 2h | [ ] preference + cached decode |

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

**Time:** reading 3h; implementation 6h; self-grill 2h.

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

### Reading & primary sources (3h total)

- 60m: [Adam, Kingma & Ba](https://arxiv.org/abs/1412.6980), Algorithm 1 and
  bias-correction discussion; compare on paper to momentum SGD.
- 60m: [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101),
  §§2–3 and AdamW algorithm; derive why the SGD equivalence needs an LR factor.
- 60m: [Mixed Precision Training](https://arxiv.org/abs/1710.03740), §§2–3:
  master weights, loss scaling, and FP32 reductions. Inspect `torch.finfo`.

### Target implementation drill (6h)

- `AdamW(torch.optim.Optimizer)`: dense real parameters, FP32 or FP64 state matching
  parameter dtype, independent parameter steps, no AMSGrad/foreach/fused mode.
  FP32 master parameters are the mixed-precision integration choice; implementing
  a master-copy optimizer is outside this component's scope.
- `unscale_clip_(grads, scale, max_norm)`: return `(preclip_norm, found_inf)`;
  finite lists are unscaled/clipped in place with FP64 norm accumulation. On any
  nonfinite input, return `(inf, True)` and leave **all** gradients untouched.
  Empty lists yield `(0, False)`; positive scale and nonnegative norm cap required.

Budget: 45m setup/derivation, 2h15 AdamW, 1h30 precision utility, 1h30 tests.
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

**Time:** reading 3h; implementation 6h; self-grill 2h.

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

### Reading & primary sources (3h total)

- 60m: [Layer Normalization](https://arxiv.org/abs/1607.06450), §§2–3 and its
  BatchNorm comparison. Write the normalized axes for `[B,T,D]` explicitly.
- 45m: [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467),
  §§3–4; derive the input and scale gradients rather than copying a kernel.
- 75m: [RoFormer](https://arxiv.org/abs/2104.09864), §3.2, rotation and relative
  inner-product equations; compare to its absolute-position discussion.

### Target implementation drill (6h)

- `RMSNormFunction`: custom `torch.autograd.Function` forward/backward, last-axis
  normalization, learnable scale, no bias. Keep float64 for gradcheck; use at least
  FP32 reductions for FP16/BF16 inputs, return the input dtype.
- `apply_rope(x, positions, base)`: adjacent pairs, `x[B,H,T,D]`, positions `[T]`,
  even `D`; support nonzero cache offsets and negative positions for inversion.

Budget: 45m derivation, 2h15 backward, 1h30 RoPE, 1h30 testing. BatchNorm and
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

**Time:** reading 3h; implementation 6h; self-grill 2h.

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

### Reading & primary sources (3h total)

- 45m: [Fast Transformer Decoding / MQA](https://arxiv.org/abs/1911.02150),
  §§2–3; count bytes per decode step.
- 45m: [GQA](https://arxiv.org/abs/2305.13245), §2 and Figure 2.
- 90m: [FlashAttention](https://arxiv.org/abs/2205.14135), §§2–3 and Algorithm 1;
  derive the tile merge with two scalar score tiles.

### Target implementation drill (6h)

- `gqa(q,k,v,allowed=None)`: stable, explicit attention; shapes
  `[B,Hq,Tq,d]`, `[B,Hkv,Tk,d]`. Boolean mask `[Tq,Tk]` uses **True = allowed**.
  No mask means bidirectional attention. Fully masked rows return zero without
  NaNs in forward/backward. Do not use SDPA in the learner implementation.
- `StaticKVCache`: preallocate K/V `[B,Hkv,capacity,d]` once; append detached
  inference tensors, expose valid-prefix views, reset length without reallocating.
  Overflow and incompatible shapes must be rejected without mutation.

Budget: 30m shapes, 2h attention, 1h30 cache, 1h30 tests, 30m memory worksheet.
The worksheet uses the byte formula on CPU. With CUDA, replace those 30m with
synchronized peak-allocation measurement; never add time or require a fused kernel.

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

**Time:** reading 3h; implementation 6h; self-grill 2h.

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

### Reading & primary sources (3h total)

- 45m: [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202),
  §2 and parameter-matched comparisons.
- 75m: [Switch Transformers](https://arxiv.org/abs/2101.03961), §2 routing,
  capacity, and auxiliary-loss equations (4)–(6).
- 60m: [GShard](https://arxiv.org/abs/2006.16668), §3 and top-2 gating discussion;
  compare communication and dropped-token behavior to the top-1 design.

### Target implementation drill (6h)

- `SwiGLU(d_model, hidden)`: bias-free `gate`, `up`, `down` linear modules, whose
  names are part of the test contract. No residual or complete block yet.
- `route_topk(logits,k,capacity_factor)`: return `indices[N,k]`, selected
  `weights[N,k]`, `keep[N,k]`, scalar auxiliary loss. Lower expert ID wins ties;
  process assignments in token-major, then slot-major order for capacity. Dropped
  weights become zero without renormalizing surviving weights. No expert dispatch,
  distributed all-to-all, or MoE model implementation this week.

Budget: 30m derivation, 1h SwiGLU, 2h30 router, 2h tests/capacity examples.

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

**Time:** reading 3h; implementation 6h; self-grill 2h.

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

### Reading & primary sources (3h total)

- 45m: [InstructGPT](https://arxiv.org/abs/2203.02155), §3 and Figure 2:
  SFT/reward-model/PPO data and model roles.
- 45m: [PPO](https://arxiv.org/abs/1707.06347), clipped objective in §3;
  identify which ratios and advantages are held fixed during an update.
- 90m: [DPO](https://arxiv.org/abs/2305.18290), §§3–4, equations (4)–(7),
  and the optimal-policy derivation. Re-derive the partition cancellation.

### Target implementation drill (6h)

- `masked_nll(logits,targets,mask)`: logits `[B,T,V]`, valid integer targets,
  boolean mask `[B,T]`; inputs are **already shifted**. All-masked batches raise
  `ValueError` rather than silently dividing by zero.
- `dpo_loss(policy_chosen,policy_rejected,ref_chosen,ref_rejected,beta)`:
  inputs `[B]` are response-only **summed** sequence log probabilities. Return
  mean stable logistic loss. Reference arguments must never receive gradients.
  No policy model, reward model, rollout loop, or PPO implementation this week.

Budget: 1h derivation/shapes, 1h masked loss, 1h30 DPO, 2h30 tests and saturation
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

**Time:** reading 3h; implementation 6h; self-grill 2h.

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

### Reading & primary sources (3h total)

- 60m: [CPC](https://arxiv.org/abs/1807.03748), §2.3 and InfoNCE bound discussion;
  identify the assumptions behind the mutual-information interpretation.
- 60m: [SimCLR](https://arxiv.org/abs/2002.05709), §2 and §4 temperature/normalization
  ablations. Note its two-view masking differs from the paired cross-modal setup.
- 60m: [CLIP](https://arxiv.org/abs/2103.00020), §2.3 and Figure 3;
  trace both directions of its paired classification objective.

### Target implementation drill (6h)

- `symmetric_infonce(a,b,temperature)`: `[N,D]`, normalize internally with fixed
  ε=1e−12, diagonal positives, stable row/column losses, positive temperature.
- `hard_negative_indices(scores,positive_indices,k)`: `[N,M]` scores, one positive
  index per row; exclude it, return descending top-k candidate indices with
  lower candidate ID breaking ties. `0≤k<M`; k=0 returns `[N,0]`.

Budget: 45m derivation, 1h30 loss, 1h15 miner, 2h30 testing and failure analysis.
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

**Time:** reading 3h; implementation 6h; self-grill 2h.

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

### Reading & primary sources (3h total)

- 75m: [Megatron-LM](https://arxiv.org/abs/1909.08053), §3 and Figure 3;
  annotate forward and backward communication around the two linears.
- 45m: [GPipe](https://arxiv.org/abs/1811.06965), §2, microbatching and bubbles.
- 60m: [Efficient Large-Scale Language Model Training / Megatron](https://arxiv.org/abs/2104.04473),
  §§2–3 and pipeline schedule figures; compare 1F1B, interleaving, and DP groups.

### Target implementation drill (6h)

- `ColumnParallelLinear(in_features,out_features,group=None)`: bias-free, replicated
  input, local output shard. Expose `.weight` and correct custom communication
  autograd semantics. Require divisible dimensions.
- `RowParallelLinear(in_features,out_features,group=None)`: bias-free, already
  input-sharded, replicated output, local `.weight`. Compose the two around a
  local activation using the supplied two-rank Gloo test harness.

Budget: 45m equations/collectives, 1h45 column, 1h45 row, 1h15 distributed tests,
30m draw a 2-stage/4-microbatch 1F1B schedule with dependency arrows. This drawing
replaces a runtime scheduler implementation. No cluster provisioning this week.

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
eight-rank 3D ownership sketch. Both sketches are within the six coding hours.

## Week 8: Memory partitioning & inference optimization

**Time:** reading 3h; implementation 6h; self-grill 2h.

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

### Reading & primary sources (3h total)

- 60m: [ZeRO](https://arxiv.org/abs/1910.02054), §§3–5; reconstruct its state
  ledger under this course's explicitly different dtype choices.
- 75m: [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192),
  Algorithm 1 and distribution-preservation argument.
- 45m: [PyTorch pin-memory and non-blocking transfer tutorial](https://pytorch.org/tutorials/intermediate/pinmem_nonblock.html),
  pinned/pageable comparisons, stream overlap conditions, and synchronization
  caveats. Read it as evidence about data movement, not as a training wrapper.

### Target implementation drill (6h)

- `zero_bytes(P,world_size,stage)`: idealized equal sharding, require `P%d=0`,
  return integer component bytes for `parameters`, `gradients`, `master`, `m`,
  `v`, and `total`. State assumptions stay explicit in the return contract.
- `speculative_distribution(p,q)`: normalized nonnegative finite `[V]` tensors;
  return vector acceptance probabilities and the normalized rejection residual.
  Set acceptance to 1 where q=0 (that event is never proposed). If p=q, return
  p as the unused residual distribution, avoiding a divide-by-zero artifact.

Budget: 45m memory derivation, 1h accounting, 1h45 distribution math/stub,
2h tests, 30m transfer/kernel worksheet. In that last half-hour, sketch a pinned
two-buffer stream timeline; if CUDA is available, use the reading's measurement
method on small buffers instead and record synchronized latency. No offload
engine, speculative serving loop, or custom kernel implementation is assigned.

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
break the 12-hour integration budget. Their memory and scheduling lessons inform
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

**Time:** reading 3h; capstone implementation 6h; self-grill 2h.

### Core concepts

1. Pre-norm residual shape/gradient paths, RoPE at each attention layer, and the
   relationship between query width and smaller KV projection width.
2. Teacher forcing, next-token shifting, causal isolation, and reproducible tiny
   overfitting as an integration diagnostic rather than a benchmark.
3. Parameter/moment/activation/cache ownership in the assembled model. Every
   trainable scale and projection must appear in the optimizer's parameter set.

### Reading & primary sources (3h total)

- 60m: [LLaMA](https://arxiv.org/abs/2302.13971), §2 architecture and pre-norm,
  SwiGLU, and RoPE choices; do not replicate its scale or data pipeline.
- 60m: [Attention Is All You Need](https://arxiv.org/abs/1706.03762), §§3.1–3.3
  and autoregressive masking, then contrast pre-norm to its original arrangement.
- 60m: revisit the Week 1 AdamW and Week 2 RMSNorm equations against your own
  shape/byte ledger; this is targeted integration reading, not another paper.

### Integration deliverable (6h)

Budget: 30m shape ledger, 2h30 `TinyDecoder`, 1h30 `fit_sft`, 1h tests/overfit,
30m save/load demonstration. Use ordinary PyTorch `state_dict` APIs in your notes
or test session; no checkpoint manager component is required.

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

**Time:** reading 3h; capstone implementation 6h; self-grill 2h.

### Core concepts

1. Clone the post-SFT model as a frozen reference. Score policy/reference on the
   same prompt-response tokens and masks. Only policy weights are optimized.
2. Incremental decoding rotates new keys at their absolute positions and stores
   them once; cached attention spans the complete valid prefix. A chunk beginning
   at offset s allows key index j for query i iff `j ≤ s+i`.
3. A serving path has request-local cache length, explicit capacity, and a reset
   contract. Count prefill separately from time per output token; synchronize
   before timing on accelerators. Performance remains observational.

### Reading & primary sources (3h total)

- 60m: revisit [DPO](https://arxiv.org/abs/2305.18290), §4 practical loss and
  reference-policy assumptions; connect the paper's sequence scores to masks.
- 60m: revisit [GQA](https://arxiv.org/abs/2305.13245), §2 and decoding motivation;
  derive this capstone's exact cache capacity and used bytes.
- 60m: revisit [FlashAttention](https://arxiv.org/abs/2205.14135), IO analysis,
  and the Week 8 transfer tutorial; explain why this unfused toy engine is a
  correctness reference rather than a production inference-kernel benchmark.

### Integration deliverable (6h)

Budget: 2h `fit_dpo`, 1h45 `CacheEngine`, 1h15 tests, 30m timing/bytes,
30m final evidence note. Preserve a frozen post-SFT reference and use 10–20 tiny
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
