# From-scratch MLSys refresher

A ten-week, self-paced course for an experienced ML systems engineer rebuilding
mathematical fluency and low-level implementation intuition. **Budget: exactly
12 hours per week, 120 hours total.** Eight weeks of small drills lead into one
two-week decoder training, alignment, and cached-inference capstone.

The repository deliberately starts with **unsolved skeletons and failing tests**.
There are no reference solutions. AI agents are restricted to Socratic mentoring
and test-driven review by `AGENTS.md`.

## Repository layout

```text
agentic-mlsys-course/
├── AGENTS.md                 # AI-only mentoring and anti-cheat instructions
├── README.md                 # Human setup and workflow
├── SYLLABUS.md               # Full schedule, sources, checkpoints, grill questions
├── pyproject.toml            # Minimal dependencies and pytest configuration
├── uv.lock                   # Resolved dependencies for the uv workflow
├── drills/
│   ├── __init__.py
│   ├── week01.py             # AdamW; unscale/check/clip
│   ├── week02.py             # RMSNorm autograd; RoPE
│   ├── week03.py             # GQA; static KV cache
│   ├── week04.py             # SwiGLU; capacity-limited router
│   ├── week05.py             # Masked SFT NLL; DPO
│   ├── week06.py             # InfoNCE; hard-negative selection
│   ├── week07.py             # Column-parallel; row-parallel linear
│   └── week08.py             # ZeRO accounting; speculative correction
├── capstone/
│   ├── __init__.py
│   └── project.py            # TinyDecoder, SFT/DPO runners, cache engine stubs
├── tests/
│   ├── conftest.py
│   ├── test_week01.py … test_week08.py
│   └── test_capstone.py
├── notes/                    # Learner-created derivations and pytest review logs
│   └── weekNN.md
├── artifacts/                # Learner-created checkpoints and small measurements
├── CLAUDE.md                 # Existing pointer to AGENTS.md
└── graft/                    # Existing repository context graph
```

`notes/` and `artifacts/` are created by you when first needed. Environment-specific
and existing hidden configuration files are omitted from the tree.

## Local study companion

For a desktop reading workspace with a dark Field Guide layout, side-by-side
papers and notes, and locally saved progress, run from the repository root:

```bash
python -m study_companion
```

Open **http://localhost:8765**. The companion uses Python's standard library and
reads all ten weeks directly from `SYLLABUS.md`. Notes autosave to `notes/weekNN.md`;
progress lives in Git-ignored `.study-companion/progress.json`. Time logging is
optional and editable, and the focus timer never logs time automatically.

See [the companion guide](study_companion/README.md) for file locations, the
reading pane, conflict recovery, startup options, and its independent test command.

## Local environment

Use Python **3.10 or newer** with a compatible PyTorch release (2.5 or newer).
CPU is sufficient for every required correctness gate, including two-process
Gloo tensor parallelism. CUDA is useful for optional physical memory experiments;
MPS is useful for exploratory model runs. Neither is needed to complete the course.
Use Linux/WSL for the Gloo exercise if your platform's PyTorch lacks Gloo.

### Verified local profile: WSL2 + RTX 3090

Learner-reported checks on **2026-09-18** confirmed Python **3.13.7**, an
**RTX 3090 with 24 GiB VRAM**, and all three capabilities below:

- `torch.cuda.is_available()` → `True`
- `torch.cuda.is_bf16_supported()` → `True`
- `torch.distributed.is_gloo_available()` → `True`

This is sufficient for the entire course on one PC. Use the existing working
CUDA installation and `uv.lock`. From the repository root inside WSL:

```bash
uv sync --locked --extra dev
uv run --locked --extra dev python -m pytest --collect-only -q
uv run --locked --extra dev python -m pytest tests/test_week01.py -q
```

`dev` is an **optional extra**, so `--extra dev` installs PyTest; it is not a
dependency group selected by `--dev`. For any `python -m pytest ...` command in
this course, the uv equivalent is `uv run --locked --extra dev python -m pytest ...`.
The lockfile records package versions; also record the runtime and driver in
your first study note:

```bash
nvidia-smi
uv run --locked --extra dev python -c "import torch; print('PyTorch:', torch.__version__); print('Wheel CUDA runtime:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0)); print('BF16:', torch.cuda.is_bf16_supported()); print('Gloo:', torch.distributed.is_gloo_available())"
```

The CUDA version displayed by `nvidia-smi` describes driver capability;
`torch.version.cuda` identifies PyTorch's CUDA build. These need not match.
The supplied drills use prebuilt PyTorch operations and do not require a
separate CUDA toolkit installation. Keep the repository and `.venv` in the WSL
Linux filesystem (such as `~/workspace/`), as in the verified setup.

| Work | Device policy for this PC |
|---|---|
| Numerical parity and float64 `gradcheck` | Keep the supplied CPU tests as the correctness baseline. |
| Week 1 precision experiments | The 3090 supports both FP16 and BF16; FP32 remains the optimizer/master-weight baseline. |
| Week 3 memory profiling | Run the existing CUDA allocation test after implementing the cache. |
| Week 7 tensor parallelism | Two CPU/Gloo ranks on this PC; the test harness launches them. |
| Week 8 pinned-memory/stream overlap | Use the CUDA experiment within the existing 30-minute worksheet slot. |
| Capstone | Mandatory integration tests stay on CPU; use CUDA for the budgeted inference measurements. |

To save Week 3 CUDA telemetry (logical bytes, live/peak allocation, and reserved
memory) after completing the cache:

```bash
uv run --locked --extra dev python -m pytest tests/test_week03.py -m cuda -q --junitxml=artifacts/week03-cuda.xml -o junit_family=xunit1
```

PyTest creates the report directory if needed. On this verified GPU the CUDA
test should execute rather than skip; it will still fail while the cache is an
unsolved stub. The GPU's availability does not automatically move other tests
to CUDA. Keep device placement explicit instead of changing PyTorch's global
default device. GPU experiments replace their CPU worksheet alternatives within
the same 11-hour week.

### Fresh environment setup (venv/pip alternative)

The following is for a fresh environment. With the verified uv/CUDA environment
above, use the uv commands rather than reinstalling a CPU-only wheel.

```bash
python --version
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# CPU baseline (Linux/Windows); macOS can use the ordinary PyPI torch wheel:
python -m pip install 'torch>=2.5' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
python -c "import torch, pytest; print(torch.__version__, pytest.__version__); print('CUDA:', torch.cuda.is_available()); print('MPS:', torch.backends.mps.is_available()); print('Gloo:', torch.distributed.is_gloo_available())"
python -m pytest --collect-only -q
```

For NVIDIA GPUs, select a wheel compatible with your driver using the official
[PyTorch installer](https://pytorch.org/get-started/locally/) **instead of the CPU
wheel command**. On Windows activate with `.venv\Scripts\Activate.ps1`.
Do not install a GPU wheel just to complete the readings. Record Python, PyTorch,
device, and dtype in your first note. GPU backend precision and kernel selection
vary; mandatory double-precision `gradcheck` runs stay on CPU, including on Macs.

The course uses `torch.nn`, `torch.autograd`, `torch.distributed`, NumPy, and
optionally Triton. Standard-library tooling and PyTest are permitted. Built-in
reference optimizers/losses/attention appear in tests; do not call them to bypass
the component being implemented. Ordinary `nn.Linear`/`nn.Embedding` are allowed
in capstone assembly. No tokenizer package or downloaded dataset is required.

## Study workflow

1. Open [SYLLABUS.md](SYLLABUS.md). Reserve **8h study + 2h coding + 2h review**.
   The reading allocation includes listening and watching. Follow the listed
   sequence: approachable explanations first, targeted paper excerpts afterward.
   Each allowance includes pauses, notes, and its comprehension question. Use a
   video's text alternative instead of the video, not in addition to it; stop at
   the named sections or timestamps rather than completing whole sites/playlists.
2. Derive the formula, annotate shapes and dtypes, and predict two edge cases in
   `notes/weekNN.md` before implementing. The syllabus budgets this in guided study.
3. Implement only that week's one or two stubs. Keep each component under 150
   implementation lines. Run `python -m pytest tests/test_weekNN.py -q` often.
4. Use the last two hours for the four grill questions and a pytest-backed review:
   30m derivation, 45m test/edge-case discussion, 30m systems trade-offs, 15m recap.
   An agent may ask questions and explain invariants, but cannot fill in code.
5. Mark the syllabus checkpoint only when numerical tests pass and you can explain
   the result without reading the implementation. Record unresolved questions.
   At hour 12, stop. If necessary, repeat the week in a later calendar slot;
   ten learning weeks need not mean ten consecutive calendar weeks.

```bash
python -m pytest tests/test_week01.py -q
python -m pytest tests/test_week07.py -q   # launches two local Gloo processes itself
python -m pytest tests/test_capstone.py -q -m week9            # week 9
python -m pytest tests/test_capstone.py -q                     # week 10
python -m pytest -q                                          # final regression
```

Do not invoke the Week 7 test under `torchrun`: its fixture already spawns ranks.
Unimplemented methods raise `NotImplementedError`; that is an expected *starting
failure*, not a passing verification. Tests are not marked `xfail`, and required
tests never skip because a learner implementation is missing. A few CUDA-specific
checks skip when CUDA is unavailable. No throughput threshold or hardware purchase
is a course gate.

### What counts as completion?

- All mandatory CPU tests pass, including gradients, state transitions, distributed
  scaling, cache invariants, and end-to-end toy training/alignment gates.
- Week 9/10 uses your AdamW, RMSNorm, RoPE, GQA, masked loss, DPO, and static cache
  in a single tiny decoder path. The module-reuse integration test must pass.
- A short final note explains memory costs, teacher-forcing vs. incremental decode,
  and what the synthetic DPO result does and does not demonstrate.

This is a mechanics curriculum. Toy loss reduction and preference improvement
verify integration; they are not evidence of useful language ability or alignment
on real-world human preferences.
