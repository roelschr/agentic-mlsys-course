import copy
import math

import pytest
import torch

from drills.week01 import AdamW, unscale_clip_


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_adamw_parameter_moment_and_missing_gradient_parity(dtype):
    params = [torch.nn.Parameter(torch.randn(3, 2, dtype=dtype)) for _ in range(2)]
    refs = [torch.nn.Parameter(p.detach().clone()) for p in params]
    kwargs = dict(lr=0.02, betas=(0.8, 0.95), eps=1e-7, weight_decay=0.2)
    optimizer = AdamW(params, **kwargs)
    reference = torch.optim.AdamW(refs, foreach=False, **kwargs)
    for step in range(6):
        for i, (p, r) in enumerate(zip(params, refs)):
            grad = None if i == 1 and step % 2 == 0 else torch.randn_like(p)
            p.grad = grad
            r.grad = None if grad is None else grad.clone()
        optimizer.step()
        reference.step()
        for p, r in zip(params, refs):
            torch.testing.assert_close(p, r, rtol=2e-6, atol=2e-7)
            if r not in reference.state:
                assert not optimizer.state.get(p)
                continue
            for key in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(
                    torch.as_tensor(optimizer.state[p][key], dtype=torch.float64),
                    torch.as_tensor(reference.state[r][key], dtype=torch.float64),
                    rtol=2e-6, atol=2e-7,
                )


def test_zero_gradient_decays_but_missing_gradient_does_not():
    p = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
    opt = AdamW([p], lr=0.1, weight_decay=0.5)
    opt.step()
    torch.testing.assert_close(p, torch.tensor([2.0], dtype=p.dtype))
    assert not opt.state.get(p)
    p.grad = torch.zeros_like(p)
    opt.step()
    torch.testing.assert_close(p, torch.tensor([1.9], dtype=p.dtype))
    assert float(opt.state[p]["step"]) == 1


def test_optimizer_state_round_trip():
    p = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64))
    first = AdamW([p], lr=0.03)
    p.grad = torch.tensor([0.2, -0.4], dtype=p.dtype)
    first.step()
    q = torch.nn.Parameter(p.detach().clone())
    resumed = AdamW([q], lr=99.0)
    resumed.load_state_dict(copy.deepcopy(first.state_dict()))
    for _ in range(3):
        p.grad = torch.tensor([-0.1, 0.3], dtype=p.dtype)
        q.grad = p.grad.clone()
        first.step()
        resumed.step()
    torch.testing.assert_close(p, q, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("scale", [1.0, 8.0, 1024.0])
def test_unscale_global_norm_and_direction(scale):
    grads = [torch.tensor([3.0], dtype=torch.float64) * scale,
             torch.tensor([4.0, 0.0], dtype=torch.float64) * scale]
    norm, overflow = unscale_clip_(grads, scale, 2.0)
    assert norm == pytest.approx(5.0)
    assert overflow is False
    torch.testing.assert_close(grads[0], torch.tensor([1.2], dtype=torch.float64))
    torch.testing.assert_close(grads[1], torch.tensor([1.6, 0.0], dtype=torch.float64))


@pytest.mark.parametrize("bad", [float("inf"), float("nan"), -float("inf")])
def test_overflow_is_atomic_and_optimizer_can_be_skipped(bad):
    p = torch.nn.Parameter(torch.tensor([2.0]))
    opt = AdamW([p])
    p.grad = torch.tensor([3.0])
    opt.step()
    state_before = copy.deepcopy(opt.state_dict())
    value_before = p.detach().clone()
    grads = [torch.tensor([8.0]), torch.tensor([bad])]
    before = [g.clone() for g in grads]
    norm, overflow = unscale_clip_(grads, 8.0, 1.0)
    assert overflow is True and math.isinf(norm)
    for actual, expected in zip(grads, before):
        torch.testing.assert_close(actual, expected, equal_nan=True)
    # The caller's skip contract is explicit: no optimizer.step() on overflow.
    torch.testing.assert_close(p, value_before)
    for key, value in state_before["state"][0].items():
        torch.testing.assert_close(torch.as_tensor(opt.state_dict()["state"][0][key]),
                                   torch.as_tensor(value))


def test_empty_zero_and_unclipped_gradients():
    assert unscale_clip_([], 2.0, 1.0) == (0.0, False)
    g = torch.zeros(3)
    assert unscale_clip_([g], 2.0, 0.0) == (0.0, False)
    torch.testing.assert_close(g, torch.zeros(3))
    g = torch.tensor([2.0, -2.0])
    norm, flag = unscale_clip_([g], 2.0, 10.0)
    assert norm == pytest.approx(math.sqrt(2)) and not flag
    torch.testing.assert_close(g, torch.tensor([1.0, -1.0]))


@pytest.mark.parametrize("scale,cap", [(0, 1), (-1, 1), (1, -1), (float("inf"), 1)])
def test_invalid_clip_controls(scale, cap):
    with pytest.raises(ValueError):
        unscale_clip_([torch.ones(2)], scale, cap)


def test_fp16_bf16_range_and_precision_are_different():
    large = torch.tensor([1e5], dtype=torch.float32)
    assert torch.isinf(large.to(torch.float16)).all()
    assert torch.isfinite(large.to(torch.bfloat16)).all()
    tiny = torch.tensor([1e-8], dtype=torch.float32)
    assert tiny.to(torch.float16).item() == 0
    assert tiny.to(torch.bfloat16).item() > 0
    near_one = torch.tensor([1.001], dtype=torch.float32)
    assert near_one.to(torch.bfloat16).item() == 1.0
    assert near_one.to(torch.float16).item() != 1.0
