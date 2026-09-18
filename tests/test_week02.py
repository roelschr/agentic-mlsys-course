import pytest
import torch
from torch.autograd import gradcheck
from torch.nn import functional as F

from drills.week02 import RMSNormFunction, apply_rope


@pytest.mark.parametrize("kind", ["random", "zero", "tiny", "noncontiguous"])
def test_rmsnorm_reference_and_gradcheck(kind):
    x = torch.randn(2, 3, 4, dtype=torch.float64)
    if kind == "zero":
        x.zero_()
    elif kind == "tiny":
        x.mul_(1e-7)
    elif kind == "noncontiguous":
        x = x.transpose(0, 1)
    x.requires_grad_()
    gamma = torch.randn(4, dtype=torch.float64, requires_grad=True)
    actual = RMSNormFunction.apply(x, gamma, 1e-5)
    expected = F.rms_norm(x, (4,), gamma, eps=1e-5)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    assert gradcheck(lambda a, b: RMSNormFunction.apply(a, b, 1e-5),
                     (x, gamma), eps=1e-6, atol=2e-4, rtol=2e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rmsnorm_low_precision_reduction(dtype):
    x = torch.full((2, 8), 1000.0, dtype=dtype, requires_grad=True)
    gamma = torch.ones(8, dtype=dtype, requires_grad=True)
    result = RMSNormFunction.apply(x, gamma, 1e-5)
    assert result.dtype == dtype
    torch.testing.assert_close(result.float(), torch.ones(2, 8), atol=0.02, rtol=0.02)
    result.float().sum().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(gamma.grad).all()


def test_rope_rotation_inverse_relative_identity_and_gradcheck():
    x = torch.randn(1, 2, 3, 4, dtype=torch.float64, requires_grad=True)
    y = torch.randn_like(x)
    positions = torch.tensor([4, 5, 6])
    rotated = apply_rope(x, positions)
    torch.testing.assert_close(rotated.square().sum(-1), x.square().sum(-1))
    torch.testing.assert_close(apply_rope(rotated, -positions), x)
    torch.testing.assert_close(apply_rope(x, torch.zeros(3)), x)
    lhs = (rotated * apply_rope(y, positions + 3)).sum(-1)
    rhs = (x * apply_rope(y, torch.full((3,), 3))).sum(-1)
    torch.testing.assert_close(lhs, rhs, rtol=1e-10, atol=1e-10)
    assert gradcheck(lambda a: apply_rope(a, positions), (x,))


def test_rope_adjacent_pair_orientation_and_frequency():
    # At position pi/2 the first pair rotates 90 degrees; the second rotates pi/200.
    x = torch.tensor([[[[1.0, 0.0, 1.0, 0.0]]]], dtype=torch.float64)
    p = torch.tensor([torch.pi / 2], dtype=torch.float64)
    angle = torch.tensor(torch.pi / 200, dtype=torch.float64)
    expected = torch.tensor([[[[0.0, 1.0, angle.cos().item(), angle.sin().item()]]]],
                            dtype=torch.float64)
    torch.testing.assert_close(apply_rope(x, p), expected, rtol=1e-10, atol=1e-10)


def test_rope_rejects_odd_head_width():
    with pytest.raises(ValueError):
        apply_rope(torch.zeros(1, 1, 2, 3), torch.arange(2))
