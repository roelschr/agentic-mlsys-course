import math

import pytest
import torch
from torch.autograd import gradcheck
from torch.nn import functional as F

from drills.week05 import dpo_loss, masked_nll


def test_masked_nll_reference_gradcheck_and_masked_gradients():
    logits = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([[1, 0, 2], [4, 1, 3]])
    mask = torch.tensor([[False, True, True], [False, False, True]])
    loss = masked_nll(logits, targets, mask)
    reference = F.cross_entropy(logits[mask], targets[mask])
    torch.testing.assert_close(loss, reference, rtol=1e-12, atol=1e-12)
    assert gradcheck(lambda a: masked_nll(a, targets, mask), (logits,))
    loss.backward()
    torch.testing.assert_close(logits.grad[~mask], torch.zeros_like(logits.grad[~mask]))
    assert logits.grad[mask].abs().sum() > 0


def test_nll_inputs_are_already_shifted_and_all_masked_is_an_error():
    logits = torch.tensor([[[0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]], dtype=torch.float64)
    targets = torch.tensor([[1, 2]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    expected = F.cross_entropy(logits.reshape(-1, 3), targets.reshape(-1))
    torch.testing.assert_close(masked_nll(logits, targets, mask), expected)
    with pytest.raises(ValueError):
        masked_nll(logits, targets, ~mask)


def test_dpo_baseline_signs_detachment_and_gradcheck():
    pc = torch.tensor([-2.0, -3.0], dtype=torch.float64, requires_grad=True)
    pr = torch.tensor([-4.0, -2.0], dtype=torch.float64, requires_grad=True)
    rc = pc.detach().clone().requires_grad_()
    rr = pr.detach().clone().requires_grad_()
    loss = dpo_loss(pc, pr, rc, rr, beta=0.3)
    assert loss.item() == pytest.approx(math.log(2))
    loss.backward()
    torch.testing.assert_close(pc.grad, torch.full_like(pc, -0.075))
    torch.testing.assert_close(pr.grad, torch.full_like(pr, 0.075))
    assert rc.grad is None and rr.grad is None
    assert gradcheck(lambda a, b: dpo_loss(a, b, rc, rr, beta=0.3), (pc, pr))


def test_dpo_reference_margin_stability_and_shift_invariance():
    pc = torch.tensor([-1.0, -10001.0], dtype=torch.float64, requires_grad=True)
    pr = torch.tensor([-10001.0, -1.0], dtype=torch.float64, requires_grad=True)
    rc = torch.tensor([-2.0, -3.0], dtype=torch.float64)
    rr = torch.tensor([-4.0, -1.0], dtype=torch.float64)
    actual = dpo_loss(pc, pr, rc, rr, beta=1.0)
    margin = (pc - pr) - (rc - rr)
    expected = F.binary_cross_entropy_with_logits(margin, torch.ones_like(margin))
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual, dpo_loss(pc, pr, rc + 19, rr + 19, beta=1.0))
    actual.backward()
    assert torch.isfinite(pc.grad).all() and torch.isfinite(pr.grad).all()


@pytest.mark.parametrize("beta", [0, -0.1, float("inf")])
def test_dpo_rejects_invalid_beta(beta):
    x = torch.zeros(2)
    with pytest.raises(ValueError):
        dpo_loss(x, x, x, x, beta)
