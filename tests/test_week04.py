import pytest
import torch
from torch.autograd import gradcheck

from drills.week04 import SwiGLU, route_topk


def test_swiglu_hand_example_and_registered_parameters():
    module = SwiGLU(3, 3).double()
    assert set(dict(module.named_parameters())) == {"gate.weight", "up.weight", "down.weight"}
    with torch.no_grad():
        for projection in (module.gate, module.up, module.down):
            assert projection.bias is None
            projection.weight.copy_(torch.eye(3, dtype=torch.float64))
    x = torch.tensor([[-1.0, 0.0, 1.0]], dtype=torch.float64, requires_grad=True)
    expected = torch.tensor([[0.2689414213699951, 0.0, 0.7310585786300049]],
                            dtype=torch.float64)
    torch.testing.assert_close(module(x), expected, rtol=1e-12, atol=1e-12)
    assert gradcheck(module, (x,))
    module(x).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in module.parameters())


def test_swiglu_rectangular_shapes():
    module = SwiGLU(4, 7)
    assert module.gate.weight.shape == (7, 4)
    assert module.up.weight.shape == (7, 4)
    assert module.down.weight.shape == (4, 7)
    assert module(torch.randn(2, 3, 4)).shape == (2, 3, 4)


def test_router_ties_capacity_and_no_postdrop_renormalization():
    logits = torch.zeros(3, 3, dtype=torch.float64, requires_grad=True)
    indices, weights, keep, aux = route_topk(logits, 2, 0.5)
    assert indices.dtype == torch.int64 and keep.dtype == torch.bool
    torch.testing.assert_close(indices, torch.tensor([[0, 1], [0, 1], [0, 1]]))
    torch.testing.assert_close(keep, torch.tensor([[True, True], [False, False], [False, False]]))
    torch.testing.assert_close(weights, torch.tensor([[0.5, 0.5], [0.0, 0.0], [0.0, 0.0]],
                                                   dtype=torch.float64))
    assert aux.item() == pytest.approx(1.0)
    # Capacity one for each expert: only one of the second token's assignments survives.
    logits2 = torch.tensor([[5.0, 4.0, 0.0], [5.0, 0.0, 4.0]], dtype=torch.float64)
    ids, w, retained, _ = route_topk(logits2, 2, 0.5)
    torch.testing.assert_close(ids, torch.tensor([[0, 1], [0, 2]]))
    torch.testing.assert_close(retained, torch.tensor([[True, True], [False, True]]))
    assert w[1, 0] == 0
    assert w[1, 1].item() == pytest.approx(0.2689414213699951)


def test_router_normalized_weights_and_aux_gradient():
    logits = torch.tensor([[8.0, 1.0, 0.0], [7.0, 0.0, 1.0]],
                          dtype=torch.float64, requires_grad=True)
    indices, weights, keep, aux = route_topk(logits, 1, 10.0)
    torch.testing.assert_close(indices, torch.zeros(2, 1, dtype=torch.int64))
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, dtype=torch.float64))
    assert keep.all() and aux.item() > 2.9
    reference = 3 * torch.softmax(logits, dim=-1)[:, 0].mean()
    torch.testing.assert_close(aux, reference)
    actual_grad, = torch.autograd.grad(aux, logits)
    ref_grad, = torch.autograd.grad(reference, logits)
    torch.testing.assert_close(actual_grad, ref_grad)
    assert actual_grad.abs().sum() > 0
    logits = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
    indices, weights, keep, _ = route_topk(logits, 2, 10.0)
    assert keep.all()
    torch.testing.assert_close(weights.sum(-1), torch.ones(5, dtype=torch.float64))
    selected = logits.gather(1, indices)
    torch.testing.assert_close(weights, torch.softmax(selected, dim=-1))
    grad, = torch.autograd.grad(weights[:, 0].sum(), logits)
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0


@pytest.mark.parametrize("k,factor", [(0, 1), (4, 1), (1, 0), (1, -1)])
def test_router_invalid_controls(k, factor):
    with pytest.raises(ValueError):
        route_topk(torch.zeros(2, 3), k, factor)
