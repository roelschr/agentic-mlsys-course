import pytest
import torch
from torch.autograd import gradcheck
from torch.nn import functional as F

from drills.week06 import hard_negative_indices, symmetric_infonce


def test_infonce_reference_gradcheck_and_invariances():
    a = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    b = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    actual = symmetric_infonce(a, b, 0.2)
    logits = F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).T / 0.2
    labels = torch.arange(3)
    expected = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    assert gradcheck(lambda x, y: symmetric_infonce(x, y, 0.2), (a, b))
    order = torch.tensor([2, 0, 1])
    torch.testing.assert_close(actual, symmetric_infonce(a[order], b[order], 0.2))
    scale = torch.tensor([[2.0], [3.0], [0.5]], dtype=torch.float64)
    torch.testing.assert_close(actual, symmetric_infonce(a * scale, b / scale, 0.2))


def test_paired_beats_shuffled_and_low_temperature_is_finite():
    a = torch.eye(4, dtype=torch.float64)
    assert symmetric_infonce(a, a, 0.1) < symmetric_infonce(a, a.roll(1, 0), 0.1)
    x = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    y = torch.randn_like(x, requires_grad=True)
    loss = symmetric_infonce(x, y, 1e-4)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(y.grad).all()


def test_infonce_zero_embeddings_and_single_pair():
    a = torch.zeros(2, 3, dtype=torch.float64, requires_grad=True)
    b = torch.zeros_like(a, requires_grad=True)
    loss = symmetric_infonce(a, b, 0.1)
    assert loss.item() == pytest.approx(0.6931471805599453)
    loss.backward()
    assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()
    one = torch.ones(1, 3, dtype=torch.float64)
    assert symmetric_infonce(one, one).item() == pytest.approx(0.0)


def test_hard_negatives_rectangular_ties_and_exclusion():
    scores = torch.tensor([[9.0, 4.0, 4.0, 1.0, 0.0], [2.0, 8.0, 6.0, 6.0, 0.0]])
    positives = torch.tensor([0, 1])
    result = hard_negative_indices(scores, positives, 3)
    assert result.dtype == torch.int64
    torch.testing.assert_close(result, torch.tensor([[1, 2, 3], [2, 3, 0]]))
    assert not (result == positives[:, None]).any()
    assert hard_negative_indices(scores, positives, 0).shape == (2, 0)


@pytest.mark.parametrize("temperature", [0, -1, float("nan")])
def test_infonce_invalid_temperature(temperature):
    with pytest.raises(ValueError):
        symmetric_infonce(torch.ones(2, 3), torch.ones(2, 3), temperature)


@pytest.mark.parametrize("k", [-1, 3])
def test_miner_invalid_k(k):
    with pytest.raises(ValueError):
        hard_negative_indices(torch.ones(2, 3), torch.tensor([0, 1]), k)
