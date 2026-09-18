import pytest
import torch

from drills.week08 import speculative_distribution, zero_bytes


@pytest.mark.parametrize("stage,expected", [
    (0, [240, 240, 480, 480, 480, 1920]),
    (1, [240, 240, 120, 120, 120, 840]),
    (2, [240, 60, 120, 120, 120, 660]),
    (3, [60, 60, 120, 120, 120, 480]),
])
def test_zero_exact_component_ledger(stage, expected):
    ledger = zero_bytes(120, 4, stage)
    names = ["parameters", "gradients", "master", "m", "v", "total"]
    assert set(ledger) == set(names)
    assert [ledger[name] for name in names] == expected
    assert all(type(value) is int for value in ledger.values())
    assert sum(ledger[name] for name in names[:-1]) == ledger["total"]


def test_zero_single_rank_and_monotonic_sharding():
    baseline = zero_bytes(120, 1, 0)
    for stage in range(4):
        assert zero_bytes(120, 1, stage) == baseline
        counts = [zero_bytes(120, size, stage)["total"] for size in (1, 2, 4)]
        assert counts[0] >= counts[1] >= counts[2]


@pytest.mark.parametrize("parameters,size,stage", [(0, 2, 1), (5, 2, 3), (10, 0, 2),
                                                   (10, 2, 4), (10.5, 2, 0)])
def test_zero_rejects_invalid_inputs(parameters, size, stage):
    with pytest.raises(ValueError):
        zero_bytes(parameters, size, stage)


def test_speculative_fixed_acceptance_and_residual():
    p = torch.tensor([0.6, 0.3, 0.1], dtype=torch.float64)
    q = torch.tensor([0.2, 0.5, 0.3], dtype=torch.float64)
    acceptance, residual = speculative_distribution(p, q)
    torch.testing.assert_close(acceptance, torch.tensor([1.0, 0.6, 1 / 3], dtype=p.dtype))
    torch.testing.assert_close(residual, torch.tensor([1.0, 0.0, 0.0], dtype=p.dtype))


@pytest.mark.parametrize("case", ["random", "equal", "disjoint", "zero_draft"])
def test_speculative_exact_target_law(case):
    for _ in range(12):
        p = torch.softmax(torch.randn(7, dtype=torch.float64), dim=0)
        q = torch.softmax(torch.randn(7, dtype=torch.float64), dim=0)
        if case == "equal":
            q = p.clone()
        elif case == "disjoint":
            p = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
            q = torch.tensor([0.0, 0.3, 0.7], dtype=torch.float64)
        elif case == "zero_draft":
            p = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
            q = torch.tensor([0.0, 0.5, 0.5], dtype=torch.float64)
        acceptance, residual = speculative_distribution(p, q)
        assert acceptance.dtype == p.dtype and residual.dtype == p.dtype
        assert torch.isfinite(acceptance).all() and torch.isfinite(residual).all()
        assert ((acceptance >= 0) & (acceptance <= 1)).all()
        assert (residual >= 0).all()
        torch.testing.assert_close(residual.sum(), torch.ones((), dtype=p.dtype))
        torch.testing.assert_close(acceptance[q == 0], torch.ones_like(acceptance[q == 0]))
        accepted_mass = q * acceptance
        output_law = accepted_mass + (1 - accepted_mass.sum()) * residual
        torch.testing.assert_close(output_law, p, rtol=1e-12, atol=1e-12)
        if case == "equal":
            torch.testing.assert_close(acceptance, torch.ones_like(p))
            torch.testing.assert_close(residual, p)


@pytest.mark.parametrize("p,q", [
    ([0.2, 0.2], [0.5, 0.5]),
    ([-0.1, 1.1], [0.5, 0.5]),
    ([float("nan"), 1.0], [0.5, 0.5]),
    ([1.0], [0.5, 0.5]),
    ([], []),
])
def test_speculative_rejects_invalid_distributions(p, q):
    with pytest.raises(ValueError):
        speculative_distribution(torch.tensor(p), torch.tensor(q))
