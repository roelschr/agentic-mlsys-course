"""Two-process CPU contracts. The harness supplies no learner communication logic."""

from datetime import timedelta
import multiprocessing
import time

import pytest
import torch
import torch.distributed as dist
from torch.nn import functional as F

from drills.week07 import ColumnParallelLinear, RowParallelLinear


def _worker(rank, store, mode):
    torch.set_num_threads(1)
    torch.manual_seed(37)
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank,
                            world_size=2, timeout=timedelta(seconds=15))
    try:
        if mode == "collectives":
            local = torch.arange(4, dtype=torch.float64) + rank
            reduced = local.clone()
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
            torch.testing.assert_close(reduced, torch.tensor([1., 3., 5., 7.],
                                                            dtype=torch.float64))
            # Gloo-portable ReduceScatter semantic oracle: reduce then owned slice.
            shard = reduced.chunk(2)[rank].contiguous()
            gathered = [torch.empty_like(shard) for _ in range(2)]
            dist.all_gather(gathered, shard)
            torch.testing.assert_close(torch.cat(gathered), reduced)
            return

        column = ColumnParallelLinear(4, 6).double()
        row = RowParallelLinear(6, 3).double()
        assert column.weight.shape == (3, 4)
        assert row.weight.shape == (3, 3)
        torch.manual_seed(37)  # Reference data is independent of constructor RNG use.
        full_column = torch.nn.Parameter(torch.randn(6, 4, dtype=torch.float64))
        full_row = torch.nn.Parameter(torch.randn(3, 6, dtype=torch.float64))
        with torch.no_grad():
            column.weight.copy_(full_column.chunk(2, dim=0)[rank])
            row.weight.copy_(full_row.chunk(2, dim=1)[rank])
        x = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
        xr = x.detach().clone().requires_grad_()
        output = row(F.gelu(column(x)))
        reference = F.linear(F.gelu(F.linear(xr, full_column)), full_row)
        torch.testing.assert_close(output, reference, rtol=1e-10, atol=1e-10)
        target = torch.randn_like(reference)
        F.mse_loss(output, target).backward()
        F.mse_loss(reference, target).backward()
        torch.testing.assert_close(x.grad, xr.grad, rtol=1e-9, atol=1e-10)
        torch.testing.assert_close(column.weight.grad, full_column.grad.chunk(2, 0)[rank],
                                   rtol=1e-9, atol=1e-10)
        torch.testing.assert_close(row.weight.grad, full_row.grad.chunk(2, 1)[rank],
                                   rtol=1e-9, atol=1e-10)
        torch.optim.SGD(list(column.parameters()) + list(row.parameters()), lr=0.03).step()
        torch.optim.SGD([full_column, full_row], lr=0.03).step()
        torch.testing.assert_close(column.weight, full_column.chunk(2, 0)[rank])
        torch.testing.assert_close(row.weight, full_row.chunk(2, 1)[rank])
        with pytest.raises(ValueError):
            ColumnParallelLinear(4, 5)
        with pytest.raises(ValueError):
            RowParallelLinear(5, 3)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("mode", ["collectives", "layers"])
def test_two_rank_contract(tmp_path, mode):
    assert dist.is_available() and dist.is_gloo_available(), (
        "Required CPU distributed gate needs a Gloo-enabled PyTorch build; use Linux/WSL."
    )
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=_worker, args=(rank, tmp_path / "store", mode))
                 for rank in range(2)]
    try:
        for process in processes:
            process.start()
        deadline = time.monotonic() + 45
        for process in processes:
            process.join(timeout=max(0, deadline - time.monotonic()))
        assert all(not p.is_alive() for p in processes), "Collective/process timeout"
        assert [p.exitcode for p in processes] == [0, 0], "See rank traceback above"
    finally:
        for process in processes:
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5)
