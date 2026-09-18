"""Deterministic CPU numerical contracts, without learner solution code."""

import pytest
import torch


@pytest.fixture(autouse=True)
def deterministic_cpu():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026)
        yield
    torch.set_num_threads(previous_threads)
