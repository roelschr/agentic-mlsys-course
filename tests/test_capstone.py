"""Integration invariants; no alternative model, training loop, or decode engine."""

import copy
import math
from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F

import capstone.project as project


def _tiny():
    return project.TinyDecoder(vocab_size=16, d_model=16, n_layers=2,
                               n_heads=4, n_kv_heads=2, hidden=24)


def _corpus():
    return torch.tensor([[1, 2, 3, 4, 1, 2, 3, 4, 1]]).repeat(4, 1)


def _preferences():
    chosen = torch.tensor([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
    rejected = torch.tensor([[1, 2, 7, 8, 9], [2, 3, 8, 9, 10]])
    # Unequal response lengths; the final token of row 1 is padding.
    mask = torch.tensor([[False, False, True, True, True],
                         [False, False, True, True, False]])
    return chosen, rejected, mask.clone(), mask.clone()


def _cache_pointers(caches):
    return [(c.k.untyped_storage().data_ptr(), c.v.untyped_storage().data_ptr())
            for c in caches]


@pytest.mark.week9
def test_forward_causality_gradients_and_state_reload():
    model = _tiny()
    tokens = torch.tensor([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
    logits = model(tokens)
    assert logits.shape == (2, 5, 16) and torch.isfinite(logits).all()
    changed = tokens.clone()
    changed[:, 3:] = 11
    torch.testing.assert_close(model(changed)[:, :3], logits[:, :3])
    F.cross_entropy(logits.reshape(-1, 16), tokens.reshape(-1)).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               and p.grad.abs().sum() > 0 for p in model.parameters())
    restored = _tiny()
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    torch.testing.assert_close(restored(tokens), logits, rtol=0, atol=0)


@pytest.mark.week9
def test_sft_tiny_batch_loss_reduction():
    model = _tiny()
    trace = project.fit_sft(model, _corpus(), steps=40, lr=0.01)
    assert len(trace) == 41 and all(math.isfinite(float(x)) for x in trace)
    assert trace[-1] < 0.7 * trace[0]
    tokens = _corpus()
    with torch.no_grad():
        final = F.cross_entropy(model(tokens[:, :-1]).reshape(-1, 16),
                                tokens[:, 1:].reshape(-1))
    assert trace[-1] == pytest.approx(final.item(), rel=1e-5, abs=1e-6)


@pytest.mark.week9
def test_sft_and_forward_reuse_custom_components():
    # Spies observe call boundaries only; they do not replace learner logic.
    with patch.object(project.RMSNormFunction, "apply", wraps=project.RMSNormFunction.apply) as norm, \
         patch.object(project, "apply_rope", wraps=project.apply_rope) as rope, \
         patch.object(project, "gqa", wraps=project.gqa) as attention, \
         patch.object(project, "masked_nll", wraps=project.masked_nll) as nll, \
         patch.object(project.AdamW, "step", autospec=True, side_effect=project.AdamW.step) as step, \
         patch.object(project.SwiGLU, "forward", autospec=True, side_effect=project.SwiGLU.forward) as ff:
        model = _tiny()
        project.fit_sft(model, _corpus(), steps=1)
    assert norm.call_count >= 1 and rope.call_count >= 1
    assert attention.call_count >= 1 and ff.call_count >= 1
    assert nll.call_count >= 2 and step.call_count == 1
    assert sum(isinstance(m, project.SwiGLU) for m in model.modules()) == model.n_layers


def test_dpo_scores_sum_responses_and_ignore_padding():
    policy = _tiny()
    reference = copy.deepcopy(policy)
    chosen, rejected, cm, rm = _preferences()
    with torch.no_grad():
        chosen_nll = F.cross_entropy(policy(chosen[:, :-1]).transpose(1, 2),
                                     chosen[:, 1:], reduction="none")
        rejected_nll = F.cross_entropy(policy(rejected[:, :-1]).transpose(1, 2),
                                       rejected[:, 1:], reduction="none")
        expected_margin = ((rejected_nll * rm[:, 1:]).sum(-1)
                           - (chosen_nll * cm[:, 1:]).sum(-1)).mean().item()
    metrics = project.fit_dpo(policy, reference, chosen, rejected, cm, rm, steps=0)
    assert metrics["initial_loss"] == pytest.approx(math.log(2), rel=1e-5)
    assert metrics["initial_margin"] == pytest.approx(expected_margin, abs=1e-5)
    padded_chosen = F.pad(chosen, (0, 2), value=13)
    padded_rejected = F.pad(rejected, (0, 2), value=14)
    padded = project.fit_dpo(policy, reference, padded_chosen, padded_rejected,
                             F.pad(cm, (0, 2), value=False),
                             F.pad(rm, (0, 2), value=False), steps=0)
    for key in ("initial_loss", "final_loss", "initial_margin", "final_margin"):
        assert padded[key] == pytest.approx(metrics[key], abs=2e-5)


def test_dpo_improves_margin_and_reference_is_frozen():
    policy = _tiny()
    reference = copy.deepcopy(policy)
    snapshot = copy.deepcopy(reference.state_dict())
    before = copy.deepcopy(policy.state_dict())
    with patch.object(project, "dpo_loss", wraps=project.dpo_loss) as loss, \
         patch.object(project.AdamW, "step", autospec=True, side_effect=project.AdamW.step) as step:
        metrics = project.fit_dpo(policy, reference, *_preferences(), steps=12, lr=0.002)
    assert loss.call_count >= 2 and step.call_count == 12
    assert all(math.isfinite(float(value)) for value in metrics.values())
    assert metrics["final_loss"] < metrics["initial_loss"]
    assert metrics["final_margin"] > metrics["initial_margin"]
    for name, tensor in reference.state_dict().items():
        torch.testing.assert_close(tensor, snapshot[name], rtol=0, atol=0)
    assert all(p.grad is None and not p.requires_grad for p in reference.parameters())
    assert any(not torch.equal(p, before[name]) for name, p in policy.state_dict().items())


def test_cached_forward_chunk_offsets_capacity_and_atomic_rejection():
    model = _tiny().eval()
    engine = project.CacheEngine(model, capacity=8)
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
    assert len(engine.caches) == model.n_layers
    pointers = _cache_pointers(engine.caches)
    expected_bytes = 2 * model.n_layers * model.n_kv_heads * 8 * model.head_dim * 4
    actual_bytes = sum(c.k.untyped_storage().nbytes() + c.v.untyped_storage().nbytes()
                       for c in engine.caches)
    assert actual_bytes == expected_bytes
    with torch.no_grad():
        full = model(tokens)
        chunks = [model(tokens[:, :2], caches=engine.caches, offset=0),
                  model(tokens[:, 2:5], caches=engine.caches, offset=2),
                  model(tokens[:, 5:], caches=engine.caches, offset=5)]
    torch.testing.assert_close(torch.cat(chunks, dim=1), full, rtol=2e-4, atol=2e-5)
    assert all(c.length == 6 for c in engine.caches)
    assert _cache_pointers(engine.caches) == pointers
    saved = [(c.k.clone(), c.v.clone()) for c in engine.caches]
    with torch.no_grad(), pytest.raises(ValueError):
        model(tokens[:, :3], caches=engine.caches, offset=6)
    with torch.no_grad(), pytest.raises(ValueError):
        model(tokens[:, :1], caches=engine.caches, offset=1)
    for cache, (k, v) in zip(engine.caches, saved):
        assert cache.length == 6
        torch.testing.assert_close(cache.k, k, equal_nan=True)
        torch.testing.assert_close(cache.v, v, equal_nan=True)


def test_engine_greedy_matches_full_logits_and_reuses_cache():
    model = _tiny().eval()
    engine = project.CacheEngine(model, capacity=9)
    pointers = _cache_pointers(engine.caches)
    prompt = torch.tensor([[1, 2, 3]])
    with patch.object(model, "forward", wraps=model.forward) as calls, \
         patch.object(project.StaticKVCache, "append", autospec=True,
                      side_effect=project.StaticKVCache.append) as append:
        result = engine.generate(prompt, max_new_tokens=4)
    assert result.shape == (1, 7) and result.dtype == torch.int64
    torch.testing.assert_close(result[:, :3], prompt)
    lengths = [call.args[0].shape[1] for call in calls.call_args_list]
    assert lengths[0] == 3 and all(length == 1 for length in lengths[1:])
    assert 4 <= len(lengths) <= 5  # The last sampled token may optionally be cached.
    assert append.call_count >= model.n_layers * 4
    # Teacher-forced greedy invariant checks every generated token at once;
    # this is not a second autoregressive decoding implementation.
    with torch.no_grad():
        full = model(result[:, :-1])
    torch.testing.assert_close(result[:, 3:], full[:, 2:].argmax(-1))
    again = engine.generate(prompt, max_new_tokens=4)
    torch.testing.assert_close(result, again)
    assert _cache_pointers(engine.caches) == pointers
    state = [(c.length, c.k.clone(), c.v.clone()) for c in engine.caches]
    with pytest.raises(ValueError):
        engine.generate(prompt, max_new_tokens=7)
    for cache, (length, k, v) in zip(engine.caches, state):
        assert cache.length == length
        torch.testing.assert_close(cache.k, k, equal_nan=True)
        torch.testing.assert_close(cache.v, v, equal_nan=True)
    torch.testing.assert_close(engine.generate(prompt, max_new_tokens=0), prompt)
    assert all(c.length == 0 for c in engine.caches)
    assert _cache_pointers(engine.caches) == pointers


def test_sft_dpo_cached_serve_single_policy_smoke():
    policy = _tiny()
    sft = project.fit_sft(policy, _corpus(), steps=8)
    assert sft[-1] < sft[0]
    reference = copy.deepcopy(policy)
    dpo = project.fit_dpo(policy, reference, *_preferences(), steps=4)
    assert dpo["final_loss"] < dpo["initial_loss"]
    engine = project.CacheEngine(policy, capacity=8)
    result = engine.generate(torch.tensor([[1, 2]]), max_new_tokens=3)
    assert result.shape == (1, 5)
    with torch.no_grad():
        logits = policy(result[:, :-1])
    torch.testing.assert_close(result[:, 2:], logits[:, 1:].argmax(-1))
