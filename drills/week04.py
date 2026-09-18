"""A gated feed-forward module and deterministic capacity-limited routing."""

import torch
from torch import nn


class SwiGLU(nn.Module):
    """down(silu(gate(x)) * up(x)); x[...,d_model], result[...,d_model].

    Expose bias-free nn.Linear modules gate/up [hidden,d_model] and
    down [d_model,hidden]. All parameters must be registered.
    """

    def __init__(self, d_model, hidden):
        """Create three learnable projections; no residual path here."""
        raise NotImplementedError("Compare gated and ordinary FFN parameter counts.")

    def forward(self, x):
        """Apply the SwiGLU formula, preserving leading dimensions."""
        raise NotImplementedError("Derive how the two hidden channels interact.")


def route_topk(logits, k, capacity_factor=1.0):
    """Return (indices, weights, keep, aux) from finite logits[N,E], N,E>0.

    indices/weights/keep have shape [N,k], int64/input dtype/bool respectively.
    Require 1<=k<=E, finite capacity_factor>0; invalid controls raise ValueError.
    Lower expert ID wins equal-score ties. Selected softmax weights normalize
    before drops. Capacity C=ceil(capacity_factor*N*k/E), admitting assignments
    in token-major then slot-major order. Dropped weights are zero; do not
    renormalize the survivors. Expert indices remain defined even when dropped.
    Aux = E*sum(f*P), where f is detached pre-drop assignment count/(N*k),
    and P is the mean FULL router softmax distribution, not top-k weights.
    """
    raise NotImplementedError("Predict a capacity-overflow example before coding.")
