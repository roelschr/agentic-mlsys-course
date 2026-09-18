"""Last-axis RMSNorm with custom backward, and adjacent-pair RoPE."""

import torch


class RMSNormFunction(torch.autograd.Function):
    """y = x * gamma / sqrt(mean(x**2, dim=-1, keepdim=True) + eps).

    x [..., D], gamma [D], eps > 0; output matches x shape/dtype.
    Accumulate in at least float32, preserve float64 for numerical checks.
    Support noncontiguous input. No nn.RMSNorm/F.rms_norm in the learner code.
    First-order gradients for x and gamma are required; double backward is not.
    """

    @staticmethod
    def forward(ctx, x, gamma, eps):
        """Record only the tensors needed for the analytic backward."""
        raise NotImplementedError("Derive the normalization statistic.")

    @staticmethod
    def backward(ctx, grad_output):
        """Return gradients for x, gamma, and None for eps; see Week 2's VJP."""
        raise NotImplementedError("Derive the last-axis reduction in the VJP.")


def apply_rope(x, positions, base=10000.0):
    """Rotate x[B,H,T,D] adjacent pairs, preserving shape/dtype and gradients.

    positions [T] contains absolute positions (negative allowed for inversion).
    Pair j rotates by positions * base**(-2*j/D). Require even D and base > 0;
    invalid shape/base raises ValueError. Positions broadcast across B and H.
    """
    raise NotImplementedError("Relate pairwise rotation to complex multiplication.")
