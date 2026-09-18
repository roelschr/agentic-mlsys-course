"""Two tensor-parallel layers; communication/autograd belongs to the learner."""

from torch import nn


class ColumnParallelLinear(nn.Module):
    """Bias-free local output shard for replicated x[...,in_features].

    Default group is the initialized distributed world group (Gloo in tests).
    Local registered weight [out_features/world_size,in_features], contiguous
    rank-ordered output partition. Forward returns [...,out_features/world_size].
    Input grad must equal the complete unsharded input gradient when composed
    with RowParallelLinear. Output dimension must divide world size; ValueError
    otherwise. Constructor initialization distribution is not prescribed.
    Communication autograd helper(s) count toward this component's line budget.
    """

    def __init__(self, in_features, out_features, group=None):
        """Register the local weight and group; no bias."""
        raise NotImplementedError("Label replicated and partitioned dimensions.")

    def forward(self, x):
        """Compute the local output with correct replicated-input gradient."""
        raise NotImplementedError("Derive the gradient ownership invariant.")


class RowParallelLinear(nn.Module):
    """Bias-free replicated output for already-sharded x[...,in_features/p].

    Local registered weight [out_features,in_features/world_size]; input feature
    dimension must divide world size or raise ValueError. Return [...,out_features].
    Every rank evaluates the same logical downstream loss; do not multiply its
    gradient by world size. No implicit input scatter or output gather API.
    Communication autograd helper(s) count toward this component's line budget.
    """

    def __init__(self, in_features, out_features, group=None):
        """Register the local weight and group; no bias."""
        raise NotImplementedError("Relate local partial outputs to the full output.")

    def forward(self, x):
        """Return a replicated full output with correctly scaled local gradients."""
        raise NotImplementedError("Predict a world-size scaling failure.")
