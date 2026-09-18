"""Explicit grouped attention and an inference-only static KV cache."""

import torch


def gqa(q, k, v, allowed=None):
    """Return [B,Hq,Tq,D] grouped attention (no dropout or projection layers).

    q[B,Hq,Tq,D], k/v[B,Hkv,Tk,D]; Hq must be a multiple of Hkv.
    Contiguous query-head groups share one KV head. Scale scores by sqrt(D).
    allowed is a boolean [Tq,Tk] tensor, True = permitted; None = all permitted.
    Fully masked query rows must yield zero outputs and finite zero gradients
    for that row. Invalid shapes raise ValueError. Support ordinary autograd.
    Preserve float64; use at least float32 score/softmax reductions otherwise.
    Do not implement this drill using scaled_dot_product_attention.
    """
    raise NotImplementedError("Write down head grouping and mask semantics.")


class StaticKVCache:
    """Single-layer detached cache; K/V are allocated once, never concatenated.

    Public k/v: [batch,kv_heads,capacity,head_dim], length: int initially zero.
    Prefix views share storage with k/v. Invalid append is atomic: ValueError
    with unchanged length AND tensors. Require positive allocation dimensions.
    Cache bytes = 2*batch*kv_heads*capacity*head_dim*element_size.
    """

    def __init__(self, batch, kv_heads, capacity, head_dim, *, dtype=torch.float32,
                 device="cpu"):
        """Allocate the backing buffers and initialize the logical length."""
        raise NotImplementedError("Distinguish capacity from valid prefix length.")

    def append(self, k, v):
        """Append matching [B,Hkv,T,D] tensors, detached, and return prefix()."""
        raise NotImplementedError("Specify append's capacity and shape invariant.")

    def prefix(self):
        """Return (k_valid, v_valid) views with time dimension self.length."""
        raise NotImplementedError("What must the views' storage identity be?")

    def reset(self):
        """Set length to zero without reallocating or clearing the buffers."""
        raise NotImplementedError("Which state is logical rather than physical?")
