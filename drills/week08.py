"""Idealized ZeRO persistent-state bytes and exact speculative correction."""


def zero_bytes(num_parameters, world_size, stage):
    """Return integer byte counts: parameters, gradients, master, m, v, total.

    BF16 parameters and gradients (2B each); FP32 master/m/v (4B each).
    ZeRO-1 shards master/m/v; ZeRO-2 also shards gradients; ZeRO-3 also shards
    parameters. ZeRO-0 shards nothing. Positive integer sizes, P%world_size=0,
    and integer stage in {0,1,2,3} required, else ValueError.
    Excludes activations, live gathers, padding, buckets, caches, allocator slack.
    This is an analytic ledger, not an allocation profiler or ZeRO runtime.
    """
    raise NotImplementedError("Assign each state tensor to an owner group.")


def speculative_distribution(p, q):
    """Return (acceptance[V], residual[V]) for target p and draft q.

    Require finite nonnegative same-shaped nonempty vectors normalized to 1
    (within 1e-6); ValueError otherwise. acceptance(x)=min(1,p(x)/q(x));
    define acceptance=1 wherever q=0. residual is normalized (p-q)_+.
    For p=q return p as the unused residual. Preserve input dtype/device.
    This is a one-step distribution primitive, not a sampling/generation loop.
    """
    raise NotImplementedError("Prove the accepted and corrected masses sum to p.")
