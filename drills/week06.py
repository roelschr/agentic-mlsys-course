"""Paired contrastive loss and discrete hard-negative mining."""


def symmetric_infonce(a, b, temperature=0.1):
    """Symmetric row/column classification loss for paired a/b[N,D], N,D>0.

    L2-normalize both embeddings internally (epsilon=1e-12); diagonal matches
    are positives. Return scalar mean of the two directional mean losses.
    Positive finite temperature required; incompatible inputs raise ValueError.
    Zero vectors must yield finite losses/gradients. Do not call cross_entropy
    in learner code; use stable log-sum-exp mechanics.
    """
    raise NotImplementedError("Derive one positive pair's softmax gradient.")


def hard_negative_indices(scores, positive_indices, k):
    """Return int64 [N,k] candidate indices for finite scores[N,M].

    Exclude positive_indices[N] (valid int64 column indices). Rank by descending
    score, lower candidate ID on ties. Require 0<=k<M and nonempty N,M;
    k=0 returns [N,0]. Invalid inputs raise ValueError. No differentiable
    selection or similarity computation is required in this component.
    """
    raise NotImplementedError("Which candidate must never be mined?")
