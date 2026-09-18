"""Response-masked supervised NLL and sequence-score DPO."""


def masked_nll(logits, targets, mask):
    """Mean NLL over True mask positions; inputs already shifted exactly once.

    logits[B,T,V], targets[B,T] int64 within [0,V), mask[B,T] bool.
    Return scalar with stable log probabilities and autograd support. All-masked
    or incompatible inputs raise ValueError. Masked targets are still valid IDs.
    Do not call cross_entropy/nll_loss to implement this exercise.
    """
    raise NotImplementedError("Explain the reduction denominator and mask.")


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1):
    """Mean -log sigmoid(beta*((pc-pr)-(rc-rr))) for nonempty [B] inputs.

    Inputs are response-only SUMMED sequence log probabilities. Reference scores
    are detached internally, even if supplied with requires_grad=True. beta must
    be finite and >0; invalid beta/shapes raise ValueError. Remain finite for
    logits of magnitude 1e4; no probability-space log(sigmoid(...)) underflow.
    """
    raise NotImplementedError("Derive the pairwise log-ratio margin.")
