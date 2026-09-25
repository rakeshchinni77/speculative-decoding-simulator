"""
Verification logic for speculative decoding.

Compares candidate draft tokens against target model predictions
to find the longest matching prefix and the first divergence.
"""

from typing import List, Tuple, Optional, Union
import torch


def verify_draft_tokens(
    draft_tokens: Union[List[int], torch.Tensor],
    target_predictions: Union[List[int], torch.Tensor],
    target_bonus_token: Optional[int] = None,
    eos_token_id: Optional[int] = None
) -> Tuple[List[int], Optional[int], int, bool]:
    """
    Compares drafted tokens with target model predictions.

    Args:
        draft_tokens: Sequence of tokens proposed by draft model (length K).
        target_predictions: Target model predictions at draft positions (length K).
        target_bonus_token: Target model prediction for position K+1 (if all K match).
        eos_token_id: Optional EOS token ID for early stopping check.

    Returns:
        Tuple containing:
        - accepted_draft_tokens (List[int]): Accepted tokens from draft proposals.
        - next_token (Optional[int]): Correction token on mismatch, or bonus token if all match.
        - accepted_count (int): Number of draft tokens accepted.
        - hit_eos (bool): True if an accepted token or next_token matches eos_token_id.
    """
    if isinstance(draft_tokens, torch.Tensor):
        draft_tokens = draft_tokens.tolist()
    if isinstance(target_predictions, torch.Tensor):
        target_predictions = target_predictions.tolist()

    k = len(draft_tokens)
    accepted_draft_tokens: List[int] = []

    for i in range(k):
        d_tok = draft_tokens[i]
        t_tok = target_predictions[i]

        if d_tok == t_tok:
            accepted_draft_tokens.append(d_tok)
            if eos_token_id is not None and d_tok == eos_token_id:
                # Reached EOS within accepted draft tokens
                return accepted_draft_tokens, None, len(accepted_draft_tokens), True
        else:
            # First divergence at position i. Target model proposes correction token t_tok.
            hit_eos = (eos_token_id is not None and t_tok == eos_token_id)
            return accepted_draft_tokens, t_tok, len(accepted_draft_tokens), hit_eos

    # All K draft tokens matched
    hit_eos = False
    if target_bonus_token is not None and eos_token_id is not None and target_bonus_token == eos_token_id:
        hit_eos = True

    return accepted_draft_tokens, target_bonus_token, len(accepted_draft_tokens), hit_eos
