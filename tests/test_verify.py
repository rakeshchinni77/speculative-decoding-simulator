"""
Unit tests for verification logic in src/verify.py.
"""

import pytest
import torch
from src.verify import verify_draft_tokens


def test_verify_all_matched():
    draft = [10, 20, 30, 40]
    target_preds = [10, 20, 30, 40]
    bonus = 50

    accepted, next_tok, count, hit_eos = verify_draft_tokens(
        draft, target_preds, target_bonus_token=bonus, eos_token_id=50256
    )

    assert accepted == [10, 20, 30, 40]
    assert next_tok == 50
    assert count == 4
    assert not hit_eos


def test_verify_mismatch_at_start():
    draft = [10, 20, 30, 40]
    target_preds = [99, 20, 30, 40]
    bonus = 50

    accepted, next_tok, count, hit_eos = verify_draft_tokens(
        draft, target_preds, target_bonus_token=bonus, eos_token_id=50256
    )

    assert accepted == []
    assert next_tok == 99
    assert count == 0
    assert not hit_eos


def test_verify_mismatch_in_middle():
    draft = [10, 20, 30, 40]
    target_preds = [10, 20, 99, 40]
    bonus = 50

    accepted, next_tok, count, hit_eos = verify_draft_tokens(
        draft, target_preds, target_bonus_token=bonus, eos_token_id=50256
    )

    assert accepted == [10, 20]
    assert next_tok == 99
    assert count == 2
    assert not hit_eos


def test_verify_mismatch_at_last_token():
    draft = [10, 20, 30, 40]
    target_preds = [10, 20, 30, 99]
    bonus = 50

    accepted, next_tok, count, hit_eos = verify_draft_tokens(
        draft, target_preds, target_bonus_token=bonus, eos_token_id=50256
    )

    assert accepted == [10, 20, 30]
    assert next_tok == 99
    assert count == 3
    assert not hit_eos


def test_verify_with_eos_in_draft():
    draft = [10, 50256, 30]
    target_preds = [10, 50256, 30]

    accepted, next_tok, count, hit_eos = verify_draft_tokens(
        draft, target_preds, target_bonus_token=40, eos_token_id=50256
    )

    assert accepted == [10, 50256]
    assert next_tok is None
    assert count == 2
    assert hit_eos is True


def test_verify_with_tensors():
    draft = torch.tensor([1, 2, 3])
    target = torch.tensor([1, 2, 9])

    accepted, next_tok, count, hit_eos = verify_draft_tokens(
        draft, target, target_bonus_token=4
    )

    assert accepted == [1, 2]
    assert next_tok == 9
    assert count == 2
    assert not hit_eos
