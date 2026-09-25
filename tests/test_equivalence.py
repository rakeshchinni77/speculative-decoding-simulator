"""
Equivalence tests asserting that speculative decoding produces the exact same output
as baseline autoregressive greedy decoding.
"""

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.generators import baseline_generate, speculative_generate


@pytest.fixture(scope="module")
def models_and_tokenizer():
    """Loads lightweight models for fast, deterministic equivalence testing."""
    draft_id = "gpt2"
    target_id = "gpt2"  # gpt2 as both draft and target or gpt2 vs gpt2 tests mathematical greedy guarantee

    tokenizer = AutoTokenizer.from_pretrained(draft_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    draft_model = AutoModelForCausalLM.from_pretrained(draft_id)
    target_model = AutoModelForCausalLM.from_pretrained(target_id)

    draft_model.eval()
    target_model.eval()

    return draft_model, target_model, tokenizer


@pytest.mark.parametrize("prompt", [
    "Translate to French: The meeting is scheduled for 3pm.",
    "The primary mechanism behind speculative decoding in large language models",
    "Once upon a time in a digital simulation",
])
@pytest.mark.parametrize("n_draft", [2, 4])
def test_speculative_greedy_equivalence(models_and_tokenizer, prompt, n_draft):
    """
    Asserts exact string match between baseline greedy generation and speculative decoding.
    """
    draft_model, target_model, tokenizer = models_and_tokenizer
    max_new_tokens = 15

    baseline_result = baseline_generate(
        model=target_model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens
    )

    speculative_result = speculative_generate(
        draft_model=draft_model,
        target_model=target_model,
        tokenizer=tokenizer,
        prompt=prompt,
        n_draft=n_draft,
        max_new_tokens=max_new_tokens
    )

    # 1. Exact text equivalence assertion
    assert baseline_result["text"] == speculative_result["text"], (
        f"Mismatch detected for prompt: '{prompt}' (n_draft={n_draft})\n"
        f"Baseline:    {baseline_result['text']!r}\n"
        f"Speculative: {speculative_result['text']!r}"
    )

    # 2. Schema assertions
    assert isinstance(baseline_result["latency"], float)
    assert isinstance(baseline_result["tokens_per_sec"], float)
    assert isinstance(speculative_result["latency"], float)
    assert isinstance(speculative_result["tokens_per_sec"], float)
    assert isinstance(speculative_result["acceptance_rate"], float)
    assert 0.0 <= speculative_result["acceptance_rate"] <= 1.0
