"""
Unit and integration tests for src/generators.py.
"""

import ast
import inspect
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer
import src.generators as generators
from src.generators import baseline_generate, speculative_generate


def test_no_forbidden_generate_api_in_generators():
    """
    Asserts via AST that model.generate or .generate() is not called in src/generators.py.
    """
    source_file = inspect.getsourcefile(generators)
    with open(source_file, "r", encoding="utf-8") as f:
        source_code = f.read()

    tree = ast.parse(source_code)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Check function calls like obj.generate(...)
            if isinstance(node.func, ast.Attribute):
                assert node.func.attr != "generate", (
                    f"Forbidden method call '{node.func.attr}' found on line {node.lineno} in {source_file}"
                )
            elif isinstance(node.func, ast.Name):
                assert node.func.id != "generate", (
                    f"Forbidden function call '{node.func.id}' found on line {node.lineno} in {source_file}"
                )


def test_import_logging_present_in_generators():
    """
    Asserts that logging is imported and used in src/generators.py.
    """
    source_file = inspect.getsourcefile(generators)
    with open(source_file, "r", encoding="utf-8") as f:
        source_code = f.read()

    assert "import logging" in source_code or "from logging import" in source_code
    assert "logger.info" in source_code or "logger.debug" in source_code


@pytest.fixture(scope="module")
def gpt2_models():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()
    return model, tokenizer


def test_baseline_generator_contract(gpt2_models):
    model, tokenizer = gpt2_models
    prompt = "Hello, my name is"
    max_new_tokens = 5

    res = baseline_generate(model, tokenizer, prompt, max_new_tokens)

    assert isinstance(res, dict)
    assert set(res.keys()) == {"text", "latency", "tokens_per_sec"}
    assert isinstance(res["text"], str)
    assert isinstance(res["latency"], float)
    assert isinstance(res["tokens_per_sec"], float)
    assert res["latency"] > 0
    assert res["tokens_per_sec"] > 0


def test_speculative_generator_contract(gpt2_models):
    model, tokenizer = gpt2_models
    prompt = "Artificial Intelligence is"
    max_new_tokens = 6
    n_draft = 2

    res = speculative_generate(model, model, tokenizer, prompt, n_draft, max_new_tokens)

    assert isinstance(res, dict)
    assert set(res.keys()) == {"text", "latency", "tokens_per_sec", "acceptance_rate"}
    assert isinstance(res["text"], str)
    assert isinstance(res["latency"], float)
    assert isinstance(res["tokens_per_sec"], float)
    assert isinstance(res["acceptance_rate"], float)
    assert 0.0 <= res["acceptance_rate"] <= 1.0
    assert res["latency"] > 0


def test_speculative_generator_various_n_draft(gpt2_models):
    model, tokenizer = gpt2_models
    prompt = "The quick brown fox"

    for n_draft in [1, 2, 4]:
        res = speculative_generate(model, model, tokenizer, prompt, n_draft=n_draft, max_new_tokens=8)
        assert isinstance(res["text"], str)
        assert 0.0 <= res["acceptance_rate"] <= 1.0
