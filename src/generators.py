"""
Inference generators for speculative decoding and baseline autoregressive decoding.

Implements manual greedy decoding loops without relying on high-level library generation APIs.
"""

import time
import logging
from typing import Dict, Any, Union, List
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from src.verify import verify_draft_tokens

# Set up module-level logger
logger = logging.getLogger("src.generators")
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)


def _get_model_device(model: PreTrainedModel) -> torch.device:
    """Helper to detect the device of a model."""
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError):
        return torch.device("cpu")


def baseline_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: Union[str, torch.Tensor],
    max_new_tokens: int
) -> Dict[str, Any]:
    """
    Performs standard, naive autoregressive greedy decoding manually.

    Args:
        model: Hugging Face causal language model.
        tokenizer: Hugging Face tokenizer.
        prompt: Input text string or input_ids tensor.
        max_new_tokens: Maximum number of new tokens to generate.

    Returns:
        Dict containing:
        - text (str): Generated text excluding prompt.
        - latency (float): Total generation time in seconds.
        - tokens_per_sec (float): Generation throughput in tokens per second.
    """
    device = _get_model_device(model)
    model.eval()

    # Encode input prompt
    if isinstance(prompt, str):
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded.input_ids.to(device)
    else:
        input_ids = prompt.to(device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

    prompt_length = input_ids.shape[1]
    eos_token_id = tokenizer.eos_token_id or getattr(model.config, "eos_token_id", None)

    generated_ids: List[int] = []
    current_ids = input_ids.clone()

    start_time = time.perf_counter()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids=current_ids)
            logits = outputs.logits  # Shape: (1, seq_len, vocab_size)
            next_token_logits = logits[:, -1, :]  # Shape: (1, vocab_size)
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # Shape: (1, 1)

            next_token_id = next_token.item()
            generated_ids.append(next_token_id)
            current_ids = torch.cat([current_ids, next_token], dim=-1)

            if eos_token_id is not None and next_token_id == eos_token_id:
                break

    end_time = time.perf_counter()
    latency = end_time - start_time
    tokens_generated = len(generated_ids)
    tokens_per_sec = float(tokens_generated / latency) if latency > 0 else 0.0

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return {
        "text": generated_text,
        "latency": float(latency),
        "tokens_per_sec": float(tokens_per_sec)
    }


def speculative_generate(
    draft_model: PreTrainedModel,
    target_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: Union[str, torch.Tensor],
    n_draft: int,
    max_new_tokens: int
) -> Dict[str, Any]:
    """
    Performs speculative decoding using a draft model and target verification in single forward passes.

    Args:
        draft_model: Smaller, faster draft language model.
        target_model: Larger, higher-quality target language model.
        tokenizer: Hugging Face tokenizer.
        prompt: Input text string or input_ids tensor.
        n_draft: Number of candidate tokens to draft per step.
        max_new_tokens: Maximum number of new tokens to generate.

    Returns:
        Dict containing:
        - text (str): Generated text excluding prompt.
        - latency (float): Total generation time in seconds.
        - tokens_per_sec (float): Generation throughput in tokens per second.
        - acceptance_rate (float): Ratio of accepted draft tokens to total proposed draft tokens.
    """
    draft_device = _get_model_device(draft_model)
    target_device = _get_model_device(target_model)

    draft_model.eval()
    target_model.eval()

    # Encode input prompt
    if isinstance(prompt, str):
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded.input_ids.to(target_device)
    else:
        input_ids = prompt.to(target_device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

    prompt_length = input_ids.shape[1]
    eos_token_id = tokenizer.eos_token_id or getattr(target_model.config, "eos_token_id", None)

    current_ids = input_ids.clone()
    generated_ids: List[int] = []

    total_proposed_draft_tokens = 0
    total_accepted_draft_tokens = 0
    step_count = 0

    start_time = time.perf_counter()

    with torch.no_grad():
        while len(generated_ids) < max_new_tokens:
            step_count += 1
            remaining_budget = max_new_tokens - len(generated_ids)
            k_draft = min(n_draft, remaining_budget)

            if k_draft <= 0:
                break

            # 1. Draft Phase: Autoregressively draft k_draft tokens with draft_model
            draft_current_ids = current_ids.to(draft_device)
            proposed_draft_tokens: List[int] = []
            draft_hit_eos = False

            for _ in range(k_draft):
                draft_out = draft_model(input_ids=draft_current_ids)
                draft_logits = draft_out.logits[:, -1, :]
                draft_next_token = torch.argmax(draft_logits, dim=-1, keepdim=True)
                draft_tok_id = draft_next_token.item()

                proposed_draft_tokens.append(draft_tok_id)
                draft_current_ids = torch.cat([draft_current_ids, draft_next_token], dim=-1)

                if eos_token_id is not None and draft_tok_id == eos_token_id:
                    draft_hit_eos = True
                    break

            num_proposed = len(proposed_draft_tokens)
            total_proposed_draft_tokens += num_proposed

            # 2. Target Forward Pass: Evaluate current sequence + proposed tokens in a single forward pass
            prefix_len = current_ids.shape[1]
            draft_tokens_tensor = torch.tensor([proposed_draft_tokens], device=target_device, dtype=torch.long)
            target_input = torch.cat([current_ids, draft_tokens_tensor], dim=-1)

            target_out = target_model(input_ids=target_input)
            target_logits = target_out.logits  # Shape: (1, prefix_len + num_proposed, vocab)

            # Target predictions at draft positions
            # Index prefix_len - 1 predicts token at prefix_len (draft token 0)
            # Index prefix_len - 1 + i predicts draft token i
            target_predictions: List[int] = []
            for i in range(num_proposed):
                pred_pos = prefix_len - 1 + i
                pred_tok = torch.argmax(target_logits[0, pred_pos, :], dim=-1).item()
                target_predictions.append(pred_tok)

            # Target prediction for position after all drafted tokens (bonus token if all match)
            bonus_pos = prefix_len - 1 + num_proposed
            target_bonus_tok = torch.argmax(target_logits[0, bonus_pos, :], dim=-1).item()

            # 3. Verification Phase
            accepted_draft_tokens, next_token, accepted_count, hit_eos = verify_draft_tokens(
                draft_tokens=proposed_draft_tokens,
                target_predictions=target_predictions,
                target_bonus_token=target_bonus_tok,
                eos_token_id=eos_token_id
            )

            total_accepted_draft_tokens += accepted_count

            # Logging telemetry per step
            all_matched = (accepted_count == num_proposed)
            divergence_idx = accepted_count if not all_matched else None
            logger.info(
                f"Speculative Step {step_count}: proposed={num_proposed}, accepted={accepted_count}, "
                f"all_matched={all_matched}, divergence_idx={divergence_idx}, remaining_budget={remaining_budget}"
            )
            logger.debug(
                f"Draft tokens: {proposed_draft_tokens}, Target predictions: {target_predictions}, "
                f"Next token: {next_token}"
            )

            # 4. Update generated sequence
            tokens_to_append: List[int] = list(accepted_draft_tokens)

            # If we haven't hit EOS and have budget, append the correction or bonus token
            if next_token is not None and len(generated_ids) + len(tokens_to_append) < max_new_tokens:
                tokens_to_append.append(next_token)

            # Append to generated list and tensor
            if tokens_to_append:
                append_tensor = torch.tensor([tokens_to_append], device=target_device, dtype=torch.long)
                current_ids = torch.cat([current_ids, append_tensor], dim=-1)
                generated_ids.extend(tokens_to_append)

            if hit_eos:
                break

    end_time = time.perf_counter()
    latency = end_time - start_time
    tokens_generated = len(generated_ids)
    tokens_per_sec = float(tokens_generated / latency) if latency > 0 else 0.0

    acceptance_rate = (
        float(total_accepted_draft_tokens / total_proposed_draft_tokens)
        if total_proposed_draft_tokens > 0
        else 0.0
    )
    # Ensure acceptance rate is within [0.0, 1.0]
    acceptance_rate = max(0.0, min(1.0, acceptance_rate))

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return {
        "text": generated_text,
        "latency": float(latency),
        "tokens_per_sec": float(tokens_per_sec),
        "acceptance_rate": float(acceptance_rate)
    }
