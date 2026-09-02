"""Optional Hugging Face adapter for the model-independent DISC core."""

from __future__ import annotations

import math
from collections.abc import Sequence

from .core import DiscDetector, DiscEncoder, token_ids_to_bits


def _import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("Hugging Face generation requires: pip install -e '.[hf]'") from exc
    return torch


def generate(
    model,
    tokenizer,
    prompt: str,
    encoder: DiscEncoder,
    *,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
) -> tuple[str, list[int]]:
    """Generate DISC-watermarked text using any causal Transformers model."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    torch = _import_torch()
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    generated: list[int] = []
    past_key_values = None
    model_input = input_ids
    model.eval()
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            output = model(input_ids=model_input, past_key_values=past_key_values, use_cache=True)
            past_key_values = output.past_key_values
            probabilities = torch.softmax(output.logits[0, -1].float() / temperature, dim=-1)
            token_id = encoder.encode_token(probabilities.cpu().numpy())
            generated.append(token_id)
            if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
                break
            model_input = torch.tensor([[token_id]], device=device)
    return tokenizer.decode(generated, skip_special_tokens=True), generated


def detect_token_ids(
    token_ids: Sequence[int],
    vocab_size: int,
    detector: DiscDetector,
    *,
    token_aligned_starts: bool = False,
):
    width = math.ceil(math.log2(vocab_size))
    bits = token_ids_to_bits(token_ids, width)
    candidates = range(width, len(bits), width) if token_aligned_starts else None
    return detector.detect(bits, n_star_candidates=candidates)
