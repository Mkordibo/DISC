"""Optional Hugging Face adapter for the model-independent DISC core.

Converts a causal LM's next-token distribution into the 1-D ``float`` vector
that ``DiscEncoder.encode_token`` expects, and converts generated token IDs
back into the 0/1 bit string that ``DiscDetector.detect`` expects.

Typical types:

    model: transformers.PreTrainedModel     # e.g. GPT-2
    tokenizer: transformers.PreTrainedTokenizer
    prompt: str                             # e.g. "Explain channel capacity:"
    encoder: DiscEncoder
    generated_ids: list[int]                # token IDs, e.g. [464, 318, ...]
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Sequence

from .core import DiscDetector, DiscEncoder, token_ids_to_bits


def _import_torch():
    """Import torch lazily so CPU-only installs without ``[hf]`` still load ``disc``.

    Returns:
        The ``torch`` module.

    Raises:
        ImportError: If PyTorch is not installed.
    """
    try:
        # Resolve the optional package dynamically so importing ``disc`` does
        # not require PyTorch and static analysis does not flag a missing local
        # Hugging Face dependency at this source line.
        return importlib.import_module("torch")
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("Hugging Face generation requires: pip install -e '.[hf]'") from exc


def generate(
    model,
    tokenizer,
    prompt: str,
    encoder: DiscEncoder,
    *,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
) -> tuple[str, list[int]]:
    """Generate DISC-watermarked text using any causal Transformers model.

    At each step the last-position softmax becomes a 1-D numpy array of shape
    ``(vocab_size,)`` with ``dtype=float32/float64`` values in ``[0, 1]``.
    ``encoder.encode_token`` walks that distribution bit by bit.

    Args:
        model: Hugging Face causal LM with a ``.forward`` that returns
            ``logits`` of shape ``(batch=1, seq, vocab_size)``. Example:
            ``AutoModelForCausalLM.from_pretrained("openai-community/gpt2")``.
        tokenizer: Matching tokenizer. Example:
            ``AutoTokenizer.from_pretrained("openai-community/gpt2")``.
        prompt: Prompt string. Example: ``"Explain channel capacity:"``.
        encoder: Stateful ``DiscEncoder`` (holds key, payload, bits so far).
        max_new_tokens: Maximum new token IDs to emit, ``int``. Default ``128``.
        temperature: Softmax temperature, ``float > 0``. Default ``1.0``.
            Values ``< 1`` sharpen the distribution; ``> 1`` flatten it.

    Returns:
        ``(text, generated_ids)`` where
            - ``text`` is ``str``, the decoded continuation (special tokens
              stripped), e.g. ``" The capacity of a channel..."``;
            - ``generated_ids`` is ``list[int]`` of new token IDs only
              (prompt IDs are not included), e.g. ``[198, 464, 318]``.
              Generation stops early if ``eos_token_id`` is sampled.
    """

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    torch = _import_torch()
    device = next(model.parameters()).device  # torch.device, e.g. cpu or cuda:0
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)  # Tensor[int64], shape (1, prompt_len)
    generated: list[int] = []  # newly sampled token IDs
    past_key_values = None  # KV cache; None on the first forward pass
    model_input = input_ids
    model.eval()
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            output = model(input_ids=model_input, past_key_values=past_key_values, use_cache=True)
            past_key_values = output.past_key_values
            # logits[0, -1]: Tensor[float], shape (vocab_size,)
            probabilities = torch.softmax(output.logits[0, -1].float() / temperature, dim=-1)
            # encode_token expects a 1-D numpy array of floats summing to ~1.
            token_id = encoder.encode_token(probabilities.cpu().numpy())  # int, e.g. 464
            generated.append(token_id)
            if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
                break
            model_input = torch.tensor([[token_id]], device=device)  # shape (1, 1)
    return tokenizer.decode(generated, skip_special_tokens=True), generated


def generate_unwatermarked(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    seed: int | None = None,
) -> tuple[str, list[int]]:
    """Sample an ordinary causal-LM continuation for a paired DISC control.

    The model, tokenizer, prompt, length, and temperature have the same
    format as :func:`generate`.  ``seed`` is an optional integer used only by
    this control sampler.  The return value is ``(text, token_ids)`` where
    ``text`` is the decoded continuation and ``token_ids`` contains only its
    newly generated vocabulary IDs.  No DISC encoder, key, PRF, or watermark
    decision is involved.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    torch = _import_torch()
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    # A per-call generator makes paired experiments reproducible without
    # changing the caller's global PyTorch RNG state.
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
    generated: list[int] = []
    past_key_values = None
    model_input = input_ids
    model.eval()
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            output = model(input_ids=model_input, past_key_values=past_key_values, use_cache=True)
            past_key_values = output.past_key_values
            probabilities = torch.softmax(output.logits[0, -1].float() / temperature, dim=-1)
            token_id = int(torch.multinomial(probabilities, 1, generator=generator).item())
            generated.append(token_id)
            if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
                break
            model_input = torch.tensor([[token_id]], device=device)
    return tokenizer.decode(generated, skip_special_tokens=True), generated


def perplexity_from_token_ids(model, token_ids: Sequence[int]) -> float:
    """Return continuation perplexity for a nonempty sequence of token IDs.

    Args:
        model: Hugging Face causal LM used for generation.
        token_ids: ``Sequence[int]`` of continuation IDs. At least two IDs
            are required because causal-LM loss predicts the next ID.

    Returns:
        ``float`` perplexity, or ``float('nan')`` if fewer than two IDs are
        supplied. This intentionally scores only the continuation, matching
        MirrorMark's checkpoint output convention.
    """
    if len(token_ids) < 2:
        return float("nan")
    torch = _import_torch()
    device = next(model.parameters()).device
    ids = torch.tensor([list(token_ids)], dtype=torch.long, device=device)
    model.eval()
    with torch.inference_mode():
        loss = model(input_ids=ids, labels=ids).loss
    return float(torch.exp(loss).item())


def detect_token_ids(
    token_ids: Sequence[int],
    vocab_size: int,
    detector: DiscDetector,
    *,
    token_aligned_starts: bool = False,
):
    """Run DISC detection on a sequence of vocabulary token IDs.

    Token IDs are expanded to ``ceil(log2(vocab_size))`` bits each (MSB first)
    before calling ``detector.detect``.

    Args:
        token_ids: Sequence of ``int`` token IDs. Example: the
            ``generated_ids`` list from ``generate``, ``[198, 464, 318]``.
            Each ID must fit in ``ceil(log2(vocab_size))`` bits.
        vocab_size: Vocabulary size, ``int``. Example: ``len(tokenizer)``
            which for GPT-2 is ``50257``. Determines bits per token
            (GPT-2 → ``ceil(log2(50257)) = 16``).
        detector: ``DiscDetector`` configured with the same key, ``m``,
            context width, and message mapping as the encoder.
        token_aligned_starts: ``bool``. If True, only try ``n_star`` values
            that fall on token boundaries (multiples of the bit width).
            Default False searches every bit offset, matching Algorithm 4.

    Returns:
        ``DetectionResult`` (see ``disc.core.DetectionResult``). Example
        fields: ``detected=True``, ``payload=3``, ``n_star=32``.
    """
    width = math.ceil(math.log2(vocab_size))  # int bits/token, e.g. 16
    bits = token_ids_to_bits(token_ids, width)  # list[int] of 0/1, length = N * width
    if not detector.use_prefix:
        candidates = [0]
    elif detector.context_mode == "prefix_token_ngram":
        # Token-prefix detection searches R in real-token units. The core then
        # converts a candidate k to its reported bit offset k * width.
        candidates = range(1, len(token_ids))
    else:
        candidates = range(width, len(bits), width) if token_aligned_starts else None
    return detector.detect(
        bits,
        n_star_candidates=candidates,
        token_ids=token_ids,
        bit_length=width,
    )
