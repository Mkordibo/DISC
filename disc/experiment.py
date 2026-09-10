"""Paired Hugging Face experiments with the same artifact layout as MirrorMark.

The runner is deliberately a framework layer around DISC, not a replacement
for its algorithm.  It uses :class:`disc.core.HmacPrf` through
``DiscEncoder`` and ``DiscDetector`` exactly as before, while standardizing
prompt loading, paired WM/NWM sampling, checkpoint detection, timing, and
JSONL records with the MirrorMark experiment driver.
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .core import DetectionResult, DiscDetector, DiscEncoder
from .huggingface import (
    detect_token_ids,
    generate,
    generate_unwatermarked,
    perplexity_from_token_ids,
)


def parse_key(value: str) -> bytes | str | int:
    """Parse a CLI key without changing DISC's supported HMAC key types.

    Decimal and ``0x``-prefixed values become nonnegative ``int`` keys, so a
    MirrorMark-style hexadecimal key can be reused directly. Every other
    nonempty value remains a UTF-8 ``str`` key. Bytes keys remain available
    through the Python API, where they cannot be represented losslessly by a
    command-line string.
    """
    if not value:
        raise ValueError("secret key cannot be empty")
    try:
        parsed = int(value, 0)
    except ValueError:
        return value
    if parsed < 0:
        raise ValueError("integer keys must be non-negative")
    return parsed


def key_type(key: bytes | str | int) -> str:
    """Return the public type name for experiment metadata without logging a secret."""
    return "bytes" if isinstance(key, bytes) else "int" if isinstance(key, int) else "str"


def load_prompts(prompt: str, prompts_file: str | None) -> list[str]:
    """Load MirrorMark-compatible JSON prompt lists or newline-delimited prompts."""
    if prompts_file is None:
        return [prompt]
    contents = Path(prompts_file).read_text(encoding="utf-8")
    try:
        parsed = json.loads(contents)
    except json.JSONDecodeError:
        parsed = [line for line in contents.splitlines() if line.strip()]
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError("prompts file must be a JSON list of strings or one prompt per line")
    if not parsed:
        raise ValueError("prompts file contains no prompts")
    return parsed


def checkpoint_path(directory: Path, kind: str, payload_bits: int, positions: int, checkpoint: int) -> Path:
    """Return the MirrorMark-style JSONL name for one output type and checkpoint."""
    return directory / f"{kind}_m{payload_bits}_pos{positions}_{checkpoint}tokens.jsonl"


def result_fields(result: DetectionResult) -> dict[str, Any]:
    """Convert DISC's detection result into shared JSONL detection fields."""
    return {
        "score": result.score,
        # DISC has one calibrated tail probability rather than MirrorMark's
        # Gaussian z statistic; retain the shared key explicitly as null.
        "z": None,
        "pvalue": result.global_p_value,
        "detected": result.detected,
        "decoded_msg": result.payload,
        "n_star": result.n_star,
        "scored_bits": result.scored_bits,
        "local_pvalue": result.local_p_value,
        "position_pvalues": list(result.position_p_values),
    }


@dataclass(frozen=True)
class ExperimentConfig:
    """DISC settings recorded with every JSONL experiment row."""

    payload_bits: int
    positions: int
    context_mode: str
    context_width: int
    use_prefix: bool | None
    message_mapping: str
    use_cabs: bool
    fpr: float
    key_type: str


def build_record(
    *,
    index: int,
    prompt: str,
    response: str,
    token_ids: Sequence[int],
    checkpoint: int,
    perplexity: float,
    generation_seconds: float,
    decoding_seconds: float,
    result: DetectionResult,
    config: ExperimentConfig,
    payload: int | None = None,
) -> dict[str, Any]:
    """Build one JSON-serializable shared MirrorMark/DISC checkpoint row."""
    record: dict[str, Any] = {
        "idx": index,
        "prompt": prompt,
        "response": response,
        "response_ids": list(token_ids),
        "checkpoint_tokens": checkpoint,
        "ppl": perplexity,
        "time": generation_seconds,
        "generation_seconds": generation_seconds,
        "decoding_seconds": decoding_seconds,
        "step_stats": None,
        "disc": asdict(config),
    }
    record.update(result_fields(result))
    if payload is not None:
        record["message"] = payload
        record["message_bits"] = config.payload_bits * config.positions
        record["message_correct"] = result.detected and result.payload == payload
    return record


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one UTF-8 JSON object to a checkpoint artifact, creating its directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False, allow_nan=True) + "\n")


def _detector(key: bytes | str | int, args: argparse.Namespace) -> DiscDetector:
    """Create a detector whose configuration exactly mirrors one DISC encoder."""
    return DiscDetector(
        key,
        args.payload_bits,
        n_positions=args.positions,
        context_mode=args.context_mode,
        use_prefix=args.use_prefix,
        context_width=args.context_width,
        fpr=args.fpr,
        message_mapping=args.message_mapping,
        use_cabs=not args.disable_cabs,
    )


def run(args: argparse.Namespace) -> None:
    """Run paired DISC Hugging Face experiments and write one JSONL row per checkpoint."""
    try:
        transformers = importlib.import_module("transformers")
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError("Hugging Face experiments require: pip install -e '.[hf]'") from exc
    key = parse_key(args.secret_key)
    prompts = load_prompts(args.prompt, args.prompts_file)
    if args.start < 0 or args.generation_num < 1:
        raise ValueError("start must be non-negative and generation-num must be positive")
    selected = [prompts[(args.start + offset) % len(prompts)] for offset in range(args.generation_num)]
    checkpoints = sorted({value for value in args.desired_checkpoints if 0 < value <= args.gen_len})
    if not checkpoints:
        checkpoints = [args.gen_len]
    model = transformers.AutoModelForCausalLM.from_pretrained(args.llm_model)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.llm_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.device:
        model = model.to(args.device)
    model.eval()
    output_dir = Path(args.save_dir)
    config = ExperimentConfig(
        args.payload_bits, args.positions, args.context_mode, args.context_width,
        args.use_prefix, args.message_mapping, not args.disable_cabs, args.fpr, key_type(key),
    )
    rng = random.Random(args.seed)
    for offset, prompt in enumerate(selected):
        index = args.start + offset
        payload = args.payload if args.payload is not None else rng.randrange(2 ** (args.payload_bits * args.positions))
        encoder = DiscEncoder(
            key, payload, args.payload_bits, n_positions=args.positions,
            context_mode=args.context_mode, use_prefix=args.use_prefix,
            context_width=args.context_width, message_mapping=args.message_mapping,
            use_cabs=not args.disable_cabs, seed=rng.randrange(2**63),
        )
        began = time.perf_counter()
        _wm_text, wm_ids = generate(
            model, tokenizer, prompt, encoder,
            max_new_tokens=args.gen_len, temperature=args.temperature,
        )
        wm_seconds = time.perf_counter() - began
        began = time.perf_counter()
        _nwm_text, nwm_ids = generate_unwatermarked(
            model, tokenizer, prompt, max_new_tokens=args.gen_len,
            temperature=args.temperature, seed=rng.randrange(2**63),
        )
        nwm_seconds = time.perf_counter() - began
        for checkpoint in checkpoints:
            wm_prefix, nwm_prefix = wm_ids[:checkpoint], nwm_ids[:checkpoint]
            began = time.perf_counter()
            wm_result = detect_token_ids(wm_prefix, len(tokenizer), _detector(key, args))
            wm_decoding = time.perf_counter() - began
            began = time.perf_counter()
            nwm_result = detect_token_ids(nwm_prefix, len(tokenizer), _detector(key, args))
            nwm_decoding = time.perf_counter() - began
            _append_jsonl(
                checkpoint_path(output_dir, "watermark", args.payload_bits, args.positions, checkpoint),
                build_record(index=index, prompt=prompt, response=tokenizer.decode(wm_prefix, skip_special_tokens=True),
                             token_ids=wm_prefix, checkpoint=checkpoint,
                             perplexity=perplexity_from_token_ids(model, wm_prefix),
                             generation_seconds=wm_seconds, decoding_seconds=wm_decoding,
                             result=wm_result, config=config, payload=payload),
            )
            _append_jsonl(
                checkpoint_path(output_dir, "nonwatermark", args.payload_bits, args.positions, checkpoint),
                build_record(index=index, prompt=prompt, response=tokenizer.decode(nwm_prefix, skip_special_tokens=True),
                             token_ids=nwm_prefix, checkpoint=checkpoint,
                             perplexity=perplexity_from_token_ids(model, nwm_prefix),
                             generation_seconds=nwm_seconds, decoding_seconds=nwm_decoding,
                             result=nwm_result, config=config),
            )
        print(json.dumps({"idx": index, "payload": payload, "wm_tokens": len(wm_ids), "nwm_tokens": len(nwm_ids), "wm_time": wm_seconds, "nwm_time": nwm_seconds}))


def main() -> None:
    """Parse a MirrorMark-style DISC experiment command and execute :func:`run`."""
    parser = argparse.ArgumentParser(description="Paired DISC Hugging Face experiment runner")
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--prompt", default="Explain how statistical watermarking works.")
    parser.add_argument("--prompts-file")
    parser.add_argument("--save-dir", default="generated_samples")
    parser.add_argument("--generation-num", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--gen-len", type=int, default=200)
    parser.add_argument("--desired-checkpoints", nargs="*", type=int, default=[100, 200])
    parser.add_argument("--secret-key", default="0xC6D7563E6652C762EA00666974452BFB")
    parser.add_argument("--payload", type=int)
    parser.add_argument("--payload-bits", "--m-bits", dest="payload_bits", type=int, default=3)
    parser.add_argument("--positions", "--message-length", dest="positions", type=int, default=1)
    parser.add_argument("--context-width", type=int, default=4)
    parser.add_argument("--context-mode", choices=["prefix_bit_ngram", "bit_ngram", "token_ngram", "prefix_token_ngram"], default="prefix_bit_ngram")
    parser.add_argument("--no-prefix", dest="use_prefix", action="store_false", default=None)
    parser.add_argument("--message-mapping", choices=["direct", "gray"], default="direct")
    parser.add_argument("--disable-cabs", action="store_true")
    parser.add_argument("--fpr", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
