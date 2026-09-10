# DISC: Multi-Bit Distortion-Free Watermarking

This repository contains a tested reference implementation of **Distribution
Interval Shift Coding (DISC)** from:

> M. Kordi Boroujeny, Y. Jiang, K. Zeng, and B. L. Mark, “Multi-Bit
> Distortion-Free Watermarking for Large Language Models,” arXiv:2402.16578.

The maintained implementation is the `disc/` package. The original research
prototypes remain for provenance, but they are not the public API.

## Implemented

- binary-language-model conversion (Equations 31–32);
- shifted-interval encoder (Equations 20–22 and Algorithm 3);
- empirical-entropy random initialization;
- HMAC-SHA256 PRF over the random prefix and binary n-gram context;
- score and Erlang-tail test (Equations 24–26 and Algorithm 4);
- optional PRF context modes (`prefix_bit_ngram`, `bit_ngram`, `token_ngram`,
  `prefix_token_ngram`);
- Context-Anchored Balanced Scheduler (CABS) for multi-position DISC;
- Fisher combination of per-position q-values so the overall FPR stays at `fpr`;
- repeated `(context, current bit)` removal during detection;
- optional Hugging Face causal-LM adapter;
- Section 5 Bernoulli simulation and Figure 5 reproduction script;
- deterministic unit tests.

## Installation

Python 3.10 or later is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

For Hugging Face generation:

```bash
pip install -e '.[hf]'
```

## Model-independent simulation

```bash
disc-simulate --runs 100 --payload-bits 4 --real-tokens 20

# Choose how DISC seeds its PRF context.
disc-simulate --context-mode prefix_token_ngram

# Optional Gray-code payload mapping
disc-simulate --runs 100 --payload-bits 4 --real-tokens 20 --message-mapping gray
```

The command prints JSON containing BER, message accuracy, detection rate, and
cumulative `generation_seconds` and `decoding_seconds` across all runs.

## Encode and detect binary tokens

```python
from disc import DiscDetector, DiscEncoder

encoder = DiscEncoder(
    key="replace-with-a-secret-key",
    payload=5,
    payload_bits=3,
    entropy_threshold=5.0,
    context_width=4,
    seed=7,
    message_mapping="direct",  # use "gray" for the optional Gray-code extension
)

for p_one in [0.2, 0.7, 0.4, 0.8] * 100:
    encoder.encode_bit(p_one)

result = DiscDetector(
    key="replace-with-a-secret-key",
    payload_bits=3,
    context_width=4,
    fpr=0.01,
    message_mapping="direct",  # must match the encoder
).detect(encoder.bits)

print(result.detected, result.payload, result.global_p_value)
```

The decoder normally searches all admissible random-prefix lengths. A known
start can be supplied in controlled tests with
`detect(bits, n_star_candidates=[encoder.n_star])`.

## CABS multi-position DISC

CABS (from MirrorMark, used here only as a scheduler) splits a payload into
`H` symbols. Each eligible token is assigned to a position and watermarked
with DISC using that position's shift. Detection scores each position, union-
corrects by `2**m`, then mixes the per-position q-values with Fisher's method
so the overall false-positive rate is still the configured `fpr`. Mirror
mapping is not implemented.

```python
from disc import CabsConfig, DiscDetector, DiscEncoder

encoder = DiscEncoder(
    "secret",
    payload=27,          # integer; with H=2 and m=4 this is symbols [1, 11]
    payload_bits=4,      # m bits per position
    n_positions=2,       # H
    context_mode="token_ngram",
    context_width=4,     # shared h previous tokens seed the PRF and CABS
    seed=7,
)
# ... encoder.encode_token(lm_probs) ...

result = DiscDetector(
    "secret",
    payload_bits=4,
    n_positions=2,
    context_mode="token_ngram",
    context_width=4,
    fpr=0.01,
).detect(encoder.bits, token_ids=encoder.tokens, bit_length=width)
```

PRF `context_mode` options:

- `prefix_bit_ngram` — `y = F(R, last h bits)` (default DISC with random init)
- `bit_ngram` — `y = F(last h bits)` (DISC without random initialization)
- `token_ngram` — `y = F(last h token IDs, bit index)` (typical with CABS)
- `prefix_token_ngram` — same as token n-gram plus a random token prefix `R`

Pass `use_prefix=False` on both `DiscEncoder` and `DiscDetector` to keep a
`prefix_*` mode but hash `R = []` and skip the decoder's `n_star` search.
The first `h` tokens/bits are still generated without a watermark so the
n-gram exists. CLI: `disc-simulate --no-prefix`.

## Hugging Face generation

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from disc import DiscDetector, DiscEncoder
from disc.huggingface import detect_token_ids, generate

name = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name)

encoder = DiscEncoder("secret", payload=3, payload_bits=2, seed=11)
text, generated_ids = generate(model, tokenizer, "Explain channel capacity:", encoder)

detector = DiscDetector("secret", payload_bits=2)
result = detect_token_ids(generated_ids, len(tokenizer), detector)
print(text)
print(result)
```

## Reproduce Figure 5

The paper used 10,000 trials per point; start smaller for a smoke test.

```bash
python experiments/reproduce_figure5.py \
  --runs 10000 \
  --payload-bits 1 2 3 4 \
  --real-tokens 4 6 8 10 12 16 20 24 30 \
  --output output/figure5.csv
```

Use `--plot output/figure5.png` after `pip install -e '.[experiments]'`.
The paper’s simulation uses 17 binary tokens per real token. Results can differ
slightly with the PRF, finite Monte Carlo runs, and random seed.

## Tests

```bash
pytest -q
```

## Layout

```text
disc/                         maintained reference implementation
experiments/                  reproducibility scripts
tests/                        deterministic unit tests
DISC.py, watermarking.py      legacy research prototypes
getData*.py, perfTests/       legacy performance experiments
REPOSITORY_AUDIT.md           audit and migration guidance
```

## Direct and Gray-code message mappings

The maintained API supports both mappings:

- `message_mapping="direct"` is the default and follows the supplied paper:
  `delta_M = M / 2^m`.
- `message_mapping="gray"` first computes `G = M xor (M >> 1)` and embeds
  `delta_G = G / 2^m`. The decoder searches the shifted intervals and applies
  the inverse Gray transform before returning the payload.

The encoder and decoder must use the same mapping. Gray mapping is a permutation
of the message space, so it does not change the number of hypotheses or the
multiple-testing correction. It can, however, make adjacent interval indices
differ by one message bit, which may reduce bit errors when the estimated shift
lands in a neighboring interval.

## Interpretation choices

- Direct mapping remains the paper-faithful default; Gray coding is retained as
  an explicit optional extension and is never applied silently.
- The global p-value uses `1 - (1 - |M| p*)^(L-h)`, not the number of
  implementation-level coarse/fine evaluations.
- Natural logarithms are used in empirical entropy and scores.
- PRF input serialization is explicit and domain separated. Python’s global
  `random` state is not used as a PRF.

This is research code, not a production key-management or provenance system.
## Paired Hugging Face experiments

`disc-experiment` uses the same evaluation shape as the supplied MirrorMark
driver: it accepts a prompt or a JSON list of prompts, produces paired
watermarked and non-watermarked continuations, detects both at each requested
checkpoint, and appends parallel JSONL files. DISC retains its HMAC PRF; a
decimal or `0x`-prefixed `--secret-key` is parsed as DISC's existing integer
key type, while any other value is a string key.

```bash
disc-experiment --llm-model openai-community/gpt2 \
  --prompt "Explain statistical watermarking." \
  --generation-num 1 --gen-len 200 --desired-checkpoints 100 200 \
  --payload-bits 3 --positions 18 --context-width 4 \
  --context-mode prefix_bit_ngram --save-dir generated_samples
```

The resulting files are named
`watermark_m3_pos18_100tokens.jsonl` and
`nonwatermark_m3_pos18_100tokens.jsonl` (and likewise for every checkpoint).
Each row has MirrorMark's common fields (`idx`, `prompt`, `response`,
`checkpoint_tokens`, `ppl`, `score`, `z`, `pvalue`, `time`, `step_stats`) plus
DISC-specific decoded payload, `n_star`, scored-bit count, p-values, and the
non-secret configuration. `z` is `null` because DISC uses an Erlang-tail
p-value rather than MirrorMark's Gaussian z-score.
