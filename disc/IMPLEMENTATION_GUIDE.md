# DISC implementation guide

This file is the maintained, implementation-level explanation of the Python
package in this directory. It describes the checked-out
`paper-reference-implementation` branch as implemented in the source files,
including extensions such as token contexts, CABS, no-prefix operation, and
the paired Hugging Face experiment runner. Update this document in the same
change that alters any algorithm, state variable, public API, file, or
cross-file call described here.

It has two deliberately separate parts:

1. **Algorithm and data model** explains what DISC computes and why each
   value exists.
2. **Implementation and call relationships** explains the classes, methods,
   properties, helper functions, calls, data ownership, and file roles that
   implement it.

## Contents and terminology

The package represents a language-model token stream in two compatible forms:

- A real token is a vocabulary ID `t`, an integer in `[0, |V|)`.
- Its binary token is its `w = ceil(log2(|V|))`-bit, MSB-first encoding.
- `bits` is the flat concatenation of all binary tokens. Its index is a bit
  index, not a real-token index.
- `tokens` or `token_ids` is the aligned list of real vocabulary IDs.
- `m` (`payload_bits`) is the number of payload bits in one DISC position.
- `H` (`n_positions`) is the number of DISC positions. The full message has
  `m * H` bits and is represented by one nonnegative Python integer.
- A position symbol is one `m`-bit integer. `split_payload` uses big-endian
  order: for `m=4, H=2, payload=27`, the symbols are `[1, 11]`.
- `M` is a position symbol after the optional Gray-code mapping. Its circular
  interval shift is `delta_M = M / 2**m`.
- `h` (`context_width`) is always configured in **real-token units**. In a
  bit context it becomes `h*w` binary bits. In a token context it remains `h`
  token IDs.
- `R` is an ordinary, unwatermarked initial prefix. It is not a random number
  or a cryptographic key. `n_star` is its bit length; for token-prefix mode
  `n_star_tokens` is also retained in real-token units.
- `S` is the context immediately before a bit. `y` is the deterministic
  HMAC-derived pseudo-uniform number computed from the key and the selected
  `R` and `S`.

`prefix_bit_ngram` is the default mode. Its PRF input is the complete
unwatermarked prefix plus the previous `h*w` bits. `bit_ngram` omits `R`.
`token_ngram` uses the previous `h` token IDs and the bit offset in the
current token. `prefix_token_ngram` additionally includes an initial prefix
of token IDs.

# Part 1 — Algorithm and data model

## 1. The single-position DISC encoder

DISC embeds one `m`-bit symbol without changing the desired next-bit
probability. Suppose the language model says the next binary bit has
probability `p = P(bit=1)`. A key and the prior generated text determine
`y` in `(0, 1)`. The payload determines `delta_M` in `[0, 1)`.

The implementation creates the bit by testing whether the shifted point lies
in the interval corresponding to bit 1:

```text
z = (y + delta_M) mod 1
bit = 1 when z is in [1 - p, 1), otherwise 0
```

`in_shifted_interval(y, p, delta_M)` implements this test, including the
case in which the interval wraps around zero. Since a uniformly distributed
`y` lands in a length-`p` region with probability `p`, the marginal next-bit
distribution stays equal to the language model's requested distribution.
The watermark affects the deterministic coupling between the key/context and
the sampled bit, not the marginal distribution.

`DiscEncoder.encode_bit(p_one)` is the direct implementation of this rule.
It first decides whether the next bit has enough history and is allowed to be
watermarked. If not, it calls `_sample_unwatermarked`, which makes an ordinary
Bernoulli draw using the encoder-local `random.Random` instance. Otherwise it
computes `y = prf_uniform(...)`, chooses the bit with the shifted interval
test, appends the bit, and records `True` in `watermark_mask`.

### Initial prefix R

With a prefix context mode and `use_prefix=True`, encoding begins with
ordinary samples. Each ordinary sampled bit adds its surprisal
`-log(P(the sampled bit))` to `empirical_entropy`. Prefix generation stays
active while either condition is true:

1. `empirical_entropy < entropy_threshold`, or
2. the necessary context does not yet exist.

For `prefix_bit_ngram`, the second condition requires `h*w` prior bits. Once
both conditions have become false, the first eligible bit starts watermarking
and the current bit length becomes `n_star`. Thus `bits[:n_star]` is exactly
`R`. For `prefix_token_ngram`, the comparable boundary is fixed at a token
boundary and stored as `n_star_tokens`; the corresponding `n_star` is
`n_star_tokens*w`.

The detector cannot know the encoder's sampled entropy trajectory. It instead
treats every valid candidate initial segment as `R`, reconstructs the PRF for
that hypothesis, and applies a multiple-testing correction for that search.

### No-prefix modes and warm-up

`bit_ngram` and `token_ngram` default to `use_prefix=False`. A prefix mode
can also be forced to use no prefix by passing `use_prefix=False`. In all
those cases, `R=[]`, `n_star=0`, and the detector has no start search.

There is still an ordinary-sampling warm-up until the n-gram exists: the first
`h*w` bits in bit modes or first `h` real tokens in token modes cannot use a
context. Warm-up bits are marked `False` in `watermark_mask`, but they are
not a prefix `R` when `use_prefix=False`.

The optional `randomize_without_prefix=True` mode supplies a second,
domain-separated PRF decision. For an otherwise eligible context, it samples
the ordinary LM with probability `1/randomization_window`; all other eligible
contexts use DISC. This decision is reproducible from the same key/context.
The encoder records the exact mask and the detector can reconstruct it with
`_randomization_mask` when the caller did not provide one.

### Repeated contexts

The encoder owns `_seen_watermark_contexts`. Before watermarking an ordinary
raw bit it builds `_watermark_context_key`, which contains the context mode,
the prefix when relevant, the current context, and, in token modes, the bit
offset. If that exact PRF-input identity has been watermarked before, the
encoder falls back to ordinary sampling. This prevents the same `y` from being
reused as correlated evidence. The detector separately deduplicates each
hypothesis using `(context, observed current bit)` so its score does not count
repeated evidence twice.

## 2. Token-level encoding

The language model supplies a complete categorical distribution
`probabilities[v] = P(next token ID=v)`. `DiscEncoder.encode_token` converts
the categorical sample problem into `w` conditional binary sample problems.

At binary offset `j` of the token, it passes the already chosen token-bit
prefix to `conditional_bit_probability`. That helper sums the categorical
mass of every token ID whose `j`th MSB is 1 and whose earlier bits equal the
already selected prefix, then divides by the mass of all IDs compatible with
that prefix. The resulting `p_one` is exactly the conditional probability of
the next bit under the original categorical distribution. `encode_token`
calls `encode_bit(p_one)` for each bit, joins the chosen bits into an integer,
and returns that vocabulary ID.

This procedure is why DISC can encode through a binary vocabulary tree while
preserving the original categorical token distribution. Invalid leaves above
the vocabulary size have zero probability and are never selected.

While executing `encode_token`, `_in_token=True` and `_token_width=w`. These
flags let `encode_bit` know the current bit offset and delay token-level state
updates until all `w` bits are known. On completion, `encode_token` appends the
chosen real token ID to `tokens`, appends its assigned CABS position to
`token_positions`, and commits a token-level CABS decision where applicable.

## 3. PRF inputs and serialization

`HmacPrf` is the keyed PRF. It accepts a nonempty `bytes`, UTF-8 `str`, or
nonnegative `int` key. Integer keys are converted to minimum-length big-endian
bytes. The key itself is never written to experiment artifacts.

The PRF uses separate HMAC-SHA256 domains:

- `DISC-v1` for bit contexts;
- `DISC-v1-tok` for token contexts;
- `DISC-v1-randomize` for optional no-prefix randomization;
- `DISC-v1-cabs-pos` for CABS tie breaking;
- `DISC-v1-cabs-frame` for CABS frame cuts.

For bit contexts, `HmacPrf._pack_bits` serializes a bit sequence as a
four-byte big-endian bit length followed by MSB-first packed bytes. Both the
prefix and context are length-delimited, so different logical sequences cannot
become ambiguous merely because their packed trailing zeros match. For token
contexts, `_pack_u32s` serializes a four-byte count then unsigned 32-bit,
big-endian IDs. `uniform` and `uniform_tokens` HMAC their domain and packed
arguments, and `_digest_to_unit` maps the digest to a value strictly inside
`(0,1)` using its high-order 64-bit integer. `integer_hash` returns an integer
from the same HMAC machinery for CABS.

`prf_uniform` is the only shared logical PRF gateway used by the encoder and
detector. It chooses the exact mode-specific slices:

| Context mode | Prefix passed to PRF | Context passed to PRF | Extra input |
| --- | --- | --- | --- |
| `prefix_bit_ngram` | `bits[:n_star]` | `bits[i-h*w:i]` | none |
| `bit_ngram` | empty | `bits[i-h*w:i]` | none |
| `prefix_token_ngram` | `tokens[:n_star_tokens]` | `tokens[k-h:k]` | 8-bit offset in token |
| `token_ngram` | empty | `tokens[k-h:k]` | 8-bit offset in token |

The token-mode offset is essential: all bits inside a real token have the
same preceding token IDs, so it separates their PRF values.

## 4. Multi-position payloads and CABS

For `H>1`, `split_payload` divides the full `m*H`-bit payload into `H`
symbols. The encoder computes one mapped symbol and one `delta_M` per
position. It must choose which position owns each watermark opportunity.

Without CABS, the multi-position encoder and detector use deterministic round
robin by real token. Enable this explicitly with `use_cabs=False` on **both**
`DiscEncoder` and `DiscDetector`. `use_cabs=None` is the default and means
`True` when `H>1`, so CABS must be turned off deliberately for this mode. In
the `disc-experiment` command, the equivalent flag is `--disable-cabs`.

For `prefix_token_ngram`, token `k` after the token-prefix `R` is assigned
`(k-n_star_tokens) mod H`. For all other modes, including a bit-prefix whose
`R` can end inside a token, token `k` is assigned `k mod H`; the detector
excludes unwatermarked prefix bits while retaining the original whole-token
group. All `w` bits of a real token use that token's position shift.

With CABS, `CabsScheduler` implements a context-anchored balanced schedule:

1. It forms an eligibility context matching the DISC PRF context representation.
2. An incomplete context is ineligible. A previously used eligibility context
   is also ineligible; it is sampled normally and receives no position.
3. For an eligible opportunity, CABS selects the least-used position in the
   current frame. A keyed uniform PRF breaks ties among the least-used slots.
4. The selected token or bit is committed after it is generated. Its ID is
   appended to a queue `Q` of the last `W` assigned IDs.
5. CABS resets per-position counts and the queue when a keyed frame hash has
   its lowest `f` bits equal to zero after `min_len` assignments, or when
   `frame_len >= max_len`. `max_len = max(H, floor(max_factor*H))`.

The split between binary and token context is intentional and must be kept
straight:

- In token modes, CABS schedules **whole real tokens**. Its eligibility
  context is the preceding `h` token IDs. Once assigned, all bits of that
  token use the selected position symbol.
- In bit modes, CABS schedules **individual binary bits**. Its eligibility
  context is the preceding `h*w` bits, which is the actual DISC PRF context.
  `_binary_cabs_settings` converts CABS lengths from real-token units to bit
  units: `W`, `min_len`, and the maximum frame scale by `w`; `frame_bits`
  increases by `ceil(log2(w))`, preserving an approximately equal cut rate per
  original token. In a prefix-bit stream it starts at bit `n_star`, so `R`
  never consumes a CABS assignment or frame state.

The detector replays exactly the same CABS operations over the observed
sequence, groups bit indices by their recovered position, and scores each
position independently.

## 5. Detection and statistical decision

For a candidate mapped symbol with shift `delta`, each scored observed bit
contributes `disc_score(bit, y, delta)`:

```text
bit = 1:  -log((1 - y - delta) mod 1)
bit = 0:  -log((y + delta) mod 1)
```

The score sums across accepted bits. Under the null hypothesis, each accepted
term is exponential with unit rate, so a sum of `N` terms has an Erlang/Gamma
distribution. The local tail probability is
`gammaincc(N, score)`, where `gammaincc` is SciPy's regularized upper gamma.

For a single position, the detector searches all `2**m` mapped symbols and
retains the smallest local p-value. It corrects for that symbol search and,
when a prefix is enabled, the number of tested `n_star` values. It reports a
payload only if the corrected p-value is at most `fpr`.

For multiple positions, `_detect_positions` replays CABS or round robin,
then searches the `2**m` mapped symbols separately for every nonempty group.
It applies the symbol-search correction at each position. It combines the
position p-values with Fisher's method: `-sum(log(p_h))` has an Erlang law
with shape equal to the number of positions, so `combine_p_values` uses
`gammaincc(H, statistic)`. It then applies the prefix-start correction to the
combined value if start candidates were searched. On success it unmaps every
symbol, joins them with `join_payload`, and returns the full payload.

The `DetectionResult` immutable dataclass contains:

- `detected`: the final boolean decision;
- `payload`: decoded full integer or `None` on failure;
- `n_star`: selected prefix length in bits, or `None` if no usable candidate;
- `local_p_value` and `global_p_value`;
- `score` and `scored_bits` for the selected hypothesis;
- `symbols` and `position_p_values` for the multi-position path.

# Part 2 — Implementation and call relationships

## File map

| File | Responsibility | Main callers |
| --- | --- | --- |
| `core.py` | DISC mathematics, PRF, encoder, detector, token/bit conversion | all other package modules |
| `cabs.py` | Stateful CABS scheduler independent of a specific watermark rule | `DiscEncoder`, `DiscDetector` |
| `huggingface.py` | Optional PyTorch/Transformers bridge | `experiment.py`, user applications |
| `simulation.py` | Model-independent Monte Carlo evaluation | `cli.py`, user applications |
| `cli.py` | `disc-simulate` argument parsing and JSON stdout | installed console script |
| `experiment.py` | Paired MirrorMark-style HF experiment runner and JSONL schema | `disc-experiment` console script |
| `__init__.py` | Selected stable public imports | `from disc import ...` users |

## `core.py`: shared primitive functions

`_resolve_use_prefix(mode, use_prefix)` normalizes prefix configuration. `None`
means true only for a `prefix_*` mode. It rejects `use_prefix=True` for a
non-prefix mode because that combination has no defined PRF input.

`_binary_cabs_settings(config, H, h, w)` creates the bit-scheduling CABS
configuration described above. The original config remains stored as
`cabs_config`; this helper derives the effective bit version rather than
mutating user configuration.

`gray_encode`, `gray_decode`, `_map_payload`, and `_unmap_payload` implement
the optional symbol permutation. They operate per position, never on the full
joined message. `split_payload` and `join_payload` provide the reversible
big-endian full-message/position-symbol conversion. `_validate_bits` verifies
that an observation contains only literal zero/one values.

`combine_p_values` provides Fisher combination. `conditional_bit_probability`
is used only by `encode_token`; `token_ids_to_bits` and `bits_to_token_ids`
provide MSB-first fixed-width conversion. `in_shifted_interval` is the
encoder's interval decision. `disc_score` is the detector's matching
per-bit evidence calculation.

## `core.py`: `HmacPrf`

`HmacPrf.__init__` normalizes and stores `_key`. `_pack_bits`, `_pack_u32s`,
and `_digest_to_unit` are static helpers because they do not use instance
state. `uniform` serves bit contexts, `uniform_tokens` token contexts, and
`integer_hash` CABS. These functions do not know the payload, CABS state,
probabilities, or whether a bit is eligible; those decisions belong to the
encoder/detector and scheduler.

## `core.py`: `DiscEncoder` state and method lifecycle

Construction validates dimensions and creates the state below.

| State/property | Meaning and writer |
| --- | --- |
| `prf`, `selector_prf` | Same secret, separate HMAC domains; created in `__init__` |
| `payload`, `payload_bits`, `n_positions` | Immutable configuration values |
| `symbols`, `mapped_symbols`, `position_deltas` | Full message split, mapped, and converted to shifts |
| `bits`, `tokens` | Growing emitted binary and real-token histories |
| `token_positions` | CABS/round-robin assignment for each completed real token where applicable |
| `watermark_mask` | Bit-parallel truth record; false for prefix/warm-up/ineligible/selector bits |
| `empirical_entropy` | Prefix sampled-surprisal sum; updated only by `_sample_unwatermarked` |
| `n_star`, `n_star_tokens` | Prefix boundary; `None` while a prefix is still being built |
| `bits_per_token` | `w`; initially caller value, reset from vocabulary size by `encode_token` |
| `_in_token`, `_token_width` | Transient markers during a token's `w` bit calls |
| `_seen_watermark_contexts` | Context identities already watermarked |
| `cabs` | `CabsScheduler` or `None`; created only when CABS is enabled |

`random_initialization` is a property returning a copy of the current/final
`R`. `binary_context_width` is a property returning `h*w`.
`_prf_context_len` returns `h*w` in bit modes and `h` in token modes.
`_cabs_eligibility_context` returns the exact complete current DISC context or
`None` during warm-up. `_needs_random_init`, `_needs_context_warmup`,
`_sample_unwatermarked`, `_watermark_context_key`, and `_scheduled_random_bit`
are decision helpers used exclusively by `encode_bit`.

`encode_bit` is the central transition. Its order is:

1. validate `p_one`;
2. if binary CABS is active, propose one position for this binary bit using
   the current `h*w` bits; set `watermark=False` when ineligible;
3. otherwise, if raw/token CABS reaches a real-token boundary, propose one
   position for the whole token and retain it in `_active_cabs_position`;
4. choose ordinary sampling for prefix construction, warm-up, ineligibility,
   a user `watermark=False`, selector randomization, or repeated context;
5. on the first usable prefix-bit context, fix `n_star`; otherwise compute
   `y`, select the DISC bit, and append it;
6. append an aligned boolean mask value;
7. commit a binary CABS bit immediately, or commit a token CABS decision only
   after its final bit is known.

`_round_robin_position` is the no-CABS counterpart to a scheduler: it maps
the real token containing the next bit to its deterministic position and is
used by both `encode_bit` and `encode_token`. `encode_token` validates and
normalizes a one-dimensional categorical vector, derives `w`, rebuilds binary
CABS if the initially assumed width changed, sets `_in_token`, derives `w`
conditional bit probabilities, and calls
`encode_bit` `w` times. It then reconstructs and returns the chosen token ID,
updates `tokens`, `token_positions`, token-prefix state, and scheduler state.

## `core.py`: `DiscDetector` lifecycle

Construction mirrors every encoder setting that affects a PRF value, bit
group, or statistical correction: key, `m`, `H`, mode, prefix setting, `h`,
`w`, mapping, CABS config, selector configuration, and `fpr`. A detector must
match its encoder in those values.

`_binary_h` computes detection `h*w`; `_prf_context_len` chooses bit or token
units. `_randomization_mask` replays the optional selector PRF. `_score` is
the single-position scorer. `_score_indices` is the equivalent for a sparse
set of bit indices belonging to one position; it intentionally returns one
aggregate `(score, count)`, not one value per input index. `_detect_positions`
performs position assignment/replay, per-position exhaustive symbol search,
Fisher combination, and payload assembly.

`detect(bits, ...)` owns the top-level branch:

1. copy and validate bits and an optional explicit mask;
2. choose the position-aware path when CABS, `H>1`, or a token context needs
   token structure; otherwise choose the optimized single-position bit path;
3. recover IDs from bits when necessary, or use supplied aligned `token_ids`;
4. choose start candidates: `[0]` without a prefix; `1..N_bits-1` for
   prefix-bit mode; `1..N_tokens-1` for prefix-token mode;
5. score each candidate, select the lowest p-value, correct over candidates,
   and construct `DetectionResult`.

For direct raw `detect` calls, prefix-bit candidates are individual bit
offsets. `huggingface.detect_token_ids` may optionally restrict that search to
token boundaries; the default preserves the complete bit-offset search.

## `cabs.py`: scheduler details

`_CabsPrf` is a typing `Protocol`, not an implementation. It declares exactly
the `HmacPrf` methods/domains CABS needs. `CabsConfig` is immutable
configuration: `window_size=W`, `frame_bits=f`, `max_factor`, and optional
`min_len`.

`CabsScheduler.__init__` validates configuration, derives `min_len` and
`max_len`, and calls `reset`. `reset` initializes `counts`, assigned-ID
`queue`, `frame_len`, repeated-context set `seen`, and deferred `_pending`
assignment. `propose(history, eligibility_context=...)` is called before
sampling. It rejects warm-up/repeated contexts; otherwise it hashes the
pre-enqueue queue, selects a least-count position, and saves context,
position, and pre-enqueue frame hash in `_pending`.

`commit(token_id)` is called after the generated value is known. It does
nothing when `propose` was ineligible. Otherwise it marks the context seen,
increments the selected count/frame length, pushes the ID into `queue`, and
performs the hash/length frame reset test. `assign_sequence` is detector
replay: it resets, skips every prefix element before `start_index`, then calls
`propose` and `commit` sequentially for every remaining known value.
`_choose_position` is the least-count operation plus PRF tie break.

## `huggingface.py`: optional model bridge

`_import_torch` imports PyTorch only when a HF function is called, so the core
package still imports without optional dependencies. `generate` tokenizes a
prompt, executes causal-LM forward passes with a key/value cache, softmaxes
the last logits at the requested temperature, and gives the CPU probability
array to `encoder.encode_token`. It returns continuation text and only newly
generated IDs, stopping at EOS.

`generate_unwatermarked` performs the matching ordinary-control loop. It
uses `torch.multinomial` and a private optional torch generator seed.
`perplexity_from_token_ids` evaluates the causal-LM loss on only continuation
IDs and returns `exp(loss)`; fewer than two IDs returns NaN.
`detect_token_ids` computes `w`, converts IDs to bits, chooses appropriate
prefix candidates, and calls `DiscDetector.detect` with the original IDs and
width. It is the correct public detector adapter for HF outputs.

## `simulation.py` and `cli.py`

`SimulationSummary` is immutable aggregate output. `run_simulation` creates
an independent encoder/detector pair for each trial, generates Uniform(0,1)
Bernoulli probabilities, measures encoding and decoding wall-clock time with
`perf_counter`, and reports bit error rate, exact message accuracy, detection
rate, and cumulative timings. It is a model-independent Monte Carlo test; it
does not use token distributions or Transformers.

`cli.main` parses `disc-simulate` flags, translates real tokens and
`bits_per_token` into `sequence_bits`, calls `run_simulation`, and prints the
dataclass as formatted JSON. It has no watermark math of its own.

## `experiment.py`: paired experiment framework

This module is the common evaluation layer shared conceptually with the
MirrorMark experiment format. It does not change the DISC PRF or algorithms.

`parse_key` parses decimal and `0x...` command-line text as an integer key;
other nonempty values remain text. `key_type` records only the key's type.
`load_prompts` accepts a JSON list of strings or one nonblank prompt per line.
`checkpoint_path` names output files. `result_fields` maps `DetectionResult`
to JSON fields. `ExperimentConfig` stores non-secret settings. `build_record`
combines common MirrorMark fields and DISC diagnostics. `_append_jsonl` writes
one UTF-8 object per line. `_detector` builds a configuration-matched detector.

`run(args)` imports Transformers lazily, loads model/tokenizer, selects the
requested prompts, samples or uses the requested payload, builds one encoder,
generates a WM continuation and a seeded ordinary NWM continuation, then for
each checkpoint detects both prefixes and appends one row to each parallel
file. It records complete-generation time with every checkpoint, just as the
MirrorMark driver does, and separately measures each checkpoint detection.
`main` only defines the `disc-experiment` command-line interface and invokes
`run`.

## `__init__.py` and safe modification checklist

`__init__.py` re-exports the intended user-facing types/functions. Add a new
public API there only after its import is safe without optional dependencies.

After a modification, verify this guide against the following invariants:

1. Encoder and detector select byte-identical PRF inputs for the same mode.
2. Every `bits`-parallel structure (`watermark_mask`, masks, scored indices)
   stays the same length and uses bit indices.
3. `tokens[k]` maps to `bits[k*w:(k+1)*w]` whenever real token structure is
   present.
4. Prefix units remain correct: bit modes search/store bits; token-prefix mode
   searches token candidates but reports `n_star` in bits.
5. CABS eligibility matches the actual PRF context representation, and its
   encoder proposal/commit sequence matches detector replay. With CABS off,
   check that encoder and detector instead use the same whole-token
   round-robin origin.
6. New selection behavior is represented in `watermark_mask` or can be
   reconstructed by the detector.
7. Statistical corrections cover every newly introduced search dimension.
8. Run `uv run ruff check disc tests` and `uv run pytest` after code changes.
