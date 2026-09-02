# Repository audit

## High-priority issues found

1. **Conflicting implementations.** `DISC.py`, `watermarking.py`, and the
   `getData*.py` family use different PRF inputs, searches, thresholds, and
   contexts. There was no canonical public API.
2. **Import-time model download.** `watermarking.py` loads GPT-2 on import and
   forces float16 even on CPU, making imports network- and GPU-dependent.
3. **Non-cryptographic PRF.** `utils.PRF` reseeds Python’s global RNG using
   string representations. It is neither a secure PRF nor portable and mutates
   global random state.
4. **Incorrect multiple-testing correction.** One decoder corrects the local
   p-value by the number of coarse/fine evaluations. Equation (26) corrects by
   the message-space size and candidate prefix lengths.
5. **Heuristic search can miss the optimum.** Refining one coarse candidate is
   not exact Algorithm 4 and should not be the statistical correctness oracle.
6. **Message mapping ambiguity.** Some files Gray-encode the payload, while the
   supplied paper uses `delta_M = M / 2^m` in Algorithms 3–4.
7. **Inconsistent logarithm bases.** Some scripts use `log2` while the entropy
   threshold and Erlang score derivations use natural logarithms.
8. **PRF context mismatch.** Implementations use token/bit indices, padded
   real-token IDs, or a binary n-gram, and some omit random prefix `R`. The
   paper specifies `(R, S_i,h)`.
9. **Hard-coded local paths.** `BER_pipeline.py` assumes private model and
   prompt paths that other users cannot reproduce.
10. **GPU imports are unconditional.** `prf.py` imports CuPy and compiles CUDA
    at import time instead of making GPU acceleration optional.
11. **No test/package boundary.** The repository had no deterministic test
    suite, installable package metadata, or CI-ready command.
12. **Generated artifacts are committed.** Profiling/data outputs such as
    `*.prof` should be ignored or published as release artifacts.

## Edits made

- Added the maintained `disc` package with a model-independent encoder,
  detector, HMAC PRF, binarized-LM utilities, and typed result.
- Added an optional Hugging Face adapter without import-time downloads.
- Added a Section 5 simulator and Figure 5 CSV/plot reproduction script.
- Added exact unit tests and project metadata (`pyproject.toml`).
- Added explicit, tested `message_mapping="gray"` support while retaining the
  paper's direct mapping as the default.
- Replaced the incomplete README with installation, API, experiment, and
  interpretation documentation.
- Added cache/output exclusions in `.gitignore`.
- Preserved legacy scripts to avoid deleting unpublished experimental work.

## Recommended next edits

1. Verify PRF serialization against the intended author implementation and
   freeze an official test vector.
2. In the paper or release notes, identify Gray mapping as an optional extension
   unless it is added to a later manuscript revision.
3. Move old scripts into `legacy/` after confirming active jobs do not import
   them; avoid multiple root-level classes named `DISC`.
4. Add `LICENSE` and `CITATION.cff` after the authors select the license and
   final bibliographic record.
5. Add GitHub Actions for `pytest` and `ruff` on Python 3.10–3.12.
6. Record model/tokenizer revisions, prompts, temperature, seeds, and hardware
   for every LLM experiment.
7. Implement the paper’s coarse/fine optimization behind a mode flag, compare
   it against exhaustive search in tests, and retain exact search as oracle.
8. Add editing-attack experiments only after defining tokenization and prefix
   synchronization recovery; Section 5 does not specify these experiments.
