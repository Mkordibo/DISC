import math

import numpy as np

from disc.cabs import CabsConfig, CabsScheduler
from disc.core import (
    DiscDetector,
    DiscEncoder,
    HmacPrf,
    combine_p_values,
    join_payload,
    split_payload,
)


def test_split_join_payload():
    assert split_payload(27, 2, 4) == [1, 11]
    assert join_payload([1, 11], 4) == 27


def test_combine_p_values_all_one_is_one():
    assert combine_p_values([1.0, 1.0, 1.0]) == 1.0


def test_combine_p_values_fisher_two_equal():
    # Two independent p=e^{-1} → score=2, gammaincc(2, 2).
    value = combine_p_values([math.exp(-1), math.exp(-1)])
    from scipy.special import gammaincc

    assert np.isclose(value, float(gammaincc(2, 2.0)))


def test_cabs_replay_is_deterministic():
    prf = HmacPrf("cabs-key")
    tokens = [3, 9, 1, 7, 4, 8, 2, 6, 5, 0, 11, 10]
    first = CabsScheduler(prf, n_positions=4, context_width=3).assign_sequence(tokens)
    second = CabsScheduler(prf, n_positions=4, context_width=3).assign_sequence(tokens)
    assert first == second
    assert any(position is not None for position in first)


def test_cabs_skips_repeated_context():
    prf = HmacPrf("elig")
    # After warm-up, repeating the same h-gram must be ineligible.
    tokens = [1, 2, 3, 1, 2, 3, 9]
    positions = CabsScheduler(prf, n_positions=2, context_width=3).assign_sequence(tokens)
    assert positions[0] is None and positions[1] is None and positions[2] is None
    assert positions[3] is not None  # first use of context (1, 2, 3)
    assert positions[6] is None  # same h-gram seen again


def test_cabs_raw_bits_starts_binary_scheduling_after_prefix():
    config = CabsConfig(window_size=3, frame_bits=2, max_factor=2.0, min_len=2)
    encoder = DiscEncoder(
        "partial-prefix",
        payload=1,
        payload_bits=1,
        n_positions=2,
        context_mode="prefix_bit_ngram",
        context_width=2,
        bits_per_token=4,
        entropy_threshold=6.1,
        seed=1,
        use_cabs=True,
        cabs_config=config,
    )
    for _ in range(20):
        encoder.encode_bit(0.5)

    # Binary CABS treats each bit as a scheduling token. All nine prefix bits
    # are ordinary samples; eligible binary-token scheduling starts afterward.
    assert encoder.n_star == 9
    assert not any(encoder.watermark_mask[: encoder.n_star])
    assert any(encoder.watermark_mask[encoder.n_star :])


def test_cabs_disc_token_round_trip():
    key = "cabs-disc"
    payload_bits = 2
    n_positions = 2
    payload = 11  # symbols [2, 3]
    encoder = DiscEncoder(
        key,
        payload,
        payload_bits,
        n_positions=n_positions,
        context_mode="token_ngram",
        context_width=3,
        entropy_threshold=0.0,
        seed=5,
        use_cabs=True,
        cabs_config=CabsConfig(window_size=3, frame_bits=2, max_factor=2.0, min_len=2),
    )
    rng = np.random.default_rng(9)
    vocab = np.array([0.1, 0.2, 0.3, 0.4])
    for _ in range(80):
        noisy = np.clip(vocab + rng.normal(0, 0.02, size=4), 1e-3, None)
        encoder.encode_token(noisy)
    detector = DiscDetector(
        key,
        payload_bits,
        n_positions=n_positions,
        context_mode="token_ngram",
        context_width=3,
        fpr=0.05,
        use_cabs=True,
        cabs_config=CabsConfig(window_size=3, frame_bits=2, max_factor=2.0, min_len=2),
    )
    result = detector.detect(
        encoder.bits, token_ids=encoder.tokens, bit_length=2, n_star_candidates=[0]
    )
    assert result.detected
    assert result.payload == payload
    assert result.symbols == (2, 3)


def test_no_cabs_uses_matching_token_round_robin_positions():
    key = "round-robin-token"
    encoder = DiscEncoder(
        key,
        payload=11,  # m=2, H=2 -> symbols [2, 3]
        payload_bits=2,
        n_positions=2,
        context_mode="token_ngram",
        context_width=3,
        seed=9,
        use_cabs=False,
    )
    rng = np.random.default_rng(81)
    for _ in range(300):
        encoder.encode_token(rng.dirichlet(np.ones(4)))
    assert encoder.token_positions[:6] == [0, 1, 0, 1, 0, 1]
    result = DiscDetector(
        key,
        payload_bits=2,
        n_positions=2,
        context_mode="token_ngram",
        context_width=3,
        fpr=0.05,
        use_cabs=False,
    ).detect(encoder.bits, token_ids=encoder.tokens, bit_length=2)
    assert result.detected
    assert result.payload == 11


def test_no_cabs_binary_round_robin_uses_whole_token_positions():
    key = "round-robin-bits"
    encoder = DiscEncoder(
        key,
        payload=6,  # m=2, H=2 -> symbols [1, 2]
        payload_bits=2,
        n_positions=2,
        context_mode="prefix_bit_ngram",
        context_width=4,
        entropy_threshold=2.0,
        seed=3,
        use_cabs=False,
    )
    rng = np.random.default_rng(82)
    for _ in range(1000):
        encoder.encode_token(rng.dirichlet(np.ones(4)))
    assert encoder.token_positions[:6] == [0, 1, 0, 1, 0, 1]
    result = DiscDetector(
        key,
        payload_bits=2,
        n_positions=2,
        context_mode="prefix_bit_ngram",
        context_width=4,
        fpr=0.05,
        use_cabs=False,
    ).detect(
        encoder.bits,
        token_ids=encoder.tokens,
        bit_length=2,
        n_star_candidates=[encoder.n_star],
        watermark_mask=encoder.watermark_mask,
    )
    assert result.detected
    assert result.payload == 6


def test_cabs_token_prefix_round_trip():
    key = "cabs-token-prefix"
    config = CabsConfig(window_size=3, frame_bits=2, max_factor=2.0, min_len=2)
    encoder = DiscEncoder(
        key,
        payload=6,
        payload_bits=2,
        n_positions=2,
        context_mode="prefix_token_ngram",
        context_width=3,
        entropy_threshold=3.0,
        seed=6,
        use_cabs=True,
        cabs_config=config,
    )
    probabilities = np.array([0.1, 0.2, 0.3, 0.4])
    for _ in range(100):
        encoder.encode_token(probabilities)

    assert encoder.n_star_tokens is not None
    assert encoder.n_star_tokens >= encoder.context_width
    detector = DiscDetector(
        key,
        payload_bits=2,
        n_positions=2,
        context_mode="prefix_token_ngram",
        context_width=3,
        fpr=0.05,
        use_cabs=True,
        cabs_config=config,
    )
    result = detector.detect(
        encoder.bits,
        token_ids=encoder.tokens,
        bit_length=2,
        watermark_mask=encoder.watermark_mask,
    )
    assert result.detected
    assert result.payload == 6
    assert result.n_star == encoder.n_star


def test_bit_ngram_context_round_trip():
    key = "bit-ngram"
    encoder = DiscEncoder(
        key,
        payload=2,
        payload_bits=2,
        context_mode="bit_ngram",
        context_width=8,
        entropy_threshold=0.0,
        seed=3,
    )
    rng = np.random.default_rng(4)
    for p_one in rng.uniform(0.15, 0.85, 350):
        encoder.encode_bit(float(p_one))
    result = DiscDetector(
        key, 2, context_mode="bit_ngram", context_width=8, fpr=0.01
    ).detect(encoder.bits)
    assert result.detected
    assert result.payload == 2


def test_token_context_is_h_times_bits_per_token():
    encoder = DiscEncoder("key", 1, 1, context_width=2, entropy_threshold=0, seed=1)
    probs = np.array([0.25, 0.25, 0.25, 0.25])  # |V|=4 → w=2
    encoder.encode_token(probs)
    assert encoder.bits_per_token == 2
    assert encoder.binary_context_width == 4  # h * ceil(log2 |V|)
