import math

import numpy as np

from disc.core import (
    DetectionResult,
    DiscDetector,
    DiscEncoder,
    HmacPrf,
    bits_to_token_ids,
    conditional_bit_probability,
    disc_score,
    gray_decode,
    gray_encode,
    in_shifted_interval,
    token_ids_to_bits,
)


def test_prefix_bit_decoder_searches_every_proper_bit_prefix(monkeypatch):
    detector = DiscDetector(
        "start-search",
        payload_bits=1,
        n_positions=2,
        context_mode="prefix_bit_ngram",
        context_width=1,
        use_cabs=False,
    )
    observed_starts = []

    def record_start(self, bits, token_ids, bit_length, n_star, n_star_tokens, watermark_mask=None):
        observed_starts.append((n_star, n_star_tokens))
        return DetectionResult(False, None, n_star, 1.0, 1.0, 0.0, 0)

    monkeypatch.setattr(DiscDetector, "_detect_positions", record_start)
    detector.detect([0, 1, 0, 1, 0, 1], token_ids=[1, 1, 1], bit_length=2)

    assert observed_starts == [(start, 0) for start in range(1, 6)]


def test_prefix_token_decoder_searches_only_token_boundaries(monkeypatch):
    detector = DiscDetector(
        "start-search",
        payload_bits=1,
        context_mode="prefix_token_ngram",
        context_width=1,
    )
    observed_starts = []

    def record_start(self, bits, token_ids, bit_length, n_star, n_star_tokens, watermark_mask=None):
        observed_starts.append((n_star, n_star_tokens))
        return DetectionResult(False, None, n_star, 1.0, 1.0, 0.0, 0)

    monkeypatch.setattr(DiscDetector, "_detect_positions", record_start)
    detector.detect(
        [0, 1, 0, 1, 0, 1, 0, 1],
        token_ids=[1, 1, 1, 1],
        bit_length=2,
    )

    assert observed_starts == [(2, 1), (4, 2), (6, 3)]


def test_conditional_probability_non_power_of_two_vocab():
    probs = np.array([0.1, 0.2, 0.3, 0.4, 0.0])
    assert np.isclose(conditional_bit_probability(probs, 0, 0), 0.0)
    assert np.isclose(conditional_bit_probability(probs, 0, 1), 0.7)
    assert np.isclose(conditional_bit_probability(probs, 1, 1), 0.0)


def test_interval_wraps():
    assert in_shifted_interval(0.95, 0.3, 0.9)
    assert in_shifted_interval(0.05, 0.3, 0.9)
    assert not in_shifted_interval(0.5, 0.3, 0.9)


def test_score_matches_equation_24():
    assert np.isclose(disc_score(1, 0.2, 0.3), -math.log(0.9))
    assert np.isclose(disc_score(1, 0.8, 0.3), -math.log(0.5))
    assert np.isclose(disc_score(0, 0.2, 0.3), -math.log(0.1))
    assert np.isclose(disc_score(0, 0.8, 0.3), -math.log(0.5))


def test_prf_is_stable_and_domain_sensitive():
    prf = HmacPrf("key")
    assert prf.uniform([1, 0], [0, 1]) == prf.uniform([1, 0], [0, 1])
    assert prf.uniform([1, 0], [0, 1]) != prf.uniform([1], [0, 0, 1])
    assert prf.uniform([], [0, 1], domain=HmacPrf.domain_randomization) != prf.uniform(
        [], [0, 1]
    )


def test_gray_mapping_is_bijective():
    assert [gray_encode(value) for value in range(8)] == [0, 1, 3, 2, 6, 7, 5, 4]
    assert [gray_decode(gray_encode(value)) for value in range(32)] == list(range(32))


def test_token_bit_round_trip():
    token_ids = [0, 1, 7, 4]
    assert bits_to_token_ids(token_ids_to_bits(token_ids, 3), 3) == token_ids


def test_encode_token_never_selects_missing_leaf():
    encoder = DiscEncoder("key", 2, 2, entropy_threshold=0, context_width=1, seed=1)
    probs = np.array([0.05, 0.15, 0.2, 0.25, 0.35])
    for _ in range(50):
        assert encoder.encode_token(probs) < len(probs)


def test_known_start_round_trip():
    key = "round-trip"
    encoder = DiscEncoder(key, 5, 3, entropy_threshold=2.0, context_width=8, seed=7)
    rng = np.random.default_rng(12)
    for p_one in rng.uniform(0.1, 0.9, 400):
        encoder.encode_bit(float(p_one))
    result = DiscDetector(key, 3, context_width=8, fpr=0.01).detect(
        encoder.bits,
        n_star_candidates=[encoder.n_star],
        watermark_mask=encoder.watermark_mask,
    )
    assert result.detected
    assert result.payload == 5
    assert result.n_star == encoder.n_star


def test_unknown_start_round_trip():
    key = "full-search"
    encoder = DiscEncoder(key, 2, 2, entropy_threshold=2.0, context_width=8, seed=4)
    for p_one in np.random.default_rng(21).uniform(0.1, 0.9, 300):
        encoder.encode_bit(float(p_one))
    result = DiscDetector(key, 2, context_width=8, fpr=0.01).detect(
        encoder.bits, watermark_mask=encoder.watermark_mask
    )
    assert result.detected
    assert result.payload == 2
    assert result.n_star == encoder.n_star


def test_gray_mapping_round_trip():
    key = "gray-round-trip"
    encoder = DiscEncoder(
        key,
        6,
        3,
        entropy_threshold=2.0,
        context_width=8,
        seed=14,
        message_mapping="gray",
    )
    assert encoder.mapped_payload == gray_encode(6)
    for p_one in np.random.default_rng(30).uniform(0.1, 0.9, 350):
        encoder.encode_bit(float(p_one))
    result = DiscDetector(key, 3, context_width=8, fpr=0.01, message_mapping="gray").detect(
        encoder.bits, watermark_mask=encoder.watermark_mask
    )
    assert result.detected
    assert result.payload == 6
    assert result.n_star == encoder.n_star


def test_empty_r_is_valid_prf_input():
    prf = HmacPrf("key")
    empty = prf.uniform([], [0, 1, 1, 0])
    nonempty = prf.uniform([1, 0], [0, 1, 1, 0])
    assert 0.0 < empty < 1.0
    assert empty == prf.uniform([], [0, 1, 1, 0])
    assert empty != nonempty


def test_use_prefix_false_round_trip():
    key = "no-prefix"
    encoder = DiscEncoder(
        key,
        5,
        3,
        context_width=8,
        seed=7,
        use_prefix=False,
    )
    rng = np.random.default_rng(12)
    for p_one in rng.uniform(0.1, 0.9, 400):
        encoder.encode_bit(float(p_one))
    assert encoder.n_star == 0
    assert encoder.random_initialization == []
    result = DiscDetector(key, 3, context_width=8, fpr=0.01, use_prefix=False).detect(
        encoder.bits, watermark_mask=encoder.watermark_mask
    )
    assert result.detected
    assert result.payload == 5
    assert result.n_star == 0


def test_no_prefix_randomization_records_mask_and_round_trips():
    encoder = DiscEncoder(
        "masked-no-prefix",
        5,
        3,
        context_width=8,
        seed=7,
        use_prefix=False,
        randomization_window=8,
        randomize_without_prefix=True,
    )
    probabilities = np.random.default_rng(12).uniform(0.1, 0.9, 400)
    for probability_one in probabilities:
        encoder.encode_bit(float(probability_one))

    assert len(encoder.watermark_mask) == len(encoder.bits)
    assert not all(encoder.watermark_mask[8:])
    result = DiscDetector("masked-no-prefix", 3, context_width=8, fpr=0.01, use_prefix=False).detect(
        encoder.bits,
        watermark_mask=encoder.watermark_mask,
    )
    assert result.detected
    assert result.payload == 5

    inferred = DiscDetector(
        "masked-no-prefix",
        3,
        context_width=8,
        fpr=0.01,
        use_prefix=False,
        randomization_window=8,
        randomize_without_prefix=True,
    ).detect(encoder.bits)
    assert inferred.detected
    assert inferred.payload == 5


def test_no_prefix_randomization_is_opt_in():
    probabilities = np.random.default_rng(9).uniform(0.1, 0.9, 120)
    deterministic_a = DiscEncoder("optional-randomness", 3, 2, context_width=8, use_prefix=False, seed=1)
    deterministic_b = DiscEncoder("optional-randomness", 3, 2, context_width=8, use_prefix=False, seed=1)
    for probability_one in probabilities:
        deterministic_a.encode_bit(float(probability_one))
        deterministic_b.encode_bit(float(probability_one))
    assert deterministic_a.bits == deterministic_b.bits
    assert deterministic_a.watermark_mask == deterministic_b.watermark_mask

    randomized = DiscEncoder(
        "optional-randomness",
        3,
        2,
        context_width=8,
        use_prefix=False,
        seed=2,
        randomize_without_prefix=True,
    )
    for probability_one in probabilities:
        randomized.encode_bit(float(probability_one))
    assert randomized.bits != deterministic_a.bits


def test_detector_rejects_mask_with_wrong_length():
    detector = DiscDetector("key", 2, context_width=2, use_prefix=False)
    try:
        detector.detect([0, 1, 0], watermark_mask=[True, False])
    except ValueError as error:
        assert "same length" in str(error)
    else:
        raise AssertionError("expected a mask length error")


def test_wrong_key_does_not_recover_known_payload():
    encoder = DiscEncoder("right", 3, 2, entropy_threshold=2.0, context_width=8, seed=8)
    for p_one in np.random.default_rng(2).uniform(0.1, 0.9, 250):
        encoder.encode_bit(float(p_one))
    result = DiscDetector("wrong", 2, context_width=8, fpr=1e-6).detect(
        encoder.bits,
        n_star_candidates=[encoder.n_star],
        watermark_mask=encoder.watermark_mask,
    )
    assert not (result.detected and result.payload == 3)
