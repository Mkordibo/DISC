import math

import numpy as np

from disc.core import (
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
        encoder.bits, n_star_candidates=[encoder.n_star]
    )
    assert result.detected
    assert result.payload == 5
    assert result.n_star == encoder.n_star


def test_unknown_start_round_trip():
    key = "full-search"
    encoder = DiscEncoder(key, 2, 2, entropy_threshold=2.0, context_width=8, seed=4)
    for p_one in np.random.default_rng(21).uniform(0.1, 0.9, 300):
        encoder.encode_bit(float(p_one))
    result = DiscDetector(key, 2, context_width=8, fpr=0.01).detect(encoder.bits)
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
        encoder.bits
    )
    assert result.detected
    assert result.payload == 6
    assert result.n_star == encoder.n_star


def test_wrong_key_does_not_recover_known_payload():
    encoder = DiscEncoder("right", 3, 2, entropy_threshold=2.0, context_width=8, seed=8)
    for p_one in np.random.default_rng(2).uniform(0.1, 0.9, 250):
        encoder.encode_bit(float(p_one))
    result = DiscDetector("wrong", 2, context_width=8, fpr=1e-6).detect(
        encoder.bits, n_star_candidates=[encoder.n_star]
    )
    assert not (result.detected and result.payload == 3)
