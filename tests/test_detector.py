"""Pure-Python checks; the CLIP weights are not downloaded."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import detector  # noqa: E402


def test_smooth_keeps_length_and_suppresses_single_spikes():
    values = np.zeros(11)
    values[5] = 1.0
    out = detector.smooth(values, 5)
    assert len(out) == len(values)
    assert np.isclose(out[5], 0.2) and out.max() < 0.5        # one frame alone cannot cross 0.5


def test_smooth_preserves_a_constant_signal_at_the_edges():
    assert np.allclose(detector.smooth(np.full(7, 0.8), 5), 0.8)


def test_frame_sampling():
    assert detector.frame_indices(100, 30.0, 0.5) == list(range(0, 100, 15))
    assert detector.even_indices(90, 8) == sorted(set(detector.even_indices(90, 8)))   # distinct, ordered
    assert 0 < detector.even_indices(90, 8)[0] and detector.even_indices(90, 8)[-1] < 89


def test_softmax_rows_sum_to_one():
    assert np.allclose(detector.softmax(np.array([[1.0, 2.0], [5.0, -1.0]])).sum(axis=1), 1)


def test_both_classes_have_several_prompts():
    assert set(detector.PROMPTS) == {"violent", "non-violent"}
    assert all(len(v) >= 3 for v in detector.PROMPTS.values())
