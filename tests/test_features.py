"""Tests for derived feature extraction."""

from __future__ import annotations

import numpy as np

from qstrainer.features.derived import (
    DerivedFeatureExtractor,
    extended_feature_count,
    extended_feature_names,
)
from qstrainer.models.frame import N_BASE_FEATURES


class TestDerivedFeatureExtractor:
    def test_output_shape(self, gen):
        ext = DerivedFeatureExtractor(window_size=10)
        for _ in range(5):
            f = gen.generate_healthy("GPU-FE", "node-fe")
            vec = ext.extract("GPU-FE", f.to_vector())
        assert vec.shape == (extended_feature_count(),)

    def test_extended_count(self):
        # 15 base + 15 delta + 15 rolling_std + 12 cross + 2 cv + 1 trend = 60
        assert extended_feature_count() == 60

    def test_names_match_count(self):
        names = extended_feature_names()
        assert len(names) == extended_feature_count()

    def test_first_15_are_base(self, gen):
        ext = DerivedFeatureExtractor()
        for _ in range(3):
            f = gen.generate_healthy("GPU-B", "node-b")
            base = f.to_vector()
            extended = ext.extract("GPU-B", base)
        # First 15 features should match the base vector
        np.testing.assert_array_almost_equal(extended[:N_BASE_FEATURES], base)
