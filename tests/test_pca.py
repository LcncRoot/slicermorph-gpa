"""Tests for PCA module."""

import numpy as np
import pytest

from slicermorph_gpa import generalized_procrustes, pca, project_to_pc_space, warp_along_pc


class TestPCA:
    def test_pca_returns_correct_shapes(self):
        # Create synthetic data: 5 landmarks, 3 dims, 10 specimens
        np.random.seed(42)
        landmarks = np.random.randn(5, 3, 10)

        result = pca(landmarks)

        assert result.scores.shape == (10, 10)  # n_specimens x n_components
        assert result.vectors.shape == (15, 10)  # n_coords x n_components
        assert result.values.shape == (10,)
        assert result.variance_explained.shape == (10,)
        assert result.mean.shape == (5, 3)

    def test_pca_eigenvalues_descending(self):
        np.random.seed(42)
        landmarks = np.random.randn(5, 3, 10)

        result = pca(landmarks)

        # Eigenvalues should be in descending order
        for i in range(len(result.values) - 1):
            assert result.values[i] >= result.values[i + 1]

    def test_pca_variance_explained_sums_to_one(self):
        np.random.seed(42)
        landmarks = np.random.randn(5, 3, 10)

        result = pca(landmarks)

        np.testing.assert_almost_equal(result.variance_explained.sum(), 1.0)

    def test_pca_n_components(self):
        np.random.seed(42)
        landmarks = np.random.randn(5, 3, 10)

        result = pca(landmarks, n_components=3)

        assert result.scores.shape[1] == 3
        assert result.vectors.shape[1] == 3
        assert result.values.shape[0] == 3

    def test_pca_with_aligned_data(self):
        # Create data with known structure
        np.random.seed(42)
        landmarks = np.random.randn(5, 3, 10)

        gpa_result = generalized_procrustes(landmarks)
        pca_result = pca(gpa_result.aligned)

        assert pca_result.scores.shape[0] == 10
        assert pca_result.values[0] >= pca_result.values[-1]


class TestWarpAlongPC:
    def test_warp_along_pc_shape(self):
        np.random.seed(42)
        landmarks = np.random.randn(5, 3, 10)
        pca_result = pca(landmarks)

        mean = landmarks.mean(axis=2)
        warped = warp_along_pc(mean, pca_result, pc=1, magnitude=1.0)

        assert warped.shape == mean.shape

    def test_warp_along_pc_zero_magnitude(self):
        np.random.seed(42)
        landmarks = np.random.randn(5, 3, 10)
        pca_result = pca(landmarks)

        mean = landmarks.mean(axis=2)
        warped = warp_along_pc(mean, pca_result, pc=1, magnitude=0.0)

        np.testing.assert_array_almost_equal(warped, mean)

    def test_warp_along_pc_opposite_magnitudes(self):
        np.random.seed(42)
        landmarks = np.random.randn(5, 3, 10)
        pca_result = pca(landmarks)

        mean = landmarks.mean(axis=2)
        warped_pos = warp_along_pc(mean, pca_result, pc=1, magnitude=1.0)
        warped_neg = warp_along_pc(mean, pca_result, pc=1, magnitude=-1.0)

        # The average of positive and negative warps should equal the mean
        avg = (warped_pos + warped_neg) / 2
        np.testing.assert_array_almost_equal(avg, mean)

    def test_warp_along_pc_invalid_pc(self):
        np.random.seed(42)
        landmarks = np.random.randn(5, 3, 10)
        pca_result = pca(landmarks)

        mean = landmarks.mean(axis=2)

        with pytest.raises(ValueError):
            warp_along_pc(mean, pca_result, pc=100, magnitude=1.0)

        with pytest.raises(ValueError):
            warp_along_pc(mean, pca_result, pc=0, magnitude=1.0)


class TestProjectToPCSpace:
    def test_project_to_pc_space_shape(self):
        np.random.seed(42)
        landmarks = np.random.randn(5, 3, 10)
        pca_result = pca(landmarks)

        projected = project_to_pc_space(landmarks, pca_result, pc_x=1, pc_y=2)

        assert projected.shape == (10, 2)

    def test_project_to_pc_space_consistent_with_scores(self):
        np.random.seed(42)
        landmarks = np.random.randn(5, 3, 10)
        pca_result = pca(landmarks)

        projected = project_to_pc_space(landmarks, pca_result, pc_x=1, pc_y=2)

        # Should match first two columns of scores
        np.testing.assert_array_almost_equal(projected[:, 0], pca_result.scores[:, 0])
        np.testing.assert_array_almost_equal(projected[:, 1], pca_result.scores[:, 1])
