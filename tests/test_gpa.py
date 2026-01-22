"""Tests for GPA module."""

import numpy as np
import pytest

from slicermorph_gpa import (
    align,
    center,
    centroid_size,
    generalized_procrustes,
    mean_shape,
    procrustes_distance,
    scale,
)


class TestCenter:
    def test_center_moves_centroid_to_origin(self):
        shape = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        centered = center(shape)

        # Centroid should be at origin
        np.testing.assert_array_almost_equal(
            centered.mean(axis=0), np.array([0.0, 0.0, 0.0])
        )

    def test_center_preserves_relative_positions(self):
        shape = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        centered = center(shape)

        # Distances between points should be preserved
        orig_dist = np.linalg.norm(shape[0] - shape[1])
        cent_dist = np.linalg.norm(centered[0] - centered[1])
        np.testing.assert_almost_equal(orig_dist, cent_dist)


class TestScale:
    def test_scale_normalizes_to_unit_norm(self):
        shape = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
        scaled = scale(shape)

        np.testing.assert_almost_equal(np.linalg.norm(scaled), 1.0)

    def test_scale_handles_zero_shape(self):
        shape = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        scaled = scale(shape)

        # Should return unchanged
        np.testing.assert_array_equal(scaled, shape)


class TestCentroidSize:
    def test_centroid_size_unit_triangle(self):
        # Equilateral triangle centered at origin
        shape = np.array([[1.0, 0.0, 0.0], [-0.5, 0.866, 0.0], [-0.5, -0.866, 0.0]])
        size = centroid_size(shape)

        assert size > 0

    def test_centroid_size_scaled_shape(self):
        shape = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        size1 = centroid_size(shape)
        size2 = centroid_size(shape * 2)

        np.testing.assert_almost_equal(size2, size1 * 2)


class TestAlign:
    def test_align_rotated_shape(self):
        # Reference shape
        ref = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        # 90 degree rotation around z-axis
        theta = np.pi / 2
        rotation = np.array(
            [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
        )
        rotated = np.dot(ref, rotation)

        # Align should recover original
        aligned = align(rotated, ref)
        np.testing.assert_array_almost_equal(aligned, ref, decimal=5)

    def test_align_preserves_shape(self):
        ref = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        shape = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])

        aligned = align(shape, ref)

        # Pairwise distances should be preserved
        orig_dists = [np.linalg.norm(shape[i] - shape[j]) for i in range(3) for j in range(i + 1, 3)]
        aligned_dists = [np.linalg.norm(aligned[i] - aligned[j]) for i in range(3) for j in range(i + 1, 3)]

        np.testing.assert_array_almost_equal(orig_dists, aligned_dists)


class TestMeanShape:
    def test_mean_shape_single_specimen(self):
        landmarks = np.array([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]], [[7.0], [8.0], [9.0]]])
        mean = mean_shape(landmarks)

        expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        np.testing.assert_array_equal(mean, expected)

    def test_mean_shape_multiple_specimens(self):
        landmarks = np.zeros((3, 3, 2))
        landmarks[:, :, 0] = [[0, 0, 0], [2, 0, 0], [0, 2, 0]]
        landmarks[:, :, 1] = [[0, 0, 0], [4, 0, 0], [0, 4, 0]]

        mean = mean_shape(landmarks)

        expected = np.array([[0, 0, 0], [3, 0, 0], [0, 3, 0]])
        np.testing.assert_array_equal(mean, expected)


class TestProcrustesDistance:
    def test_procrustes_distance_identical(self):
        landmarks = np.zeros((3, 3, 2))
        landmarks[:, :, 0] = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        landmarks[:, :, 1] = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]

        ref = mean_shape(landmarks)
        dists = procrustes_distance(landmarks, ref)

        np.testing.assert_array_almost_equal(dists, [0.0, 0.0])

    def test_procrustes_distance_different(self):
        landmarks = np.zeros((3, 3, 2))
        landmarks[:, :, 0] = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        landmarks[:, :, 1] = [[0, 0, 0], [2, 0, 0], [0, 2, 0]]

        ref = landmarks[:, :, 0]
        dists = procrustes_distance(landmarks, ref)

        assert dists[0] == 0.0
        assert dists[1] > 0.0


class TestGeneralizedProcrustes:
    def test_gpa_aligns_translated_shapes(self):
        # Two identical shapes, one translated
        landmarks = np.zeros((3, 3, 2))
        landmarks[:, :, 0] = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        landmarks[:, :, 1] = [[10, 10, 10], [11, 10, 10], [10, 11, 10]]

        result = generalized_procrustes(landmarks)

        # After alignment, shapes should be nearly identical
        diff = np.linalg.norm(result.aligned[:, :, 0] - result.aligned[:, :, 1])
        assert diff < 0.01

    def test_gpa_aligns_scaled_shapes(self):
        landmarks = np.zeros((3, 3, 2))
        landmarks[:, :, 0] = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        landmarks[:, :, 1] = [[0, 0, 0], [2, 0, 0], [0, 2, 0]]

        result = generalized_procrustes(landmarks, scale=True)

        # After scaling, shapes should be similar
        diff = np.linalg.norm(result.aligned[:, :, 0] - result.aligned[:, :, 1])
        assert diff < 0.01

    def test_gpa_no_scale_preserves_size_differences(self):
        landmarks = np.zeros((3, 3, 2))
        landmarks[:, :, 0] = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        landmarks[:, :, 1] = [[0, 0, 0], [2, 0, 0], [0, 2, 0]]

        result = generalized_procrustes(landmarks, scale=False)

        # Without scaling, size difference should remain
        size0 = np.linalg.norm(result.aligned[:, :, 0])
        size1 = np.linalg.norm(result.aligned[:, :, 1])
        assert size1 > size0

    def test_gpa_returns_centroid_sizes(self):
        landmarks = np.zeros((3, 3, 2))
        landmarks[:, :, 0] = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        landmarks[:, :, 1] = [[0, 0, 0], [2, 0, 0], [0, 2, 0]]

        result = generalized_procrustes(landmarks)

        assert len(result.centroid_sizes) == 2
        assert result.centroid_sizes[1] > result.centroid_sizes[0]

    def test_gpa_does_not_modify_input(self):
        landmarks = np.zeros((3, 3, 2))
        landmarks[:, :, 0] = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        landmarks[:, :, 1] = [[10, 10, 10], [11, 10, 10], [10, 11, 10]]
        original = landmarks.copy()

        generalized_procrustes(landmarks)

        np.testing.assert_array_equal(landmarks, original)
