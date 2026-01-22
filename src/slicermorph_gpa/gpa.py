"""
Generalized Procrustes Analysis (GPA) functions.

This module provides functions for performing Generalized Procrustes Analysis
on 3D landmark data, including centering, scaling, and alignment operations.

Based on Dryden and Mardia (2016) "Statistical Shape Analysis".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg as sp

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class GPAResult:
    """Result of Generalized Procrustes Analysis.

    Attributes:
        aligned: Aligned landmark coordinates, shape (n_landmarks, n_dims, n_specimens)
        mean_shape: Mean shape after alignment, shape (n_landmarks, n_dims)
        centroid_sizes: Centroid size of each specimen before alignment
    """

    aligned: NDArray[np.floating]
    mean_shape: NDArray[np.floating]
    centroid_sizes: NDArray[np.floating]


def center(shape: NDArray[np.floating]) -> NDArray[np.floating]:
    """Center a shape by subtracting the centroid.

    Args:
        shape: Landmark coordinates, shape (n_landmarks, n_dims)

    Returns:
        Centered shape with centroid at origin
    """
    return shape - shape.mean(axis=0)


def scale(shape: NDArray[np.floating]) -> NDArray[np.floating]:
    """Scale a shape to unit centroid size (Frobenius norm).

    Args:
        shape: Landmark coordinates, shape (n_landmarks, n_dims)

    Returns:
        Scaled shape with unit centroid size
    """
    norm = np.linalg.norm(shape)
    if norm == 0:
        return shape
    return shape / norm


def centroid_size(shape: NDArray[np.floating]) -> float:
    """Compute the centroid size of a shape.

    Centroid size is the square root of the sum of squared distances
    from each landmark to the centroid.

    Args:
        shape: Landmark coordinates, shape (n_landmarks, n_dims)

    Returns:
        Centroid size (scalar)
    """
    centered = center(shape)
    return float(np.linalg.norm(centered))


def align(
    shape: NDArray[np.floating],
    reference: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Align a shape to a reference shape using optimal rotation.

    Uses Singular Value Decomposition (SVD) to find the optimal
    rotation matrix that minimizes the Procrustes distance.

    Args:
        shape: Shape to align, shape (n_landmarks, n_dims)
        reference: Reference shape to align to, shape (n_landmarks, n_dims)

    Returns:
        Rotated shape aligned to reference
    """
    u, s, v = sp.svd(np.dot(reference.T, shape), full_matrices=True)
    rotation_matrix = np.dot(v.T, u.T)
    return np.dot(shape, rotation_matrix)


def mean_shape(landmarks: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute the mean shape from multiple specimens.

    Args:
        landmarks: Landmark coordinates, shape (n_landmarks, n_dims, n_specimens)

    Returns:
        Mean shape, shape (n_landmarks, n_dims)
    """
    return landmarks.mean(axis=2)


def procrustes_distance(
    landmarks: NDArray[np.floating],
    reference: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute Procrustes distances from each specimen to a reference shape.

    Args:
        landmarks: Aligned coordinates, shape (n_landmarks, n_dims, n_specimens)
        reference: Reference shape (e.g. mean), shape (n_landmarks, n_dims)

    Returns:
        Array of Procrustes distances, shape (n_specimens,)
    """
    n_specimens = landmarks.shape[2]
    distances = np.zeros(n_specimens)
    for i in range(n_specimens):
        diff = landmarks[:, :, i] - reference
        distances[i] = np.linalg.norm(diff, "fro")
    return distances


def generalized_procrustes(
    landmarks: NDArray[np.floating],
    scale: bool = True,
    max_iterations: int = 5,
    tolerance: float = 0.0001,
) -> GPAResult:
    """Perform Generalized Procrustes Analysis on a set of landmark configurations.

    This function aligns multiple specimen landmark configurations to minimize
    the total Procrustes distance. The algorithm iteratively:
    1. Centers (and optionally scales) each specimen
    2. Aligns all specimens to the current mean shape
    3. Recomputes the mean shape
    4. Repeats until convergence

    Args:
        landmarks: Landmark coordinates, shape (n_landmarks, n_dims, n_specimens).
            Will be copied, not modified in place.
        scale: If True (default), scale specimens to unit centroid size.
            If False, perform Boas coordinates (no scaling).
        max_iterations: Maximum number of alignment iterations
        tolerance: Convergence threshold for mean shape change

    Returns:
        GPAResult containing aligned coordinates, mean shape, and centroid sizes
    """
    # Work on a copy
    aligned = landmarks.copy()
    n_landmarks, n_dims, n_specimens = aligned.shape

    # Compute centroid sizes before any transformations
    centroid_sizes = np.zeros(n_specimens)
    for i in range(n_specimens):
        centroid_sizes[i] = np.linalg.norm(
            aligned[:, :, i] - aligned[:, :, i].mean(axis=0)
        )

    # Center (and optionally scale) each specimen
    for i in range(n_specimens):
        aligned[:, :, i] = center(aligned[:, :, i])
        if scale:
            aligned[:, :, i] = _scale_shape(aligned[:, :, i])

    # Initial alignment to first specimen
    aligned = _procrustes_align_all(aligned[:, :, 0], aligned, scale=scale)

    # Compute initial mean
    current_mean = mean_shape(aligned)
    if scale:
        current_mean = _scale_shape(current_mean)
    else:
        current_mean = center(current_mean)

    # Iterate until convergence
    for _ in range(max_iterations):
        aligned = _procrustes_align_all(current_mean, aligned, scale=scale)
        new_mean = mean_shape(aligned)

        if scale:
            new_mean = _scale_shape(new_mean)
        else:
            new_mean = center(new_mean)

        diff = np.linalg.norm(current_mean - new_mean)
        current_mean = new_mean

        if diff < tolerance:
            break

    # Final re-centering for no-scale case
    if not scale:
        for i in range(n_specimens):
            aligned[:, :, i] = center(aligned[:, :, i])

    return GPAResult(
        aligned=aligned,
        mean_shape=current_mean,
        centroid_sizes=centroid_sizes,
    )


def _scale_shape(shape: NDArray[np.floating]) -> NDArray[np.floating]:
    """Internal scale function (avoids name collision with scale parameter)."""
    norm = np.linalg.norm(shape)
    if norm == 0:
        return shape
    return shape / norm


def _procrustes_align_all(
    reference: NDArray[np.floating],
    landmarks: NDArray[np.floating],
    scale: bool = True,
) -> NDArray[np.floating]:
    """Align all specimens to a reference shape."""
    n_specimens = landmarks.shape[2]

    if scale:
        ref = _scale_shape(reference)
    else:
        ref = center(reference)

    for i in range(n_specimens):
        aligned = align(landmarks[:, :, i], ref)
        if not scale:
            aligned = center(aligned)
        landmarks[:, :, i] = aligned

    return landmarks
