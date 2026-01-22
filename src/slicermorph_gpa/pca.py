"""
Principal Component Analysis (PCA) for shape data.

This module provides functions for performing PCA on Procrustes-aligned
landmark coordinates, enabling dimensionality reduction and visualization
of shape variation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg as sp

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class PCAResult:
    """Result of Principal Component Analysis on shape data.

    Attributes:
        scores: PC scores for each specimen, shape (n_specimens, n_components)
        vectors: Principal component vectors (loadings), shape (n_coords, n_components)
        values: Eigenvalues in descending order, shape (n_components,)
        variance_explained: Proportion of variance explained by each PC
        mean: Mean shape used for centering, shape (n_landmarks, n_dims)
    """

    scores: NDArray[np.floating]
    vectors: NDArray[np.floating]
    values: NDArray[np.floating]
    variance_explained: NDArray[np.floating]
    mean: NDArray[np.floating]


def pca(
    landmarks: NDArray[np.floating],
    n_components: int | None = None,
) -> PCAResult:
    """Perform Principal Component Analysis on aligned landmark data.

    Args:
        landmarks: Aligned coordinates, shape (n_landmarks, n_dims, n_specimens)
        n_components: Number of components to retain. If None, retains
            min(n_specimens, n_landmarks * n_dims) components.

    Returns:
        PCAResult containing scores, loadings, eigenvalues, and variance explained
    """
    n_landmarks, n_dims, n_specimens = landmarks.shape
    n_coords = n_landmarks * n_dims

    # Flatten to 2D matrix: (n_coords, n_specimens)
    flat = _flatten_landmarks(landmarks)

    # Center the data
    mean_vec = flat.mean(axis=1, keepdims=True)
    centered = flat - mean_vec

    # Compute covariance matrix
    cov_matrix = _covariance(centered)

    # Determine number of components
    if n_components is None:
        n_components = min(n_specimens, n_coords)

    # Compute eigendecomposition
    if n_specimens > n_coords:
        # More specimens than coordinates: full eigendecomposition
        eigenvalues, eigenvectors = sp.eigh(cov_matrix)
    else:
        # More coordinates than specimens: compute only needed eigenvalues
        eigenvalues, eigenvectors = sp.eigh(
            cov_matrix,
            subset_by_index=(n_coords - n_components, n_coords - 1),
        )

    # Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Ensure we have the right number of components
    eigenvalues = eigenvalues[:n_components]
    eigenvectors = eigenvectors[:, :n_components]

    # Compute PC scores: project centered data onto eigenvectors
    scores = np.dot(centered.T, eigenvectors)

    # Compute variance explained
    total_variance = eigenvalues.sum()
    if total_variance > 0:
        variance_explained = eigenvalues / total_variance
    else:
        variance_explained = np.zeros_like(eigenvalues)

    # Reshape mean back to landmark format
    mean_shape = mean_vec.reshape(n_landmarks, n_dims, order="F")

    return PCAResult(
        scores=np.real(scores),
        vectors=np.real(eigenvectors),
        values=np.real(eigenvalues),
        variance_explained=np.real(variance_explained),
        mean=mean_shape,
    )


def warp_along_pc(
    mean_shape: NDArray[np.floating],
    pca_result: PCAResult,
    pc: int,
    magnitude: float,
) -> NDArray[np.floating]:
    """Warp the mean shape along a principal component.

    This is useful for visualizing what shape changes are captured
    by each principal component.

    Args:
        mean_shape: Mean shape, shape (n_landmarks, n_dims)
        pca_result: PCA result containing eigenvectors
        pc: Principal component number (1-indexed, like PC1, PC2, etc.)
        magnitude: How far to warp along the PC (in units of standard deviation)

    Returns:
        Warped shape, shape (n_landmarks, n_dims)
    """
    n_landmarks, n_dims = mean_shape.shape
    pc_index = pc - 1  # Convert to 0-indexed

    if pc_index < 0 or pc_index >= pca_result.vectors.shape[1]:
        raise ValueError(
            f"PC {pc} is out of range. Available: 1-{pca_result.vectors.shape[1]}"
        )

    # Get the eigenvector for this PC
    eigenvector = pca_result.vectors[:, pc_index]

    # Scale by magnitude and standard deviation
    std = np.sqrt(pca_result.values[pc_index]) if pca_result.values[pc_index] > 0 else 1
    scaled_vec = eigenvector * magnitude * std

    # Reshape eigenvector to landmark format
    shift = scaled_vec.reshape(n_landmarks, n_dims, order="F")

    return mean_shape + shift


def project_to_pc_space(
    landmarks: NDArray[np.floating],
    pca_result: PCAResult,
    pc_x: int = 1,
    pc_y: int = 2,
) -> NDArray[np.floating]:
    """Project landmarks onto a 2D PC space for visualization.

    Args:
        landmarks: Landmark coordinates, shape (n_landmarks, n_dims, n_specimens)
        pca_result: PCA result to use for projection
        pc_x: PC number for x-axis (1-indexed)
        pc_y: PC number for y-axis (1-indexed)

    Returns:
        2D coordinates, shape (n_specimens, 2)
    """
    n_landmarks, n_dims, n_specimens = landmarks.shape

    # Flatten landmarks
    flat = _flatten_landmarks(landmarks)

    # Center using PCA mean
    mean_flat = pca_result.mean.reshape(-1, 1, order="F")
    centered = flat - mean_flat

    # Get the two PC vectors
    pc_x_idx = pc_x - 1
    pc_y_idx = pc_y - 1

    vec_x = pca_result.vectors[:, pc_x_idx]
    vec_y = pca_result.vectors[:, pc_y_idx]

    # Project
    coords_x = np.dot(vec_x, centered)
    coords_y = np.dot(vec_y, centered)

    return np.column_stack((coords_x, coords_y))


def _flatten_landmarks(
    landmarks: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Flatten 3D landmark array to 2D matrix.

    Args:
        landmarks: Shape (n_landmarks, n_dims, n_specimens)

    Returns:
        Flattened array, shape (n_landmarks * n_dims, n_specimens)
    """
    n_landmarks, n_dims, n_specimens = landmarks.shape
    flat = np.zeros((n_landmarks * n_dims, n_specimens))
    for i in range(n_specimens):
        flat[:, i] = landmarks[:, :, i].reshape(-1, order="F")
    return flat


def _covariance(centered: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute covariance matrix from centered data.

    Args:
        centered: Centered data, shape (n_features, n_samples)

    Returns:
        Covariance matrix, shape (n_features, n_features)
    """
    n_features, n_samples = centered.shape
    cov = np.zeros((n_features, n_features))
    for i in range(n_samples):
        col = centered[:, i : i + 1]
        cov += np.dot(col, col.T) / n_samples
    return cov
