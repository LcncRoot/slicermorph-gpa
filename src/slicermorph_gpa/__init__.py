"""
SlicerMorph GPA - Generalized Procrustes Analysis library.

A standalone Python library for performing Generalized Procrustes Analysis
(GPA) and Principal Component Analysis (PCA) on 3D landmark data.

Extracted from the SlicerMorph project (https://github.com/SlicerMorph/SlicerMorph).

Example usage:
    >>> import slicermorph_gpa as gpa
    >>>
    >>> # Load landmark data
    >>> dataset = gpa.load_dataset("specimens/*.fcsv")
    >>>
    >>> # Perform GPA
    >>> result = gpa.generalized_procrustes(dataset)
    >>>
    >>> # Perform PCA on aligned data
    >>> pca_result = gpa.pca(result.aligned)
    >>>
    >>> # Visualize shape variation
    >>> warped = gpa.warp_along_pc(result.mean_shape, pca_result, pc=1, magnitude=2.0)
"""

from slicermorph_gpa.gpa import (
    GPAResult,
    align,
    center,
    centroid_size,
    generalized_procrustes,
    mean_shape,
    procrustes_distance,
    scale,
)
from slicermorph_gpa.io import (
    get_filenames,
    load_dataset,
    read_landmarks,
    write_landmarks,
)
from slicermorph_gpa.pca import (
    PCAResult,
    pca,
    project_to_pc_space,
    warp_along_pc,
)

__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    # GPA functions
    "GPAResult",
    "generalized_procrustes",
    "center",
    "scale",
    "align",
    "mean_shape",
    "centroid_size",
    "procrustes_distance",
    # PCA functions
    "PCAResult",
    "pca",
    "warp_along_pc",
    "project_to_pc_space",
    # I/O functions
    "read_landmarks",
    "write_landmarks",
    "load_dataset",
    "get_filenames",
]
