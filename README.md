# SlicerMorph GPA

A standalone Python library for Generalized Procrustes Analysis (GPA) and Principal Component Analysis (PCA) on 3D landmark data.

Extracted from the [SlicerMorph](https://github.com/SlicerMorph/SlicerMorph) project to enable programmatic morphometric analysis without requiring 3D Slicer.

## Installation

```bash
pip install slicermorph-gpa
```

For JSON markup file support (`.mrk.json`), install with the optional pandas dependency:

```bash
pip install slicermorph-gpa[json]
```

## Quick Start

```python
import slicermorph_gpa as gpa

# Load landmark data from multiple specimens
dataset = gpa.load_dataset("specimens/*.fcsv")

# Perform Generalized Procrustes Analysis
result = gpa.generalized_procrustes(dataset)

# Access results
aligned = result.aligned          # Aligned coordinates
mean = result.mean_shape          # Mean shape
sizes = result.centroid_sizes     # Original centroid sizes

# Perform PCA on aligned data
pca_result = gpa.pca(result.aligned)

# View variance explained
print(pca_result.variance_explained[:5])  # First 5 PCs

# Visualize shape variation along PC1
warped_pos = gpa.warp_along_pc(mean, pca_result, pc=1, magnitude=2.0)
warped_neg = gpa.warp_along_pc(mean, pca_result, pc=1, magnitude=-2.0)
```

## Features

### I/O Functions

- `read_landmarks(filepath)` - Read landmarks from .fcsv or .mrk.json files
- `write_landmarks(landmarks, filepath)` - Write landmarks to file
- `load_dataset(source)` - Load multiple files into a single array
- `get_filenames(source)` - Get specimen names for labeling

### GPA Functions

- `generalized_procrustes(landmarks, scale=True)` - Full GPA with optional scaling
- `center(shape)` - Center a shape at the origin
- `scale(shape)` - Scale to unit centroid size
- `align(shape, reference)` - Align via optimal rotation
- `mean_shape(landmarks)` - Compute mean shape
- `centroid_size(shape)` - Compute centroid size
- `procrustes_distance(landmarks, reference)` - Compute Procrustes distances

### PCA Functions

- `pca(landmarks, n_components=None)` - Principal Component Analysis
- `warp_along_pc(mean_shape, pca_result, pc, magnitude)` - Visualize PC shape changes
- `project_to_pc_space(landmarks, pca_result, pc_x, pc_y)` - Project to 2D PC space

## Data Format

### Input: Landmark Arrays

Landmarks are represented as NumPy arrays:
- Single specimen: `(n_landmarks, 3)` - rows are landmarks, columns are X, Y, Z
- Multiple specimens: `(n_landmarks, 3, n_specimens)` - 3D array

### File Formats

**FCSV** (3D Slicer fiducial CSV):
```
# Markups fiducial file version = 4.11
# CoordinateSystem = LPS
# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID
vtkMRMLMarkupsFiducialNode_0,10.5,20.3,5.1,0,0,0,1,1,1,0,Nasion,,
```

**Markup JSON** (3D Slicer 5.x):
```json
{
  "markups": [{
    "controlPoints": [
      {"label": "Nasion", "position": [10.5, 20.3, 5.1]}
    ]
  }]
}
```

## Boas Coordinates (No Scaling)

For analyses where size matters (e.g., allometry studies), use `scale=False`:

```python
result = gpa.generalized_procrustes(dataset, scale=False)
```

This performs Procrustes alignment without scaling to unit centroid size, preserving size differences between specimens.

## Example: Full Analysis Pipeline

```python
import slicermorph_gpa as gpa
import numpy as np

# Load data
dataset = gpa.load_dataset("data/")
filenames = gpa.get_filenames("data/")

# GPA
result = gpa.generalized_procrustes(dataset)

# PCA
pca_result = gpa.pca(result.aligned)

# Get PC scores for plotting
pc_coords = gpa.project_to_pc_space(result.aligned, pca_result, pc_x=1, pc_y=2)

# Find specimen closest to mean
distances = gpa.procrustes_distance(result.aligned, result.mean_shape)
closest_idx = np.argmin(distances)
print(f"Closest to mean: {filenames[closest_idx]}")

# Export mean shape
gpa.write_landmarks(result.mean_shape, "mean_shape.fcsv")
```

## References

- Dryden, I.L. and Mardia, K.V. (2016). *Statistical Shape Analysis with Applications in R*. Wiley.
- SlicerMorph: Rolfe et al. (2021). SlicerMorph: An open and extensible platform to retrieve, visualize and analyze 3D morphology. *Methods in Ecology and Evolution*.

## License

BSD 2-Clause License. See [LICENSE](LICENSE) for details.

## Acknowledgments

This library is part of the [SlicerMorph](https://slicermorph.github.io/) project, supported by NSF Advances in Bioinformatics grants (DBI-1759883, DBI-1759637, DBI-1759839) awarded to Murat Maga (University of Washington), Adam Summers (University of Washington), and Douglas Boyer (Duke University).
