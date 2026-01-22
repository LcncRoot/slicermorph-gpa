"""
I/O functions for reading and writing landmark files.

Supports:
- FCSV format (.fcsv) - 3D Slicer fiducial CSV format
- Markup JSON format (.mrk.json) - 3D Slicer 5.x markup format

Both formats are used by 3D Slicer and SlicerMorph for storing 3D landmark
coordinates.
"""

from __future__ import annotations

import glob as glob_module
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Optional pandas import for JSON support
try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def read_landmarks(
    filepath: str | Path,
) -> NDArray[np.floating]:
    """Read landmarks from a file.

    Automatically detects the file format based on extension.

    Args:
        filepath: Path to landmark file (.fcsv or .mrk.json)

    Returns:
        Landmark coordinates, shape (n_landmarks, 3)

    Raises:
        ValueError: If file format is not supported
        FileNotFoundError: If file does not exist
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    suffix = filepath.suffix.lower()
    name = filepath.name.lower()

    if name.endswith(".mrk.json"):
        return _read_markup_json(filepath)
    elif suffix == ".fcsv":
        return _read_fcsv(filepath)
    elif suffix == ".json":
        return _read_markup_json(filepath)
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. "
            "Supported formats: .fcsv, .mrk.json"
        )


def write_landmarks(
    landmarks: NDArray[np.floating],
    filepath: str | Path,
    labels: list[str] | None = None,
) -> None:
    """Write landmarks to a file.

    Args:
        landmarks: Landmark coordinates, shape (n_landmarks, 3)
        filepath: Output file path (.fcsv or .mrk.json)
        labels: Optional labels for each landmark

    Raises:
        ValueError: If file format is not supported
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()
    name = filepath.name.lower()

    if name.endswith(".mrk.json"):
        _write_markup_json(landmarks, filepath, labels)
    elif suffix == ".fcsv":
        _write_fcsv(landmarks, filepath, labels)
    elif suffix == ".json":
        _write_markup_json(landmarks, filepath, labels)
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. "
            "Supported formats: .fcsv, .mrk.json"
        )


def load_dataset(
    source: str | Path | list[str] | list[Path],
) -> NDArray[np.floating]:
    """Load multiple landmark files into a single dataset array.

    Args:
        source: Either:
            - A glob pattern (e.g., "data/*.fcsv")
            - A directory path (loads all .fcsv and .mrk.json files)
            - A list of file paths

    Returns:
        Landmark coordinates, shape (n_landmarks, 3, n_specimens)

    Raises:
        ValueError: If no files found or landmarks have inconsistent shapes
    """
    # Resolve file list
    if isinstance(source, (str, Path)):
        source_path = Path(source)
        if source_path.is_dir():
            # Load all landmark files from directory
            files = list(source_path.glob("*.fcsv")) + list(
                source_path.glob("*.mrk.json")
            )
        else:
            # Treat as glob pattern
            files = [Path(f) for f in glob_module.glob(str(source))]
    else:
        files = [Path(f) for f in source]

    if not files:
        raise ValueError(f"No landmark files found: {source}")

    # Sort for reproducibility
    files = sorted(files)

    # Load first file to get shape
    first = read_landmarks(files[0])
    n_landmarks, n_dims = first.shape
    n_specimens = len(files)

    # Allocate array
    dataset = np.zeros((n_landmarks, n_dims, n_specimens))
    dataset[:, :, 0] = first

    # Load remaining files
    for i, filepath in enumerate(files[1:], start=1):
        lm = read_landmarks(filepath)
        if lm.shape != (n_landmarks, n_dims):
            raise ValueError(
                f"Inconsistent landmark shape in {filepath}: "
                f"expected {(n_landmarks, n_dims)}, got {lm.shape}"
            )
        dataset[:, :, i] = lm

    return dataset


def get_filenames(
    source: str | Path | list[str] | list[Path],
) -> list[str]:
    """Get list of filenames from a source (for labeling specimens).

    Args:
        source: Same as load_dataset

    Returns:
        List of filenames (without directory path)
    """
    if isinstance(source, (str, Path)):
        source_path = Path(source)
        if source_path.is_dir():
            files = list(source_path.glob("*.fcsv")) + list(
                source_path.glob("*.mrk.json")
            )
        else:
            files = [Path(f) for f in glob_module.glob(str(source))]
    else:
        files = [Path(f) for f in source]

    return [f.name for f in sorted(files)]


def _read_fcsv(filepath: Path) -> NDArray[np.floating]:
    """Read landmarks from FCSV format.

    FCSV is a CSV format where:
    - Lines starting with # are headers/comments
    - Data columns: id, x, y, z, ...
    """
    data = []
    with open(filepath) as f:
        for row in f:
            # Skip comment/header lines
            if row.startswith("#"):
                continue
            parts = row.strip().split(",")
            if len(parts) >= 4:
                # Columns 1-3 are x, y, z (0 is id/label)
                try:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    data.append([x, y, z])
                except (ValueError, IndexError):
                    continue

    if not data:
        raise ValueError(f"No valid landmarks found in {filepath}")

    return np.array(data)


def _read_markup_json(filepath: Path) -> NDArray[np.floating]:
    """Read landmarks from 3D Slicer markup JSON format."""
    if not HAS_PANDAS:
        # Fall back to pure json parsing
        return _read_markup_json_no_pandas(filepath)

    with open(filepath) as f:
        content = json.load(f)

    # Navigate to control points
    try:
        control_points = content["markups"][0]["controlPoints"]
    except (KeyError, IndexError) as e:
        raise ValueError(f"Invalid markup JSON format in {filepath}") from e

    df = pd.DataFrame.from_dict(control_points)

    # Extract positions
    positions = df["position"].to_numpy()
    landmarks = np.array([p for p in positions])

    return landmarks


def _read_markup_json_no_pandas(filepath: Path) -> NDArray[np.floating]:
    """Read markup JSON without pandas dependency."""
    with open(filepath) as f:
        content = json.load(f)

    try:
        control_points = content["markups"][0]["controlPoints"]
    except (KeyError, IndexError) as e:
        raise ValueError(f"Invalid markup JSON format in {filepath}") from e

    landmarks = []
    for point in control_points:
        pos = point.get("position", [0, 0, 0])
        landmarks.append(pos)

    return np.array(landmarks)


def _write_fcsv(
    landmarks: NDArray[np.floating],
    filepath: Path,
    labels: list[str] | None = None,
) -> None:
    """Write landmarks to FCSV format."""
    n_landmarks = landmarks.shape[0]

    if labels is None:
        labels = [f"F-{i + 1}" for i in range(n_landmarks)]

    with open(filepath, "w") as f:
        # Write header
        f.write("# Markups fiducial file version = 4.11\n")
        f.write("# CoordinateSystem = LPS\n")
        f.write("# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,assocNodeID\n")

        for i in range(n_landmarks):
            x, y, z = landmarks[i]
            label = labels[i] if i < len(labels) else f"F-{i + 1}"
            f.write(
                f"vtkMRMLMarkupsFiducialNode_{i},{x},{y},{z},"
                f"0,0,0,1,1,1,0,{label},,\n"
            )


def _write_markup_json(
    landmarks: NDArray[np.floating],
    filepath: Path,
    labels: list[str] | None = None,
) -> None:
    """Write landmarks to 3D Slicer markup JSON format."""
    n_landmarks = landmarks.shape[0]

    if labels is None:
        labels = [f"F-{i + 1}" for i in range(n_landmarks)]

    control_points = []
    for i in range(n_landmarks):
        x, y, z = landmarks[i]
        label = labels[i] if i < len(labels) else f"F-{i + 1}"
        control_points.append(
            {
                "id": str(i + 1),
                "label": label,
                "description": "",
                "associatedNodeID": "",
                "position": [float(x), float(y), float(z)],
                "orientation": [-1.0, -0.0, -0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 1.0],
                "selected": True,
                "locked": False,
                "visibility": True,
                "positionStatus": "defined",
            }
        )

    markup = {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
        "markups": [
            {
                "type": "Fiducial",
                "coordinateSystem": "LPS",
                "coordinateUnits": "mm",
                "locked": False,
                "fixedNumberOfControlPoints": False,
                "labelFormat": "%N-%d",
                "lastUsedControlPointNumber": n_landmarks,
                "controlPoints": control_points,
            }
        ],
    }

    with open(filepath, "w") as f:
        json.dump(markup, f, indent=2)
