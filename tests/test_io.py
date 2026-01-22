"""Tests for I/O module."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from slicermorph_gpa import get_filenames, load_dataset, read_landmarks, write_landmarks


@pytest.fixture
def sample_fcsv(tmp_path):
    """Create a sample FCSV file."""
    content = """# Markups fiducial file version = 4.11
# CoordinateSystem = LPS
# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID
vtkMRMLMarkupsFiducialNode_0,1.0,2.0,3.0,0,0,0,1,1,1,0,F-1,,
vtkMRMLMarkupsFiducialNode_1,4.0,5.0,6.0,0,0,0,1,1,1,0,F-2,,
vtkMRMLMarkupsFiducialNode_2,7.0,8.0,9.0,0,0,0,1,1,1,0,F-3,,
"""
    filepath = tmp_path / "sample.fcsv"
    filepath.write_text(content)
    return filepath


@pytest.fixture
def sample_mrk_json(tmp_path):
    """Create a sample markup JSON file."""
    content = {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
        "markups": [
            {
                "type": "Fiducial",
                "coordinateSystem": "LPS",
                "controlPoints": [
                    {"id": "1", "label": "F-1", "position": [1.0, 2.0, 3.0], "positionStatus": "defined"},
                    {"id": "2", "label": "F-2", "position": [4.0, 5.0, 6.0], "positionStatus": "defined"},
                    {"id": "3", "label": "F-3", "position": [7.0, 8.0, 9.0], "positionStatus": "defined"},
                ],
            }
        ],
    }
    filepath = tmp_path / "sample.mrk.json"
    filepath.write_text(json.dumps(content))
    return filepath


class TestReadLandmarks:
    def test_read_fcsv(self, sample_fcsv):
        landmarks = read_landmarks(sample_fcsv)

        expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        np.testing.assert_array_equal(landmarks, expected)

    def test_read_mrk_json(self, sample_mrk_json):
        landmarks = read_landmarks(sample_mrk_json)

        expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        np.testing.assert_array_equal(landmarks, expected)

    def test_read_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            read_landmarks("/nonexistent/file.fcsv")

    def test_read_unsupported_format(self, tmp_path):
        filepath = tmp_path / "file.xyz"
        filepath.write_text("data")

        with pytest.raises(ValueError, match="Unsupported file format"):
            read_landmarks(filepath)


class TestWriteLandmarks:
    def test_write_fcsv(self, tmp_path):
        landmarks = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        filepath = tmp_path / "output.fcsv"

        write_landmarks(landmarks, filepath)

        # Read back and verify
        loaded = read_landmarks(filepath)
        np.testing.assert_array_almost_equal(loaded, landmarks)

    def test_write_mrk_json(self, tmp_path):
        landmarks = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        filepath = tmp_path / "output.mrk.json"

        write_landmarks(landmarks, filepath)

        # Read back and verify
        loaded = read_landmarks(filepath)
        np.testing.assert_array_almost_equal(loaded, landmarks)

    def test_write_with_labels(self, tmp_path):
        landmarks = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        labels = ["Nasion", "Bregma"]
        filepath = tmp_path / "output.fcsv"

        write_landmarks(landmarks, filepath, labels=labels)

        # Verify labels are in file
        content = filepath.read_text()
        assert "Nasion" in content
        assert "Bregma" in content

    def test_roundtrip_fcsv(self, tmp_path):
        original = np.random.randn(10, 3)
        filepath = tmp_path / "roundtrip.fcsv"

        write_landmarks(original, filepath)
        loaded = read_landmarks(filepath)

        np.testing.assert_array_almost_equal(loaded, original)

    def test_roundtrip_mrk_json(self, tmp_path):
        original = np.random.randn(10, 3)
        filepath = tmp_path / "roundtrip.mrk.json"

        write_landmarks(original, filepath)
        loaded = read_landmarks(filepath)

        np.testing.assert_array_almost_equal(loaded, original)


class TestLoadDataset:
    def test_load_from_list(self, tmp_path):
        # Create multiple files
        for i in range(3):
            landmarks = np.array([[i, 0, 0], [0, i, 0], [0, 0, i]], dtype=float)
            filepath = tmp_path / f"specimen_{i}.fcsv"
            write_landmarks(landmarks, filepath)

        files = list(tmp_path.glob("*.fcsv"))
        dataset = load_dataset(files)

        assert dataset.shape == (3, 3, 3)

    def test_load_from_glob_pattern(self, tmp_path):
        # Create multiple files
        for i in range(3):
            landmarks = np.array([[i, 0, 0], [0, i, 0], [0, 0, i]], dtype=float)
            filepath = tmp_path / f"specimen_{i}.fcsv"
            write_landmarks(landmarks, filepath)

        dataset = load_dataset(str(tmp_path / "*.fcsv"))

        assert dataset.shape == (3, 3, 3)

    def test_load_from_directory(self, tmp_path):
        # Create multiple files
        for i in range(3):
            landmarks = np.array([[i, 0, 0], [0, i, 0], [0, 0, i]], dtype=float)
            filepath = tmp_path / f"specimen_{i}.fcsv"
            write_landmarks(landmarks, filepath)

        dataset = load_dataset(tmp_path)

        assert dataset.shape == (3, 3, 3)

    def test_load_empty_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No landmark files found"):
            load_dataset(tmp_path / "nonexistent")

    def test_load_inconsistent_shapes_raises(self, tmp_path):
        # Create files with different numbers of landmarks
        write_landmarks(np.array([[0, 0, 0], [1, 1, 1]]), tmp_path / "a.fcsv")
        write_landmarks(np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]), tmp_path / "b.fcsv")

        with pytest.raises(ValueError, match="Inconsistent landmark shape"):
            load_dataset(tmp_path)


class TestGetFilenames:
    def test_get_filenames(self, tmp_path):
        for name in ["alpha.fcsv", "beta.fcsv", "gamma.fcsv"]:
            (tmp_path / name).write_text("# placeholder")

        names = get_filenames(tmp_path)

        assert sorted(names) == ["alpha.fcsv", "beta.fcsv", "gamma.fcsv"]
