import numpy as np
import numpy.testing as npt
import pytest

from ..graph_analysis.triplets import MotifReader


def test_matrix_to_name():
    motif_reader = MotifReader()

    # Test: Valid matrix
    motif_matrix_A = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0]])
    assert motif_reader.matrix_to_name(motif_matrix_A) == 'A'

    # Test: Invalid matrix
    invalid_matrix = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    assert motif_reader.matrix_to_name(invalid_matrix) is None


def test_name_to_matrix():
    motif_reader = MotifReader()

    # Test: Valid name
    motif_name_A = 'A'
    motif_matrix_A = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0]])
    npt.assert_array_equal(motif_reader.name_to_matrix(motif_name_A), motif_matrix_A)

    # Test: Invalid name
    invalid_name = 'Z'
    assert motif_reader.name_to_matrix(invalid_name) is None


def test_index_to_name():
    motif_reader = MotifReader()

    # Test: Valid index
    motif_index_A = 3
    motif_name_A = 'A'
    assert motif_reader.index_to_name(motif_index_A) == motif_name_A

    # Test: Invalid index
    invalid_index = 100
    assert motif_reader.index_to_name(invalid_index) is None


def test_name_to_index():
    motif_reader = MotifReader()

    # Test: Valid name
    motif_name_A = 'A'
    motif_index_A = 3
    assert motif_reader.name_to_index(motif_name_A) == motif_index_A

    # Test: Invalid name
    invalid_name = 'Z'
    assert motif_reader.name_to_index(invalid_name) is None


@pytest.mark.parametrize(
    'motif_matrix, expected_name',
    [
        (np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0]]), 'A'),
        (np.array([[0, 0, 0], [0, 0, 1], [1, 1, 0]]), 'D'),
        (np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]]), 'C'),
    ]
)
def test_matrix_to_name_rotation_equivalence(motif_matrix, expected_name):
    motif_reader = MotifReader()

    # Test: Rotation equivalence
    motif_matrix_rotated = np.rot90(motif_matrix)
    assert motif_reader.matrix_to_name(motif_matrix_rotated) == expected_name


def test_invalid_input():
    motif_reader = MotifReader()

    # Test: Invalid input for matrix_to_name
    invalid_matrix = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    assert motif_reader.matrix_to_name(invalid_matrix) is None

    # Test: Invalid input for name_to_matrix
    invalid_name = 'Z'
    assert motif_reader.name_to_matrix(invalid_name) is None

    # Test: Invalid input for index_to_name
    invalid_index = 100
    assert motif_reader.index_to_name(invalid_index)
