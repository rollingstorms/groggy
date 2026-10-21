import math
import statistics

import pytest

from groggy import GraphMatrix


@pytest.mark.parametrize("data", [
    [[1., 2., 3.], [4., 6., 8.]],
    [[1e12, 1e12 + 1, 1e12 + 2]],
    [[-5., 0., 11.], [3., 9., 1.], [8., -2., 7.]],
])
def test_sample_statistics_match_independent_reference_on_each_axis(data):
    # Existing nested-list factory treats its outer sequence as columns.
    matrix = GraphMatrix.from_data([list(column) for column in zip(*data)])
    assert matrix.shape == (len(data), len(data[0]))
    for axis, series in [(0, data), (1, list(zip(*data)))]:
        expected_var = [statistics.variance(row) if len(row) > 1 else 0. for row in series]
        expected_std = [statistics.stdev(row) if len(row) > 1 else 0. for row in series]
        assert matrix.var_axis(axis).to_list() == pytest.approx(expected_var)
        assert matrix.std_axis(axis).to_list() == pytest.approx(expected_std)


def test_identity_and_degenerate_or_nonfinite_statistics():
    identity = GraphMatrix.identity(3)
    for row in range(3):
        for col in range(3):
            assert identity.get(row, col) == (1. if row == col else 0.)
    constant = GraphMatrix.from_data([[7.], [7.], [7.]])
    assert constant.std_axis(0).to_list() == [0.]
    assert constant.std_axis(1).to_list() == [0., 0., 0.]
    nonfinite = GraphMatrix.from_data([[1., 1.], [float("nan"), float("inf")]])
    assert all(math.isnan(v) for v in nonfinite.std_axis(0).to_list())
    with pytest.raises(ValueError):
        constant.std_axis(2)


def test_empty_matrix_shapes_and_checked_overflow():
    empty = GraphMatrix.zeros(3, 0)
    assert empty.shape == (3, 0)
    assert empty.std_axis(0).to_list() == [0., 0., 0.]
    assert empty.std_axis(1).to_list() == []
    assert GraphMatrix.identity(0).shape == (0, 0)
    with pytest.raises(RuntimeError):
        GraphMatrix.identity(2**64 - 1)
