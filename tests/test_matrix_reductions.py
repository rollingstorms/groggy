import math
import pytest
from groggy import GraphMatrix


def test_shared_matrix_reductions_preserve_finite_means_and_independent_std_range():
    a = GraphMatrix.from_data([[1e20], [1.], [-1e20]])
    assert a.sum() == 1.
    assert a.sum_axis(0).to_list() == [1.]
    assert a.mean() == 1 / 3
    largest = float.fromhex('0x1.fffffffffffffp1023')
    a = GraphMatrix.from_data([[largest], [largest]])
    assert a.mean_axis(0).to_list() == [largest]
    a = GraphMatrix.from_data([[0.], [largest]])
    assert math.isinf(a.var_axis(0).to_list()[0])
    assert a.std_axis(0).to_list()[0] == pytest.approx(largest / math.sqrt(2), rel=1e-15)
    a = GraphMatrix.from_data([[0.], [1e-300]])
    assert a.std_axis(0).to_list()[0] == pytest.approx(1e-300 / math.sqrt(2), rel=1e-15, abs=0.)


def test_later_nan_propagates_to_extrema_and_empty_means_preserve_axis_length():
    a = GraphMatrix.from_data([[1.], [math.nan]])
    assert math.isnan(a.min())
    assert math.isnan(a.max())
    assert math.isnan(a.min_axis(0).to_list()[0])
    assert GraphMatrix.zeros(3, 0).mean_axis(0).to_list() == [0.] * 3
