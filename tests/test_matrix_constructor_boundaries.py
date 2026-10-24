import sys
import pytest
import groggy
from groggy import GraphMatrix, BaseArray, NumArray


@pytest.mark.parametrize('constructor', [GraphMatrix.zeros, GraphMatrix.ones])
def test_shape_and_label_overflow_reject_without_fallback_matrix(constructor):
    for rows, cols in [(sys.maxsize, 3), (0, 2**64-1)]:
        with pytest.raises((ValueError, OverflowError)):
            constructor(rows, cols)
    assert constructor(0, 3).shape == (0, 3)
    for dtype in ['int', 'int64', 'bool', 'str', 'float32']:
        with pytest.raises(ValueError, match='float64'):
            constructor(2, 3, dtype=dtype)


@pytest.mark.parametrize('value', [None, 'not numeric', 2**63-1])
def test_all_matrix_constructor_paths_reject_missing_nonnumeric_or_inexact_cells(value):
    for build in [lambda: groggy.matrix([[value]]),
                  lambda: GraphMatrix.from_data([[value]]),
                  lambda: GraphMatrix([BaseArray([value])]),
                  lambda: GraphMatrix.from_base_array(BaseArray([value]), 1, 1)]:
        with pytest.raises((TypeError, ValueError, OverflowError)):
            build()
    if isinstance(value, int):
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            GraphMatrix([NumArray([value], dtype='int64')])


def test_compact_integer_columns_and_explicit_row_major_constructors():
    matrix = GraphMatrix([BaseArray([3, True]), NumArray([4, 5])])
    assert matrix.shape == (2, 2)
    assert matrix.get(0, 0) == 3 and matrix.get(1, 0) == 1
    a = GraphMatrix.from_base_array(BaseArray([1, 2, 3, 4, 5, 6]), 2, 3)
    b = GraphMatrix.from_flattened(NumArray([1, 2, 3, 4, 5, 6]), 2, 3)
    assert a.get(1, 2) == b.get(1, 2) == 6
    assert NumArray([]).reshape(0, 3).shape == (0, 3)
    with pytest.raises((ValueError, OverflowError)):
        GraphMatrix.from_flattened(NumArray([1]), sys.maxsize, 3)


def test_attribute_export_preserves_f64_width_and_representable_range():
    a = GraphMatrix.from_data([[1. / 3]])
    assert a.to_base_array(exact=True).to_list() == [1. / 3]
    assert a.to_base_array().to_list() == [1. / 3]
    assert GraphMatrix.from_data([[1e300]]).to_base_array().to_list() == [1e300]
