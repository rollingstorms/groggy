import itertools
import pytest
from groggy import BaseArray, NumArray, GraphMatrix


@pytest.mark.parametrize('kind', ['base', 'numeric', 'bool'])
@pytest.mark.parametrize('length', [0, 1, 5, 10])
def test_array_slices_match_python_bound_defaults_clamping_and_extreme_steps(kind, length):
    values = [i % 2 == 0 for i in range(length)] if kind == 'bool' else list(range(length))
    a = BaseArray(values) if kind == 'base' else NumArray(values, dtype='bool' if kind == 'bool' else 'int64')
    bounds = [None, -12, -3, -1, 0, 1, 3, 12, -2**63, 2**63-1]
    steps = [None, -4, -1, 1, 3, -2**63, 2**63-1]
    for start, stop, step in itertools.product(bounds, bounds, steps):
        index = slice(start, stop, step)
        assert a[index].to_list() == values[index], (kind, length, start, stop, step)
    with pytest.raises(ValueError):
        a[::0]


def test_matrix_ranges_keep_shared_negative_bounds_and_empty_selection_shapes():
    m = GraphMatrix.from_data([[0., 2., 4., 6., 8.], [1., 3., 5., 7., 9.]])
    assert m[-2:, ::-1].to_list() == [[7., 6.], [9., 8.]]
    assert m[1::2**63-1, :].to_list() == [[2., 3.]]
    assert m[:-10, :].shape == (0, 2)
