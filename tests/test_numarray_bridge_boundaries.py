import math
import pytest
from groggy import NumArray, BaseArray, Graph


@pytest.mark.parametrize('make', [NumArray, BaseArray])
def test_array_fancy_indices_require_exact_integer_positions(make):
    a = make([10, 20, 30])
    for values in [[0.5], [-0.5], [math.nan], [math.inf], [float(2**63)]]:
        with pytest.raises(ValueError):
            a[NumArray(values, dtype='float64')]
    assert a[NumArray([2., 0., -1.])].to_list() == [30, 10, 30]
    assert a[NumArray([True, False, True], dtype='bool')].to_list() == [10, 30]


def test_graph_id_arrays_preserve_values_and_signed_integer_dtype():
    graph = Graph()
    a = graph.add_node()
    b = graph.add_node()
    edge = graph.add_edge(a, b)
    assert graph.node_ids.dtype == 'int64'
    assert sorted(graph.node_ids.to_list()) == sorted([a, b])
    assert graph.edge_ids.to_list() == [edge]


def test_base_array_numeric_conversions_check_ranges_precision_and_missing_values():
    assert BaseArray([-2**63, 2**63-1]).to_num_array().to_list() == [-2**63, 2**63-1]
    assert BaseArray([True, False]).to_num_array().dtype == 'bool'
    assert BaseArray([1, 2]).to_num_array().dtype == 'int64'
    assert BaseArray([]).to_num_array().dtype == 'float64'
    assert BaseArray([2.5, -2.5]).to_type('int64').to_list() == [3, -3]
    with pytest.raises(ValueError):
        BaseArray([1.5]).to_type('int64', exact=True)
    with pytest.raises(ValueError):
        BaseArray([2**63-1]).to_type('float64', exact=True)
    for value in [math.nan, math.inf, float(2**63)]:
        with pytest.raises(ValueError):
            BaseArray([value]).to_type('int64')
    for values in [[None], [1, None]]:
        with pytest.raises(TypeError):
            BaseArray(values).to_num_array()
