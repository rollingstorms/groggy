import math
import struct
import pytest
from groggy import Graph


def grouped_values():
    graph = Graph()
    for kind, value in [('a', 1), ('a', 2), ('b', 4), ('b', 6)]:
        graph.add_node(kind=kind, value=value)
    return graph.nodes.table().group_by('kind')['value']


def keyed(table, column):
    return dict(zip(table.column('kind').to_list(), table.column(column).to_list()))


def test_arrayarray_group_keys_and_documented_f32_results_survive_single_and_multi_aggregation():
    groups = grouped_values()
    assert keyed(groups.sum(), 'sum') == {'a': 3., 'b': 10.}
    f32 = lambda value: struct.unpack('f', struct.pack('f', value))[0]
    expected = {'a': f32(1 / math.sqrt(2)), 'b': f32(math.sqrt(2))}
    actual = keyed(groups.std(), 'std')
    for key in expected:
        assert actual[key] == pytest.approx(expected[key], rel=1e-14)
    result = groups.agg({'deviation': 'std', 'average': 'mean', 'n': 'count'})
    assert keyed(result, 'average') == {'a': 1.5, 'b': 5.}
    assert keyed(result, 'n') == {'a': 2., 'b': 2.}
    for key, value in keyed(result, 'deviation').items():
        assert value == pytest.approx(expected[key], rel=1e-14)
    assert sorted(groups.to_type('float64').to_list()) == [1., 2., 4., 6.]
    with pytest.raises(ValueError, match='conflicting'):
        groups.agg({'kind': 'sum'})
    with pytest.raises(ValueError, match='Unknown aggregation'):
        groups.agg({'x': 'unknown'})


def test_grouped_i64_cancellation_and_high_offset_standard_deviation_survive_both_dispatches():
    graph = Graph()
    largest = 2**63 - 1
    for kind, sequence in [('offset', [largest - 2, largest - 1, largest]),
                           ('cancel', [largest, 1, -largest])]:
        for value in sequence:
            graph.add_node(kind=kind, value=value)
    groups = graph.nodes.table().group_by('kind')['value']
    assert keyed(groups.sum(), 'sum')['cancel'] == 1.
    assert keyed(groups.std(), 'std')['offset'] == 1.
    result = groups.agg({'total': 'sum', 'deviation': 'std'})
    assert keyed(result, 'total')['cancel'] == 1.
    assert keyed(result, 'deviation')['offset'] == 1.
    assert keyed(groups.count(), 'count') == {'offset': 3., 'cancel': 3.}
