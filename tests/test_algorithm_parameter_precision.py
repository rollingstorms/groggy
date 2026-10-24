import pytest

from groggy import Graph, algorithms
from groggy import _groggy


@pytest.mark.parametrize("resolution", [1e-300, 1e100, 1.0000000000000002])
def test_scalar_algorithm_configuration_keeps_python_float_precision(resolution):
    algo = algorithms.community.louvain(resolution=resolution)
    assert algo.to_spec()["params"]["resolution"] == resolution
    assert isinstance(algo.to_spec()["params"]["resolution"], float)
    assert algo.validate()
    graph = Graph()
    a, b = graph.add_node(), graph.add_node()
    graph.add_edge(a, b)
    graph.induced_subgraph([a, b]).apply(algo)
    assert graph.get_node_attr(a, "community") is not None


@pytest.mark.parametrize("resolution", [1e-300, 1e100, 1.0000000000000002,
                                        _groggy.AttrValue.float32(1)])
def test_explicit_property_wrapped_parameters_remain_supported(resolution):
    algo = algorithms.algorithm(
        "community.louvain", resolution=_groggy.AttrValue(resolution),
        max_iter=_groggy.AttrValue(20),
    )
    assert algo.validate()
    graph = Graph()
    node = graph.add_node()
    graph.induced_subgraph([node]).apply(algo)
    assert graph.get_node_attr(node, "community") == 0
