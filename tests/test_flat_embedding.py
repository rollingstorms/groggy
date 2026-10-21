import numpy as np
import pytest
from groggy import Graph, GraphMatrix


def run(g, matrix, **kwargs):
    return g.flat_embedding(matrix, repulsion_weight=0.0, spread_weight=0.0, **kwargs)


def test_actual_sparse_topology_changes_only_connected_rows_and_retains_identity():
    g = Graph()
    nodes = [g.add_node() for _ in range(4)]
    g.remove_node(nodes[1])
    matrix = GraphMatrix.zeros(3, 4)
    initial = run(g, matrix, iterations=0).to_numpy()
    assert np.array_equal(run(g, matrix, iterations=20).to_numpy(), initial)
    for _ in range(120):
        g.add_edge(nodes[0], nodes[0])
    edge = g.add_edge(nodes[2], nodes[3])
    out = run(g, matrix, iterations=20)
    coords = out.to_numpy()
    assert np.linalg.norm(coords[1] - coords[2]) < np.linalg.norm(initial[1] - initial[2])
    assert np.array_equal(coords[0], initial[0])
    assert out.row_labels == [f"node_{n}" for n in [nodes[0], nodes[2], nodes[3]]]
    g.remove_edge(edge)
    assert np.array_equal(run(g, matrix, iterations=20).to_numpy(), initial)


def test_layout_is_seeded_bounded_owned_and_does_not_modify_input():
    g = Graph()
    nodes = [g.add_node() for _ in range(4)]
    g.add_edge(nodes[0], nodes[2])
    matrix = GraphMatrix.zeros(4, 2)
    first = g.flat_embedding(matrix, iterations=10, radius_clip=0.1, seed=42)
    assert np.array_equal(first.to_numpy(), g.flat_embedding(matrix, iterations=10, radius_clip=0.1, seed=42).to_numpy())
    assert np.all(np.linalg.norm(first.to_numpy(), axis=1) <= 0.1 * (1 + 1e-12))
    assert np.array_equal(matrix.to_numpy(), np.zeros((4, 2)))
    before = first.to_numpy().copy()
    g.remove_node(nodes[2])
    assert np.array_equal(first.to_numpy(), before)


@pytest.mark.parametrize("name", ["learning_rate", "repulsion_power", "radius_clip", "edge_cohesion_weight", "repulsion_weight", "spread_weight"])
@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_invalid_config_rejects(name, value):
    g = Graph()
    g.add_node()
    with pytest.raises(ValueError):
        g.flat_embedding(GraphMatrix.zeros(1, 2), **{name: value})


def test_mismatched_rows_reject_and_empty_singletons_work():
    g = Graph()
    assert run(g, GraphMatrix.zeros(0, 2)).shape == (0, 2)
    with pytest.raises(ValueError, match="rows"):
        run(g, GraphMatrix.zeros(1, 2))
    g.add_node()
    assert run(g, GraphMatrix.zeros(1, 2)).shape == (1, 2)
