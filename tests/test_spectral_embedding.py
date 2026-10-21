import numpy as np
import pytest
from groggy import Graph


def fixture(mask):
    g = Graph()
    nodes = [g.add_node() for _ in range(3)]
    a = np.zeros((3, 3))
    for bit, (u, v) in enumerate([(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]):
        if not mask & (1 << bit):
            continue
        for _ in range(1 + bit % 2):
            g.add_edge(nodes[u], nodes[v])
            a[u, v] += 1
            if u != v:
                a[v, u] += 1
    return g, nodes, a


@pytest.mark.parametrize("normalized", [False, True])
def test_all_three_node_edge_sets_match_numpy_laplacian_eigenpairs(normalized):
    for mask in range(64):
        g, nodes, adjacency = fixture(mask)
        degree = adjacency.sum(axis=1)
        laplacian = np.diag(degree) - adjacency
        if normalized:
            inverse = np.zeros(3)
            inverse[degree > 0] = 1 / np.sqrt(degree[degree > 0])
            laplacian = inverse[:, None] * laplacian * inverse[None, :]
        values, _ = np.linalg.eigh(laplacian)
        matrix = g.spectral_embedding(3, normalized=normalized, skip_nullspace=False)
        vectors = np.asarray(matrix.to_numpy()).reshape(3, 3)
        assert matrix.row_labels == [f"node_{n}" for n in nodes]
        assert np.allclose(vectors.T @ vectors, np.eye(3), atol=1e-10)
        assert np.allclose(laplacian @ vectors, vectors * values, atol=1e-10), mask
        count = int(np.sum(values > 1e-8))
        if count:
            selected = g.spectral_embedding(count, normalized=normalized, eigenvalue_threshold=1e-300)
            selected = np.asarray(selected.to_numpy()).reshape(3, count)
            assert np.allclose(laplacian @ selected, selected * values[-count:], atol=1e-9)
        else:
            with pytest.raises(ValueError, match="non-constant|single node"):
                g.spectral_embedding(1, normalized=normalized)


def test_sparse_labels_are_copied_and_old_coordinates_are_owned():
    g = Graph()
    nodes = [g.add_node() for _ in range(5)]
    g.remove_node(nodes[1])
    g.remove_node(nodes[4])
    g.add_edge(nodes[0], nodes[2])
    g.add_edge(nodes[2], nodes[3])
    matrix = g.spectral_embedding(2)
    before = np.asarray(matrix.to_numpy()).copy()
    labels = matrix.row_labels
    labels[0] = "changed"
    g.remove_node(nodes[2])
    assert matrix.row_labels == [f"node_{n}" for n in [nodes[0], nodes[2], nodes[3]]]
    assert np.array_equal(matrix.to_numpy(), before)
    # The matrix can flow into the shared PCA caller without losing identity.
    projected = matrix.pca_projection()
    assert projected.row_labels == matrix.row_labels


@pytest.mark.parametrize("threshold", [0, -1, float("nan"), float("inf")])
def test_bad_thresholds_reject(threshold):
    g, _, _ = fixture(2)
    with pytest.raises(ValueError, match="threshold"):
        g.spectral_embedding(1, eigenvalue_threshold=threshold)


def test_domain_dimension_and_explicit_nullspace_options():
    g, _, _ = fixture(2)  # One pair plus an isolate: only one positive mode.
    for dimensions in [0, 2, 3, 4]:
        with pytest.raises(ValueError):
            g.spectral_embedding(dimensions)
    with pytest.raises(ValueError):
        g.spectral_embedding(1, eigenvalue_threshold=10.0)
    directed = Graph(directed=True)
    a, b = directed.add_node(), directed.add_node()
    directed.add_edge(a, b)
    with pytest.raises(ValueError, match="undirected"):
        directed.spectral_embedding(1)
    singleton = Graph()
    singleton.add_node()
    with pytest.raises(ValueError, match="single node"):
        singleton.spectral_embedding(1)
    assert np.array_equal(singleton.spectral_embedding(1, skip_nullspace=False).to_numpy(), [[1.0]])
    with pytest.raises(ValueError, match="empty"):
        Graph().spectral_embedding(1)
