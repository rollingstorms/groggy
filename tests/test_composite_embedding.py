import numpy as np
import pytest
from groggy import Graph, GraphMatrix


def graph(n=2):
    g = Graph()
    for _ in range(n):
        g.add_node()
    return g


def core(values):
    return GraphMatrix.from_data(np.asarray(values, dtype=float).T.tolist())


@pytest.mark.parametrize("chunk", [1, 2, 3, 100])
def test_interleave_ragged_columns_matches_round_robin_reference(chunk):
    g = graph()
    arrays = [np.arange(10).reshape(2, 5), 100 + np.arange(6).reshape(2, 3)]
    matrices = [core(a) for a in arrays]
    order = sorted([(child, col) for child, a in enumerate(arrays) for col in range(a.shape[1])],
                   key=lambda pair: (pair[1] // chunk, pair[0], pair[1]))
    expected = np.column_stack([arrays[child][:, col] for child, col in order])
    result = g.combine_embeddings(matrices, chunk_size=chunk)
    assert np.array_equal(result.to_numpy(), expected)
    assert result.row_labels == ["node_0", "node_1"]
    assert all(np.array_equal(matrix.to_numpy(), data) for matrix, data in zip(matrices, arrays))


def test_concatenation_and_average_preserve_existing_values():
    g = graph()
    a, b = core([[1, 2], [3, 4]]), core([[5, 6], [7, 8]])
    assert np.array_equal(g.combine_embeddings([a, b], strategy="concatenate").to_numpy(), [[1, 2, 5, 6], [3, 4, 7, 8]])
    assert np.array_equal(g.combine_embeddings([a, b], strategy="weighted_average", weights=[1, 3]).to_numpy(), [[4, 5], [6, 7]])
    assert np.array_equal(g.combine_embeddings([a, b], strategy="weighted_average").to_numpy(), [[3, 4], [5, 6]])


def test_extreme_finite_average_does_not_overflow_weights_or_values():
    g = graph(1)
    maximum = np.finfo(float).max
    a, b = core([[maximum, -maximum]]), core([[maximum, maximum]])
    out = g.combine_embeddings([a, b], strategy="weighted_average", weights=[maximum, maximum]).to_numpy()
    assert np.array_equal(out, [[maximum, 0]])


@pytest.mark.parametrize("kwargs", [
    {"chunk_size": 0}, {"strategy": "unknown"}, {"weights": []},
    {"weights": [-1]}, {"weights": [float("nan")]}, {"weights": [float("inf")]},
    {"strategy": "weighted_average", "weights": [0]},
])
def test_bad_controls_reject(kwargs):
    with pytest.raises(ValueError):
        graph().combine_embeddings([core([[1, 2], [3, 4]])], **kwargs)


def test_empty_wrong_width_rows_and_nonfinite_inputs_reject():
    g = graph()
    with pytest.raises(ValueError):
        g.combine_embeddings([])
    with pytest.raises(ValueError):
        g.combine_embeddings([core([[1, 2]])])
    with pytest.raises(ValueError):
        g.combine_embeddings([core([[np.nan, 2], [3, 4]])])
    with pytest.raises(ValueError, match="width"):
        g.combine_embeddings([core([[1], [2]]), core([[1, 2], [3, 4]])], strategy="weighted_average")


def test_labeled_stale_identity_rejects_even_when_row_count_matches():
    g = graph(2)
    g.add_edge(0, 1)
    labeled = g.spectral_embedding(1)
    g.remove_node(0)
    g.add_node()
    with pytest.raises(ValueError, match="Inactive"):
        g.combine_embeddings([labeled])
