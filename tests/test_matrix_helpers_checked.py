import numpy as np
import pytest
from groggy import GraphMatrix


def core(a):
    return GraphMatrix.from_data(np.asarray(a, dtype=float).T.tolist())


@pytest.mark.parametrize('sparse', [False, True])
def test_concatenate_split_keep_owned_values_labels_and_sparse_storage(sparse):
    a = np.array([[1., 0., 2.], [0., 3., 0.]])
    m = core(a)
    if sparse:
        m = m.to_csr()
    for axis in [0, 1]:
        result = m.concatenate(m, axis)
        assert result.is_sparse == sparse
        assert np.array_equal(np.asarray(result.to_numpy()).reshape(result.shape),
                              np.concatenate([a, a], axis=axis))
        for chunk in result.split([a.shape[axis]], axis):
            assert chunk.is_sparse == sparse
            assert np.array_equal(np.asarray(chunk.to_numpy()).reshape(chunk.shape), a)
    with pytest.raises(Exception):
        m.split([0], 0)
    with pytest.raises(Exception):
        m.concatenate(m, 2)
