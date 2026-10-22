import pytest
from groggy import GraphMatrix


def test_native_csr_construction_mutation_and_dense_consumers():
    a = GraphMatrix.from_csr(2, 3, [2., 5., -1.], [0, 2, 1], [0, 2, 3])
    assert a.is_sparse
    assert a.get(0, 1) == 0.
    b = GraphMatrix.from_data([[1., 2., 3.]])
    product = a * b
    assert product.shape == (2, 1)
    assert product.get(0, 0) == 17.
    assert product.get(1, 0) == -2.
    assert not product.is_sparse
    assert a.transpose().get(2, 0) == 5.
    assert a.reshape(3, 2).get(1, 0) == 5.
    assert a[:, [2, 0]].get(0, 0) == 5.
    with pytest.raises(ValueError):
        GraphMatrix.from_csr(1, 3, [1., 2.], [1, 1], [0, 2])
