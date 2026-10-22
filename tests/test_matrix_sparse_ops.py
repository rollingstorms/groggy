import numpy as np
import pytest
from groggy import GraphMatrix


def core(a):return GraphMatrix.from_data(np.asarray(a,dtype=float).T.tolist())
def array(a):return np.asarray(a.to_numpy()).reshape(a.shape)


@pytest.mark.parametrize('shape',[(4,3),(3,5),(4,4)])
def test_sparse_products_arithmetic_and_owned_selections_match_dense(shape):
    rng=np.random.default_rng(981+sum(shape));a=rng.integers(-3,4,size=shape).astype(float);a[rng.random(shape)<0.7]=0
    b=rng.integers(-3,4,size=shape).astype(float);b[rng.random(shape)<0.7]=0
    sa,sb=core(a).to_csr(),core(b).to_csr()
    for result,expected in [(sa+sb,a+b),(sa-sb,a-b),(sa.multiply(sb.transpose()),a@b.T),(sa.transpose(),a.T)]:
        assert result.is_sparse
        assert np.allclose(array(result),expected)
    assert np.allclose(array(sa.multiply(core(b.T))),a@b.T)
    assert np.allclose(array(core(a).multiply(sb.transpose())),a@b.T)
    assert np.allclose(array(sa[[shape[0]-1,0,shape[0]-1],:]),a[[shape[0]-1,0,shape[0]-1],:])
    assert not sa.to_dense().is_sparse


def test_sparse_ieee_and_cancellation_rules_match_dense_reference():
    a=GraphMatrix.from_csr(1,3,[1e308,1.,-1e308],[0,1,2],[0,3])
    b=GraphMatrix.from_csr(3,1,[1.,1.,1.],[0,0,0],[0,1,2,3])
    assert a.multiply(b).get(0,0)==1.
    a=core([[np.inf,0.]]).to_csr();b=core([[1.,0.],[0.,0.]]).to_csr()
    out=a.multiply(b)
    assert np.isinf(out.get(0,0)) and np.isnan(out.get(0,1))


def test_sparse_scalar_maps_and_reshape_retain_storage_and_values():
    a=np.array([[2.,0.,0.],[0.,0.,3.]])
    sparse=core(a).to_csr()
    for result,expected in [(sparse.multiply(2.),a*2),(sparse+1.,a+1),(sparse.reshape(3,2),a.reshape(3,2))]:
        assert result.is_sparse
        assert np.allclose(array(result),expected)
