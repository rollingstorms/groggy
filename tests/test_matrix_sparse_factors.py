import numpy as np
import pytest
from groggy import GraphMatrix


def core(a):
    a=np.asarray(a,dtype=float)
    return GraphMatrix.zeros(*a.shape) if not a.size else GraphMatrix.from_data(a.T.tolist())
def array(a):return np.asarray(a.to_numpy()).reshape(a.shape)


@pytest.mark.parametrize('n',[1,3,7,12])
@pytest.mark.parametrize('scale',[1e-200,1.,1e200])
def test_sparse_lu_solve_and_cholesky_match_independent_references(n,scale):
    rng=np.random.default_rng(349+n);a=rng.normal(size=(n,n));a[rng.random((n,n))<0.7]=0;a+=np.eye(n)*3
    b=rng.normal(size=(n,2));sparse=core(a*scale).to_csr()
    p,l,u=sparse.lu_decomposition()
    assert p.is_sparse and l.is_sparse and u.is_sparse
    assert np.allclose(array(p)@a,array(l)@(array(u)/scale),atol=1e-11)
    assert np.allclose(array(sparse.solve(core(b*scale))),np.linalg.solve(a,b),atol=1e-11,rtol=1e-11)
    spd=a@a.T+np.eye(n);factor=core(spd*scale).to_csr().cholesky_decomposition()
    assert factor.is_sparse
    assert np.allclose(array(factor)/np.sqrt(scale),np.linalg.cholesky(spd),atol=1e-11,rtol=1e-11)


def test_sparse_factor_invalid_empty_and_saved_gradient_cases():
    for a in [[[0.]],[[1.,2.],[2.,4.]],[[1.,2.],[0.,1.]],[[np.inf]]]:
        with pytest.raises(RuntimeError):core(a).to_csr().cholesky_decomposition()
    empty=GraphMatrix.zeros(0,0).to_csr()
    assert empty.cholesky_decomposition().shape==(0,0)
    assert all(x.is_sparse for x in empty.lu_decomposition())
    a=core([[4.,1.],[1.,3.]]).to_csr().requires_grad_(True)
    factor=a.cholesky_decomposition();factor.backward()
    assert np.allclose(array(a.grad),array(core([[4.,1.],[1.,3.]]).cholesky_vjp(core(np.ones((2,2))))))
