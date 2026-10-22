import numpy as np
import pytest
from groggy import GraphMatrix


def core(a):
    a=np.asarray(a,dtype=float)
    return GraphMatrix.zeros(*a.shape) if not a.size else GraphMatrix.from_data(a.T.tolist())


def array(a):
    return np.asarray(a.to_numpy()).reshape(a.shape)


@pytest.mark.parametrize('n',[1,2,5])
def test_cholesky_backward_matches_explicit_pullback_and_independent_loss_derivative(n):
    rng=np.random.default_rng(n+973)
    x=rng.normal(size=(n,n)); a=x@x.T+np.eye(n)*2
    tracked=core(a).requires_grad_(True)
    factor=tracked.cholesky_decomposition()
    assert factor.requires_grad
    factor.backward()
    result=array(tracked.grad)
    expected=array(core(a).cholesky_vjp(core(np.ones((n,n)))))
    assert np.allclose(result,expected,atol=1e-13)
    direction=rng.normal(size=(n,n)); direction=(direction+direction.T)/2
    eps=1e-5
    fd=(np.linalg.cholesky(a+eps*direction).sum()-np.linalg.cholesky(a-eps*direction).sum())/(2*eps)
    assert np.sum(result*direction)==pytest.approx(fd,rel=2e-7,abs=1e-9)
    factor.backward()
    assert np.allclose(array(tracked.grad),result)
    tracked.zero_grad()
    assert tracked.grad is None


def test_stale_binding_rejects_and_forward_cache_survives_later_mutation():
    a=core([[4.]]).requires_grad_(True)
    a.set(0,0,9.)
    with pytest.raises(RuntimeError,match='changed since gradient binding'):
        a.cholesky_decomposition()
    a=a.requires_grad_(False).requires_grad_(True)
    factor=a.cholesky_decomposition()
    a.set(0,0,16.)
    factor.backward()
    assert a.grad.get(0,0)==pytest.approx(1/6)


def test_chained_cholesky_and_empty_tape():
    a=core([[16.]]).requires_grad_(True)
    out=a.cholesky_decomposition().cholesky_decomposition()
    out.backward()
    assert a.grad.get(0,0)==pytest.approx(1/32)
    a=GraphMatrix.zeros(0,0).requires_grad_(True)
    a.cholesky_decomposition().backward()
    assert a.grad.shape==(0,0)
