import numpy as np
import pytest
from groggy import GraphMatrix


def core(a):
    a=np.asarray(a,dtype=float)
    return GraphMatrix.zeros(*a.shape) if not a.size else GraphMatrix.from_data(a.T.tolist())


def array(a):
    return np.asarray(a.to_numpy()).reshape(a.shape)


@pytest.mark.parametrize('n',[1,2,4,7])
def test_jvp_matches_independent_finite_difference_and_vjp_adjoint(n):
    rng=np.random.default_rng(762+n)
    x=rng.normal(size=(n,n)); a=x@x.T+np.eye(n)*2
    h=rng.normal(size=(n,n)); h=(h+h.T)/2
    g=rng.normal(size=(n,n))
    jvp=array(core(a).cholesky_jvp(core(h)))
    vjp=array(core(a).cholesky_vjp(core(g)))
    eps=1e-5
    expected=(np.linalg.cholesky(a+eps*h)-np.linalg.cholesky(a-eps*h))/(2*eps)
    assert np.allclose(jvp,expected,atol=2e-10,rtol=2e-8)
    assert np.allclose(vjp,vjp.T,atol=1e-14)
    assert np.sum(g*jvp)==pytest.approx(np.sum(vjp*h),rel=2e-12,abs=1e-12)
    # Vary each independent symmetric input coefficient; off-diagonal entries vary twice.
    for row in range(n):
        for col in range(row+1):
            basis=np.zeros((n,n)); basis[row,col]=basis[col,row]=1.
            fd=np.sum(g*(np.linalg.cholesky(a+eps*basis)-np.linalg.cholesky(a-eps*basis)))/(2*eps)
            actual=vjp[row,col]*(1 if row==col else 2)
            assert actual==pytest.approx(fd,rel=2e-7,abs=5e-10)


@pytest.mark.parametrize('scale',[1e-300,1e-150,1.,1e150,1e300])
def test_common_scales_preserve_derivatives_without_intermediate_products_overflowing(scale):
    a=np.array([[4.,1.],[1.,3.]])
    h=np.array([[2.,-1.],[-1.,3.]])
    g=np.array([[1.,9.],[-2.,3.]])
    expected_jvp=array(core(a).cholesky_jvp(core(h)))
    expected_vjp=array(core(a).cholesky_vjp(core(g)))
    assert np.allclose(array(core(a*scale).cholesky_jvp(core(h*scale)))/np.sqrt(scale),expected_jvp,atol=1e-13)
    assert np.allclose(array(core(a*scale).cholesky_vjp(core(g)))*np.sqrt(scale),expected_vjp,atol=1e-13)


def test_upper_cotangents_are_ignored_and_extreme_rescaling_keeps_representable_outputs():
    a=core([[4.,0.],[0.,9.]])
    g=np.array([[1e-300,1e308],[0.,1e-300]])
    result=array(a.cholesky_vjp(core(g)))
    assert result[0,0]/1e-300==pytest.approx(0.25)
    assert result[1,1]/1e-300==pytest.approx(1/6)
    assert core([[1e300]]).cholesky_vjp(core([[1e300]])).get(0,0)==pytest.approx(5e149)
    v=core([[1e300]]).cholesky_vjp(core([[1e-160]])).get(0,0)
    assert v/1e-310==pytest.approx(0.5,rel=1e-12)


def test_invalid_domains_and_empty_derivatives_fail_or_return_explicit_shapes():
    a=core(np.eye(2))
    for direction in [[[1.]],[[1.,1.],[0.,1.]],[[np.nan,0.],[0.,1.]]]:
        with pytest.raises(RuntimeError):
            a.cholesky_jvp(core(direction))
    for a,g in [([[0.]],[[1.]]),([[1.]],[[np.inf]]),([[1e-300]],[[1e300]])]:
        with pytest.raises(RuntimeError):
            core(a).cholesky_vjp(core(g))
    empty=GraphMatrix.zeros(0,0)
    assert empty.cholesky_jvp(empty).shape==(0,0)
    assert empty.cholesky_vjp(empty).shape==(0,0)
