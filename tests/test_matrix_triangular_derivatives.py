import numpy as np
import pytest
from groggy import GraphMatrix


def core(a):
    a=np.asarray(a,dtype=float)
    return GraphMatrix.zeros(*a.shape) if not a.size else GraphMatrix.from_data(a.T.tolist())


def array(a):return np.asarray(a.to_numpy()).reshape(a.shape)


def qr(a,base):
    q,r=np.linalg.qr(a,mode='reduced');signs=np.sign(np.sum(q*base,axis=0));return q*signs,r*signs[:,None]


def lu(a):
    # Independent test reference with partial pivoting.
    n=len(a);u=a.copy();l=np.eye(n);p=np.eye(n)
    for i in range(n):
        pivot=i+np.argmax(np.abs(u[i:,i]))
        u[[i,pivot]]=u[[pivot,i]];p[[i,pivot]]=p[[pivot,i]]
        l[[i,pivot],:i]=l[[pivot,i],:i]
        for j in range(i+1,n):
            l[j,i]=u[j,i]/u[i,i];u[j,i:]-=l[j,i]*u[i,i:]
    return p,l,u


@pytest.mark.parametrize('shape',[(1,1),(5,3),(3,5),(4,4)])
def test_qr_derivatives_reconstruct_and_match_independent_finite_differences(shape):
    rng=np.random.default_rng(962+sum(shape));a=rng.normal(size=shape);h=rng.normal(size=shape);m,n=shape;k=min(shape)
    q,r=core(a).qr_decomposition();q,r=array(q),array(r)
    gq=rng.normal(size=(m,k));gr=rng.normal(size=(k,n))
    dq,dr=core(a).qr_jvp(core(h));dq,dr=array(dq),array(dr)
    grad=array(core(a).qr_vjp(core(gq),core(gr)))
    assert np.allclose(dq@r+q@dr,h,atol=2e-11)
    assert np.allclose(q.T@dq+dq.T@q,0,atol=2e-11)
    eps=1e-5;qp,rp=qr(a+eps*h,q);qm,rm=qr(a-eps*h,q)
    assert np.allclose(dq,(qp-qm)/(2*eps),atol=2e-8,rtol=2e-6)
    assert np.allclose(dr,(rp-rm)/(2*eps),atol=2e-8,rtol=2e-6)
    lhs=np.sum(gq*dq)+np.sum(gr*dr)
    assert lhs==pytest.approx(np.sum(grad*h),rel=2e-11,abs=2e-11)


@pytest.mark.parametrize('n',[1,2,4,7])
def test_fixed_pivot_lu_derivatives_reconstruct_and_match_independent_finite_differences(n):
    rng=np.random.default_rng(349+n);a=rng.normal(size=(n,n));h=rng.normal(size=(n,n))
    p,l,u=lu(a);gl=rng.normal(size=(n,n));gu=rng.normal(size=(n,n))
    dl,du=core(a).lu_jvp(core(h));dl,du=array(dl),array(du)
    grad=array(core(a).lu_vjp(core(gl),core(gu)))
    assert np.allclose(p@h,dl@u+l@du,atol=1e-11)
    eps=1e-5;pp,lp,up=lu(a+eps*h);pm,lm,um=lu(a-eps*h)
    assert np.array_equal(pp,pm)
    assert np.allclose(dl,(lp-lm)/(2*eps),atol=1e-8,rtol=2e-6)
    assert np.allclose(du,(up-um)/(2*eps),atol=1e-8,rtol=2e-6)
    lhs=np.sum(gl*dl)+np.sum(gu*du)
    assert lhs==pytest.approx(np.sum(grad*h),rel=2e-11,abs=2e-11)


@pytest.mark.parametrize('scale',[1e-250,1e250])
def test_factor_derivative_scales(scale):
    a=np.array([[4.,1.],[2.,3.]]);h=np.array([[2.,3.],[-1.,2.]])
    for method in ['qr_jvp','lu_jvp']:
        x,y=getattr(core(a),method)(core(h));xs,ys=getattr(core(a*scale),method)(core(h*scale))
        assert np.allclose(array(xs),array(x),atol=1e-12)
        assert np.allclose(array(ys)/scale,array(y),atol=1e-12)


def test_factor_derivative_domains_and_empty_shapes():
    for a,method in [(np.zeros((2,2)),'qr_jvp'),(np.zeros((2,2)),'lu_jvp'),(np.array([[1.,0.],[1.,1.]]),'lu_jvp'),(np.array([[0.,1.],[1.,0.]]),'qr_jvp')]:
        with pytest.raises(RuntimeError):getattr(core(a),method)(core(np.eye(2)))
    for shape in [(0,0),(0,3),(3,0)]:
        a=core(np.zeros(shape));q,r=a.qr_jvp(a);assert a.qr_vjp(q,r).shape==shape
