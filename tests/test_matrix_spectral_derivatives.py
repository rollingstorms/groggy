import numpy as np
import pytest
from groggy import GraphMatrix


def core(a):
    a=np.asarray(a,dtype=float)
    return GraphMatrix.zeros(*a.shape) if not a.size else GraphMatrix.from_data(a.T.tolist())


def array(a):
    return np.asarray(a.to_numpy()).reshape(a.shape)


def eigen(a,base):
    values,vectors=np.linalg.eigh(a)
    values,vectors=values[::-1],vectors[:,::-1]
    signs=np.sign(np.sum(vectors*base,axis=0)); vectors*=signs
    return values,vectors


def svd(a,base):
    u,s,vt=np.linalg.svd(a,full_matrices=False)
    signs=np.sign(np.sum(u*base,axis=0));u*=signs;vt*=signs[:,None]
    return u,s,vt


@pytest.mark.parametrize('n',[1,2,4,6])
def test_symmetric_eigen_jvp_vjp_match_independent_differences(n):
    rng=np.random.default_rng(935+n);a=rng.normal(size=(n,n));a=(a+a.T)/2
    h=rng.normal(size=(n,n));h=(h+h.T)/2
    gs=rng.normal(size=n);gv=rng.normal(size=(n,n))
    _,base=core(a).eigenvalue_decomposition();base=array(base)
    ds,dv=core(a).eigen_jvp(core(h));dv=array(dv)
    gradient=array(core(a).eigen_vjp(gs.tolist(),core(gv)))
    eps=1e-5;sp,vp=eigen(a+eps*h,base);sm,vm=eigen(a-eps*h,base)
    assert np.allclose(ds,(sp-sm)/(2*eps),rtol=1e-7,atol=1e-9)
    assert np.allclose(dv,(vp-vm)/(2*eps),rtol=1e-6,atol=1e-8)
    lhs=np.dot(gs,ds)+np.sum(gv*dv)
    assert lhs==pytest.approx(np.sum(gradient*h),rel=2e-11,abs=2e-11)
    assert lhs==pytest.approx((np.dot(gs,sp-sm)+np.sum(gv*(vp-vm)))/(2*eps),rel=1e-6,abs=1e-8)


@pytest.mark.parametrize('shape',[(1,1),(5,3),(3,5),(4,4)])
def test_thin_svd_jvp_vjp_match_independent_differences(shape):
    rng=np.random.default_rng(452+sum(shape));a=rng.normal(size=shape);h=rng.normal(size=shape);m,n=shape;k=min(shape)
    base,_,_=core(a).svd();base=array(base)[:,:k]
    gu=rng.normal(size=(m,k));gs=rng.normal(size=k);gvt=rng.normal(size=(k,n))
    du,ds,dvt=core(a).svd_jvp(core(h));du,dvt=array(du),array(dvt)
    gradient=array(core(a).svd_vjp(core(gu),gs.tolist(),core(gvt)))
    eps=1e-5;up,sp,vp=svd(a+eps*h,base);um,sm,vm=svd(a-eps*h,base)
    assert np.allclose(du,(up-um)/(2*eps),atol=1e-8,rtol=2e-6)
    assert np.allclose(ds,(sp-sm)/(2*eps),atol=1e-8,rtol=2e-6)
    assert np.allclose(dvt,(vp-vm)/(2*eps),atol=1e-8,rtol=2e-6)
    lhs=np.sum(gu*du)+np.dot(gs,ds)+np.sum(gvt*dvt)
    assert lhs==pytest.approx(np.sum(gradient*h),rel=2e-11,abs=2e-11)
    fd=(np.sum(gu*(up-um))+np.dot(gs,sp-sm)+np.sum(gvt*(vp-vm)))/(2*eps)
    assert lhs==pytest.approx(fd,rel=2e-6,abs=2e-8)


@pytest.mark.parametrize('scale',[1e-250,1e250])
def test_derivative_scales_and_invalid_spectra(scale):
    a=np.diag([3.,1.]);h=np.array([[1.,2.],[2.,-1.]])
    ds,dv=core(a).eigen_jvp(core(h));ss,sv=core(a*scale).eigen_jvp(core(h*scale))
    assert np.allclose(np.array(ss)/scale,ds)
    assert np.allclose(array(sv),array(dv))
    du,ds,dvt=core(a).svd_jvp(core(h));su,ss,svt=core(a*scale).svd_jvp(core(h*scale))
    assert np.allclose(np.array(ss)/scale,ds)
    assert np.allclose(array(su),array(du));assert np.allclose(array(svt),array(dvt))
    for a in [np.eye(2),np.zeros((2,2))]:
        with pytest.raises(RuntimeError): core(a).eigen_jvp(core(np.eye(2)))
        with pytest.raises(RuntimeError): core(a).svd_jvp(core(np.eye(2)))


def test_domains_empty_shapes_and_nonfinite_cotangents():
    for shape in [(0,0),(0,3),(3,0)]:
        a=core(np.zeros(shape));du,ds,dvt=a.svd_jvp(a)
        assert du.shape==(shape[0],0) and dvt.shape==(0,shape[1]) and ds==[]
        assert a.svd_vjp(du,ds,dvt).shape==shape
    a=core(np.diag([3.,1.]))
    with pytest.raises(RuntimeError): a.eigen_jvp(core([[0.,1.],[0.,0.]]))
    with pytest.raises(RuntimeError): a.eigen_vjp([np.nan,1.],core(np.eye(2)))
    with pytest.raises(RuntimeError): a.svd_vjp(core(np.eye(2)),[np.inf,1.],core(np.eye(2)))
