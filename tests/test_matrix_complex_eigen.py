import numpy as np
import pytest
from groggy import ComplexEigenResult, GraphMatrix


def core(a):
    a = np.asarray(a, dtype=float)
    return GraphMatrix.zeros(*a.shape) if not a.size else GraphMatrix.from_data(a.T.tolist())


def pairs(result):
    assert isinstance(result, ComplexEigenResult)
    values = np.array(result.eigenvalues_real) + 1j*np.array(result.eigenvalues_imag)
    n = len(values)
    vectors = np.array(result.eigenvectors_real.to_numpy()).reshape(n,n) + 1j*np.array(result.eigenvectors_imag.to_numpy()).reshape(n,n)
    return values, vectors


@pytest.mark.parametrize('n', [1,2,3,4,6,8])
@pytest.mark.parametrize('scale', [1e-200,1.,1e200])
def test_random_real_input_matches_independent_eigenpairs(n,scale):
    rng = np.random.default_rng(470+n)
    a = rng.normal(size=(n,n))
    values,vectors = pairs(core(a*scale).complex_eigenvalue_decomposition())
    values = values/scale
    reference = np.linalg.eigvals(a)
    # Compare unordered spectra without relying on conjugate sorting roundoff.
    remaining = list(reference)
    for value in values:
        index = min(range(len(remaining)),key=lambda i: abs(remaining[i]-value))
        assert abs(remaining.pop(index)-value) < 2e-10*max(1.,np.linalg.norm(a))
    assert np.allclose(a@vectors,vectors*values,atol=2e-10,rtol=2e-10)
    assert np.allclose(np.linalg.norm(vectors,axis=0),1.,atol=1e-12)
    assert np.linalg.matrix_rank(vectors) == n


@pytest.mark.parametrize('a', [np.array([[0.,-1.],[1.,0.]]),np.diag([2.,2.,-1.]),np.zeros((3,3)),np.zeros((0,0)),np.array([[1.,2.,3.],[0.,4.,5.],[0.,0.,6.]]),np.roll(np.eye(4),1,axis=0)])
def test_rotation_repeated_triangular_zero_and_cyclic_inputs(a):
    values,vectors = pairs(core(a).complex_eigenvalue_decomposition())
    assert np.allclose(a@vectors,vectors*values,atol=2e-10)
    if len(a):
        assert np.linalg.matrix_rank(vectors) == len(a)
        assert np.allclose(np.linalg.norm(vectors,axis=0),1.)


def test_symmetric_api_preserved_and_invalid_or_defective_input_is_explicit():
    a=core([[0.,-1.],[1.,0.]])
    with pytest.raises(RuntimeError,match='symmetric'):
        a.eigenvalue_decomposition()
    values,vectors=pairs(a.complex_eigenvalue_decomposition())
    assert np.allclose(np.sort(values.imag),[-1.,1.])
    for a in [[[1.,1.],[0.,1.]],[[np.nan]],[[np.inf]],[[1.,2.]]]:
        with pytest.raises(RuntimeError):
            core(a).complex_eigenvalue_decomposition()


def test_mixed_scale_resolved_complex_block_is_not_deflated_against_global_scale():
    a = np.array([[1.,0.,0.],[0.,0.,-1e-300],[0.,1e-300,0.]])
    values,vectors = pairs(core(a).complex_eigenvalue_decomposition())
    small = values[np.abs(values)<0.5]
    assert np.allclose(np.sort(small.imag/1e-300),[-1.,1.],atol=1e-12)
    block = vectors[1:,np.abs(values)<0.5]
    assert np.allclose(np.array([[0.,-1.],[1.,0.]])@block,block*(small/1e-300),atol=1e-12)


def test_diagonalizable_repeated_spectrum_preserves_independent_basis():
    rng=np.random.default_rng(29)
    q,_=np.linalg.qr(rng.normal(size=(5,5)))
    a=q@np.diag([3.,3.,-2.,-2.,1.])@q.T
    values,vectors=pairs(core(a).complex_eigenvalue_decomposition())
    assert np.allclose(a@vectors,vectors*values,atol=1e-10)
    assert np.linalg.matrix_rank(vectors)==5
