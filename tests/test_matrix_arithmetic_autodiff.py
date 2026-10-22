import numpy as np
import pytest
from groggy import GraphMatrix


def core(a):
    return GraphMatrix.from_data(np.asarray(a, dtype=float).T.tolist())


def array(a):
    return np.asarray(a.to_numpy()).reshape(a.shape)


def finite_difference(fn, a):
    result = np.zeros_like(a)
    for position in np.ndindex(a.shape):
        plus, minus = a.copy(), a.copy()
        plus[position] += 1e-5
        minus[position] -= 1e-5
        result[position] = (fn(plus) - fn(minus)) / 2e-5
    return result


@pytest.mark.parametrize('sparse', [False, True])
def test_arithmetic_composes_cloned_and_independent_tapes(sparse):
    a = core([[2., 3.], [4., 5.]])
    if sparse:
        a = a.to_csr()
    a = a.requires_grad_(True)
    b = core([[5., 7.], [8., 9.]]).requires_grad_(True)
    output = (a * 2).elementwise_multiply(a * 3) + b
    assert output.requires_grad
    output.backward()
    assert np.array_equal(array(a.grad), 12 * array(a))
    assert np.array_equal(array(b.grad), np.ones((2, 2)))
    output.backward()
    assert np.array_equal(array(a.grad), 12 * array(a))


def test_matrix_products_transpose_and_storage_conversion_preserve_the_tape():
    a = core([[1., 2.], [3., 4.]]).requires_grad_(True)
    output = a.transpose().multiply(a).to_csr().to_dense()
    output.backward()
    assert np.array_equal(array(a.grad), [[6., 6.], [14., 14.]])


def test_composed_math_chain_and_independent_reference():
    values = np.array([[2., 3.], [4., 5.]])
    a = core(values).requires_grad_(True)
    output = ((a + 1).log().exp().sqrt() / 2).abs()
    output.backward()
    assert np.allclose(array(a.grad), 0.25 / np.sqrt(values + 1))
    for method in ['abs', 'sqrt']:
        a = core([[0.]]).requires_grad_(True)
        with pytest.raises(RuntimeError, match='undefined|singular'):
            getattr(a, method)().backward()


def test_joint_factor_outputs_work_in_one_loss_against_numpy_finite_differences():
    values = np.array([[4., 1.], [2., 3.], [.5, 2.]])
    a = core(values).requires_grad_(True)
    u, s, vt = a.svd_tracked()
    output = u.multiply(vt) + core(np.ones((3, 1))).multiply(s.transpose())
    output.backward()

    def reference(x):
        u, s, vt = np.linalg.svd(x, full_matrices=False)
        return (u @ vt + np.ones((3, 1)) @ s[None, :]).sum()

    assert np.allclose(array(a.grad), finite_difference(reference, values), atol=1e-7)
    q, r = a.qr_decomposition()
    q.multiply(r).backward()
    assert np.allclose(array(a.grad), np.ones_like(values), atol=1e-10)


def test_saved_bindings_reject_changes_before_a_new_operation():
    a = core([[2.]]).requires_grad_(True)
    a.set(0, 0, 3.)
    with pytest.raises(RuntimeError, match='changed since gradient binding'):
        _ = a + a


def test_scalar_division_backward_does_not_overflow_an_unneeded_reciprocal():
    tiny = float.fromhex('0x0.0000000000001p-1022')
    a = core([[tiny]]).requires_grad_(True)
    output = a / tiny
    output.backward_with_gradient(core([[tiny]]))
    assert array(a.grad)[0, 0] == 1.
