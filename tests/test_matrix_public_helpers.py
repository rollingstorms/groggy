import numpy as np
import pytest
from groggy import GraphMatrix


def core(a):
    a = np.asarray(a, dtype=float)
    return GraphMatrix.zeros(*a.shape) if not a.size else GraphMatrix.from_data(a.T.tolist())


def array(a):
    return np.asarray(a.to_numpy()).reshape(a.shape)


@pytest.mark.parametrize('sparse', [False, True])
def test_public_math_and_operators_share_native_domains_storage_and_shapes(sparse):
    a = np.array([[-1., 0., 4.], [np.nan, np.inf, -0.]])
    m = core(a)
    if sparse:
        m = m.to_csr()
    with np.errstate(all='ignore'):
        for output, expected in [(m.abs(), np.abs(a)), (abs(m), np.abs(a)),
                                 (m.sqrt(), np.sqrt(a)), (m.log(), np.log(a)),
                                 (m.exp(), np.exp(a)), (m.scalar_multiply(2.), a*2)]:
            assert output.shape == a.shape
            assert output.is_sparse == sparse
            assert np.allclose(array(output), expected, equal_nan=True)
    for output, expected in [(m>0, a>0), (m<0, a<0), (m>=0, a>=0), (m<=0, a<=0)]:
        assert output.shape == a.shape and output.is_sparse == sparse
        assert np.array_equal(array(output), expected)


@pytest.mark.parametrize('shape', [(2, 3), (0, 3), (3, 0)])
def test_tile_repeat_and_math_retain_empty_dimensions_and_reference_order(shape):
    a = np.arange(np.prod(shape)).reshape(shape).astype(float)
    m = core(a)
    for reps in [(0, 2), (2, 3), (1, 0)]:
        output = m.tile(reps)
        assert output.shape == np.tile(a, reps).shape
        assert np.array_equal(array(output), np.tile(a, reps))
    for axis in [0, 1]:
        for repeats in [0, 2]:
            output = m.repeat(repeats, axis)
            assert output.shape == np.repeat(a, repeats, axis=axis).shape
            assert np.array_equal(array(output), np.repeat(a, repeats, axis=axis))
    for output in [m.abs(), abs(m), m.sqrt(), m>0]:
        assert output.shape == shape


def test_matrix_scalar_comparison_preserves_large_integer_rhs_precision():
    a = core([[float(2**63)]])
    assert (a > (2**63-1)).get(0, 0) == 1.
    assert (a <= (2**63-1)).get(0, 0) == 0.
