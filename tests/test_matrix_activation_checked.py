"""Faithful activation math, tracking and storage-independent dropout."""
import math

import numpy as np
import pytest
from groggy import GraphMatrix


def core(values):
    return GraphMatrix.from_data(np.asarray(values, dtype=float).T.tolist())


def array(matrix):
    return np.asarray(matrix.to_numpy()).reshape(matrix.shape)


def gelu(values):
    return 0.5 * values * (1 + np.tanh(math.sqrt(2 / math.pi) * (values + 0.044715 * values**3)))


def test_gelu_gradient_matches_forward_difference_and_composes():
    values = np.array([[-3., -1., 0., 0.5, 2., 6.]])
    source = core(values).requires_grad_(True)
    result = source.gelu() * 3
    assert result.requires_grad
    result.backward()
    h = 1e-5
    reference = 3 * (gelu(values + h) - gelu(values - h)) / (2 * h)
    assert np.allclose(array(source.grad), reference, rtol=1e-7, atol=1e-8)
    source.relu().backward()
    assert np.array_equal(array(source.grad), [[0., 0., 0., 1., 1., 1.]])


def test_gelu_extreme_input_and_stable_elu_near_zero():
    maximum = np.finfo(float).max
    assert np.array_equal(array(core([[maximum, -maximum]]).gelu()), [[maximum, 0.]])
    assert array(core([[-1e-15]]).elu())[0, 0] == pytest.approx(math.expm1(-1e-15), rel=1e-14)


def test_seeded_dropout_matches_dense_csr_and_repeated_calls():
    source = core(np.ones((8, 8)))
    result = array(source.dropout(0.5, True, seed=91))
    assert np.array_equal(result, array(source.to_csr().dropout(0.5, True, seed=91)))
    assert np.array_equal(result, array(source.dropout(0.5, True, seed=91)))
    assert set(result.ravel()) == {0., 2.}
    assert not np.array_equal(result, array(source.dropout(0.5, True, seed=92)))
    assert np.array_equal(array(source.dropout(0.8, False)), np.ones((8, 8)))
    assert np.array_equal(array(source.dropout(0., True)), np.ones((8, 8)))
    assert np.array_equal(array(source.dropout(1., True)), np.zeros((8, 8)))


@pytest.mark.parametrize("training", [False, True])
def test_dropout_validates_probability_in_both_modes(training):
    source = core([[1.]])
    for probability in [math.nan, -0.1, 1.1]:
        with pytest.raises(RuntimeError):
            source.dropout(probability, training)


def test_unsupported_tracking_is_explicit_and_eval_dropout_keeps_tape():
    source = core([[1., 2.]]).requires_grad_(True)
    for operation in (lambda: source.leaky_relu(), lambda: source.elu(),
                      lambda: source.dropout(0.5, True)):
        with pytest.raises(RuntimeError, match="Tracked|tracked"):
            operation()
    output = source.dropout(0.5, False)
    assert output.requires_grad
    output.backward()
    assert np.array_equal(array(source.grad), [[1., 1.]])


def test_activation_controls_and_nonfinite_inputs_do_not_silently_zero():
    for operation in (lambda: core([[math.nan]]).relu(),
                      lambda: core([[-1.]]).elu(math.inf),
                      lambda: core([[-1.]]).leaky_relu(math.nan),
                      lambda: core([[-np.finfo(float).max]]).leaky_relu(2.)):
        with pytest.raises(RuntimeError):
            operation()


@pytest.mark.parametrize("operation", ["sigmoid", "tanh", "softmax"])
def test_other_native_activation_tapes_match_independent_derivatives(operation):
    data = np.array([[-1., 0.5, 2.], [0.2, -0.4, 0.7]])
    weights = np.array([[2., -1., 3.], [1., 4., -2.]])
    source = core(data).requires_grad_(True)
    output = getattr(source, operation)()
    assert output.requires_grad
    output.elementwise_multiply(core(weights)).backward()
    if operation == "sigmoid":
        probabilities = 1 / (1 + np.exp(-data))
        expected = weights * probabilities * (1 - probabilities)
    elif operation == "tanh":
        expected = weights * (1 - np.tanh(data)**2)
    else:
        probabilities = np.exp(data - data.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        expected = probabilities * (weights - (probabilities * weights).sum(axis=1, keepdims=True))
    assert np.allclose(array(source.grad), expected, rtol=1e-13, atol=1e-14)
