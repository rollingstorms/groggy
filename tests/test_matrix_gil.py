import sys
import threading
import numpy as np
import pytest
from groggy import GraphMatrix


def core(a):
    return GraphMatrix.from_data(np.asarray(a, dtype=float).T.tolist())


def progress_during(operation):
    ready, begin, progress, done = (threading.Event() for _ in range(4))

    def worker():
        ready.set()
        begin.wait()
        progress.set()
        done.wait()

    thread = threading.Thread(target=worker)
    thread.start()
    ready.wait()
    interval = sys.getswitchinterval()
    try:
        # Prevent ordinary Python bytecode switching between setting begin and
        # entering the native call. The worker can proceed when Rust releases GIL.
        sys.setswitchinterval(10.)
        begin.set()
        output = operation()
        progressed = progress.is_set()
    finally:
        sys.setswitchinterval(interval)
        done.set()
        thread.join(timeout=5)
    assert progressed, 'Python worker did not progress during the numerical call'
    assert not thread.is_alive()
    return output


@pytest.mark.parametrize('operation', ['multiply', 'matmul', 'qr', 'cholesky'])
def test_owned_long_numerical_calls_release_gil_and_preserve_results(operation):
    rng = np.random.default_rng(872)
    a = rng.normal(size=(96, 96))
    if operation == 'cholesky':
        a = a @ a.T + np.eye(96)
    m = core(a)
    if operation in ('multiply', 'matmul'):
        b = core(a.T)
        result = progress_during(lambda: m.multiply(b) if operation == 'multiply' else m @ b)
        expected = a @ a.T
    elif operation == 'qr':
        q, r = progress_during(m.qr_decomposition)
        result = q.multiply(r)
        expected = a
    else:
        l = progress_during(m.cholesky_decomposition)
        result = l.multiply(l.transpose())
        expected = a
    actual = np.asarray(result.to_numpy()).reshape(result.shape)
    assert np.allclose(actual, expected, rtol=1e-11, atol=1e-11)


def test_sparse_transfer_and_tracked_tape_keep_their_contracts():
    n = 10000
    a = GraphMatrix.from_csr(n, n, [4.] * n, list(range(n)), list(range(n + 1)))
    result = progress_during(a.cholesky_decomposition)
    assert result.is_sparse and result.get(n-1, n-1) == 2.
    a = core([[16.]]).requires_grad_(True)
    output = a.cholesky_decomposition().cholesky_decomposition()
    output.backward()
    assert a.grad.get(0, 0) == pytest.approx(1 / 32)
