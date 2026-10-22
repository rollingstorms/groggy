# Composing matrix gradients

Floating GraphMatrix arithmetic now retains its saved-input computation tape:
addition/subtraction, elementwise multiplication, matrix products, transpose,
dense/CSR conversions, scalar arithmetic and abs/exp/log/sqrt. Existing matrix
power and real factor routes compose with these operations. Positional labels
and the numerical operation's existing storage rules are retained.

Tapes use process-unique node IDs. Combining cloned sibling branches or independent
tracked inputs unions shared ancestor nodes instead of overwriting colliding
local IDs. Repeated operands accumulate both contributions. Constants are copied
into the resulting tape; later mutation does not alter saved forward values.

```python
x = matrix.requires_grad_(True)
q, r = x.qr_decomposition()
reconstructed = q.multiply(r)
reconstructed.backward()  # all-ones output cotangent
# x.grad is approximately an all-ones matrix in the supported QR domain.
```

One composite loss can include several factor outputs. Explicit joint VJPs remain
available. Every backward call replaces gradients for nodes in that tape;
separate backward calls do not sum into one loss. Shared gradients should not be
used for concurrent backward calls across overlapping tapes. Tracked paths retain
the GIL and saved-input copies; this is not a zero-allocation computation graph.

A new operation checks that tracked input values still match their binding.
After mutation, explicitly rebind with `requires_grad(false).requires_grad(true)`
(native) or `requires_grad_(False).requires_grad_(True)` (Python). Backward uses
saved forward values. Floating finite inputs/outputs are required; integer/bool
automatic differentiation is rejected. abs at zero, log outside its positive
real domain and sqrt at nonpositive values have undefined/singular derivative
errors. Existing factor spectral/rank/chart restrictions still apply.

Owned scalar/Vec reductions are values rather than differentiable matrix nodes.
`materialize()` and `requires_grad(false)` remain explicit detach operations.
This supported composition set does not imply gradients for every selection,
reshape, activation, solve or arbitrary graph operator.

Evidence: native cloned/independent/repeated-input, sparse, stale-binding and
joint-factor fixtures; Python NumPy finite differences for a composite SVD loss
using U, S and Vt together, plus QR reconstruction and chain-rule checks.

Scalar shifts and division retain faithful forward operations for tape replay.
Division pullbacks divide the cotangent directly instead of forming a possibly
overflowing reciprocal. Topological execution uses ascending ready-node IDs;
missing input dependencies reject before inserting an operation node.
