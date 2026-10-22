# Checked matrix powers and derivatives

`power(n)` / Python `** n` computes a nonnegative integer **matrix** power.
Exponentiation by squaring bounds forward multiplication count by O(log n),
including u32::MAX. Squaring can round differently from sequential floating
multiplication. Integer operations remain checked and return overflow errors.
Inputs must be square; zero order returns identity and empty shapes remain
empty. CSR inputs retain sparse outputs, including zero-order identity, without
allocating a dense identity. Output positional labels follow the input.

`power_jvp(n, tangent)` and `power_vjp(n, cotangent)` use floating dtypes and
shape-matching square directions. For positive n, the directional derivative is

```
D(A^n)[E] = sum(A^i E A^(n-1-i), i=0..n-1)
```

The pullback transposes the left/right factors. Matrices need not commute;
`n * A^(n-1) * E` is generally incorrect. The implementation differentiates the
squaring recursion, with O(log² n) multiplication work and O(log n) recursion
depth, rather than enumerating n terms. Zero/one exponents give zero/identity
derivative maps. Forward and explicit derivative outputs own their storage.

Tracked `power` outputs route backward through this matrix pullback and retain
the saved forward input. Mutation before producing a tracked output requires
rebinding; mutation afterward leaves its saved derivative unchanged. This fixes
the former elementwise power rule in matrix backpropagation. Untracked Python
powers/explicit derivatives use the numerical owned-data GIL boundary; tracked
calls retain their tape ownership.

Evidence: `tests/matrix_power_checked.rs` checks polynomial sums, adjoints,
saved inputs, integer errors, empty shapes, sparse identity and large exponents.
`tests/test_matrix_power_checked.py` compares independent NumPy matrix powers,
finite differences, adjoints and seeded automatic gradients. General arithmetic
tape composition remains a separate follow-up.
