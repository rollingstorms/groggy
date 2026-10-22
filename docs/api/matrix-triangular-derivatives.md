# QR and LU derivatives

`qr_jvp(H)` returns derivatives `(dQ, dR)` of the economy factors; `qr_vjp(GQ,
GR)` returns their full-matrix Frobenius input gradient. Shapes follow the
forward economy API, including wide inputs. A wide derivative requires its
leading square coefficient block to be resolved; a full-row-rank matrix with
a singular leading block still has a valid forward factorization but rejects
this derivative chart.

`lu_jvp(H)` returns `(dL, dU)` for the existing partial-pivot convention PA=LU.
P is locally fixed and its derivative is zero. `lu_vjp(GL, GU)` pulls the two
factor cotangents back to the original input coordinates. Strictly lower L and
upper U entries vary; fixed unit diagonal/upper L and lower U cotangents have
no effect. QR lower-triangular R entries are likewise fixed zero.

Finite real floating inputs and matching tangent/cotangent shapes are required.
QR rejects unresolved Householder sign decisions and a singular/numerically
unresolved leading R. LU rejects unresolved zero pivots and pivot-selection
ties. Resolution thresholds use 64*epsilon(f64)*dimension and relative factor
scales. These are explicit smooth-domain policies; this API does not invent
a derivative across discrete pivot/sign/rank changes. Upper/fixed cotangents
must still be finite. Empty axes/native f32 are supported.

Native kernels differentiate the factor reconstruction using triangular solves,
accurate dot products, analytic adjoints and independently normalized/exponent
rescaled directions. There is no inverse or finite-difference implementation.
Working/output range, normalization, domain or convergence failures return
errors; ordinary rounding/underflow and f32 narrowing remain. Explicit calls
return owned results without inherited tape, and input gradients retain source
positional labels. Automatic factor-output routes remain in the completion pass.

Dense O(m²+n²+mn) scratch and repeated forward factorization remain explicit
costs; production performance is a separate gate. Tests check dA=dQ*R+Q*dR,
P*dA=dL*U+L*dU, orthogonality tangents, independent finite differences and
JVP/VJP adjoint identities across shape/scale/domain fixtures.

Mathematical context: [Auto-Differentiating Linear Algebra](https://arxiv.org/abs/1710.08717).
