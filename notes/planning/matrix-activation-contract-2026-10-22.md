# Matrix activation/dropout closeout — 2026-10-22

Checklist 29.62 closes the NC2 activation/control/tracking findings. This is the
existing matrix helper surface, not a promise of a complete neural framework or
universal automatic differentiation.

## Checked activation kernels

ReLU, tanh-GELU, sigmoid, tanh, swish, leaky ReLU, ELU and their existing
pointwise derivative paths use logical matrix storage and checked conversions.
Nonfinite numeric inputs/results and unsupported nonintegral integer outputs
fail; they cannot become silent zeros. ReLU retains exact integer values. Alpha
must be finite; ELU uses exp_m1 for small negative inputs. In-place activation
replacement occurs only after successful evaluation.

GELU's derivative now differentiates its actual tanh approximation. Finite
saturated extremes produce 0/x and 0/1 slopes without cubic overflow. Existing
native sigmoid/tanh derivative kernels are also used by the tape.

Row softmax reads/writes actual logical rows, including column-major input. Its
old placeholder row/set helpers are removed. Softmax requires floating storage,
uses maximum subtraction, and rejects nonfinite input. Zero-preserving activation
maps keep CSR; nonzero zero-images and dense sparse-softmax output use the common
implicit expansion guard. This includes derivative images such as GELU'(0)=0.5.

## Tracking and metadata

GraphMatrix activation outputs retain column names and row labels. ReLU/GELU/sigmoid/tanh/row-softmax
append native tape operations and their pullbacks compose with existing arithmetic;
ReLU uses slope zero at zero. Row softmax uses a direct Jacobian-vector pullback without building a Jacobian. Tracked leaky ReLU/ELU and actual training dropout
currently return explicit unsupported errors; detach explicitly to evaluate them.
These are capability boundaries, not implemented gradients. Duplicate FFI activation
math is removed: both Python methods and nn functions delegate to the native
matrix API. Untracked calls use the existing owned-data/GIL-release boundary;
tracked calls retain their tape ownership and the GIL.

## Dropout

Inverted training dropout scales retained values by 1/(1-p). Evaluation is an
identity. All modes validate finite p in [0,1]; p=0 is identity and p=1 training
returns zeros without division. The new native dropout_seeded method and optional
Python seed argument allow reproducible masks, preserving existing call forms.
Default calls draw a fresh seed from the existing RNG dependency. Coordinate
mixing makes dense, logical-layout and CSR masks agree for the same seed; sparse
work touches only stored cells and row pointers. Errors are propagated and
metadata is retained. Evaluation/p=0 preserve the input tape.

## Evidence

Native/Python tests cover finite differences of the actual GELU formula, tape
composition/reset, labels, integer domains, nonfinite/overflow errors, atomic
in-place failure, real column-major softmax, sparse expansion, deterministic
storage-equivalent masks and probability endpoints. Existing numerical/array/
table suites and strict core lint are rerun on the rebuilt extension.

Executed: full locked offline native suites; rebuilt FFI; numerical/array/table
Python gate **570 passed, 1 existing skip**; strict all-target core Clippy. The
idle release corpus completed **55 cases**, matching all prior 52 checksums to
1e-13. See `notes/benchmarks/numerical-activations-2026-10-22.md` for sparse
storage/metadata allocation and timing receipts.
