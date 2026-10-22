# Owned positional matrix selection and shape transforms

Item 29.19 shares checked native selection among select_columns, select_indices and
2D matrix slicing. Validate every supplied index, even if the other axis is empty;
invalid columns no longer silently become zeros. Preserve requested order and repeats,
row labels and selected column names. Results own their values; writes do not change
the source. Malformed source row-label counts reject explicitly.

Empty selections retain their declared 0xN or Nx0 shapes. Python integer lists,
negative slices and bool masks resolve against actual dimensions in the Rust core.
The bridge now marshals raw slice start/stop/step instead of prematurely normalizing
against a synthetic maximum length; nonempty all-bool Python lists become bool masks.
Empty lists remain integer index lists. Native usize row/column wrappers reject values
outside signed index range instead of wrapping them into negative indices.

Reshape copies values in logical row-major iteration order, independent of physical
row/column-major storage. Element counts and allocation sizes are checked. A same-shape
reshape retains labels; a changed shape resets coordinate labels to defaults.
Transpose checks allocation/cells and maps original columns to result row labels,
with original row labels becoming result column names where present. Neither operation
inherits graph bindings, gradient tapes or cached properties.

These are owned results, not live views or graph mutations. Low-level UnifiedMatrix
reshape/view operations, lazy storage, arbitrary malformed public backend structs and
all remaining matrix indexing APIs are not certified by this slice. No throughput
claim is made.

Evidence: tests/matrix_indexing.rs tests physical column-major inputs, repeated/permuted
selection, source isolation, coordinate identity, empty shapes, range/overflow errors,
logical reshape and transpose. tests/test_matrix_indexing.py exercises rebuilt Python
empty/reverse/bool indexing and shape transforms.
