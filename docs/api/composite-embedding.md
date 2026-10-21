# Composite embedding

The existing native EmbeddingMethod::Composite/factory now runs all three
strategies without an interleave TODO panic. A shared checked native
`combine_embedding_matrices` also serves a thin Python entry point:

```python
combined = graph.combine_embeddings(
    [first_matrix, second_matrix], strategy="interleave", chunk_size=2,
)
mean = graph.combine_embeddings(
    [first_matrix, second_matrix], strategy="weighted_average", weights=[1.0, 3.0],
)
```

## Columns and dimensions

- `concatenate`: append every column from child 0, then child 1, etc.
- `interleave`: take a chunk from child 0, then child 1, etc.; repeat with the next
  chunks until **all columns** are included. Preserve partial final chunks and
  unequal matrix widths. `chunk_size` must be positive; a chunk larger than every
  width has concatenation order.
- `weighted_average`: combine equal-width matrices by column position using
  normalized weights. Omitted weights mean equal weights.

Example: A has columns a0..a4 and B has b0..b2. Chunk two yields
`a0,a1,b0,b1,a2,a3,b2,a4`. Interleaving does not silently discard columns.

The existing factory `dimensions` remains the requested **per-child width**.
Concatenation/interleaving return the sum of actual child widths; averaging
returns their common width. This preserves the prior concatenation convention.
The direct matrix combiner can accept unequal widths for concatenation/interleave;
the factory still asks each child engine for the same width. Nested composites
can produce larger actual widths and are checked at their next combination.

## Identity and validation

Require a nonempty graph and at least one nonzero-width matrix. Validate complete
row coverage against active nodes. Unlabeled matrices mean ascending active IDs;
canonical `node_<id>` row labels may be permuted. Align every child's rows by ID
and return owned coordinates in ascending active-ID order, with copied identity
labels. Reject inactive/duplicate/malformed labels and wrong row counts.

Columns are positional features, not automatically aligned by names or rotated
into a common spectral/PCA basis. Concatenation/interleaving retain feature-name
provenance as `embedding_<child>.<name>`; absent native source names receive a
column-index fallback. Averaging uses the first matrix's feature names.

Weights must match child count and be finite nonnegative; they affect only
averaging. Averaging requires at least one positive weight. Normalize through the
largest weight so a finite-weight total cannot overflow; use scaled compensated
cell accumulation and enforce the mathematical input-range bound against rounding.
All supplied matrix values must be finite, including zero-weight children. Reject
zero chunk sizes, empty inputs, mismatched averaging widths, invalid weights and
output-size overflow explicitly. Factory controls are checked before running
children. No valid input uses unchecked empty indexing or an interleave panic.

## Evidence and limits

`tests/composite_embedding.rs` compares ragged interleaving to a separately sorted
round/child/column reference across chunk sizes, checks all-strategy row alignment,
factory execution, invalid controls/identity/shapes and extreme finite weighted
means. `tests/test_composite_embedding.py` checks the rebuilt Python path,
concatenation/average references, ownership and stale labeled identities.

Matrix composition costs O(total input/output cells), plus row-ID maps, with
owned output and O(children * nodes) row-map memory. Dense output byte/element
arithmetic is checked; ordinary allocation failure is not a recoverable-memory
guarantee. Child engine costs still apply. No sparse acceleration, gradients,
streaming/incremental updates, safe GIL release, cancellation or production-scale
claim is added. Child random-seed propagation and general preprocessing/pipeline
semantics remain separate audits. No graph properties/history are mutated.
