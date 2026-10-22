//! Limits on implicit dense logical work, not a total-memory guarantee.
use super::{MatrixError, MatrixResult};
/// Maximum logical cells expanded/materialized implicitly from CSR storage.
/// Explicit dense conversion opts into larger dense numerical work.
pub const IMPLICIT_DENSE_ELEMENT_LIMIT: usize = 1_048_576;
pub(crate) fn implicit_dense(rows: usize, cols: usize, operation: &str) -> MatrixResult<()> {
    let cells = rows
        .checked_mul(cols)
        .ok_or_else(|| MatrixError::InvalidInput(format!("{operation}: matrix shape overflow")))?;
    if cells > IMPLICIT_DENSE_ELEMENT_LIMIT {
        return Err(MatrixError::InvalidInput(format!("{operation} would expand or scan {rows}x{cols} sparse logical cells beyond the implicit {IMPLICIT_DENSE_ELEMENT_LIMIT}-cell limit; call to_dense() explicitly to opt in")));
    }
    Ok(())
}
