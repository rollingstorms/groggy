//! Owned, thread-transferable numerical data without graph references or tapes.
use super::*;
#[derive(Debug)]
enum Storage<T: NumericType> {
    Dense(Vec<T>),
    Csr {
        values: Vec<T>,
        indices: Vec<usize>,
        pointers: Vec<usize>,
    },
}
/// Validated owned numerical snapshot. Transferring it deliberately excludes
/// graph references, caches and differentiation tape ownership.
#[derive(Debug)]
pub struct NumericalMatrixData<T: NumericType> {
    rows: usize,
    cols: usize,
    columns: Vec<String>,
    labels: Option<Vec<String>>,
    storage: Storage<T>,
}
impl<T: NumericType> NumericalMatrixData<T> {
    /// Rebuild an owned, untracked matrix on the receiving thread.
    pub fn into_matrix(self) -> GraphResult<GraphMatrix<T>> {
        let mut matrix = match self.storage {
            Storage::Dense(values) => {
                GraphMatrix::from_row_major_data(values, self.rows, self.cols, None)?
            }
            Storage::Csr {
                values,
                indices,
                pointers,
            } => GraphMatrix::from_csr(self.rows, self.cols, values, indices, pointers)?,
        };
        matrix.column_names = self.columns;
        matrix.row_labels = self.labels;
        Ok(matrix)
    }
}
impl<T: NumericType> GraphMatrix<T> {
    /// Copy checked owned numerical data, retaining CSR and positional labels.
    /// The returned snapshot is Send; the graph/tape-bearing matrix is not.
    pub fn numerical_data(&self) -> GraphResult<NumericalMatrixData<T>> {
        let (rows, cols) = self.shape();
        if self.column_names.len() != cols
            || self
                .row_labels
                .as_ref()
                .is_some_and(|labels| labels.len() != rows)
        {
            return Err(GraphError::InvalidInput(
                "Numerical snapshot labels do not match shape".into(),
            ));
        }
        let storage = if let Some(sparse) = self
            .storage
            .checked_sparse()
            .map_err(|e| GraphError::InvalidInput(e.to_string()))?
        {
            Storage::Csr {
                values: sparse.values.clone(),
                indices: sparse.col_indices.clone(),
                pointers: sparse.row_pointers.clone(),
            }
        } else {
            Storage::Dense(
                self.storage
                    .to_vec()
                    .map_err(|e| GraphError::InvalidInput(e.to_string()))?,
            )
        };
        Ok(NumericalMatrixData {
            rows,
            cols,
            columns: self.column_names.clone(),
            labels: self.row_labels.clone(),
            storage,
        })
    }
}
