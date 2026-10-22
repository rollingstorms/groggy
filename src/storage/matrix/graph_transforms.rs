//! Checked graph-derived matrix transforms; owned results retain coordinates.
use super::*;
impl<T: NumericType> GraphMatrix<T> {
    fn graph_matrix_shape(&self) -> GraphResult<usize> {
        if self.requires_grad_enabled() {
            return Err(GraphError::InvalidInput("Graph matrix transforms produce owned operators; detach gradient tracking explicitly".into()));
        }
        let (rows, cols) = self.shape();
        if rows != cols {
            return Err(GraphError::InvalidInput(
                "Graph matrix transform requires square input".into(),
            ));
        }
        if self.column_names.len() != cols
            || self
                .row_labels
                .as_ref()
                .is_some_and(|labels| labels.len() != rows)
        {
            return Err(GraphError::InvalidInput(
                "Graph matrix labels do not match shape".into(),
            ));
        }
        Ok(rows)
    }
    fn diagonal_values(&self, diagonal: &[T]) -> GraphResult<Self> {
        let n = self.graph_matrix_shape()?;
        if diagonal.len() != n {
            return Err(GraphError::InvalidInput(
                "Diagonal length does not match shape".into(),
            ));
        }
        let mut values = Vec::new();
        let mut columns = Vec::new();
        let mut pointers = vec![0];
        for (row, &value) in diagonal.iter().enumerate() {
            if value != T::zero() {
                values.push(value);
                columns.push(row);
            }
            pointers.push(values.len());
        }
        let sparse = UnifiedMatrix::from_csr(n, n, values, columns, pointers);
        let storage = if self.is_sparse() {
            sparse
        } else {
            sparse.and_then(|matrix| matrix.to_dense())
        };
        self.numerical_result(storage)
    }
    /// Weighted row-degree diagonal. Self-loop weight occurs once in row degree.
    pub fn to_degree_matrix(&self) -> GraphResult<Self> {
        if T::DTYPE == crate::storage::advanced_matrix::DType::Bool {
            return Err(GraphError::InvalidInput(
                "Cast boolean adjacency to a numeric dtype for row degrees".into(),
            ));
        }
        self.graph_matrix_shape()?;
        let stored = self
            .storage
            .reduction_values()
            .map_err(|e| GraphError::InvalidInput(e.to_string()))?;
        if stored.iter().any(|value| !value.is_finite()) {
            return Err(GraphError::InvalidInput(
                "Degree matrix requires finite values".into(),
            ));
        }
        let diagonal = self.sum_axis(Axis::Rows)?;
        if diagonal.iter().any(|value| !value.is_finite()) {
            return Err(GraphError::InvalidInput(
                "Degree exceeds finite dtype range".into(),
            ));
        }
        self.diagonal_values(&diagonal)
    }
    /// Legacy parameterized normalized adjacency, (D^eps A D^eps)^k.
    /// Zero-degree rows use a zero scale; k=0 returns identity.
    pub fn normalized_adjacency(&self, eps: f64, k: u32) -> GraphResult<Self> {
        let n = self.graph_matrix_shape()?;
        if !T::DTYPE.is_float() {
            return Err(GraphError::InvalidInput(
                "Cast adjacency to floating dtype before normalization".into(),
            ));
        }
        if !eps.is_finite() {
            return Err(GraphError::InvalidInput(
                "Degree exponent must be finite".into(),
            ));
        }
        if !self.is_adjacency_matrix() {
            return Err(GraphError::InvalidInput(
                "Normalization requires finite nonnegative adjacency values".into(),
            ));
        }
        let diagonal = self.to_degree_matrix()?;
        let mut powers = Vec::with_capacity(n);
        for row in 0..n {
            let degree = diagonal.get_checked(row, row)?.to_f64();
            let powered = if degree == 0. { 0. } else { degree.powf(eps) };
            powers.push(
                T::from_f64(powered)
                    .filter(|value| value.is_finite())
                    .ok_or_else(|| {
                        GraphError::InvalidInput(
                            "Degree power exceeds finite dtype working range".into(),
                        )
                    })?,
            );
        }
        let diagonal = self.diagonal_values(&powers)?;
        let normalized = diagonal.multiply(self)?.multiply(&diagonal)?;
        normalized.power(k)
    }
    /// True normalized Laplacian, diag(positive row degree)-D^-1/2 A D^-1/2.
    /// Isolate rows/columns remain zero; directed input need not produce symmetry.
    pub fn normalized_laplacian(&self) -> GraphResult<Self> {
        let normalized = self.normalized_adjacency(-0.5, 1)?;
        let degrees = self.sum_axis(Axis::Rows)?;
        let diagonal: Vec<_> = degrees
            .iter()
            .map(|value| {
                if *value > T::zero() {
                    T::one()
                } else {
                    T::zero()
                }
            })
            .collect();
        self.diagonal_values(&diagonal)?.subtract(&normalized)
    }
    /// Row-stochastic transition operator; zero-degree rows remain zero.
    /// Per-row scaling avoids overflowing the degree before division.
    pub fn transition_matrix(&self) -> GraphResult<Self> {
        let n = self.graph_matrix_shape()?;
        if !T::DTYPE.is_float() {
            return Err(GraphError::InvalidInput(
                "Cast adjacency to floating dtype before transition normalization".into(),
            ));
        }
        fn constants<T: NumericType>(row: &[T]) -> GraphResult<(f64, f64)> {
            let mut maximum = 0.0_f64;
            for value in row {
                let value = value.to_f64();
                if !value.is_finite() || value < 0. {
                    return Err(GraphError::InvalidInput(
                        "Transitions require finite nonnegative weights".into(),
                    ));
                }
                maximum = maximum.max(value);
            }
            if maximum == 0. {
                return Ok((0., 0.));
            }
            let mut total = crate::storage::matrix::dot_product::Dot::new();
            for value in row {
                total.add(value.to_f64() / maximum, 1.);
            }
            Ok((maximum, total.total()))
        }
        let convert = |value: T, scale: f64, total: f64| -> GraphResult<T> {
            if scale == 0. {
                return Ok(T::zero());
            }
            T::from_f64((value.to_f64() / scale) / total).ok_or_else(|| {
                GraphError::InvalidInput("Transition output exceeds dtype range".into())
            })
        };
        if let Some(source) = self
            .storage
            .checked_sparse()
            .map_err(|e| GraphError::InvalidInput(e.to_string()))?
        {
            let mut values = Vec::new();
            let mut columns = Vec::new();
            let mut pointers = vec![0];
            for row in 0..n {
                let start = source.row_pointers[row];
                let end = source.row_pointers[row + 1];
                let (scale, total) = constants(&source.values[start..end])?;
                for entry in start..end {
                    let value = convert(source.values[entry], scale, total)?;
                    if value != T::zero() {
                        values.push(value);
                        columns.push(source.col_indices[entry]);
                    }
                }
                pointers.push(values.len());
            }
            return self.numerical_result(UnifiedMatrix::from_csr(n, n, values, columns, pointers));
        }
        let mut values = self.to_vec()?;
        if n > 0 {
            for row in values.chunks_mut(n) {
                let (scale, total) = constants(row)?;
                for value in row {
                    *value = convert(*value, scale, total)?;
                }
            }
        }
        self.numerical_result(UnifiedMatrix::from_data(values, n, n))
    }
}
