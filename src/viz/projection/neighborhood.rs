//! Shared checked matrix geometry for native neighborhood projections.
use crate::viz::streaming::data_source::Position;
use crate::{GraphError, GraphMatrix, GraphResult};

pub(super) fn invalid(message: &str) -> GraphError {
    GraphError::InvalidInput(message.into())
}

/// Scale all features together before squaring, retaining Euclidean neighbor order.
pub(super) fn scaled_features(input: &GraphMatrix) -> GraphResult<Vec<f64>> {
    let (rows, cols) = input.shape();
    if rows == 0 || cols < 2 {
        return Err(invalid(
            "Neighborhood projection requires nonempty input with at least two features",
        ));
    }
    if input
        .row_labels()
        .is_some_and(|labels| labels.len() != rows)
    {
        return Err(invalid("Projection row-label count must match input rows"));
    }
    let mut scale: f64 = 0.0;
    let mut data = Vec::new();
    let count = rows
        .checked_mul(cols)
        .ok_or_else(|| invalid("Projection feature count overflow"))?;
    data.try_reserve_exact(count)
        .map_err(|_| invalid("Cannot allocate projection features"))?;
    for row in 0..rows {
        for col in 0..cols {
            let value = input.get_checked(row, col)?;
            if !value.is_finite() {
                return Err(invalid("Projection input must be finite"));
            }
            scale = scale.max(value.abs());
            data.push(value);
        }
    }
    if scale != 0.0 {
        for value in &mut data {
            *value /= scale;
        }
    }
    Ok(data)
}

pub(super) fn squared_distance(
    features: &[f64],
    cols: usize,
    i: usize,
    j: usize,
) -> GraphResult<f64> {
    let mut squared = 0.0;
    for col in 0..cols {
        let difference = features[i * cols + col] - features[j * cols + col];
        squared += difference * difference;
    }
    if !squared.is_finite() {
        return Err(invalid("Projection distance overflow"));
    }
    Ok(squared)
}

pub(super) fn squared_distances(input: &GraphMatrix) -> GraphResult<Vec<f64>> {
    let features = scaled_features(input)?;
    let (rows, cols) = input.shape();
    let count = rows
        .checked_mul(rows)
        .and_then(|n| n.checked_mul(std::mem::size_of::<f64>()))
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or_else(|| invalid("Neighborhood pair matrix size overflow"))?
        / std::mem::size_of::<f64>();
    let mut result = vec![0.0; count];
    for i in 0..rows {
        for j in i + 1..rows {
            let squared = squared_distance(&features, cols, i, j)?;
            result[i * rows + j] = squared;
            result[j * rows + i] = squared;
        }
    }
    Ok(result)
}

pub(super) fn coordinates(
    input: &GraphMatrix,
    data: Vec<f64>,
    prefix: &str,
) -> GraphResult<GraphMatrix> {
    let mut result = GraphMatrix::from_row_major_data(data, input.shape().0, 2, None)?;
    result.set_row_labels(input.row_labels().map(<[String]>::to_vec));
    result.set_column_names(vec![format!("{prefix}_x"), format!("{prefix}_y")]);
    Ok(result)
}
pub(super) fn positions(matrix: GraphMatrix) -> GraphResult<Vec<Position>> {
    (0..matrix.shape().0)
        .map(|row| {
            Ok(Position {
                x: matrix.get_checked(row, 0)?,
                y: matrix.get_checked(row, 1)?,
            })
        })
        .collect()
}
