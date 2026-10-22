//! Checked matrix powers and noncommuting Frechet derivatives.
use crate::storage::advanced_matrix::{MatrixError, MatrixResult, NumericType, UnifiedMatrix};
fn owned<T: NumericType>(a: &UnifiedMatrix<T>) -> MatrixResult<UnifiedMatrix<T>> {
    if a.rows() != a.cols() {
        return Err(MatrixError::InvalidShape(
            "Matrix power requires square input".into(),
        ));
    }
    if a.checked_sparse()?.is_some() {
        a.to_csr()
    } else {
        a.to_dense()
    }
}
fn constant<T: NumericType>(
    a: &UnifiedMatrix<T>,
    identity: bool,
) -> MatrixResult<UnifiedMatrix<T>> {
    let n = a.rows();
    if a.checked_sparse()?.is_some() {
        let values = if identity { vec![T::one(); n] } else { vec![] };
        let columns = if identity { (0..n).collect() } else { vec![] };
        let pointers = if identity {
            (0..=n).collect()
        } else {
            vec![0; n + 1]
        };
        UnifiedMatrix::from_csr(n, n, values, columns, pointers)
    } else if identity {
        UnifiedMatrix::identity(n)
    } else {
        UnifiedMatrix::zeros(n, n)
    }
}
fn power<T: NumericType>(a: &UnifiedMatrix<T>, n: u32) -> MatrixResult<UnifiedMatrix<T>> {
    if n == 0 {
        return constant(a, true);
    }
    if n == 1 {
        return Ok(a.clone());
    }
    let half = power(a, n / 2)?;
    let square = half.matmul(&half)?;
    if n.is_multiple_of(2) {
        Ok(square)
    } else {
        square.matmul(a)
    }
}
pub(crate) fn forward<T: NumericType>(
    a: &UnifiedMatrix<T>,
    n: u32,
) -> MatrixResult<UnifiedMatrix<T>> {
    power(&owned(a)?, n)
}
fn derivative<T: NumericType>(
    a: &UnifiedMatrix<T>,
    n: u32,
    direction: &UnifiedMatrix<T>,
    reverse: bool,
) -> MatrixResult<UnifiedMatrix<T>> {
    if n == 0 {
        return constant(a, false);
    }
    if n == 1 {
        return Ok(direction.clone());
    }
    if n.is_multiple_of(2) {
        let half = power(a, n / 2)?;
        if reverse {
            let half_t = half.transpose()?;
            let seed = direction.matmul(&half_t)?.add(&half_t.matmul(direction)?)?;
            derivative(a, n / 2, &seed, true)
        } else {
            let tangent = derivative(a, n / 2, direction, false)?;
            tangent.matmul(&half)?.add(&half.matmul(&tangent)?)
        }
    } else {
        let previous = power(a, n - 1)?;
        if reverse {
            let seed = direction.matmul(&a.transpose()?)?;
            derivative(a, n - 1, &seed, true)?.add(&previous.transpose()?.matmul(direction)?)
        } else {
            derivative(a, n - 1, direction, false)?
                .matmul(a)?
                .add(&previous.matmul(direction)?)
        }
    }
}
pub(crate) fn apply<T: NumericType>(
    a: &UnifiedMatrix<T>,
    n: u32,
    direction: &UnifiedMatrix<T>,
    reverse: bool,
) -> MatrixResult<UnifiedMatrix<T>> {
    if !T::DTYPE.is_float() {
        return Err(MatrixError::UnsupportedOperation(
            "Matrix power derivatives require floating dtype".into(),
        ));
    }
    if direction.shape() != a.shape() {
        return Err(MatrixError::DimensionMismatch {
            expected: a.shape().as_tuple(),
            got: direction.shape().as_tuple(),
        });
    }
    let a = owned(a)?;
    let direction = owned(direction)?;
    derivative(&a, n, &direction, reverse)
}
