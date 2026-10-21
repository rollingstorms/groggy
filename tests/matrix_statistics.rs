use groggy::storage::{Axis, GraphMatrix};

fn matrix(values: Vec<f64>, rows: usize, cols: usize) -> GraphMatrix<f64> {
    GraphMatrix::from_row_major_data(values, rows, cols, None).unwrap()
}
fn close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert!((a - b).abs() <= 1e-12 * b.abs().max(1.0), "{a} != {b}");
    }
}
#[test]
fn eye_is_identity_for_zero_single_and_larger_shapes() {
    for n in [0, 1, 4] {
        let eye = GraphMatrix::<f64>::eye(n);
        assert_eq!(eye.shape(), (n, n));
        for row in 0..n {
            for col in 0..n {
                assert_eq!(eye.get(row, col), Some(if row == col { 1.0 } else { 0.0 }));
            }
        }
        let a = matrix((0..n * n).map(|v| v as f64).collect(), n, n);
        if n > 0 {
            let product = a.multiply(&eye).unwrap();
            for row in 0..n {
                for col in 0..n {
                    assert_eq!(product.get(row, col), a.get(row, col));
                }
            }
        }
    }
}
#[test]
fn sample_statistics_reduce_the_declared_axis_instead_of_returning_sums() {
    let a = matrix(vec![1., 2., 3., 4., 6., 8.], 2, 3);
    close(&a.var_axis(Axis::Rows).unwrap(), &[1., 4.]);
    close(&a.std_axis(Axis::Rows).unwrap(), &[1., 2.]);
    close(&a.var_axis(Axis::Columns).unwrap(), &[4.5, 8., 12.5]);
    close(
        &a.std_axis(Axis::Columns).unwrap(),
        &[4.5f64.sqrt(), 8f64.sqrt(), 12.5f64.sqrt()],
    );
}
#[test]
fn statistics_handle_large_offsets_f32_nonfinite_and_degenerate_axes() {
    let a = matrix(vec![1e12, 1e12 + 1., 1e12 + 2.], 1, 3);
    close(&a.var_axis(Axis::Rows).unwrap(), &[1.]);
    close(&a.std_axis(Axis::Rows).unwrap(), &[1.]);
    close(&a.std_axis(Axis::Columns).unwrap(), &[0., 0., 0.]);
    let single_precision =
        GraphMatrix::<f32>::from_row_major_data(vec![1., 2., 3., f32::NAN, 0., 1.], 2, 3, None)
            .unwrap();
    let result = single_precision.std_axis(Axis::Rows).unwrap();
    assert_eq!(result[0], 1.);
    assert!(result[1].is_nan());
    let infinite = matrix(vec![1., f64::INFINITY], 1, 2);
    assert!(infinite.var_axis(Axis::Rows).unwrap()[0].is_nan());
    let zero = matrix(vec![7., 7., 7.], 1, 3);
    assert_eq!(zero.std_axis(Axis::Rows).unwrap(), vec![0.]);
    let empty = matrix(vec![], 3, 0);
    assert_eq!(empty.var_axis(Axis::Rows).unwrap(), vec![0.; 3]);
    assert!(empty.std_axis(Axis::Columns).unwrap().is_empty());
}

#[test]
fn checked_constructors_reject_overflow_and_mismatched_buffer_lengths() {
    use groggy::storage::advanced_matrix::{MatrixLayout, SharedBuffer};
    assert!(GraphMatrix::<f64>::from_row_major_data(vec![], usize::MAX, 2, None).is_err());
    assert!(SharedBuffer::<f64>::new((usize::MAX, 2), MatrixLayout::RowMajor).is_err());
    assert!(SharedBuffer::<f64>::from_data(vec![1.], (0, 0), MatrixLayout::RowMajor).is_err());
    let empty = SharedBuffer::<f64>::new((0, 3), MatrixLayout::RowMajor).unwrap();
    assert!(empty.is_empty());
    assert!(empty.to_vec().unwrap().is_empty());
    assert_eq!(empty.clone().shape(), (0, 3));
}

#[test]
fn axis_statistics_follow_column_major_storage_strides() {
    use groggy::storage::advanced_matrix::{MatrixLayout, UnifiedMatrix};
    let mut storage =
        UnifiedMatrix::<f64>::new_with_layout(2, 3, MatrixLayout::ColumnMajor).unwrap();
    for (row, values) in [[1., 2., 3.], [4., 6., 8.]].iter().enumerate() {
        for (col, value) in values.iter().enumerate() {
            storage.set(row, col, *value).unwrap();
        }
    }
    let a = GraphMatrix::from_storage(storage);
    close(&a.std_axis(Axis::Rows).unwrap(), &[1., 2.]);
    close(&a.var_axis(Axis::Columns).unwrap(), &[4.5, 8., 12.5]);
}
