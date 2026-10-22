use groggy::storage::GraphMatrix;
fn check(data: Vec<f64>, rows: usize, cols: usize) {
    let matrix = GraphMatrix::from_row_major_data(data.clone(), rows, cols, None).unwrap();
    let (q, r) = matrix.qr_decomposition().unwrap();
    let k = rows.min(cols);
    assert_eq!(q.shape(), (rows, k));
    assert_eq!(r.shape(), (k, cols));
    let reconstructed = q.multiply(&r).unwrap();
    let scale = data.iter().fold(1e-300_f64, |s, v| s.max(v.abs()));
    for row in 0..rows {
        for col in 0..cols {
            assert!(
                (reconstructed.get(row, col).unwrap() / scale - data[row * cols + col] / scale)
                    .abs()
                    < 2e-13
            );
        }
    }
    let orthogonal = q.transpose().unwrap().multiply(&q).unwrap();
    for row in 0..k {
        for col in 0..k {
            assert!(
                (orthogonal.get(row, col).unwrap() - if row == col { 1. } else { 0. }).abs()
                    < 2e-13
            );
        }
    }
    for row in 0..k {
        for col in 0..row {
            assert_eq!(r.get(row, col), Some(0.));
        }
    }
}
#[test]
fn householder_qr_reconstructs_rectangular_rank_deficient_scaled_and_empty_inputs() {
    for (rows, cols) in [(5, 3), (3, 5), (3, 3), (0, 3), (3, 0), (0, 0)] {
        for scale in [1e-250, 1., 1e250] {
            check(
                (0..rows * cols)
                    .map(|i| ((i * 7 % 13) as f64 - 6.) * scale)
                    .collect(),
                rows,
                cols,
            );
        }
    }
    check(vec![0.; 12], 4, 3);
    check(vec![1., 2., 3., 2., 4., 6., 3., 6., 9., 4., 8., 12.], 4, 3);
}
#[test]
fn qr_rejects_nonfinite_integer_and_unrepresentable_output_domains_and_keeps_identity() {
    for value in [f64::NAN, f64::INFINITY] {
        assert!(GraphMatrix::from_row_major_data(vec![value], 1, 1, None)
            .unwrap()
            .qr_decomposition()
            .is_err());
    }
    assert!(GraphMatrix::<i64>::from_row_major_data(vec![1], 1, 1, None)
        .unwrap()
        .qr_decomposition()
        .is_err());
    assert!(
        GraphMatrix::from_row_major_data(vec![f64::MAX; 2], 2, 1, None)
            .unwrap()
            .qr_decomposition()
            .is_err()
    );
    let mut a = GraphMatrix::from_row_major_data(vec![1., 2., 3., 4.], 2, 2, None).unwrap();
    a.set_row_labels(Some(vec!["r0".into(), "r1".into()]));
    a.set_column_names(vec!["a".into(), "b".into()]);
    let (q, r) = a.qr_decomposition().unwrap();
    assert_eq!(q.row_labels(), a.row_labels());
    assert_eq!(r.column_names(), a.column_names());
    assert_eq!(q.column_names(), r.row_labels().unwrap());
    assert_eq!(a.get(0, 0), Some(1.));
    let (q, r) = GraphMatrix::<f32>::from_row_major_data(vec![1., 2., 3., 4.], 2, 2, None)
        .unwrap()
        .qr_decomposition()
        .unwrap();
    assert!((q.multiply(&r).unwrap().get(1, 1).unwrap() - 4.).abs() < 1e-5);
}
