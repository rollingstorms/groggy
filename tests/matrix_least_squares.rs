use groggy::storage::GraphMatrix;
fn matrix(values: Vec<f64>, n: usize, m: usize) -> GraphMatrix<f64> {
    GraphMatrix::from_row_major_data(values, n, m, None).unwrap()
}
#[test]
fn rectangular_and_rank_deficient_solutions_are_minimum_norm_and_keep_identity() {
    for (values, rows, cols, rhs, expected) in [
        (
            vec![1., 0., 0., 1., 1., 1.],
            3,
            2,
            vec![1., 2., 2.],
            vec![2. / 3., 5. / 3.],
        ),
        (
            vec![1., 0., 1., 0., 1., 1.],
            2,
            3,
            vec![1., 1.],
            vec![1. / 3., 1. / 3., 2. / 3.],
        ),
        (vec![1., 2., 2., 4.], 2, 2, vec![3., 6.], vec![0.6, 1.2]),
    ] {
        for scale in [1e-250, 1., 1e250] {
            let mut a = matrix(values.iter().map(|v| v * scale).collect(), rows, cols);
            a.set_column_names((0..cols).map(|i| format!("x{i}")).collect());
            let mut b = matrix(rhs.iter().map(|v| v * scale).collect(), rows, 1);
            b.set_column_names(vec!["target".into()]);
            let result = a.least_squares(&b, None).unwrap();
            for (row, expected) in expected.iter().enumerate() {
                assert!((result.get(row, 0).unwrap() - expected).abs() < 1e-12);
            }
            assert_eq!(result.row_labels(), Some(a.column_names()));
            assert_eq!(result.column_names(), b.column_names());
        }
    }
}
#[test]
fn cutoffs_empty_domains_and_float32_output_are_explicit() {
    let a = matrix(vec![1., 0., 0., 1e-20], 2, 2);
    let b = matrix(vec![2., 1e-20], 2, 1);
    assert_eq!(a.least_squares(&b, None).unwrap().get(1, 0), Some(0.));
    assert_eq!(a.least_squares(&b, Some(0.)).unwrap().get(1, 0), Some(1.));
    for cutoff in [-1., f64::INFINITY, f64::NAN] {
        assert!(a.least_squares(&b, Some(cutoff)).is_err());
    }
    for (rows, cols, rhs) in [(0, 3, 2), (3, 0, 2), (3, 3, 0), (0, 0, 0)] {
        let a = matrix(vec![0.; rows * cols], rows, cols);
        let b = matrix(vec![0.; rows * rhs], rows, rhs);
        assert_eq!(a.least_squares(&b, None).unwrap().shape(), (cols, rhs));
    }
    assert!(a.least_squares(&matrix(vec![1.], 1, 1), None).is_err());
    assert!(matrix(vec![f64::INFINITY], 1, 1)
        .least_squares(&matrix(vec![1.], 1, 1), None)
        .is_err());
    assert!(matrix(vec![1.], 1, 1)
        .least_squares(&matrix(vec![f64::NAN], 1, 1), None)
        .is_err());
    let a = GraphMatrix::<f32>::from_row_major_data(vec![3.], 1, 1, None).unwrap();
    let b = GraphMatrix::<f32>::from_row_major_data(vec![1.], 1, 1, None).unwrap();
    assert_eq!(
        a.least_squares(&b, None).unwrap().get(0, 0),
        Some(1f32 / 3.)
    );
    let a = GraphMatrix::<i64>::from_row_major_data(vec![1], 1, 1, None).unwrap();
    assert!(a.least_squares(&a, None).is_err());
}
