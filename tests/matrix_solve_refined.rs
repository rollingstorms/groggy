use groggy::storage::GraphMatrix;
fn matrix(values: Vec<f64>, n: usize, m: usize) -> GraphMatrix<f64> {
    GraphMatrix::from_row_major_data(values, n, m, None).unwrap()
}
#[test]
fn reports_error_accepts_only_improvements_and_preserves_coordinates() {
    for scale in [1e-250, 1., 1e250] {
        let n = 7;
        let mut a = matrix(
            (0..n * n)
                .map(|i| scale / ((i / n + i % n + 1) as f64))
                .collect(),
            n,
            n,
        );
        a.set_column_names((0..n).map(|i| format!("unknown{i}")).collect());
        let mut b = matrix(
            (0..n * 2).map(|i| scale * ((i % 5) as f64 - 2.)).collect(),
            n,
            2,
        );
        b.set_column_names(vec!["one".into(), "two".into()]);
        let initial = a.solve_refined(&b, 0).unwrap();
        let refined = a.solve_refined(&b, 4).unwrap();
        assert!(refined.backward_error.is_finite());
        assert!(refined.backward_error <= initial.backward_error);
        assert!(refined.accepted_refinements <= 4);
        assert_eq!(refined.solution.row_labels(), Some(a.column_names()));
        assert_eq!(refined.solution.column_names(), b.column_names());
    }
}
#[test]
fn checks_domains_empty_shapes_and_dtype_rounding() {
    let a = matrix(vec![1.], 1, 1);
    let report = a.solve_refined(&a, 16).unwrap();
    assert_eq!(report.backward_error, 0.);
    assert_eq!(report.accepted_refinements, 0);
    assert!(a.solve_refined(&a, 17).is_err());
    assert!(matrix(vec![0.], 1, 1).solve_refined(&a, 1).is_err());
    assert!(matrix(vec![f64::NAN], 1, 1).solve_refined(&a, 1).is_err());
    assert!(a.solve_refined(&matrix(vec![], 0, 1), 1).is_err());
    let empty = matrix(vec![], 0, 0)
        .solve_refined(&matrix(vec![], 0, 3), 2)
        .unwrap();
    assert_eq!(empty.solution.shape(), (0, 3));
    assert_eq!(empty.backward_error, 0.);
    let a = GraphMatrix::<f32>::from_row_major_data(vec![3.], 1, 1, None).unwrap();
    let b = GraphMatrix::<f32>::from_row_major_data(vec![1.], 1, 1, None).unwrap();
    let report = a.solve_refined(&b, 2).unwrap();
    assert!(report.backward_error > 0.);
    assert_eq!(report.accepted_refinements, 0);
    let a = GraphMatrix::<i64>::from_row_major_data(vec![1], 1, 1, None).unwrap();
    assert!(a.solve_refined(&a, 0).is_err());
}
