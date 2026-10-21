use groggy::storage::GraphMatrix;
fn matrix(data: Vec<f64>, n: usize, m: usize) -> GraphMatrix<f64> {
    GraphMatrix::from_row_major_data(data, n, m, None).unwrap()
}
#[test]
fn direct_solve_and_inverse_are_scale_independent_with_pivoting_and_multiple_rhs() {
    for scale in [1e-300, 1., 1e300] {
        let mut a = matrix(vec![0., 2. * scale, scale, 3. * scale], 2, 2);
        a.set_column_names(vec!["x".into(), "y".into()]);
        a.set_row_labels(Some(vec!["equation0".into(), "equation1".into()]));
        let b = matrix(vec![4. * scale, 8. * scale, 7. * scale, 14. * scale], 2, 2);
        let x = a.solve(&b).unwrap();
        for row in 0..2 {
            for col in 0..2 {
                assert!(
                    (x.get(row, col).unwrap() - ((row + 1) as f64 * (col + 1) as f64)).abs()
                        < 1e-12
                );
            }
        }
        assert_eq!(x.row_labels(), Some(a.column_names()));
        let inverse = a.inverse().unwrap();
        let identity = a.multiply(&inverse).unwrap();
        for row in 0..2 {
            for col in 0..2 {
                assert!(
                    (identity.get(row, col).unwrap() - if row == col { 1. } else { 0. }).abs()
                        < 1e-14
                );
            }
        }
        assert_eq!(inverse.column_names(), a.row_labels().unwrap());
        assert_eq!(a.get(0, 0), Some(0.));
    }
}
#[test]
fn rejects_singular_nonfinite_integer_and_unrepresentable_results_and_handles_empty() {
    assert!(matrix(vec![1., 2., 2., 4.], 2, 2).inverse().is_err());
    assert!(matrix(vec![0.], 1, 1).inverse().is_err());
    for value in [f64::NAN, f64::INFINITY] {
        assert!(matrix(vec![value], 1, 1).inverse().is_err());
        assert!(matrix(vec![1.], 1, 1)
            .solve(&matrix(vec![value], 1, 1))
            .is_err());
    }
    assert!(matrix(vec![f64::from_bits(1)], 1, 1).inverse().is_err());
    let a = GraphMatrix::<i64>::from_row_major_data(vec![2], 1, 1, None).unwrap();
    assert!(a.inverse().is_err());
    let f = GraphMatrix::<f32>::from_row_major_data(vec![f32::from_bits(1)], 1, 1, None).unwrap();
    assert!(f.inverse().is_err());
    assert_eq!(matrix(vec![], 0, 0).inverse().unwrap().shape(), (0, 0));
    assert_eq!(
        matrix(vec![], 0, 0)
            .solve(&matrix(vec![], 0, 3))
            .unwrap()
            .shape(),
        (0, 3)
    );
    assert!(matrix(vec![1., 2.], 1, 2).inverse().is_err());
    assert!(matrix(vec![1.], 1, 1).solve(&matrix(vec![], 0, 0)).is_err());
}
