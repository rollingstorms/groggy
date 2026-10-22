use groggy::storage::GraphMatrix;
fn matrix(values: Vec<f64>) -> GraphMatrix<f64> {
    GraphMatrix::from_row_major_data(values, 2, 2, None).unwrap()
}
fn values(a: &GraphMatrix<f64>) -> Vec<f64> {
    (0..2)
        .flat_map(|r| (0..2).map(move |c| a.get(r, c).unwrap()))
        .collect()
}
fn inner(a: &GraphMatrix<f64>, b: &GraphMatrix<f64>) -> f64 {
    values(a)
        .into_iter()
        .zip(values(b))
        .map(|(a, b)| a * b)
        .sum()
}
#[test]
fn powers_and_noncommuting_derivatives_match_independent_polynomial_and_adjoint() {
    let a = matrix(vec![2., 1., 0., 3.]);
    let d = matrix(vec![0., 0.5, -0.25, 0.]);
    let g = matrix(vec![0.2, -0.3, 0.7, 0.4]);
    for n in [0, 1, 2, 3, 4, 7, 12] {
        let mut reference = GraphMatrix::identity(2).unwrap();
        for _ in 0..n {
            reference = reference.multiply(&a).unwrap();
        }
        assert_eq!(values(&a.power(n).unwrap()), values(&reference));
        let jvp = a.power_jvp(n, &d).unwrap();
        let vjp = a.power_vjp(n, &g).unwrap();
        let mut expected = GraphMatrix::<f64>::zeros(2, 2);
        for i in 0..n {
            expected = expected
                .add(
                    &a.power(i)
                        .unwrap()
                        .multiply(&d)
                        .unwrap()
                        .multiply(&a.power(n - 1 - i).unwrap())
                        .unwrap(),
                )
                .unwrap();
        }
        for (a, b) in values(&jvp).into_iter().zip(values(&expected)) {
            assert!((a - b).abs() < 1e-12 * b.abs().max(1.));
        }
        assert!((inner(&jvp, &g) - inner(&d, &vjp)).abs() < 1e-10 * inner(&jvp, &g).abs().max(1.));
    }
}
#[test]
fn automatic_power_uses_saved_input_and_correct_matrix_pullback() {
    let a = matrix(vec![2., 1., 0., 3.]);
    let mut tracked = a.clone().requires_grad(true);
    let mut out = tracked.power(3).unwrap();
    let g = matrix(vec![0.2, -0.3, 0.7, 0.4]);
    tracked.set(0, 0, 99.).unwrap();
    out.backward_with_gradient(&g).unwrap();
    assert_eq!(
        values(&tracked.grad().unwrap()),
        values(&a.power_vjp(3, &g).unwrap())
    );
    assert!(tracked.power(2).is_err());
    let tape_a = matrix(vec![2., 1., 0., 3.]).requires_grad(true);
    tape_a.power(0).unwrap().backward().unwrap();
    assert_eq!(values(&tape_a.grad().unwrap()), vec![0.; 4]);
}
#[test]
fn sparse_identity_large_exponents_checked_integers_and_empty_shapes_are_supported() {
    let n = 10000;
    let a = GraphMatrix::from_csr(n, n, vec![1.; n], (0..n).collect(), (0..=n).collect()).unwrap();
    for exponent in [0, 3, u32::MAX] {
        let result = a.power(exponent).unwrap();
        assert!(result.is_sparse());
        assert_eq!(result.get(n - 1, n - 1), Some(1.));
    }
    let a = GraphMatrix::from_row_major_data(vec![i64::MAX], 1, 1, None).unwrap();
    assert!(a.power(2).is_err());
    assert!(a.power_jvp(1, &a).is_err());
    assert!(GraphMatrix::<f64>::ones(2, 3).power(2).is_err());
    let a = GraphMatrix::<f64>::zeros(0, 0).requires_grad(true);
    a.power(0).unwrap().backward().unwrap();
    assert_eq!(a.grad().unwrap().shape(), (0, 0));
}
