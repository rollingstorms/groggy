use groggy::storage::GraphMatrix;
fn matrix(values: Vec<f64>, rows: usize, cols: usize) -> GraphMatrix<f64> {
    GraphMatrix::from_row_major_data(values, rows, cols, None).unwrap()
}
fn seed(a: &GraphMatrix<f64>) -> GraphMatrix<f64> {
    let (m, n) = a.shape();
    matrix((0..m * n).map(|i| (i as f64 + 1.) / 7.).collect(), m, n)
}
fn compare(a: &GraphMatrix<f64>, b: &GraphMatrix<f64>) {
    assert_eq!(a.shape(), b.shape());
    for r in 0..a.shape().0 {
        for c in 0..a.shape().1 {
            assert!((a.get(r, c).unwrap() - b.get(r, c).unwrap()).abs() < 1e-12);
        }
    }
}
#[test]
fn qr_lu_and_cholesky_outputs_pull_back_explicit_seeds() {
    let a = matrix(vec![4., 1., 2., 3.], 2, 2);
    let tracked = a.clone().requires_grad(true);
    let (mut q, mut r) = tracked.qr_decomposition().unwrap();
    let gq = seed(&q);
    let gr = seed(&r);
    q.backward_with_gradient(&gq).unwrap();
    compare(
        &tracked.grad().unwrap(),
        &a.qr_vjp(&gq, &GraphMatrix::zeros(2, 2)).unwrap(),
    );
    r.backward_with_gradient(&gr).unwrap();
    compare(
        &tracked.grad().unwrap(),
        &a.qr_vjp(&GraphMatrix::zeros(2, 2), &gr).unwrap(),
    );
    let (p, mut l, mut u) = tracked.lu_decomposition().unwrap();
    assert!(!p.requires_grad_enabled());
    let gl = seed(&l);
    let gu = seed(&u);
    l.backward_with_gradient(&gl).unwrap();
    compare(
        &tracked.grad().unwrap(),
        &a.lu_vjp(&gl, &GraphMatrix::zeros(2, 2)).unwrap(),
    );
    u.backward_with_gradient(&gu).unwrap();
    compare(
        &tracked.grad().unwrap(),
        &a.lu_vjp(&GraphMatrix::zeros(2, 2), &gu).unwrap(),
    );
    assert!(u.backward_with_gradient(&GraphMatrix::zeros(1, 1)).is_err());
    assert!(u
        .backward_with_gradient(&matrix(vec![f64::NAN; 4], 2, 2))
        .is_err());
}
#[test]
fn spectral_outputs_and_saved_input_snapshots_retain_supported_gradients() {
    let a = matrix(vec![4., 1., 1., 2.], 2, 2);
    let mut tracked = a.clone().requires_grad(true);
    let (mut values, mut vectors) = tracked.eigen_tracked().unwrap();
    let gv = seed(&values);
    let gq = seed(&vectors);
    values.backward_with_gradient(&gv).unwrap();
    compare(
        &tracked.grad().unwrap(),
        &a.eigen_vjp(&[1. / 7., 2. / 7.], &GraphMatrix::zeros(2, 2))
            .unwrap(),
    );
    vectors.backward_with_gradient(&gq).unwrap();
    compare(
        &tracked.grad().unwrap(),
        &a.eigen_vjp(&[0., 0.], &gq).unwrap(),
    );
    let (mut u, mut s, mut vt) = tracked.svd_tracked().unwrap();
    let gu = seed(&u);
    let gs = seed(&s);
    let gvt = seed(&vt);
    u.backward_with_gradient(&gu).unwrap();
    compare(
        &tracked.grad().unwrap(),
        &a.svd_vjp(&gu, &[0., 0.], &GraphMatrix::zeros(2, 2))
            .unwrap(),
    );
    s.backward_with_gradient(&gs).unwrap();
    compare(
        &tracked.grad().unwrap(),
        &a.svd_vjp(
            &GraphMatrix::zeros(2, 2),
            &[1. / 7., 2. / 7.],
            &GraphMatrix::zeros(2, 2),
        )
        .unwrap(),
    );
    tracked.set(0, 0, 99.).unwrap();
    vt.backward_with_gradient(&gvt).unwrap();
    compare(
        &tracked.grad().unwrap(),
        &a.svd_vjp(&GraphMatrix::zeros(2, 2), &[0., 0.], &gvt)
            .unwrap(),
    );
    assert!(tracked.qr_decomposition().is_err());
    assert!(tracked.svd_tracked().is_err());
}
#[test]
fn rectangular_thin_shapes_and_unsupported_derivative_domains_are_explicit() {
    let a = matrix(vec![3., 1., 0., 2., 1., 0.], 3, 2).requires_grad(true);
    let (u, mut s, vt) = a.svd_tracked().unwrap();
    assert_eq!((u.shape(), s.shape(), vt.shape()), ((3, 2), (2, 1), (2, 2)));
    s.backward().unwrap();
    assert_eq!(a.grad().unwrap().shape(), (3, 2));
    assert!(a.svd().is_err());
    let a = GraphMatrix::<f64>::identity(2).unwrap().requires_grad(true);
    assert!(a.eigen_tracked().unwrap().0.backward().is_err());
    assert!(a.svd_tracked().unwrap().1.backward().is_err());
    let a = GraphMatrix::<f64>::zeros(0, 0).requires_grad(true);
    a.svd_tracked().unwrap().1.backward().unwrap();
    assert_eq!(a.grad().unwrap().shape(), (0, 0));
}
