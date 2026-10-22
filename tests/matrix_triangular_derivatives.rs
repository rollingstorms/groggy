use groggy::storage::GraphMatrix;
fn matrix(v: Vec<f64>, m: usize, n: usize) -> GraphMatrix<f64> {
    GraphMatrix::from_row_major_data(v, m, n, None).unwrap()
}
#[test]
fn scalar_factor_derivatives_and_zero_shapes_are_correct() {
    let a = matrix(vec![4.], 1, 1);
    let h = matrix(vec![2.], 1, 1);
    let (dq, dr) = a.qr_jvp(&h).unwrap();
    assert_eq!(dq.get(0, 0), Some(0.));
    assert_eq!(dr.get(0, 0), Some(-2.));
    let (dl, du) = a.lu_jvp(&h).unwrap();
    assert_eq!(dl.get(0, 0), Some(0.));
    assert_eq!(du.get(0, 0), Some(2.));
    assert_eq!(a.qr_vjp(&h, &h).unwrap().get(0, 0), Some(-2.));
    assert_eq!(a.lu_vjp(&h, &h).unwrap().get(0, 0), Some(2.));
    for (m, n) in [(0, 0), (0, 3), (3, 0)] {
        let a = matrix(vec![], m, n);
        let (q, r) = a.qr_jvp(&a).unwrap();
        assert_eq!(q.shape(), (m, 0));
        assert_eq!(r.shape(), (0, n));
        assert_eq!(a.qr_vjp(&q, &r).unwrap().shape(), (m, n));
    }
    let a = matrix(vec![], 0, 0);
    assert!(a.lu_jvp(&a).is_ok());
    assert!(a.lu_vjp(&a, &a).is_ok());
}
#[test]
fn factor_derivative_domains_reject_rank_loss_pivot_ties_and_sign_boundaries() {
    let tie = matrix(vec![1., 0., 1., 1.], 2, 2);
    assert!(tie.lu_jvp(&tie).is_err());
    let boundary = matrix(vec![0., 1., 1., 0.], 2, 2);
    assert!(boundary.qr_jvp(&boundary).is_err());
    let zero = matrix(vec![0.; 4], 2, 2);
    assert!(zero.qr_jvp(&zero).is_err());
    assert!(zero.lu_jvp(&zero).is_err());
    let a = matrix(vec![2.], 1, 1);
    assert!(a.qr_vjp(&zero, &a).is_err());
    assert!(a.lu_jvp(&zero).is_err());
    let a = matrix(vec![f64::NAN], 1, 1);
    assert!(a.lu_jvp(&a).is_err());
    let a = GraphMatrix::<i64>::from_row_major_data(vec![2], 1, 1, None).unwrap();
    assert!(a.qr_jvp(&a).is_err());
    let a = GraphMatrix::<f32>::from_row_major_data(vec![2.], 1, 1, None).unwrap();
    assert!(a.lu_jvp(&a).is_ok());
}
