use groggy::storage::{Axis, GraphMatrix};
fn matrix<T: groggy::storage::advanced_matrix::NumericType>(
    v: Vec<T>,
    r: usize,
    c: usize,
) -> GraphMatrix<T> {
    GraphMatrix::from_row_major_data(v, r, c, None).unwrap()
}
#[test]
fn global_axis_and_column_reductions_share_integer_and_floating_precision() {
    let a = matrix(vec![i64::MAX, 1, -i64::MAX], 1, 3);
    assert_eq!(a.sum().unwrap(), 1);
    assert_eq!(a.sum_axis(Axis::Rows).unwrap(), vec![1]);
    assert_eq!(a.mean().unwrap(), 1. / 3.);
    assert!(matrix(vec![i64::MAX, 1], 1, 2).sum().is_err());
    let a = matrix(vec![i64::MAX - 2, i64::MAX - 1, i64::MAX], 3, 1);
    assert_eq!(a.var_axis(Axis::Columns).unwrap(), vec![1]);
    assert_eq!(a.std_axis(Axis::Columns).unwrap(), vec![1]);
    assert!((a.column_variances().unwrap()[0] - 2. / 3.).abs() < 1e-15);
    let a = matrix(vec![1e20, 1., -1e20], 1, 3);
    assert_eq!(a.sum().unwrap(), 1.);
    assert_eq!(a.sum_axis(Axis::Rows).unwrap(), vec![1.]);
    assert_eq!(
        matrix(vec![f64::MAX, f64::MAX], 1, 2)
            .mean_axis(Axis::Rows)
            .unwrap(),
        vec![f64::MAX]
    );
    let a = matrix(vec![0., f64::MAX], 1, 2);
    assert!(a.var_axis(Axis::Rows).unwrap()[0].is_infinite());
    assert!((a.std_axis(Axis::Rows).unwrap()[0] / (f64::MAX / 2f64.sqrt()) - 1.).abs() < 1e-15);
}
#[test]
fn extrema_empty_ieee_and_trace_conventions_are_explicit() {
    let a = matrix(vec![1., f64::NAN], 1, 2);
    assert!(a.min().unwrap().unwrap().is_nan());
    assert!(a.max_axis(Axis::Rows).unwrap()[0].is_nan());
    assert!(matrix(vec![f64::INFINITY], 1, 1)
        .std_axis(Axis::Rows)
        .unwrap()[0]
        .is_nan());
    let empty = matrix(Vec::<f64>::new(), 3, 0);
    assert_eq!(empty.mean_axis(Axis::Rows).unwrap(), vec![0.; 3]);
    assert_eq!(empty.min().unwrap(), None);
    assert_eq!(empty.sum().unwrap(), 0.);
    let a = matrix(vec![i64::MAX, 0, 0, 0, 1, 0, 0, 0, -i64::MAX], 3, 3);
    assert_eq!(a.trace().unwrap(), 1);
    let a = matrix(vec![1., 3.], 2, 1);
    assert_eq!(a.column_variances().unwrap(), vec![1.]); // Population, unlike sample axis variance=2.
    assert_eq!(a.var_axis(Axis::Columns).unwrap(), vec![2.]);
}
