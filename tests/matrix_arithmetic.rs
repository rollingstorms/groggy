use groggy::storage::GraphMatrix;
fn matrix(values: Vec<f64>, rows: usize, cols: usize) -> GraphMatrix<f64> {
    GraphMatrix::from_row_major_data(values, rows, cols, None).unwrap()
}
#[test]
fn checked_binary_arithmetic_rejects_mismatch_and_retains_positional_identity() {
    let mut a = matrix(vec![1., 2., 3., 4., 5., 6.], 2, 3);
    a.set_row_labels(Some(vec!["r1".into(), "r2".into()]));
    a.set_column_names(vec!["a".into(), "b".into(), "c".into()]);
    let b = matrix(vec![6., 5., 4., 3., 2., 1.], 2, 3);
    let sum = a.add(&b).unwrap();
    let diff = a.subtract(&b).unwrap();
    for row in 0..2 {
        for col in 0..3 {
            assert_eq!(sum.get(row, col), Some(7.));
            assert_eq!(diff.get(row, col), Some(2. * (row * 3 + col) as f64 - 5.));
        }
    }
    for output in [&sum, &diff] {
        assert_eq!(output.row_labels(), a.row_labels());
        assert_eq!(output.column_names(), a.column_names());
    }
    assert!(a.add(&matrix(vec![1.; 6], 3, 2)).is_err());
    assert!(a.subtract(&matrix(vec![], 0, 0)).is_err());
    assert_eq!(a.get(0, 0), Some(1.));
}
#[test]
fn scalar_arithmetic_preserves_degenerate_shapes_and_uses_direct_division() {
    for (rows, cols) in [(0, 0), (3, 0), (0, 3), (1, 1), (2, 3)] {
        let a = matrix(vec![2.; rows * cols], rows, cols);
        for output in [
            a.scalar_add(3.).unwrap(),
            a.scalar_subtract(3.).unwrap(),
            a.scalar_multiply(-1.).unwrap(),
            a.scalar_divide(2.).unwrap(),
        ] {
            assert_eq!(output.shape(), (rows, cols));
        }
        assert!(a.scalar_divide(0.).is_err());
        assert!(a.scalar_divide(-0.).is_err());
    }
    let tiny = f64::from_bits(1);
    let a = matrix(vec![0., tiny, -tiny], 1, 3);
    let divided = a.scalar_divide(tiny).unwrap();
    assert_eq!(divided.get(0, 0), Some(0.));
    assert_eq!(divided.get(0, 1), Some(1.));
    assert_eq!(divided.get(0, 2), Some(-1.));
    assert!(a
        .scalar_divide(f64::NAN)
        .unwrap()
        .get(0, 0)
        .unwrap()
        .is_nan());
    assert_eq!(a.scalar_divide(f64::INFINITY).unwrap().get(0, 1), Some(0.));
}

#[test]
fn borrowed_and_owned_operators_return_results_and_never_substitute_zero_data() {
    let mut a = matrix(vec![1., 2., 3., 4.], 2, 2);
    a.set_row_labels(Some(vec!["left0".into(), "left1".into()]));
    a.set_column_names(vec!["x".into(), "y".into()]);
    let mut b = matrix(vec![5., 6., 7., 8.], 2, 2);
    b.set_column_names(vec!["right0".into(), "right1".into()]);
    for sum in [(&a + &b).unwrap(), (a.clone() + b.clone()).unwrap()] {
        assert_eq!(sum.get(0, 0), Some(6.));
        assert_eq!(sum.get(1, 1), Some(12.));
        assert_eq!(sum.row_labels(), a.row_labels());
        assert_eq!(sum.column_names(), a.column_names());
    }
    for diff in [(&a - &b).unwrap(), (a.clone() - b.clone()).unwrap()] {
        for row in 0..2 {
            for col in 0..2 {
                assert_eq!(diff.get(row, col), Some(-4.));
            }
        }
    }
    for product in [(&a * &b).unwrap(), (a.clone() * b.clone()).unwrap()] {
        for (row, expected) in [[19., 22.], [43., 50.]].iter().enumerate() {
            for (col, &value) in expected.iter().enumerate() {
                assert_eq!(product.get(row, col), Some(value));
            }
        }
        assert_eq!(product.row_labels(), a.row_labels());
        assert_eq!(product.column_names(), b.column_names());
    }
    let bad = matrix(vec![9.; 3], 3, 1);
    assert!((&a + &bad).is_err());
    assert!((a.clone() + bad.clone()).is_err());
    assert!((&a - &bad).is_err());
    assert!((a.clone() - bad.clone()).is_err());
    assert!((&a * &bad).is_err());
    assert!((a.clone() * bad).is_err());
    assert_eq!(a.get(0, 0), Some(1.));
}

#[test]
fn zero_inner_dimension_products_keep_shape_and_overflow_is_an_error() {
    let a = matrix(vec![], 3, 0);
    let b = matrix(vec![], 0, 2);
    let product = (&a * &b).unwrap();
    assert_eq!(product.shape(), (3, 2));
    for row in 0..3 {
        for col in 0..2 {
            assert_eq!(product.get(row, col), Some(0.));
        }
    }
    let huge = matrix(vec![], usize::MAX, 0);
    assert!((&huge * &b).is_err());
    // Empty inner dimensions permit cheap inputs whose output byte size cannot fit.
    let byte_overflow = matrix(vec![], isize::MAX as usize / 8 + 1, 0);
    assert!(byte_overflow.multiply(&matrix(vec![], 0, 1)).is_err());
}
