use groggy::storage::array::{numeric_reductions as k, NumArray};
fn close(actual: Option<f64>, expected: f64) {
    assert!(
        (actual.unwrap() - expected).abs() < 2e-14,
        "{actual:?} != {expected}"
    );
}
#[test]
fn pearson_matches_hand_computed_reference_and_affine_invariance() {
    // Centered x=(-2,0,2), y=(-1,-1,2): dot=6, norms=sqrt(8*6).
    let expected = 3.0_f64.sqrt() / 2.0;
    close(k::correlate(&[1.0, 3.0, 5.0], &[2.0, 2.0, 5.0]), expected);
    for scale in [1e-300, 1.0, 1e300] {
        let x = [scale, 3.0 * scale, 5.0 * scale];
        let y = [2.0 * scale, 2.0 * scale, 5.0 * scale];
        close(k::correlate(&x, &y), expected);
        close(
            NumArray::new(x.to_vec()).correlate(&NumArray::new(y.to_vec())),
            expected,
        );
        close(k::correlate(&y, &x), expected);
        close(k::correlate(&x, &y.map(|v| -v)), -expected);
    }
    close(
        k::correlate(&[-f64::MAX, 0.0, f64::MAX], &[-1.0, 0.0, 1.0]),
        1.0,
    );
    close(
        k::correlate(&[1e16, 1e16 + 2.0, 1e16 + 4.0], &[0.0, 1.0, 2.0]),
        1.0,
    );
    close(
        k::correlate(&[i64::MAX - 2, i64::MAX - 1, i64::MAX], &[0_i32, 1, 2]),
        1.0,
    );
    close(
        k::correlate(&[i64::MIN, i64::MIN + 1, i64::MIN + 2], &[2_i32, 1, 0]),
        -1.0,
    );
    close(
        k::correlate(&[false, true, false, true], &[0_i32, 1, 0, 1]),
        1.0,
    );
    close(k::correlate(&[-1.0, 0.0, 1.0], &[1.0, -2.0, 1.0]), 0.0);
}
#[test]
fn undefined_and_nonfinite_cases_follow_explicit_precedence_without_mutation() {
    for (x, y) in [
        (vec![], vec![]),
        (vec![1.0], vec![2.0]),
        (vec![1.0, 2.0], vec![3.0]),
        (vec![1.0, 1.0], vec![1.0, 2.0]),
        (vec![1.0, 2.0], vec![2.0, 2.0]),
    ] {
        assert_eq!(k::correlate(&x, &y), None);
    }
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(k::correlate(&[bad, 1.0], &[2.0, 2.0]).unwrap().is_nan());
        assert!(k::correlate(&[1.0, 2.0], &[3.0, bad]).unwrap().is_nan());
        assert_eq!(k::correlate(&[bad], &[1.0]), None);
    }
    let a = NumArray::new(vec![1.0, 3.0, 5.0]);
    let b = NumArray::new(vec![2.0, 2.0, 5.0]);
    let saved = (a.as_slice().to_vec(), b.as_slice().to_vec());
    close(a.correlate(&b), 3.0_f64.sqrt() / 2.0);
    assert_eq!(a.as_slice(), saved.0);
    assert_eq!(b.as_slice(), saved.1);
}
