use groggy::storage::{
    matrix::{neural::activations::ActivationOps, MatrixLayout, UnifiedMatrix},
    GraphMatrix,
};
fn m(values: Vec<f64>, rows: usize, cols: usize) -> GraphMatrix<f64> {
    GraphMatrix::from_row_major_data(values, rows, cols, None).unwrap()
}
fn scalar(value: f64) -> UnifiedMatrix<f64> {
    UnifiedMatrix::from_data(vec![value], 1, 1).unwrap()
}
#[test]
fn gelu_derivative_matches_its_tanh_forward_and_extreme_values_are_finite() {
    for x in [-8., -3., -1., 0., 0.5, 2., 6.] {
        let h = 1e-5;
        let plus = ActivationOps::gelu(&scalar(x + h))
            .unwrap()
            .get(0, 0)
            .unwrap();
        let minus = ActivationOps::gelu(&scalar(x - h))
            .unwrap()
            .get(0, 0)
            .unwrap();
        let actual = ActivationOps::gelu_derivative(&scalar(x))
            .unwrap()
            .get(0, 0)
            .unwrap();
        assert!(
            (actual - (plus - minus) / (2. * h)).abs() < 2e-9,
            "{x}: {actual}"
        );
    }
    for (x, expected, slope) in [(f64::MAX, f64::MAX, 1.), (-f64::MAX, 0., 0.)] {
        assert_eq!(
            ActivationOps::gelu(&scalar(x)).unwrap().get(0, 0).unwrap(),
            expected
        );
        assert_eq!(
            ActivationOps::gelu_derivative(&scalar(x))
                .unwrap()
                .get(0, 0)
                .unwrap(),
            slope
        );
    }
}
#[test]
fn activation_tapes_compose_and_keep_labels_with_explicit_unsupported_boundaries() {
    let mut input = m(vec![-1., 0., 2.], 1, 3);
    input.set_column_names(vec!["a".into(), "b".into(), "c".into()]);
    input.set_row_labels(Some(vec!["r".into()]));
    let input = input.requires_grad(true);
    let mut output = input.gelu().unwrap().scalar_multiply(3.).unwrap();
    assert!(output.requires_grad_enabled());
    assert_eq!(output.column_names(), input.column_names());
    assert_eq!(output.row_labels(), input.row_labels());
    output.backward().unwrap();
    let slopes =
        ActivationOps::gelu_derivative(&UnifiedMatrix::from_data(vec![-1., 0., 2.], 1, 3).unwrap())
            .unwrap()
            .to_vec()
            .unwrap();
    for (got, expected) in input.grad().unwrap().to_vec().unwrap().iter().zip(slopes) {
        assert!((got - 3. * expected).abs() < 1e-12);
    }
    let mut output = input.relu().unwrap();
    output.backward().unwrap();
    assert_eq!(input.grad().unwrap().to_vec().unwrap(), vec![0., 0., 1.]);
    assert!(input.leaky_relu(None).is_err());
    assert!(input.elu(None).is_err());
    assert!(input.dropout_seeded(0.5, true, 42).is_err());
    let mut identity = input.dropout_seeded(0.5, false, 42).unwrap();
    identity.backward().unwrap();
    assert_eq!(input.grad().unwrap().to_vec().unwrap(), vec![1.; 3]);
}
#[test]
fn activation_errors_do_not_turn_bad_values_into_zero() {
    let int = UnifiedMatrix::from_data(vec![-1i64], 1, 1).unwrap();
    assert!(ActivationOps::leaky_relu(&int, 0.1).is_err());
    assert!(ActivationOps::gelu(&int).is_err());
    let maximum = UnifiedMatrix::from_data(vec![i64::MAX], 1, 1).unwrap();
    assert_eq!(
        ActivationOps::relu(&maximum).unwrap().get(0, 0).unwrap(),
        i64::MAX
    );
    assert!(ActivationOps::elu(&scalar(-1.), f64::NAN).is_err());
    assert!(ActivationOps::leaky_relu(&scalar(-f64::MAX), 2.).is_err());
    let mut input = scalar(f64::NAN);
    assert!(ActivationOps::relu_inplace(&mut input).is_err());
    assert!(input.get(0, 0).unwrap().is_nan());
}
#[test]
fn softmax_uses_actual_logical_rows_including_column_major_storage() {
    let mut input = UnifiedMatrix::new_with_layout(2, 3, MatrixLayout::ColumnMajor).unwrap();
    for row in 0..2 {
        for col in 0..3 {
            input
                .set(row, col, 1000. + row as f64 + col as f64)
                .unwrap();
        }
    }
    let result = ActivationOps::softmax(&input).unwrap();
    let denominator = 1. + 1f64.exp() + 2f64.exp();
    for row in 0..2 {
        for col in 0..3 {
            assert!(
                (result.get(row, col).unwrap() - (col as f64).exp() / denominator).abs() < 1e-15
            );
        }
    }
    assert!(ActivationOps::softmax(&scalar(f64::INFINITY)).is_err());
}
#[test]
fn seeded_dropout_is_storage_independent_preserves_metadata_and_handles_endpoints() {
    let mut input = m(vec![1.; 64], 8, 8);
    input.set_row_labels(Some((0..8).map(|i| format!("n{i}")).collect()));
    let dense = input.dropout_seeded(0.5, true, 91).unwrap();
    let sparse = input
        .to_csr()
        .unwrap()
        .dropout_seeded(0.5, true, 91)
        .unwrap();
    assert!(sparse.is_sparse());
    assert_eq!(dense.to_vec().unwrap(), sparse.to_vec().unwrap());
    assert_eq!(dense.row_labels(), input.row_labels());
    let values = dense.to_vec().unwrap();
    assert!(values.contains(&0.));
    assert!(values.contains(&2.));
    assert_eq!(
        input
            .dropout_seeded(0.9, false, 0)
            .unwrap()
            .to_vec()
            .unwrap(),
        vec![1.; 64]
    );
    assert_eq!(
        input.dropout_seeded(0., true, 0).unwrap().to_vec().unwrap(),
        vec![1.; 64]
    );
    assert_eq!(
        input.dropout_seeded(1., true, 0).unwrap().to_vec().unwrap(),
        vec![0.; 64]
    );
    for p in [f64::NAN, -0.1, 1.1] {
        for training in [false, true] {
            assert!(input.dropout_seeded(p, training, 1).is_err());
        }
    }
}
#[test]
fn sparse_activation_zero_images_and_expansion_policy_are_consistent() {
    let input = UnifiedMatrix::from_csr(
        2000,
        2000,
        vec![-2., 3.],
        vec![0, 1],
        std::iter::once(0)
            .chain(std::iter::repeat_n(2, 2000))
            .collect(),
    )
    .unwrap();
    assert!(matches!(
        ActivationOps::relu(&input).unwrap().storage,
        groggy::storage::matrix::unified_matrix::MatrixStorage::Sparse(_)
    ));
    assert!(ActivationOps::sigmoid(&input).is_err());
    assert!(ActivationOps::softmax(&input).is_err());
}

#[test]
fn softmax_tape_pullback_matches_the_row_jacobian_and_empty_rows_are_total() {
    let input = m(vec![-1., 0.5, 2.], 1, 3).requires_grad(true);
    let probabilities = input.softmax(1).unwrap();
    let weights = m(vec![2., -1., 3.], 1, 3);
    let mut output = probabilities.elementwise_multiply(&weights).unwrap();
    output.backward().unwrap();
    let p = probabilities.to_vec().unwrap();
    let w = weights.to_vec().unwrap();
    let mean = p.iter().zip(&w).map(|(p, w)| p * w).sum::<f64>();
    for (index, got) in input.grad().unwrap().to_vec().unwrap().iter().enumerate() {
        assert!((got - p[index] * (w[index] - mean)).abs() < 1e-14);
    }
    assert!(input.softmax(0).is_err());
    let empty = UnifiedMatrix::<f64>::zeros(3, 0).unwrap();
    assert!(ActivationOps::softmax(&empty)
        .unwrap()
        .to_vec()
        .unwrap()
        .is_empty());
}
