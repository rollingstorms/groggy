use groggy::storage::{
    advanced_matrix::{
        neural::autodiff::Operation, unified_matrix::Shape, ComputationGraph, UnifiedMatrix,
    },
    GraphMatrix,
};
fn matrix(a: Vec<f64>, n: usize) -> GraphMatrix<f64> {
    GraphMatrix::from_row_major_data(a, n, n, None).unwrap()
}
#[test]
fn tracked_cholesky_matches_explicit_pullback_and_repeated_backward_does_not_accumulate() {
    let mut a = matrix(vec![4., 1., 1., 3.], 2).requires_grad(true);
    let mut l = a.cholesky_decomposition().unwrap();
    assert!(l.requires_grad_enabled());
    let expected = a.cholesky_vjp(&matrix(vec![1.; 4], 2)).unwrap();
    l.backward().unwrap();
    for r in 0..2 {
        for c in 0..2 {
            assert!(
                (a.grad().unwrap().get(r, c).unwrap() - expected.get(r, c).unwrap()).abs() < 1e-14
            );
        }
    }
    l.backward().unwrap();
    assert_eq!(a.grad().unwrap().get(0, 0), expected.get(0, 0));
    a.zero_grad().unwrap();
    assert!(a.grad().is_none());
}
#[test]
fn saved_forward_state_is_stable_and_stale_bindings_reject_until_explicit_rebinding() {
    let mut a = matrix(vec![4.], 1).requires_grad(true);
    a.set(0, 0, 9.).unwrap();
    assert!(a.cholesky_decomposition().is_err());
    a = a.requires_grad(false).requires_grad(true);
    let mut l = a.cholesky_decomposition().unwrap();
    a.set(0, 0, 16.).unwrap();
    l.backward().unwrap();
    assert_eq!(a.grad().unwrap().get(0, 0), Some(1. / 6.));
    // Chain two valid diagonal Cholesky operations: d sqrt(sqrt(a))/da at a=16.
    let a = matrix(vec![16.], 1).requires_grad(true);
    let mut out = a
        .cholesky_decomposition()
        .unwrap()
        .cholesky_decomposition()
        .unwrap();
    out.backward().unwrap();
    assert_eq!(a.grad().unwrap().get(0, 0), Some(1. / 32.));
}
#[test]
fn lower_level_tape_executes_native_cholesky_and_reports_invalid_arity() {
    let mut tape = ComputationGraph::<f64>::new();
    let id = tape.create_leaf(UnifiedMatrix::from_data(vec![4.], 1, 1).unwrap(), true);
    let out = tape
        .create_operation(Operation::Cholesky, vec![id], Shape::new(1, 1), true)
        .unwrap();
    assert_eq!(tape.forward(out).unwrap().get(0, 0).unwrap(), 2.);
    tape.backward(out).unwrap();
    let node = tape.get_node(id).unwrap();
    assert_eq!(
        node.lock()
            .unwrap()
            .gradient
            .as_ref()
            .unwrap()
            .get(0, 0)
            .unwrap(),
        0.25
    );
    let invalid = tape
        .create_operation(Operation::Cholesky, vec![], Shape::new(1, 1), true)
        .unwrap();
    assert!(tape.forward(invalid).is_err());
    assert!(tape.backward(invalid).is_err());
    let a = GraphMatrix::<f32>::from_row_major_data(vec![4.], 1, 1, None)
        .unwrap()
        .requires_grad(true);
    let mut out = a.cholesky_decomposition().unwrap();
    out.backward().unwrap();
    assert_eq!(a.grad().unwrap().get(0, 0), Some(0.25));
    let a = matrix(vec![], 0).requires_grad(true);
    let mut out = a.cholesky_decomposition().unwrap();
    out.backward().unwrap();
    assert_eq!(a.grad().unwrap().shape(), (0, 0));
}
