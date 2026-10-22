use groggy::storage::GraphMatrix;
fn m(values: Vec<f64>, rows: usize, cols: usize) -> GraphMatrix<f64> {
    GraphMatrix::from_row_major_data(values, rows, cols, None).unwrap()
}
fn close(a: &GraphMatrix<f64>, b: &[f64], tolerance: f64) {
    let values = a.to_vec().unwrap();
    assert_eq!(values.len(), b.len());
    for (a, b) in values.iter().zip(b) {
        assert!((a - b).abs() < tolerance, "{a} vs {b}");
    }
}
#[test]
fn independent_and_cloned_branches_share_the_right_leaves() {
    let a = m(vec![2., 3.], 1, 2).requires_grad(true);
    let b = m(vec![5., 7.], 1, 2).requires_grad(true);
    let left = a.scalar_multiply(2.).unwrap();
    let right = a.scalar_multiply(3.).unwrap();
    let mut result = left.elementwise_multiply(&right).unwrap().add(&b).unwrap();
    assert!(result.requires_grad_enabled());
    result.backward().unwrap();
    close(&a.grad().unwrap(), &[24., 36.], 1e-12);
    close(&b.grad().unwrap(), &[1., 1.], 1e-12);
    // A second pass resets the shared gradients instead of doubling them.
    result.backward().unwrap();
    close(&a.grad().unwrap(), &[24., 36.], 1e-12);
    let mut zero = a.subtract(&a).unwrap();
    zero.backward().unwrap();
    close(&a.grad().unwrap(), &[0., 0.], 1e-12);
}
#[test]
fn matmul_transpose_and_storage_conversions_compose_with_repeated_input() {
    for sparse in [false, true] {
        let a = m(vec![1., 2., 3., 4.], 2, 2);
        let a = if sparse { a.to_csr().unwrap() } else { a };
        let a = a.requires_grad(true);
        let mut result = a
            .transpose()
            .unwrap()
            .matmul(&a)
            .unwrap()
            .to_csr()
            .unwrap()
            .to_dense()
            .unwrap();
        result.backward().unwrap();
        close(&a.grad().unwrap(), &[6., 6., 14., 14.], 1e-12);
    }
}
#[test]
fn stale_binding_and_nondifferentiable_dtype_fail_explicitly() {
    let mut a = m(vec![1., 2.], 1, 2).requires_grad(true);
    a.set(0, 0, 3.).unwrap();
    assert!(a.add(&a).is_err());
    assert!(a.transpose().is_err());
    let a = GraphMatrix::from_row_major_data(vec![1i64], 1, 1, None)
        .unwrap()
        .requires_grad(true);
    assert!(a.add(&a).is_err());
    let a = m(vec![f64::INFINITY], 1, 1).requires_grad(true);
    assert!(a.scalar_add(1.).is_err());
}
#[test]
fn scalar_and_elementwise_math_follow_chain_rule_and_singularities_reject() {
    let a = m(vec![2., 3.], 1, 2).requires_grad(true);
    let mut result = a
        .scalar_add(1.)
        .unwrap()
        .log()
        .unwrap()
        .exp()
        .unwrap()
        .sqrt()
        .unwrap()
        .scalar_divide(2.)
        .unwrap()
        .abs()
        .unwrap();
    result.backward().unwrap();
    close(&a.grad().unwrap(), &[0.25 / 3f64.sqrt(), 0.125], 1e-12);
    for method in [GraphMatrix::abs, GraphMatrix::sqrt] {
        let a = m(vec![0.], 1, 1).requires_grad(true);
        let mut result = method(&a).unwrap();
        assert!(result.backward().is_err());
    }
}
#[test]
fn singular_value_sibling_paths_accumulate_into_the_same_input() {
    let a = m(vec![4., 1., 2., 3., 0.5, 2.], 3, 2).requires_grad(true);
    let (_u, s, _vt) = a.svd_tracked().unwrap();
    let mut result = s
        .scalar_multiply(2.)
        .unwrap()
        .add(&s.scalar_multiply(3.).unwrap())
        .unwrap();
    result.backward().unwrap();
    let base = a.clone().requires_grad(false);
    let expected = base
        .svd_vjp(
            &GraphMatrix::zeros(3, 2),
            &[5.; 2],
            &GraphMatrix::zeros(2, 2),
        )
        .unwrap();
    close(&a.grad().unwrap(), &expected.to_vec().unwrap(), 1e-10);
}

#[test]
fn joint_qr_outputs_compose_and_cancel_to_original_input() {
    let a = m(vec![4., 1., 2., 3., 0.5, 2.], 3, 2).requires_grad(true);
    let (q, r) = a.qr_decomposition().unwrap();
    let mut reconstructed = q.matmul(&r).unwrap();
    reconstructed.backward().unwrap();
    close(&a.grad().unwrap(), &[1.; 6], 1e-10);
}

#[test]
fn division_pullback_uses_direct_division_when_reciprocal_overflows() {
    let tiny = f64::from_bits(1);
    let a = m(vec![tiny], 1, 1).requires_grad(true);
    let mut result = a.scalar_divide(tiny).unwrap();
    result.backward_with_gradient(&m(vec![tiny], 1, 1)).unwrap();
    close(&a.grad().unwrap(), &[1.], 1e-12);
}
#[test]
fn tape_scalar_primals_replay_and_invalid_dependencies_do_not_publish_nodes() {
    use groggy::storage::matrix::{
        neural::autodiff::Operation, unified_matrix::Shape, ComputationGraph, UnifiedMatrix,
    };
    let mut graph = ComputationGraph::new();
    let input = graph.create_leaf(UnifiedMatrix::from_data(vec![6.], 1, 1).unwrap(), true);
    for (operation, expected) in [
        (
            Operation::ScalarShift {
                scalar_bits: 2f64.to_bits(),
                subtract: false,
            },
            8.,
        ),
        (
            Operation::ScalarShift {
                scalar_bits: 2f64.to_bits(),
                subtract: true,
            },
            4.,
        ),
        (
            Operation::ScalarDivide {
                scalar_bits: 2f64.to_bits(),
            },
            3.,
        ),
    ] {
        let output = graph
            .create_operation(operation, vec![input], Shape::new(1, 1), true)
            .unwrap();
        // Unrelated later IDs must not break forward traversal before output.
        for _ in 0..8 {
            graph.create_leaf(UnifiedMatrix::zeros(1, 1).unwrap(), false);
        }
        graph.update_execution_order().unwrap();
        assert_eq!(
            graph.forward(output).unwrap().to_vec().unwrap(),
            vec![expected]
        );
        assert!(graph
            .execution_order()
            .windows(2)
            .all(|ids| ids[0] < ids[1]));
    }
    let before = graph.nodes().len();
    assert!(graph
        .create_operation(
            Operation::Identity,
            vec![usize::MAX],
            Shape::new(1, 1),
            true
        )
        .is_err());
    assert_eq!(graph.nodes().len(), before);
}
