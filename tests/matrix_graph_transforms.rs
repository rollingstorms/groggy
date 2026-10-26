use groggy::{
    storage::{
        adjacency::{MatrixFormat, MatrixType},
        GraphMatrix,
    },
    Graph,
};
fn core(a: Vec<f64>, n: usize) -> GraphMatrix<f64> {
    GraphMatrix::from_row_major_data(a, n, n, None).unwrap()
}
#[test]
fn normalized_adjacency_laplacian_and_transition_are_distinct_and_power_is_correct() {
    let a = core(vec![0., 2., 0., 2., 0., 0., 0., 0., 0.], 3);
    let d = a.to_degree_matrix().unwrap();
    assert_eq!(d.get(0, 0), Some(2.));
    let s = a.normalized_adjacency(-0.5, 1).unwrap();
    for power in [0, 1, 2, 3, 7] {
        let x = a.to_normalized_laplacian(-0.5, power).unwrap();
        let expected = s.power(power).unwrap();
        for r in 0..3 {
            for c in 0..3 {
                assert_eq!(x.get(r, c), expected.get(r, c));
            }
        }
    }
    let l = a.normalized_laplacian().unwrap();
    assert_eq!(l.get(2, 2), Some(0.));
    assert!((l.get(0, 1).unwrap() + 1.).abs() < 1e-14);
    assert_eq!(l.get(0, 0), Some(1.));
    assert_eq!(
        a.to_normalized_laplacian_standard().unwrap().get(0, 0),
        Some(1.)
    );
    let p = a.transition_matrix().unwrap();
    assert_eq!(p.get(0, 1), Some(1.));
    assert_eq!(p.get(2, 2), Some(0.));
    let a = core(vec![f64::MAX, f64::MAX, 0., 0.], 2);
    let p = a.transition_matrix().unwrap();
    assert_eq!(p.get(0, 0), Some(0.5));
    assert_eq!(p.get(0, 1), Some(0.5));
}
#[test]
fn sparse_operators_stay_sparse_and_domains_return_errors() {
    let n = 10000;
    let a = GraphMatrix::from_csr(n, n, vec![4.; n], (0..n).collect(), (0..=n).collect()).unwrap();
    for result in [
        a.to_degree_matrix().unwrap(),
        a.to_laplacian().unwrap(),
        a.normalized_laplacian().unwrap(),
        a.transition_matrix().unwrap(),
    ] {
        assert!(result.is_sparse());
    }
    assert!(!core(vec![f64::NAN], 1).is_adjacency_matrix());
    assert!(!core(vec![f64::INFINITY], 1).is_adjacency_matrix());
    for value in [-1., f64::NAN, f64::INFINITY] {
        let a = core(vec![value], 1);
        assert!(a.normalized_adjacency(-0.5, 1).is_err());
        assert!(a.transition_matrix().is_err());
    }
    assert!(core(vec![1.], 1).normalized_adjacency(f64::NAN, 1).is_err());
    assert!(
        GraphMatrix::from_row_major_data(vec![i64::MAX, 1, 0, 0], 2, 2, None)
            .unwrap()
            .to_degree_matrix()
            .is_err()
    );
    assert!(GraphMatrix::<i64>::ones(2, 2).transition_matrix().is_err());
    assert!(GraphMatrix::<f64>::ones(2, 2)
        .requires_grad(true)
        .normalized_laplacian()
        .is_err());
    assert_eq!(
        GraphMatrix::<f64>::zeros(0, 0)
            .normalized_laplacian()
            .unwrap()
            .shape(),
        (0, 0)
    );
}
#[test]
fn graph_high_level_routes_execute_true_operators() {
    let mut graph = Graph::new_manual();
    let a = graph.add_node().unwrap();
    let b = graph.add_node().unwrap();
    graph.add_node().unwrap();
    graph.add_edge(a, b).unwrap();
    let l = graph.laplacian_matrix(true).unwrap();
    let names = l.column_names();
    let ia = names
        .iter()
        .position(|v| v == &format!("node_{a}"))
        .unwrap();
    let ib = names
        .iter()
        .position(|v| v == &format!("node_{b}"))
        .unwrap();
    assert_eq!(l.get(ia, ia), Some(1.));
    assert_eq!(l.get(ia, ib), Some(-1.));
    let p = graph.transition_matrix().unwrap();
    assert_eq!(p.get(ia, ib), Some(1.));
    let custom = graph
        .custom_adjacency_matrix(MatrixFormat::Dense, MatrixType::Transition, true, None)
        .unwrap();
    assert_eq!(custom.get(ia, ib), Some(1.));
}
