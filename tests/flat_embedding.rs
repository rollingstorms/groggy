use groggy::viz::embeddings::flat_embedding::{
    compute_flat_embedding_from_matrix, compute_flat_embedding_matrix, FlatEmbedConfig,
};
use groggy::{Graph, GraphMatrix};
fn config() -> FlatEmbedConfig {
    FlatEmbedConfig {
        iterations: 20,
        repulsion_weight: 0.0,
        spread_weight: 0.0,
        ..Default::default()
    }
}
fn input(n: usize) -> GraphMatrix {
    GraphMatrix::zeros(n, 3)
}
fn dist(m: &GraphMatrix, u: usize, v: usize) -> f64 {
    (m.get(u, 0).unwrap() - m.get(v, 0).unwrap()).hypot(m.get(u, 1).unwrap() - m.get(v, 1).unwrap())
}
#[test]
fn actual_topology_changes_only_its_endpoints_without_fake_or_truncated_edges() {
    let mut graph = Graph::new();
    graph.add_nodes(4);
    graph.remove_node(1).unwrap();
    // Three rows map to IDs 0,2,3. Directed orientation has the same geometric energy.
    let embedding = input(3);
    let mut init = config();
    init.iterations = 0;
    let before = compute_flat_embedding_matrix(&embedding, &graph, &init).unwrap();
    let empty = compute_flat_embedding_matrix(&embedding, &graph, &config()).unwrap();
    assert_eq!(before.to_vec().unwrap(), empty.to_vec().unwrap());
    // More than the previous artificial 100-edge cap, followed by the only nonloop.
    for _ in 0..120 {
        graph.add_edge(0, 0).unwrap();
    }
    graph.add_edge(2, 3).unwrap();
    let result = compute_flat_embedding_matrix(&embedding, &graph, &config()).unwrap();
    assert!(dist(&result, 1, 2) < dist(&before, 1, 2));
    for axis in 0..2 {
        assert_eq!(result.get(0, axis), before.get(0, axis));
    }
    assert_eq!(result.row_labels().unwrap(), ["node_0", "node_2", "node_3"]);
    let edge = graph.edge_ids().into_iter().max().unwrap();
    graph.remove_edge(edge).unwrap();
    assert_eq!(
        compute_flat_embedding_matrix(&embedding, &graph, &config())
            .unwrap()
            .to_vec()
            .unwrap(),
        before.to_vec().unwrap()
    );
}
#[test]
fn labeled_permutations_validate_identity_and_preserve_row_order() {
    let mut graph = Graph::new();
    graph.add_nodes(4);
    graph.remove_node(1).unwrap();
    graph.add_edge(0, 3).unwrap();
    let mut embedding = input(3);
    embedding.set_row_labels(Some(vec![
        "node_3".into(),
        "node_0".into(),
        "node_2".into(),
    ]));
    let mut init = config();
    init.iterations = 0;
    let before = compute_flat_embedding_matrix(&embedding, &graph, &init).unwrap();
    let after = compute_flat_embedding_matrix(&embedding, &graph, &config()).unwrap();
    assert!(dist(&after, 0, 1) < dist(&before, 0, 1));
    assert_eq!(after.row_labels(), embedding.row_labels());
    for axis in 0..2 {
        assert_eq!(after.get(2, axis), before.get(2, axis));
    }
    for labels in [
        vec!["node_0", "node_0", "node_3"],
        vec!["node_0", "node_1", "node_3"],
        vec!["bad", "node_2", "node_3"],
        vec!["node_0"],
    ] {
        embedding.set_row_labels(Some(labels.into_iter().map(String::from).collect()));
        assert!(compute_flat_embedding_matrix(&embedding, &graph, &config()).is_err());
    }
    assert!(compute_flat_embedding_matrix(&input(2), &graph, &config()).is_err());
}
#[test]
fn graph_free_empty_singleton_and_bounds_are_explicit() {
    let graph = Graph::new();
    assert_eq!(
        compute_flat_embedding_matrix(&input(0), &graph, &config())
            .unwrap()
            .shape(),
        (0, 2)
    );
    assert!(compute_flat_embedding_matrix(&input(1), &graph, &config()).is_err());
    assert!(compute_flat_embedding_from_matrix(&input(3), &config()).is_err());
    let mut cfg = config();
    cfg.edge_cohesion_weight = 0.0;
    cfg.repulsion_weight = 0.1;
    cfg.radius_clip = 0.01;
    let result = compute_flat_embedding_from_matrix(&input(3), &cfg).unwrap();
    for row in 0..3 {
        assert!(
            result
                .get(row, 0)
                .unwrap()
                .hypot(result.get(row, 1).unwrap())
                <= 0.01 * (1.0 + 1e-12)
        );
    }
    let mut graph = Graph::new();
    graph.add_node();
    assert_eq!(
        compute_flat_embedding_matrix(&input(1), &graph, &config())
            .unwrap()
            .shape(),
        (1, 2)
    );
}
#[test]
fn controls_reject_invalid_values_and_rng_is_local_and_repeatable() {
    let mut graph = Graph::new();
    graph.add_nodes(3);
    graph.add_edge(0, 2).unwrap();
    for key in 0..6 {
        for bad in [-1., f64::NAN, f64::INFINITY] {
            let mut cfg = config();
            match key {
                0 => cfg.learning_rate = bad,
                1 => cfg.repulsion_power = bad,
                2 => cfg.radius_clip = bad,
                3 => cfg.repulsion_weight = bad,
                4 => cfg.spread_weight = bad,
                _ => cfg.edge_cohesion_weight = bad,
            };
            assert!(compute_flat_embedding_matrix(&input(3), &graph, &cfg).is_err());
        }
    }
    fastrand::seed(42);
    let next = fastrand::u64(..);
    fastrand::seed(42);
    let a = compute_flat_embedding_matrix(
        &input(3),
        &graph,
        &FlatEmbedConfig {
            iterations: 5,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(fastrand::u64(..), next);
    let b = compute_flat_embedding_matrix(
        &input(3),
        &graph,
        &FlatEmbedConfig {
            iterations: 5,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(a.to_vec().unwrap(), b.to_vec().unwrap());
}
