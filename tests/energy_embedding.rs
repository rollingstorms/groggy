use groggy::viz::embeddings::energy::{
    EnergyEmbedding, EnergyEmbeddingBuilder, ForceDirectedEmbedding,
};
use groggy::viz::embeddings::{
    EmbeddingConfig, EmbeddingEngine, EmbeddingEngineFactory, EmbeddingMethod, EnergyFunction,
    StressType,
};
use groggy::{Graph, GraphMatrix};

fn spring(iterations: usize) -> EnergyEmbedding {
    EnergyEmbeddingBuilder::new()
        .iterations(iterations)
        .learning_rate(0.05)
        .with_spring_electric(1.0, 0.0, 0.0)
        .seed(91)
        .build()
}
fn coordinates() -> GraphMatrix {
    GraphMatrix::from_row_major_data(vec![-2., 1., 4., 3., 7., -5.], 3, 2, None).unwrap()
}
#[test]
fn actual_edges_beyond_old_cap_and_sparse_ids_control_only_their_endpoints() {
    let mut graph = Graph::new_manual();
    graph.add_nodes(4).unwrap();
    graph.remove_node(1).unwrap();
    let input = coordinates();
    let before = input.to_vec().unwrap();
    for _ in 0..30 {
        graph.add_edge(0, 0).unwrap();
    }
    let edge = graph.add_edge(2, 3).unwrap();
    let result = spring(1).optimize_positions(&input, &graph).unwrap();
    assert_eq!(result.row_labels().unwrap(), ["node_0", "node_2", "node_3"]);
    assert_eq!(result.column_names(), &["energy_0", "energy_1"]);
    assert_eq!(result.to_vec().unwrap(), [-2., 1., 4.15, 2.6, 6.85, -4.6]);
    assert_eq!(input.to_vec().unwrap(), before);
    assert_eq!(graph.edge_count(), 31);
    graph.remove_edge(edge).unwrap();
    assert_eq!(
        spring(2)
            .optimize_positions(&input, &graph)
            .unwrap()
            .to_vec()
            .unwrap(),
        before
    );
}
#[test]
fn row_permutations_preserve_identity_and_parallel_instances_add_forces() {
    let mut graph = Graph::new_manual();
    graph.add_nodes(3).unwrap();
    graph.add_edge(0, 2).unwrap();
    let input = coordinates();
    let result = spring(1).optimize_positions(&input, &graph).unwrap();
    let mut permuted =
        GraphMatrix::from_row_major_data(vec![7., -5., -2., 1., 4., 3.], 3, 2, None).unwrap();
    permuted.set_row_labels(Some(vec![
        "node_2".into(),
        "node_0".into(),
        "node_1".into(),
    ]));
    let reordered = spring(1).optimize_positions(&permuted, &graph).unwrap();
    for (row, other) in [1, 2, 0].into_iter().enumerate() {
        for axis in 0..2 {
            assert_eq!(result.get(row, axis), reordered.get(other, axis));
        }
    }
    assert_eq!(reordered.row_labels(), permuted.row_labels());
    graph.add_edge(2, 0).unwrap();
    let doubled = spring(1).optimize_positions(&input, &graph).unwrap();
    for row in 0..3 {
        for axis in 0..2 {
            let initial = input.get(row, axis).unwrap();
            let step = result.get(row, axis).unwrap() - initial;
            assert!((doubled.get(row, axis).unwrap() - initial - 2.0 * step).abs() < 1e-14);
        }
    }
}
#[test]
fn controls_identity_and_unsupported_modes_are_checked_even_without_iterations() {
    let mut graph = Graph::new_manual();
    graph.add_nodes(3).unwrap();
    for rate in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(EnergyEmbeddingBuilder::new()
            .iterations(0)
            .learning_rate(rate)
            .compute(&graph, 2)
            .is_err());
    }
    for value in [-1.0, f64::NAN, f64::INFINITY] {
        for (a, b, d) in [(value, 1., 1.), (1., value, 1.), (1., 1., value)] {
            assert!(EnergyEmbeddingBuilder::new()
                .iterations(0)
                .with_spring_electric(a, b, d)
                .compute(&graph, 2)
                .is_err());
        }
    }
    for function in [
        EnergyFunction::Custom {
            attraction_fn: "x".into(),
            repulsion_fn: "x".into(),
        },
        EnergyFunction::StressMinimization {
            stress_type: StressType::Raw,
        },
    ] {
        assert!(EnergyEmbeddingBuilder::new()
            .iterations(0)
            .energy_function(function)
            .compute(&graph, 2)
            .is_err());
    }
    assert!(spring(0).compute_embedding(&graph, 0).is_err());
    assert!(spring(0)
        .compute_embedding(&Graph::new_manual(), 2)
        .is_err());
    assert!(spring(0).compute_embedding(&graph, usize::MAX).is_err());
    assert!(spring(0)
        .optimize_positions(&GraphMatrix::zeros(2, 2), &graph)
        .is_err());
    for labels in [
        vec!["node_0", "node_0", "node_2"],
        vec!["node_0", "node_1", "node_8"],
        vec!["bad", "node_1", "node_2"],
        vec!["node_0"],
    ] {
        let mut input = coordinates();
        input.set_row_labels(Some(labels.into_iter().map(String::from).collect()));
        assert!(spring(0).optimize_positions(&input, &graph).is_err());
    }
    for value in [f64::NAN, f64::INFINITY] {
        let mut input = coordinates();
        input.set(0, 0, value).unwrap();
        assert!(spring(0).optimize_positions(&input, &graph).is_err());
    }
    let mut singleton = Graph::new_manual();
    singleton.add_node().unwrap();
    let result = spring(10).compute_embedding(&singleton, 1).unwrap();
    assert_eq!(result.shape(), (1, 1));
    assert_eq!(result.row_labels().unwrap(), ["node_0"]);
    assert!(ForceDirectedEmbedding::new(-1., 1., 0)
        .compute_embedding(&graph, 2)
        .is_err());
    assert!(!ForceDirectedEmbedding::new(1., 1., 0).supports_incremental());
}
#[test]
fn factory_energy_and_force_directed_honor_seed_without_global_rng_mutation() {
    let mut graph = Graph::new_manual();
    graph.add_nodes(3).unwrap();
    graph.add_edge(0, 2).unwrap();
    for method in [
        EmbeddingMethod::EnergyND {
            iterations: 15,
            learning_rate: 0.01,
            annealing: true,
        },
        EmbeddingMethod::ForceDirectedND {
            spring_constant: 1.0,
            repulsion_strength: 0.1,
            iterations: 15,
        },
    ] {
        let config = EmbeddingConfig {
            method,
            dimensions: 3,
            seed: Some(35),
            ..Default::default()
        };
        let engine = EmbeddingEngineFactory::create_engine(&config).unwrap();
        fastrand::seed(41);
        let expected = fastrand::u64(..);
        fastrand::seed(41);
        let first = engine.compute_embedding(&graph, 3).unwrap();
        assert_eq!(fastrand::u64(..), expected);
        let second = engine.compute_embedding(&graph, 3).unwrap();
        assert_eq!(first.to_vec().unwrap(), second.to_vec().unwrap());
        assert!(first.to_vec().unwrap().iter().all(|x| x.is_finite()));
        assert_eq!(first.row_labels().unwrap(), ["node_0", "node_1", "node_2"]);
    }
}
