use groggy::viz::embeddings::{
    combine_embedding_matrices, CombinationStrategy, EmbeddingConfig, EmbeddingEngineFactory,
    EmbeddingMethod,
};
use groggy::{Graph, GraphMatrix};
fn matrix(rows: usize, cols: usize, base: f64) -> GraphMatrix {
    GraphMatrix::from_row_major_data(
        (0..rows)
            .flat_map(|r| (0..cols).map(move |c| base + 10. * r as f64 + c as f64))
            .collect(),
        rows,
        cols,
        None,
    )
    .unwrap()
}
#[test]
fn interleave_all_columns_matches_sorted_round_reference_for_ragged_widths() {
    let mut graph = Graph::new_manual();
    graph.add_nodes(2).unwrap();
    for count in 1..5 {
        let matrices: Vec<_> = (0..count)
            .map(|child| matrix(2, child * 2 + 1, 100. * child as f64))
            .collect();
        for chunk in [1, 2, 3, 6, 100, usize::MAX] {
            let result = combine_embedding_matrices(
                &graph,
                &matrices,
                None,
                &CombinationStrategy::Interleave { chunk_size: chunk },
            )
            .unwrap();
            let mut order: Vec<_> = matrices
                .iter()
                .enumerate()
                .flat_map(|(child, m)| (0..m.shape().1).map(move |col| (child, col)))
                .collect();
            order.sort_by_key(|&(child, col)| (col / chunk, child, col));
            assert_eq!(result.shape(), (2, order.len()));
            for (out, &(child, col)) in order.iter().enumerate() {
                for row in 0..2 {
                    assert_eq!(result.get(row, out), matrices[child].get(row, col));
                }
            }
        }
    }
}
#[test]
fn row_identity_is_aligned_for_all_strategies_and_inputs_remain_owned() {
    let mut graph = Graph::new_manual();
    graph.add_nodes(3).unwrap();
    graph.remove_node(1).unwrap();
    let mut a = GraphMatrix::from_row_major_data(vec![10., 20., 30., 40.], 2, 2, None).unwrap();
    a.set_row_labels(Some(vec!["node_2".into(), "node_0".into()]));
    let b = GraphMatrix::from_row_major_data(vec![50., 60., 70., 80.], 2, 2, None).unwrap();
    let matrices = [a, b];
    for (strategy, want) in [
        (
            CombinationStrategy::Concatenate,
            vec![30., 40., 50., 60., 10., 20., 70., 80.],
        ),
        (
            CombinationStrategy::Interleave { chunk_size: 1 },
            vec![30., 50., 40., 60., 10., 70., 20., 80.],
        ),
        (
            CombinationStrategy::WeightedAverage,
            vec![45., 55., 55., 65.],
        ),
    ] {
        let result =
            combine_embedding_matrices(&graph, &matrices, Some(&[1., 3.]), &strategy).unwrap();
        assert_eq!(result.to_vec().unwrap(), want);
        assert_eq!(result.row_labels().unwrap(), ["node_0", "node_2"]);
    }
    assert_eq!(matrices[0].to_vec().unwrap(), vec![10., 20., 30., 40.]);
}
#[test]
fn composite_factory_executes_interleave_and_preserves_existing_dimension_rules() {
    let mut graph = Graph::new_manual();
    graph.add_nodes(2).unwrap();
    for (strategy, width) in [
        (CombinationStrategy::Interleave { chunk_size: 1 }, 6),
        (CombinationStrategy::Concatenate, 6),
        (CombinationStrategy::WeightedAverage, 3),
    ] {
        let config = EmbeddingConfig {
            method: EmbeddingMethod::Composite {
                methods: vec![
                    (
                        EmbeddingMethod::CustomMatrix {
                            matrix: Box::new(matrix(2, 3, 0.)),
                        },
                        1.,
                    ),
                    (
                        EmbeddingMethod::CustomMatrix {
                            matrix: Box::new(matrix(2, 3, 100.)),
                        },
                        1.,
                    ),
                ],
                combination_strategy: strategy,
            },
            dimensions: 3,
            ..Default::default()
        };
        let engine = EmbeddingEngineFactory::create_engine(&config).unwrap();
        let out = engine.compute_embedding(&graph, 3).unwrap();
        assert_eq!(out.shape(), (2, width));
        assert!(engine.compute_embedding(&graph, 0).is_err());
    }
    let bad = EmbeddingConfig {
        method: EmbeddingMethod::Composite {
            methods: vec![],
            combination_strategy: CombinationStrategy::WeightedAverage,
        },
        ..Default::default()
    };
    assert!(EmbeddingEngineFactory::create_engine(&bad)
        .unwrap()
        .compute_embedding(&graph, 2)
        .is_err());
    let bad = EmbeddingConfig {
        method: EmbeddingMethod::Composite {
            methods: vec![(
                EmbeddingMethod::CustomMatrix {
                    matrix: Box::new(matrix(1, 1, 0.)),
                },
                1.,
            )],
            combination_strategy: CombinationStrategy::Interleave { chunk_size: 0 },
        },
        ..Default::default()
    };
    let error = EmbeddingEngineFactory::create_engine(&bad)
        .unwrap()
        .compute_embedding(&graph, 2)
        .unwrap_err();
    assert!(error.to_string().contains("chunk_size"));
}
#[test]
fn invalid_shapes_labels_weights_and_values_return_errors_without_panics() {
    let mut graph = Graph::new_manual();
    graph.add_nodes(2).unwrap();
    let a = matrix(2, 2, 0.);
    for strategy in [
        CombinationStrategy::Concatenate,
        CombinationStrategy::Interleave { chunk_size: 1 },
        CombinationStrategy::WeightedAverage,
    ] {
        assert!(combine_embedding_matrices(&graph, &[], None, &strategy).is_err());
        for weight in [-1., f64::NAN, f64::INFINITY] {
            assert!(combine_embedding_matrices(
                &graph,
                std::slice::from_ref(&a),
                Some(&[weight]),
                &strategy
            )
            .is_err());
        }
        assert!(
            combine_embedding_matrices(&graph, std::slice::from_ref(&a), Some(&[]), &strategy)
                .is_err()
        );
        let mut bad = a.clone();
        bad.set_row_labels(Some(vec!["node_0".into(), "node_0".into()]));
        assert!(combine_embedding_matrices(&graph, &[bad], None, &strategy).is_err());
        let mut bad = a.clone();
        bad.set(0, 0, f64::NAN).unwrap();
        assert!(combine_embedding_matrices(&graph, &[bad], None, &strategy).is_err());
        assert!(combine_embedding_matrices(&graph, &[matrix(1, 2, 0.)], None, &strategy).is_err());
    }
    assert!(combine_embedding_matrices(
        &graph,
        std::slice::from_ref(&a),
        None,
        &CombinationStrategy::Interleave { chunk_size: 0 }
    )
    .is_err());
    assert!(combine_embedding_matrices(
        &graph,
        &[a.clone(), matrix(2, 3, 0.)],
        None,
        &CombinationStrategy::WeightedAverage
    )
    .is_err());
    assert!(combine_embedding_matrices(
        &graph,
        &[a],
        Some(&[0.]),
        &CombinationStrategy::WeightedAverage
    )
    .is_err());
}
#[test]
fn weighted_average_handles_extreme_finite_weights_and_values() {
    let mut graph = Graph::new_manual();
    graph.add_node().unwrap();
    let a = GraphMatrix::from_row_major_data(vec![f64::MAX, -f64::MAX, 1e-300], 1, 3, Some(&[0]))
        .unwrap();
    let b = GraphMatrix::from_row_major_data(vec![f64::MAX, f64::MAX, 3e-300], 1, 3, None).unwrap();
    let result = combine_embedding_matrices(
        &graph,
        &[a.clone(), b],
        Some(&[f64::MAX, f64::MAX]),
        &CombinationStrategy::WeightedAverage,
    )
    .unwrap();
    assert_eq!(result.get(0, 0), Some(f64::MAX));
    assert_eq!(result.get(0, 1), Some(0.));
    assert!((result.get(0, 2).unwrap() / 1e-300 - 2.).abs() < 1e-14);
    // Sparse constructor metadata can omit feature names: it must not index-panic.
    let concatenated =
        combine_embedding_matrices(&graph, &[a], None, &CombinationStrategy::Concatenate).unwrap();
    assert_eq!(concatenated.shape(), (1, 3));
    assert_eq!(concatenated.column_names().len(), 3);
}
