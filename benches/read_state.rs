//! Bounded current-state capture versus shared read-pin clone scaling.
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use groggy::{storage::table::Table, AttrValue, Graph};
use std::time::Duration;
fn read_state(c: &mut Criterion) {
    let mut group = c.benchmark_group("read_state");
    for count in [1_000, 10_000] {
        let mut graph = Graph::new();
        let nodes = graph.add_nodes(count);
        for (index, &node) in nodes.iter().enumerate() {
            graph.add_edge(node, nodes[(index + 1) % count]).unwrap();
        }
        graph
            .set_node_attrs(std::collections::HashMap::from([(
                "value".into(),
                nodes
                    .iter()
                    .map(|&id| (id, AttrValue::Int(id as i64)))
                    .collect(),
            )]))
            .unwrap();
        let pin = graph.pin_current_state().unwrap();
        group.bench_with_input(BenchmarkId::new("capture", count), &graph, |b, graph| {
            b.iter(|| black_box(graph.pin_current_state().unwrap()));
        });
        group.bench_with_input(BenchmarkId::new("clone", count), &pin, |b, pin| {
            b.iter(|| black_box(pin.clone()));
        });
        group.bench_with_input(BenchmarkId::new("full_view", count), &pin, |b, pin| {
            b.iter(|| black_box(groggy::GraphView::all(pin.clone())));
        });
        let view = groggy::GraphView::all(pin.clone());
        group.bench_with_input(
            BenchmarkId::new("live_features", count),
            &graph,
            |b, graph| {
                b.iter(|| black_box(graph.to_matrix::<f64>().unwrap()));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("frozen_features", count),
            &view,
            |b, view| {
                b.iter(|| black_box(view.to_matrix::<f64>().unwrap()));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("frozen_sparse_adjacency", count),
            &view,
            |b, view| {
                b.iter(|| black_box(view.adjacency_matrix::<f64>(true, None).unwrap()));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("live_tables", count),
            &graph,
            |b, graph| {
                b.iter(|| black_box(graph.table().unwrap()));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("frozen_tables", count),
            &view,
            |b, view| {
                b.iter(|| black_box(view.table().unwrap()));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("algorithm_prepare", count),
            &view,
            |b, view| {
                b.iter(|| black_box(view.algorithm_input().unwrap()));
            },
        );
        let algorithm_input = view.algorithm_input().unwrap();
        group.bench_with_input(
            BenchmarkId::new("algorithm_clone", count),
            &algorithm_input,
            |b, input| {
                b.iter(|| black_box(input.clone()));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("prepared_bfs", count),
            &algorithm_input,
            |b, input| {
                b.iter(|| black_box(input.bfs_distances(nodes[0], None).unwrap()));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("prepared_dfs", count),
            &algorithm_input,
            |b, input| {
                b.iter(|| black_box(input.dfs_order(nodes[0], None).unwrap()));
            },
        );
        group.bench_with_input(BenchmarkId::new("frozen_bfs", count), &view, |b, view| {
            b.iter(|| black_box(view.bfs_distances(nodes[0]).unwrap()));
        });
        group.bench_with_input(
            BenchmarkId::new("prepared_dijkstra", count),
            &algorithm_input,
            |b, input| {
                b.iter(|| black_box(input.dijkstra_distances(nodes[0], None, None).unwrap()));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("frozen_dijkstra", count),
            &view,
            |b, view| {
                b.iter(|| black_box(view.dijkstra_distances(nodes[0], None).unwrap()));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("prepared_astar", count),
            &algorithm_input,
            |b, input| {
                b.iter(|| {
                    black_box(
                        input
                            .astar_path(nodes[0], nodes[count / 2], None, None, None)
                            .unwrap(),
                    )
                });
            },
        );
        group.bench_with_input(BenchmarkId::new("frozen_astar", count), &view, |b, view| {
            b.iter(|| {
                black_box(
                    view.astar_path(nodes[0], nodes[count / 2], None, None)
                        .unwrap(),
                )
            });
        });
        for (name, mode) in [
            (
                "components_weak",
                groggy::algorithms::community::ComponentMode::Weak,
            ),
            (
                "components_strong",
                groggy::algorithms::community::ComponentMode::Strong,
            ),
        ] {
            group.bench_with_input(
                BenchmarkId::new(format!("prepared_{name}"), count),
                &algorithm_input,
                |b, input| b.iter(|| black_box(input.connected_components(mode, None).unwrap())),
            );
            group.bench_with_input(
                BenchmarkId::new(format!("frozen_{name}"), count),
                &view,
                |b, view| b.iter(|| black_box(view.connected_components(mode).unwrap())),
            );
        }
        for (name, personalization) in [
            ("pagerank_uniform", None),
            ("pagerank_personalized", Some("value".to_string())),
        ] {
            let algorithm = groggy::algorithms::centrality::PageRank::new(
                0.85,
                50,
                1e-6,
                personalization.clone(),
                "unused".into(),
            )
            .unwrap();
            group.bench_with_input(
                BenchmarkId::new(format!("prepared_{name}"), count),
                &algorithm_input,
                |b, input| b.iter(|| black_box(input.pagerank(&algorithm, None).unwrap())),
            );
            group.bench_with_input(
                BenchmarkId::new(format!("frozen_{name}"), count),
                &view,
                |b, view| {
                    b.iter(|| {
                        black_box(
                            view.pagerank(0.85, 50, 1e-6, personalization.clone())
                                .unwrap(),
                        )
                    })
                },
            );
        }
        // Bound all-source centrality timing to 100/1000-node induced paths.
        let centrality_view = view.select_nodes(&nodes[..count / 10]).unwrap();
        let centrality_input = centrality_view.algorithm_input().unwrap();
        let closeness =
            groggy::algorithms::centrality::ClosenessCentrality::new(true, None, "unused".into());
        group.bench_with_input(
            BenchmarkId::new("prepared_closeness", count / 10),
            &centrality_input,
            |b, input| b.iter(|| black_box(input.closeness(&closeness, None).unwrap())),
        );
        group.bench_with_input(
            BenchmarkId::new("frozen_closeness", count / 10),
            &centrality_view,
            |b, view| {
                b.iter(|| {
                    black_box(
                        view.closeness(
                            true,
                            None,
                            groggy::algorithms::centrality::ClosenessDirection::Outgoing,
                            false,
                        )
                        .unwrap(),
                    )
                })
            },
        );
        for (name, weight_attr) in [
            ("betweenness", None),
            ("betweenness_weighted", Some("length".to_string())),
        ] {
            let algorithm = groggy::algorithms::centrality::BetweennessCentrality::new(
                true,
                weight_attr.clone(),
                "unused".into(),
            );
            group.bench_with_input(
                BenchmarkId::new(format!("prepared_{name}"), count / 10),
                &centrality_input,
                |b, input| b.iter(|| black_box(input.betweenness(&algorithm, None).unwrap())),
            );
            group.bench_with_input(
                BenchmarkId::new(format!("frozen_{name}"), count / 10),
                &centrality_view,
                |b, view| {
                    b.iter(|| black_box(view.betweenness(true, weight_attr.clone()).unwrap()))
                },
            );
        }
        for (name, value) in [
            (
                "capture_f32_vector",
                AttrValue::FloatVec((0..count).map(|i| i as f32 / count as f32).collect()),
            ),
            (
                "capture_f64_vector",
                AttrValue::Float64Vec((0..count).map(|i| i as f64 / count as f64).collect()),
            ),
        ] {
            let mut property_graph = Graph::new();
            let node = property_graph.add_node();
            property_graph
                .set_node_attr(node, "embedding".into(), value)
                .unwrap();
            group.bench_with_input(
                BenchmarkId::new(name, count),
                &property_graph,
                |b, graph| b.iter(|| black_box(graph.pin_current_state().unwrap())),
            );
        }
        let predicate_values: Vec<_> = (0..count)
            .map(|i| AttrValue::Float64(1.0 + i as f64 * 2f64.powi(-40)))
            .collect();
        let evaluator = groggy::storage::array::QueryEvaluator::new("value > 1.0000000005");
        assert_eq!(
            predicate_values
                .iter()
                .filter(|value| evaluator.evaluate(value).unwrap())
                .count(),
            count - 550
        );
        group.bench_with_input(
            BenchmarkId::new("array_f64_predicate", count),
            &predicate_values,
            |b, values| {
                b.iter(|| {
                    black_box(
                        values
                            .iter()
                            .filter(|value| evaluator.evaluate(value).unwrap())
                            .count(),
                    )
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("table_f64_predicate", count),
            &predicate_values,
            |b, values| {
                b.iter(|| {
                    black_box(
                        values
                            .iter()
                            .filter(|value| {
                                groggy::storage::table::BaseTable::compare_attr_value(
                                    value,
                                    ">",
                                    "1.0000000005",
                                )
                                .unwrap()
                            })
                            .count(),
                    )
                })
            },
        );
        let numeric_column = groggy::storage::array::BaseArray::new(predicate_values.clone());
        let numeric_tables = groggy::storage::table::TableArray::from_tables(
            predicate_values
                .chunks(count / 10)
                .map(|values| {
                    groggy::storage::table::BaseTable::from_columns(
                        std::collections::HashMap::from([(
                            "x".into(),
                            groggy::storage::array::BaseArray::new(values.to_vec()),
                        )]),
                    )
                    .unwrap()
                })
                .collect(),
        );
        let expected_mean = 1.0 + (count - 1) as f64 * 2f64.powi(-41);
        assert!((numeric_column.mean().unwrap() - expected_mean).abs() < 1e-14);
        assert!((numeric_tables.mean_column("x").unwrap() - expected_mean).abs() < 1e-14);
        group.bench_with_input(
            BenchmarkId::new("attribute_f64_mean", count),
            &numeric_column,
            |b, column| b.iter(|| black_box(column.mean().unwrap())),
        );
        group.bench_with_input(
            BenchmarkId::new("attribute_f64_sum", count),
            &numeric_column,
            |b, column| b.iter(|| black_box(column.sum().unwrap())),
        );
        group.bench_with_input(
            BenchmarkId::new("tables_f64_mean", count),
            &numeric_tables,
            |b, tables| b.iter(|| black_box(tables.mean_column("x").unwrap())),
        );
        group.bench_with_input(
            BenchmarkId::new("tables_f64_sum", count),
            &numeric_tables,
            |b, tables| b.iter(|| black_box(tables.sum_column("x").unwrap())),
        );
        for (name, values) in [
            ("profile_f64", predicate_values.clone()),
            ("profile_high_integer", (0..count).map(|i| AttrValue::Int(i64::MAX - count as i64 + i as i64)).collect()),
        ] {
            let table = groggy::storage::table::BaseTable::from_columns(
                std::collections::HashMap::from([("x".into(), groggy::storage::array::BaseArray::new(values))]),
            ).unwrap();
            let result = table.profile().unwrap();
            assert!(matches!(result.column("mean").unwrap().data()[0], AttrValue::Float64(v) if v.is_finite()));
            group.bench_with_input(BenchmarkId::new(name, count), &table, |b, table| {
                b.iter(|| black_box(table.profile().unwrap()));
            });
        }
        let selected = groggy::GraphView::all(pin.clone())
            .select_nodes(&nodes[..count / 2])
            .unwrap();
        group.bench_with_input(
            BenchmarkId::new("selected_clone", count),
            &selected,
            |b, view| {
                b.iter(|| black_box(view.clone()));
            },
        );
        let live = groggy::subgraphs::Subgraph::new(
            std::rc::Rc::new(std::cell::RefCell::new(graph)),
            nodes.iter().copied().collect(),
            pin.existence().edges().clone(),
            "query_fixture".into(),
        );
        let attribute = groggy::query::NodeFilter::AttributeFilter {
            name: "value".into(),
            filter: groggy::query::AttributeFilter::GreaterThan(AttrValue::Int((count / 2) as i64)),
        };
        let degree = groggy::query::NodeFilter::DegreeRange { min: 2, max: 2 };
        for (name, predicate) in [("attribute_query", attribute), ("degree_query", degree)] {
            group.bench_with_input(
                BenchmarkId::new(format!("live_{name}"), count),
                &live,
                |b, live| {
                    b.iter(|| black_box(live.filter_nodes(&predicate).unwrap()));
                },
            );
            group.bench_with_input(
                BenchmarkId::new(format!("frozen_{name}"), count),
                &view,
                |b, view| {
                    b.iter(|| black_box(view.filter_nodes(&predicate).unwrap()));
                },
            );
        }
    }
    group.finish();
}
criterion_group! {name=benches;config=Criterion::default().sample_size(10).warm_up_time(Duration::from_millis(200)).measurement_time(Duration::from_secs(1));targets=read_state}
criterion_main!(benches);
