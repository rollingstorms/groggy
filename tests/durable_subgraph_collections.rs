//! Portable collections preserve selected identity, source pins and fixed child writes.
use groggy::{
    algorithms::{global_registry, AlgorithmOutput, Context, Pipeline, PipelineBuilder},
    subgraphs::Subgraph,
    AttrValue as V, BatchNodeRef as N, DurableGraph, Graph, GraphBatch,
};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
};
fn pipeline(steps: Value) -> Pipeline {
    PipelineBuilder::new()
        .with_algorithm("builder.step_pipeline", |p| {
            p.set_text("steps", json!({"atomic":true,"steps":steps}).to_string());
        })
        .build(global_registry())
        .unwrap()
}
fn fixture(directed: bool) -> Graph {
    let mut g = if directed {
        Graph::new_directed()
    } else {
        Graph::new()
    };
    g.apply_batch(
        GraphBatch {
            add_nodes: 6,
            add_edges: [(0, 2), (0, 2), (2, 2), (2, 4), (4, 0), (4, 5)]
                .into_iter()
                .map(|(a, b)| (N::Added(a), N::Added(b)))
                .collect(),
            effective_at: Some(0),
            ..Default::default()
        },
        0,
        "fixture".into(),
        "test".into(),
    )
    .unwrap();
    g
}
fn increment() -> Value {
    json!({"id":"iter.loop","params":{"iterations":2,"body":[
      {"id":"core.load_node_attr","params":{"attr":"counter","default":0,"target":"old"}},
      {"id":"core.init_scalar","params":{"value":1,"target":"one"}},
      {"id":"core.add","params":{"left":"old","right":"one","target":"next"}},
      {"id":"core.attach_node_attr","params":{"source":"next","attr":"counter","effective_at":20}}
    ]}})
}
#[test]
fn collection_matrix_matches_independent_membership_memory_workers_and_recovery() {
    // Explicit independent geometry after pending endpoint replacement/removal/addition.
    let edges = BTreeMap::from([
        (0, (4, 2)),
        (2, (2, 2)),
        (3, (2, 4)),
        (4, (4, 0)),
        (6, (6, 0)),
    ]);
    for directed in [false, true] {
        for edge_seeds in [false, true] {
            for hops in [0, 1] {
                for unified in [false, true] {
                    for induced in [false, true] {
                        let mut expected: Vec<(BTreeSet<usize>, BTreeSet<usize>)> = if edge_seeds {
                            edges
                                .iter()
                                .map(|(&e, &(a, b))| (BTreeSet::from([a, b]), BTreeSet::from([e])))
                                .collect()
                        } else {
                            [0, 2, 4, 6]
                                .iter()
                                .map(|&n| (BTreeSet::from([n]), BTreeSet::new()))
                                .collect()
                        };
                        for (nodes, selected_edges) in &mut expected {
                            let original = nodes.clone();
                            if hops == 1 {
                                for &(a, b) in edges.values() {
                                    if original.contains(&a) {
                                        nodes.insert(b);
                                    }
                                    if !directed && original.contains(&b) {
                                        nodes.insert(a);
                                    }
                                }
                            }
                            *selected_edges = edges
                                .iter()
                                .filter(|(_, (a, b))| nodes.contains(a) && nodes.contains(b))
                                .map(|(&e, _)| e)
                                .collect();
                        }
                        if unified {
                            expected = vec![(
                                expected
                                    .iter()
                                    .flat_map(|(n, _)| n.iter().copied())
                                    .collect(),
                                expected
                                    .iter()
                                    .flat_map(|(_, e)| e.iter().copied())
                                    .collect(),
                            )];
                        }
                        if induced {
                            for (nodes, selected_edges) in &mut expected {
                                *selected_edges = edges
                                    .iter()
                                    .filter(|(_, (a, b))| nodes.contains(a) && nodes.contains(b))
                                    .map(|(&e, _)| e)
                                    .collect();
                            }
                        }
                        let graph = fixture(directed);
                        let dir = tempfile::tempdir().unwrap();
                        let path = dir.path().join("collections.journal");
                        let mut store =
                            DurableGraph::create_from_graph(&path, &graph, None).unwrap();
                        let source = store.read_view().unwrap().select_nodes(&[0, 2, 4]).unwrap();
                        let p = pipeline(json!([
                          {"id":"graph.add_nodes","params":{"count":1,"output":"new","effective_at":10}},
                          {"id":"graph.add_edges","params":{"edges":[[{"added":0},0]],"output":"edge","effective_at":10}},
                          {"id":"graph.set_edge_endpoints","params":{"revisions":[[0,4,2]],"effective_at":10}},
                          {"id":"graph.remove_edges","params":{"ids":[1],"effective_at":10}},
                          {"id":if edge_seeds {"sample.iterate_edges"} else {"sample.iterate_nodes"},"params":{"target":"seeds"}},
                          {"id":"alias","params":{"source":"seeds","target":"saved"}},
                          {"id":"sample.neighbors","params":{"source":"seeds","hops":hops,"target":"neighbors"}},
                          {"id":"sample.emit_subgraphs","params":{"source":"neighbors","target":"emitted","mode":if unified {"unified"} else {"per_item"},"induced":induced}},
                          {"id":"sample.for_each","params":{"source":"emitted","target":"mapped","body":[increment(),
                             {"id":"graph.capture_staged","params":{"output":"pin"}},
                             {"id":"graph.emit","params":{"source":"pin","name":"child"}}
                          ]}},
                          {"id":"sample.for_each","params":{"source":"saved","target":"preserved","body":[]}}
                        ]));
                        let plan = p.prepare_durable(&store, &source).unwrap();
                        let (pending, outputs) = std::thread::scope(|s| {
                            s.spawn(move || {
                                let mut ctx = Context::new();
                                let pending = plan.execute(&mut ctx).unwrap();
                                (pending, ctx.take_outputs())
                            })
                            .join()
                            .unwrap()
                        });
                        assert_eq!(
                            store
                                .read_view()
                                .unwrap()
                                .node_attr(0, &"counter".into())
                                .unwrap(),
                            None
                        );
                        let result = pending.accept(&mut store).unwrap();
                        let mut totals = BTreeMap::from([(0, 0i64), (2, 0), (4, 0), (6, 0)]);
                        for (index, (nodes, selected_edges)) in expected.iter().enumerate() {
                            for n in nodes {
                                *totals.get_mut(n).unwrap() += 2;
                            }
                            let AlgorithmOutput::GraphView(view) =
                                &outputs[&format!("for_each.mapped.{index}.child")]
                            else {
                                panic!("graph view")
                            };
                            assert_eq!(
                                view.node_ids().into_iter().collect::<BTreeSet<_>>(),
                                *nodes
                            );
                            assert_eq!(
                                view.edge_ids().into_iter().collect::<BTreeSet<_>>(),
                                *selected_edges
                            );
                            for n in nodes {
                                assert_eq!(
                                    view.node_attr(*n, &"counter".into()).unwrap(),
                                    Some(V::Int(totals[n]))
                                );
                            }
                        }
                        for (&n, &count) in &totals {
                            assert_eq!(
                                result.node_attr(n, &"counter".into()).unwrap(),
                                if count == 0 {
                                    None
                                } else {
                                    Some(V::Int(count))
                                }
                            );
                        }
                        let owner = graph.into_shared();
                        let selected = Subgraph::from_nodes(
                            owner.clone(),
                            [0, 2, 4].into_iter().collect(),
                            "selected".into(),
                        )
                        .unwrap()
                        .freeze()
                        .unwrap();
                        let mut ctx = Context::new();
                        p.run(&mut ctx, selected).unwrap();
                        for (&n, &count) in &totals {
                            assert_eq!(
                                owner.borrow().get_node_attr(n, &"counter".into()).unwrap(),
                                if count == 0 {
                                    None
                                } else {
                                    Some(V::Int(count))
                                }
                            );
                        }
                        assert_eq!(
                            owner.borrow().get_node_attr(5, &"counter".into()).unwrap(),
                            None
                        );
                        drop(store);
                        let recovered = DurableGraph::open(&path).unwrap();
                        for (&n, &count) in &totals {
                            assert_eq!(
                                recovered
                                    .read_view_at(20, None)
                                    .unwrap()
                                    .node_attr(n, &"counter".into())
                                    .unwrap(),
                                if count == 0 {
                                    None
                                } else {
                                    Some(V::Int(count))
                                }
                            );
                        }
                        assert_eq!(
                            recovered
                                .read_view_at(19, None)
                                .unwrap()
                                .node_attr(0, &"counter".into())
                                .unwrap(),
                            None
                        );
                    }
                }
            }
        }
    }
}
#[test]
fn invalid_reserved_outputs_and_child_topology_reject_without_publication() {
    for body in [
        json!([{"id":"core.init_nodes","params":{"target":"__sample_output__","value":1}}]),
        json!([{"id":"iter.loop","params":{"iterations":0,"body":[{"id":"graph.add_nodes","params":{"count":1,"output":"new"}}]}}]),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rejected.journal");
        let mut store = DurableGraph::create_from_graph(&path, &fixture(false), None).unwrap();
        let source = store.read_view().unwrap();
        let head = source.write_state_id();
        let before = fs::read(&path).unwrap();
        let steps = json!([
          {"id":"graph.add_nodes","params":{"count":1,"output":"new"}},
          {"id":"sample.iterate_nodes","params":{"target":"seeds"}},
          {"id":"sample.for_each","params":{"source":"seeds","target":"out","body":body}}
        ]);
        let proposed = PipelineBuilder::new()
            .with_algorithm("builder.step_pipeline", |params| {
                params.set_text("steps", json!({"atomic":true,"steps":steps}).to_string());
            })
            .build(global_registry());
        if let Ok(p) = proposed {
            assert!(p
                .run_durable(&mut Context::new(), &mut store, &source)
                .is_err());
        }
        assert_eq!(fs::read(&path).unwrap(), before);
        assert_eq!(store.read_view().unwrap().write_state_id(), head);
    }
}
