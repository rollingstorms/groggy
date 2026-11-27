//! Decay transforms owned selected maps/columns, not retained history.
use groggy::{
    algorithms::{global_registry, Context, Pipeline, PipelineBuilder},
    subgraphs::Subgraph,
    AttrValue as V, BatchNodeRef as N, DurableGraph, Graph, GraphBatch,
};
use serde_json::{json, Value};
use std::fs;
fn pipeline(steps: Value) -> Pipeline {
    PipelineBuilder::new()
        .with_algorithm("builder.step_pipeline", |p| {
            p.set_text("steps", json!({"atomic":true,"steps":steps}).to_string());
        })
        .build(global_registry())
        .unwrap()
}
fn fixture() -> Graph {
    let mut g = Graph::new_directed();
    g.apply_batch(
        GraphBatch {
            add_nodes: 4,
            add_edges: [(0, 2), (0, 2), (2, 2), (2, 3)]
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
#[test]
fn decay_layouts_match_workers_memory_pending_selections_and_recovery() {
    for layout in ["nodes", "column", "edges"] {
        let graph = fixture();
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("decay.journal");
        let mut store = DurableGraph::create_from_graph(&path, &graph, None).unwrap();
        let selected = store.read_view().unwrap().select_nodes(&[0, 2]).unwrap();
        let head = selected.write_state_id().unwrap();
        let load = match layout {
            "column" => {
                json!({"id":"core.community_seed","params":{"strategy":"singleton","target":"values"}})
            }
            "edges" => json!({"id":"core.init_edges","params":{"value":8,"target":"values"}}),
            _ => json!({"id":"core.load_node_attr","params":{"attr":"node_id","target":"values"}}),
        };
        let attach = if layout == "edges" {
            "core.attach_edge_attr"
        } else {
            "core.attach_node_attr"
        };
        let p = pipeline(json!([
         {"id":"graph.add_nodes","params":{"count":1,"output":"new","effective_at":10}},
         {"id":"graph.add_edges","params":{"edges":[[2,{"added":0}]],"output":"new_edge","effective_at":10}},
         {"id":"graph.remove_edges","params":{"ids":[1],"effective_at":10}},load,
         {"id":"alias","params":{"source":"values","target":"saved"}},
         {"id":"iter.loop","params":{"iterations":2,"body":[{"id":"temporal.decay","params":{"attr":"values","output":"values","half_life":2.0,"time_delta":2.0}}]}},
         {"id":attach,"params":{"source":"values","attr":"result","effective_at":20}},
         {"id":attach,"params":{"source":"saved","attr":"original","effective_at":20}}
        ]));
        let plan = p.prepare_durable(&store, &selected).unwrap();
        let pending = std::thread::scope(|s| {
            s.spawn(move || plan.execute(&mut Context::new()).unwrap())
                .join()
                .unwrap()
        });
        let accepted = pending.accept(&mut store).unwrap();
        assert_eq!(accepted.write_state_id(), Some(head + 1));
        let ids = if layout == "edges" {
            accepted.edge_ids()
        } else {
            accepted.node_ids()
        };
        for id in &ids {
            let expected = if layout == "edges" {
                2.0
            } else {
                *id as f64 / 4.0
            };
            assert_eq!(
                if layout == "edges" {
                    accepted.edge_attr(*id, &"result".into()).unwrap()
                } else {
                    accepted.node_attr(*id, &"result".into()).unwrap()
                },
                Some(V::Float64(expected))
            );
        }
        let owner = graph.into_shared();
        let memory = Subgraph::from_frozen_view(
            owner.clone(),
            owner
                .borrow()
                .read_view()
                .unwrap()
                .select_nodes(&[0, 2])
                .unwrap(),
            "test".into(),
        );
        p.run(&mut Context::new(), memory).unwrap();
        for id in &ids {
            for attr in ["result", "original"] {
                if layout == "edges" {
                    assert_eq!(
                        owner.borrow().get_edge_attr(*id, &attr.into()).unwrap(),
                        accepted.edge_attr(*id, &attr.into()).unwrap()
                    );
                } else {
                    assert_eq!(
                        owner.borrow().get_node_attr(*id, &attr.into()).unwrap(),
                        accepted.node_attr(*id, &attr.into()).unwrap()
                    );
                }
            }
        }
        drop(store);
        let reopened = DurableGraph::open(&path).unwrap();
        assert_eq!(
            reopened
                .read_view_at(19, None)
                .unwrap()
                .node_attr(0, &"result".into())
                .unwrap(),
            None
        );
        for id in ids {
            let after = reopened.read_view_at(20, None).unwrap();
            assert_eq!(
                if layout == "edges" {
                    after.edge_attr(id, &"result".into()).unwrap()
                } else {
                    after.node_attr(id, &"result".into()).unwrap()
                },
                Some(V::Float64(if layout == "edges" {
                    2.0
                } else {
                    id as f64 / 4.0
                }))
            );
        }
    }
}
#[test]
fn removed_source_ids_reject_even_at_zero_elapsed_without_consuming_allocations() {
    for edges in [false, true] {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("reject.journal");
        let mut store = DurableGraph::create_from_graph(&path, &fixture(), None).unwrap();
        let source = store.read_view().unwrap();
        let before = fs::read(&path).unwrap();
        let p = pipeline(json!([
         {"id":if edges{"core.init_edges"}else{"core.init_nodes"},"params":{"value":8,"target":"old"}},
         {"id":"graph.add_nodes","params":{"count":1,"output":"new"}},
         {"id":if edges{"graph.remove_edges"}else{"graph.remove_nodes"},"params":{"ids":[0]}},
         {"id":"temporal.decay","params":{"attr":"old","output":"result","half_life":2.0,"time_delta":0.0}}
        ]));
        assert!(p
            .run_durable(&mut Context::new(), &mut store, &source)
            .is_err());
        assert_eq!(fs::read(&path).unwrap(), before);
        assert_eq!(store.graph().unwrap().pool().statistics().node_count, 4);
    }
}
#[test]
fn unattached_decay_keeps_history_unchanged() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("read.journal");
    let mut store = DurableGraph::create_from_graph(&path, &fixture(), None).unwrap();
    let source = store.read_view().unwrap();
    let before = fs::read(&path).unwrap();
    let p = pipeline(
        json!([{"id":"core.init_nodes","params":{"value":8,"target":"values"}},{"id":"temporal.decay","params":{"attr":"values","output":"result","half_life":2.0,"time_delta":2.0}}]),
    );
    let result = p
        .run_durable(&mut Context::new(), &mut store, &source)
        .unwrap();
    assert_eq!(result.write_state_id(), source.write_state_id());
    assert_eq!(fs::read(&path).unwrap(), before);
}
