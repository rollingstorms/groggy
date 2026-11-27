//! The edge-weight transform shares selected pending reads and checked output.
use groggy::{
    algorithms::{global_registry, Context, Pipeline, PipelineBuilder},
    subgraphs::Subgraph,
    AttrValue as V, BatchEdgeRef as E, BatchNodeRef as N, DurableGraph, Graph, GraphBatch,
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
            add_edges: [(0, 1), (0, 1), (1, 1), (1, 2), (2, 3), (0, 2)]
                .into_iter()
                .map(|(a, b)| (N::Added(a), N::Added(b)))
                .collect(),
            edge_attrs: [(
                "weight".into(),
                vec![
                    (E::Added(0), V::Int(3)),
                    (E::Added(1), V::Float64(-0.5)),
                    (E::Added(3), V::Text("pass".into())),
                    (E::Added(4), V::Int(99)),
                    (E::Added(5), V::Null),
                ],
            )]
            .into(),
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
fn edge_scaling_matches_memory_worker_pending_topology_and_effective_time_recovery() {
    for factor in [0.0, -2.0, 0.5] {
        let graph = fixture();
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("scale.journal");
        let mut store = DurableGraph::create_from_graph(&path, &graph, None).unwrap();
        let selected = store.read_view().unwrap().select_nodes(&[0, 1, 2]).unwrap();
        let head = selected.write_state_id().unwrap();
        let p = pipeline(json!([
         {"id":"graph.add_nodes","params":{"count":1,"output":"new","effective_at":10}},
         {"id":"graph.add_edges","params":{"edges":[[0,{"added":0}]],"attributes":{"weight":3.5},"output":"new_edge","effective_at":10}},
         {"id":"graph.set_edge_endpoints","params":{"revisions":[[0,0,2]],"effective_at":10}},
         {"id":"graph.remove_edges","params":{"ids":[1],"effective_at":10}},
         {"id":"iter.loop","params":{"iterations":2,"body":[{"id":"core.edge_weight_scale","params":{"attr":"weight","factor":factor,"target":"scaled"}}]}},
         {"id":"core.attach_edge_attr","params":{"source":"scaled","attr":"scaled","effective_at":20}}
        ]));
        let prepared = p.prepare_durable(&store, &selected).unwrap();
        let pending = std::thread::scope(|s| {
            s.spawn(move || prepared.execute(&mut Context::new()).unwrap())
                .join()
                .unwrap()
        });
        let result = pending.accept(&mut store).unwrap();
        assert_eq!(result.write_state_id(), Some(head + 1));
        assert_eq!(result.edge_ids(), vec![0, 2, 3, 5, 6]);
        for (edge, value) in [
            (0, V::Float64(3.0 * factor)),
            (2, V::Float64(factor)),
            (3, V::Text("pass".into())),
            (5, V::Null),
            (6, V::Float64(3.5 * factor)),
        ] {
            assert_eq!(
                result.edge_attr(edge, &"scaled".into()).unwrap(),
                Some(value)
            );
        }
        assert_eq!(
            store
                .read_view()
                .unwrap()
                .edge_attr(4, &"scaled".into())
                .unwrap(),
            None
        );
        let owner = graph.into_shared();
        let memory = Subgraph::from_frozen_view(
            owner.clone(),
            owner
                .borrow()
                .read_view()
                .unwrap()
                .select_nodes(&[0, 1, 2])
                .unwrap(),
            "test".into(),
        );
        p.run(&mut Context::new(), memory).unwrap();
        for edge in result.edge_ids() {
            assert_eq!(
                owner
                    .borrow()
                    .get_edge_attr(edge, &"scaled".into())
                    .unwrap(),
                result.edge_attr(edge, &"scaled".into()).unwrap()
            );
        }
        drop(store);
        let reopened = DurableGraph::open(&path).unwrap();
        assert_eq!(
            reopened
                .read_view_at(19, None)
                .unwrap()
                .edge_attr(0, &"scaled".into())
                .unwrap(),
            None
        );
        assert_eq!(
            reopened
                .read_view_at(20, None)
                .unwrap()
                .edge_attr(0, &"scaled".into())
                .unwrap(),
            Some(V::Float64(3.0 * factor))
        );
    }
}
#[test]
fn overflowing_edge_scale_rejects_staged_publication_and_discards_allocations() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("reject.journal");
    let mut store = DurableGraph::create_from_graph(&path, &fixture(), None).unwrap();
    let source = store.read_view().unwrap();
    let before = fs::read(&path).unwrap();
    let p = pipeline(json!([
     {"id":"graph.add_nodes","params":{"count":1,"output":"new"}},
     {"id":"core.init_edges","params":{"value":1,"target":"prior"}},
     {"id":"core.attach_edge_attr","params":{"source":"prior","attr":"tentative"}},
     {"id":"core.edge_weight_scale","params":{"attr":"weight","factor":1e308,"target":"bad"}}
    ]));
    assert!(p
        .run_durable(&mut Context::new(), &mut store, &source)
        .is_err());
    assert_eq!(fs::read(&path).unwrap(), before);
    assert_eq!(store.graph().unwrap().pool().statistics().node_count, 4);
}
#[test]
fn unattached_edge_scale_does_not_create_a_retained_state() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("read.journal");
    let mut store = DurableGraph::create_from_graph(&path, &fixture(), None).unwrap();
    let source = store.read_view().unwrap();
    let before = fs::read(&path).unwrap();
    let result=pipeline(json!([{"id":"core.edge_weight_scale","params":{"attr":"weight","factor":2.0,"target":"out"}}])).run_durable(&mut Context::new(),&mut store,&source).unwrap();
    assert_eq!(result.write_state_id(), source.write_state_id());
    assert_eq!(fs::read(&path).unwrap(), before);
}
