//! Seeded mask samplers use the same selected population on memory and workers.
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
    let mut graph = Graph::new_directed();
    graph
        .apply_batch(
            GraphBatch {
                add_nodes: 6,
                add_edges: [(0, 2), (0, 2), (2, 2), (2, 4), (4, 5)]
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
    graph
}
#[test]
fn seeded_masks_match_memory_workers_pending_rows_and_recovery() {
    for seed in [0, 1, 42, 999] {
        let graph = fixture();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sample.journal");
        let mut store = DurableGraph::create_from_graph(&path, &graph, None).unwrap();
        let selected = store.read_view().unwrap().select_nodes(&[0, 2, 4]).unwrap();
        let head = selected.write_state_id().unwrap();
        let body = json!([
            {"id":"core.sample_nodes","params":{"target":"nodes","count":3,"seed":seed}},
            {"id":"core.sample_edges","params":{"target":"edges","fraction":0.5,"seed":seed}},
            {"id":"core.reservoir_sample","params":{"source":"nodes","target":"node_reservoir","k":2,"seed":seed}},
            {"id":"core.reservoir_sample","params":{"source":"edges","target":"edge_reservoir","entity_type":"edges","k":99,"seed":seed}},
            {"id":"core.attach_node_attr","params":{"source":"nodes","attr":"sample","effective_at":20}},
            {"id":"core.attach_node_attr","params":{"source":"node_reservoir","attr":"reservoir","effective_at":20}},
            {"id":"core.attach_edge_attr","params":{"source":"edges","attr":"sample","effective_at":20}},
            {"id":"core.attach_edge_attr","params":{"source":"edge_reservoir","attr":"reservoir","effective_at":20}}
        ]);
        let p = pipeline(json!([
            {"id":"graph.add_nodes","params":{"count":1,"output":"new","effective_at":10}},
            {"id":"graph.add_edges","params":{"edges":[[4,{"added":0}]],"output":"new_edges","effective_at":10}},
            {"id":"iter.loop","params":{"iterations":2,"body":body}}
        ]));
        let plan = p.prepare_durable(&store, &selected).unwrap();
        let pending = std::thread::scope(|s| {
            s.spawn(move || plan.execute(&mut Context::new()).unwrap())
                .join()
                .unwrap()
        });
        let accepted = pending.accept(&mut store).unwrap();
        assert_eq!(accepted.write_state_id(), Some(head + 1));
        let sampled_nodes: Vec<_> = accepted
            .node_ids()
            .into_iter()
            .filter(|&n| accepted.node_attr(n, &"sample".into()).unwrap() == Some(V::Int(1)))
            .collect();
        assert_eq!(sampled_nodes.len(), 3);
        assert!(sampled_nodes.iter().all(|n| [0, 2, 4, 6].contains(n)));
        assert_eq!(
            accepted
                .node_ids()
                .into_iter()
                .filter(|&n| accepted
                    .node_attr(n, &"reservoir".into())
                    .unwrap()
                    .is_some())
                .count(),
            2
        );
        // Five selected edge instances: Rust round(2.5) = 3, with parallel/loop instances distinct.
        let sampled_edges: Vec<_> = accepted
            .edge_ids()
            .into_iter()
            .filter(|&e| accepted.edge_attr(e, &"sample".into()).unwrap().is_some())
            .collect();
        assert_eq!(sampled_edges.len(), 3);
        for e in &sampled_edges {
            assert_eq!(
                accepted.edge_attr(*e, &"reservoir".into()).unwrap(),
                Some(V::Int(1))
            );
        }
        let owner = graph.into_shared();
        let memory = Subgraph::from_frozen_view(
            owner.clone(),
            owner
                .borrow()
                .read_view()
                .unwrap()
                .select_nodes(&[0, 2, 4])
                .unwrap(),
            "test".into(),
        );
        p.run(&mut Context::new(), memory).unwrap();
        for n in accepted.node_ids() {
            for attr in ["sample", "reservoir"] {
                assert_eq!(
                    owner.borrow().get_node_attr(n, &attr.into()).unwrap(),
                    accepted.node_attr(n, &attr.into()).unwrap()
                );
            }
        }
        for e in accepted.edge_ids() {
            for attr in ["sample", "reservoir"] {
                assert_eq!(
                    owner.borrow().get_edge_attr(e, &attr.into()).unwrap(),
                    accepted.edge_attr(e, &attr.into()).unwrap()
                );
            }
        }
        assert_eq!(
            store
                .read_view()
                .unwrap()
                .node_attr(5, &"sample".into())
                .unwrap(),
            None
        );
        assert_eq!(
            store
                .read_view()
                .unwrap()
                .edge_attr(4, &"sample".into())
                .unwrap(),
            None
        );
        drop(store);
        let reopened = DurableGraph::open(&path).unwrap();
        for n in sampled_nodes {
            assert_eq!(
                reopened
                    .read_view_at(19, None)
                    .unwrap()
                    .node_attr(n, &"sample".into())
                    .unwrap(),
                None
            );
            assert_eq!(
                reopened
                    .read_view_at(20, None)
                    .unwrap()
                    .node_attr(n, &"sample".into())
                    .unwrap(),
                Some(V::Int(1))
            );
        }
    }
}
#[test]
fn invalid_late_sampling_discards_pending_ids_and_journal_changes() {
    for bad in [
        json!({"id":"core.sample_nodes","params":{"count":99,"target":"bad"}}),
        json!({"id":"core.sample_edges","params":{"count":99,"target":"bad"}}),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("reject.journal");
        let mut store = DurableGraph::create_from_graph(&path, &fixture(), None).unwrap();
        let selected = store.read_view().unwrap();
        let before = fs::read(&path).unwrap();
        let p = pipeline(json!([{"id":"graph.add_nodes","params":{"count":1,"output":"new"}},bad]));
        assert!(p
            .run_durable(&mut Context::new(), &mut store, &selected)
            .is_err());
        assert_eq!(fs::read(&path).unwrap(), before);
        assert_eq!(store.graph().unwrap().pool().statistics().node_count, 6);
    }
}
#[test]
fn masks_without_attachments_do_not_publish_history() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("read.journal");
    let mut store = DurableGraph::create_from_graph(&path, &fixture(), None).unwrap();
    let selected = store.read_view().unwrap();
    let before = fs::read(&path).unwrap();
    let p = pipeline(json!([
        {"id":"core.sample_nodes","params":{"count":0,"target":"empty","seed":1}},
        {"id":"core.sample_edges","params":{"fraction":1.0,"target":"all","seed":1}},
        {"id":"core.reservoir_sample","params":{"source":"empty","target":"still_empty","k":99,"seed":1}}
    ]));
    let result = p
        .run_durable(&mut Context::new(), &mut store, &selected)
        .unwrap();
    assert_eq!(result.write_state_id(), selected.write_state_id());
    assert_eq!(fs::read(&path).unwrap(), before);
}
