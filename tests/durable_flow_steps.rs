//! Immutable flow maps and aliases share checked arithmetic and durable staging.
use groggy::{
    algorithms::{global_registry, AlgorithmOutput, Context, Pipeline, PipelineBuilder},
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
            add_nodes: 3,
            add_edges: [(0, 1), (0, 1), (1, 2), (2, 2)]
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
fn scalar_and_map_deltas_keep_exact_ids_integers_aliases_and_recovery() {
    for map_delta in [false, true] {
        let graph = fixture();
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("flow.journal");
        let mut store = DurableGraph::create_from_graph(&path, &graph, None).unwrap();
        let selected = store.read_view().unwrap().select_edges(&[0, 3]).unwrap();
        let head = selected.write_state_id().unwrap();
        let initial = (1i64 << 60) + 1;
        let capacity = initial + 9;
        let p = pipeline(json!([
            {"id":"graph.add_edges","params":{"edges":[[0,2]],"output":"new","effective_at":10}},
            {"id":"core.init_edges","params":{"value":initial,"target":"flow"}},
            {"id":"alias","params":{"source":"flow","target":"saved"}},
            {"id":"core.init_edges","params":{"value":capacity,"target":"capacity"}},
            {"id":if map_delta{"core.init_edges"}else{"core.init_scalar"},"params":{"value":1,"target":"delta"}},
            {"id":"iter.loop","params":{"iterations":2,"body":[{"id":"core.flow_update","params":{"flow":"flow","delta":"delta","target":"flow"}}]}},
            {"id":"core.residual_capacity","params":{"capacity":"capacity","flow":"flow","target":"residual"}},
            {"id":"core.attach_edge_attr","params":{"source":"flow","attr":"flow","effective_at":20}},
            {"id":"core.attach_edge_attr","params":{"source":"saved","attr":"saved","effective_at":20}},
            {"id":"core.attach_edge_attr","params":{"source":"residual","attr":"residual","effective_at":30}}
        ]));
        let plan = p.prepare_durable(&store, &selected).unwrap();
        let pending = std::thread::scope(|threads| {
            threads
                .spawn(move || plan.execute(&mut Context::new()).unwrap())
                .join()
                .unwrap()
        });
        let accepted = pending.accept(&mut store).unwrap();
        assert_eq!(accepted.edge_ids(), vec![0, 3, 4]);
        assert_eq!(accepted.write_state_id(), Some(head + 1));
        for edge in [0, 3, 4] {
            assert_eq!(
                accepted.edge_attr(edge, &"flow".into()).unwrap(),
                Some(V::Int(initial + 2))
            );
            assert_eq!(
                accepted.edge_attr(edge, &"saved".into()).unwrap(),
                Some(V::Int(initial))
            );
            assert_eq!(
                accepted.edge_attr(edge, &"residual".into()).unwrap(),
                Some(V::Int(7))
            );
        }
        assert_eq!(
            store
                .read_view()
                .unwrap()
                .edge_attr(1, &"flow".into())
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
                .select_edges(&[0, 3])
                .unwrap(),
            "test".into(),
        );
        p.run(&mut Context::new(), memory).unwrap();
        for edge in [0, 3, 4] {
            assert_eq!(
                owner.borrow().get_edge_attr(edge, &"flow".into()).unwrap(),
                accepted.edge_attr(edge, &"flow".into()).unwrap()
            );
        }
        drop(store);
        let reopened = DurableGraph::open(&path).unwrap();
        assert_eq!(
            reopened
                .read_view_at(19, None)
                .unwrap()
                .edge_attr(0, &"flow".into())
                .unwrap(),
            None
        );
        assert_eq!(
            reopened
                .read_view_at(30, None)
                .unwrap()
                .edge_attr(0, &"residual".into())
                .unwrap(),
            Some(V::Int(7))
        );
    }
}
#[test]
fn graph_and_table_aliases_keep_the_original_pin_across_pending_mutation() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("pins.journal");
    let mut store = DurableGraph::create_from_graph(&path, &fixture(), None).unwrap();
    let selected = store.read_view().unwrap();
    let p = pipeline(json!([
        {"id":"graph.capture_staged","params":{"output":"original"}},
        {"id":"alias","params":{"source":"original","target":"view"}},
        {"id":"graph.nodes_table","params":{"source":"view","output":"table"}},
        {"id":"alias","params":{"source":"table","target":"copied"}},
        {"id":"graph.add_nodes","params":{"count":1,"output":"new"}},
        {"id":"graph.emit","params":{"source":"view","name":"view"}},
        {"id":"table.emit","params":{"source":"copied","name":"table"}}
    ]));
    let mut ctx = Context::new();
    let accepted = p.run_durable(&mut ctx, &mut store, &selected).unwrap();
    assert_eq!(accepted.node_ids(), vec![0, 1, 2, 3]);
    let outputs = ctx.take_outputs();
    drop(store);
    let AlgorithmOutput::GraphView(view) = &outputs["view"] else {
        panic!("view")
    };
    assert_eq!(view.node_ids(), vec![0, 1, 2]);
    let AlgorithmOutput::Table(table) = &outputs["table"] else {
        panic!("table")
    };
    use groggy::storage::table::Table;
    assert_eq!(table.nrows(), 3);
}
#[test]
fn bad_alias_and_flow_arithmetic_restore_prior_outputs_allocations_and_journal() {
    for suffix in [
        json!({"id":"alias","params":{"source":"missing","target":"bad"}}),
        json!({"id":"core.flow_update","params":{"flow":"large","delta":"one","target":"bad"}}),
        json!({"id":"core.residual_capacity","params":{"capacity":"negative","flow":"one_map","target":"bad"}}),
    ] {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("reject.journal");
        let mut store = DurableGraph::create_from_graph(&path, &fixture(), None).unwrap();
        let selected = store.read_view().unwrap();
        let bytes = fs::read(&path).unwrap();
        let p = pipeline(json!([
            {"id":"graph.add_nodes","params":{"count":1,"output":"new"}},
            {"id":"core.init_edges","params":{"target":"large","value":i64::MAX}},
            {"id":"core.init_edges","params":{"target":"negative","value":i64::MIN}},
            {"id":"core.init_edges","params":{"target":"one_map","value":1}},
            {"id":"core.init_scalar","params":{"target":"one","value":1}},suffix
        ]));
        let mut ctx = Context::new();
        ctx.add_output("retained", AlgorithmOutput::GraphView(selected.clone()));
        assert!(p.run_durable(&mut ctx, &mut store, &selected).is_err());
        assert_eq!(ctx.take_outputs().len(), 1);
        assert_eq!(fs::read(&path).unwrap(), bytes);
        assert_eq!(store.graph().unwrap().pool().statistics().node_count, 3);
    }
}
