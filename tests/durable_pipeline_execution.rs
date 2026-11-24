//! The same sealed native kernels execute against memory and durable storage.
use groggy::{
    algorithms::{global_registry, AlgorithmOutput, Context, Pipeline, PipelineBuilder},
    storage::{array::ArrayOps, table::Table},
    subgraphs::Subgraph,
    utils::config::GraphConfig,
    AttrValue as V, BatchNodeRef as N, DurableGraph, Graph, GraphBatch,
};
use serde_json::{json, Value};
use std::{
    fs,
    sync::{atomic::AtomicBool, Arc},
};

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
            add_edges: vec![(N::Added(0), N::Added(2))],
            node_attrs: [(
                "untouched".into(),
                vec![(N::Added(3), V::Int(9_007_199_254_740_993))],
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
    g.apply_batch(
        GraphBatch {
            remove_nodes: vec![N::Existing(1)],
            effective_at: Some(0),
            ..Default::default()
        },
        g.current_state_id(),
        "sparse".into(),
        "test".into(),
    )
    .unwrap(); // Sparse membership with allocation high-water 4.
    g
}
fn program() -> Value {
    json!([
        {"id":"graph.capture_staged","params":{"output":"before"}},
        {"id":"graph.add_nodes","params":{"count":2,"output":"new","attributes":{"x":1.0000000000000002},"effective_at":10}},
        {"id":"graph.add_edges","params":{"edges":[[0,{"added":0}],[{"added":0},{"added":1}],[{"added":0},{"added":1}],[{"added":1},{"added":1}]],"output":"new_edges","effective_at":20}},
        {"id":"core.init_nodes_with_index","params":{"target":"index"}},
        {"id":"core.attach_node_attr","params":{"source":"index","attr":"ordinal","effective_at":25}},
        {"id":"graph.capture_staged","params":{"output":"pending"}},
        {"id":"graph.algorithm_table","params":{"algorithm":"bfs","source":"pending","output":"distance","start":0}},
        {"id":"table.emit","params":{"source":"distance","name":"distance"}},
        {"id":"graph.degrees_table","params":{"source":"pending","output":"degree"}},
        {"id":"table.emit","params":{"source":"degree","name":"degree"}},
        {"id":"graph.emit","params":{"source":"before","name":"before"}},
        {"id":"graph.emit","params":{"source":"pending","name":"pending"}}
    ])
}
fn table<'a>(
    outputs: &'a std::collections::HashMap<String, AlgorithmOutput>,
    name: &str,
) -> &'a groggy::storage::table::BaseTable {
    match &outputs[name] {
        AlgorithmOutput::Table(t) => t,
        _ => panic!("table"),
    }
}

#[test]
fn durable_program_matches_memory_publishes_once_and_reopens_with_exact_history() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("graph.journal");
    let initial = fixture();
    let mut store = DurableGraph::create_from_graph(&path, &initial, None).unwrap();
    let before = store.read_view().unwrap().select_nodes(&[0, 2]).unwrap();
    let head = store.graph().unwrap().current_state_id();
    let bytes = fs::metadata(&path).unwrap().len();
    let program = pipeline(program());
    let mut durable_ctx = Context::new();
    let accepted = program
        .run_durable(&mut durable_ctx, &mut store, &before)
        .unwrap();
    assert_eq!(accepted.node_ids(), vec![0, 2, 4, 5]);
    assert_eq!(accepted.edge_ids(), vec![0, 1, 2, 3, 4]);
    assert_eq!(accepted.write_state_id(), Some(head + 1));
    assert_eq!(store.graph().unwrap().current_state_id(), head + 1);
    assert!(fs::metadata(&path).unwrap().len() > bytes);
    assert_eq!(before.node_ids(), vec![0, 2]);
    assert_eq!(before.node_attr(0, &"ordinal".into()).unwrap(), None);
    assert_eq!(
        store
            .graph()
            .unwrap()
            .get_node_attr(3, &"ordinal".into())
            .unwrap(),
        None
    );
    let durable_outputs = durable_ctx.take_outputs();
    let AlgorithmOutput::GraphView(pending) = &durable_outputs["pending"] else {
        panic!("view")
    };
    assert_eq!(pending.write_state_id(), None);
    assert_eq!(
        pending.node_attr(4, &"x".into()).unwrap(),
        Some(V::Float64(1.0000000000000002))
    );
    let memory = initial.into_shared();
    let selected = Subgraph::from_frozen_view(
        memory.clone(),
        memory
            .borrow()
            .read_view()
            .unwrap()
            .select_nodes(&[0, 2])
            .unwrap(),
        "test".into(),
    );
    let mut memory_ctx = Context::new();
    program.run(&mut memory_ctx, selected).unwrap();
    let memory_outputs = memory_ctx.take_outputs();
    for name in ["degree", "distance"] {
        assert_eq!(
            table(&durable_outputs, name).column_names(),
            table(&memory_outputs, name).column_names()
        );
        for column in table(&durable_outputs, name).column_names() {
            assert_eq!(
                table(&durable_outputs, name)
                    .column(column)
                    .unwrap()
                    .to_vec(),
                table(&memory_outputs, name)
                    .column(column)
                    .unwrap()
                    .to_vec()
            );
        }
    }
    assert_eq!(
        accepted.node_ids(),
        memory
            .borrow()
            .read_view()
            .unwrap()
            .select_nodes(&[0, 2, 4, 5])
            .unwrap()
            .node_ids()
    );
    drop(store);
    let reopened = DurableGraph::open(&path).unwrap();
    assert_eq!(reopened.graph().unwrap().current_state_id(), head + 1);
    assert_eq!(
        reopened.read_view_at(9, None).unwrap().node_ids(),
        vec![0, 2, 3]
    );
    assert_eq!(
        reopened.read_view_at(15, None).unwrap().node_ids(),
        vec![0, 2, 3, 4, 5]
    );
    assert_eq!(reopened.read_view_at(15, None).unwrap().edge_ids(), vec![0]);
    assert_eq!(
        reopened.read_view_at(21, None).unwrap().edge_ids(),
        vec![0, 1, 2, 3, 4]
    );
    assert_eq!(
        reopened
            .read_view_at(24, None)
            .unwrap()
            .node_attr(4, &"ordinal".into())
            .unwrap(),
        None
    );
    assert_eq!(
        reopened
            .read_view_at(25, None)
            .unwrap()
            .node_attr(4, &"ordinal".into())
            .unwrap(),
        Some(V::Int(2))
    );
    assert_eq!(reopened.graph().unwrap().pool().statistics().node_count, 6);
    assert_eq!(
        accepted.node_attr(4, &"ordinal".into()).unwrap(),
        Some(V::Int(2))
    );
    assert_eq!(pending.node_ids(), vec![0, 2, 4, 5]);
}

#[test]
fn late_execution_failure_restores_outputs_journal_ids_and_scope() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("graph.journal");
    let mut store = DurableGraph::create_from_graph(&path, &fixture(), None).unwrap();
    let source = store.read_view().unwrap();
    let head = store.graph().unwrap().current_state_id();
    let bytes = fs::read(&path).unwrap();
    let mut steps = program().as_array().unwrap().clone();
    steps.push(json!({"id":"graph.algorithm_table","params":{"source":"pending","output":"bad","algorithm":"bfs","start":999}}));
    let mut ctx = Context::new();
    ctx.add_output("old", AlgorithmOutput::GraphView(source.clone()));
    assert!(pipeline(json!(steps))
        .run_durable(&mut ctx, &mut store, &source)
        .is_err());
    let outputs = ctx.take_outputs();
    assert_eq!(outputs.len(), 1);
    assert!(outputs.contains_key("old"));
    assert_eq!(fs::read(&path).unwrap(), bytes);
    assert_eq!(store.graph().unwrap().current_state_id(), head);
    assert_eq!(store.graph().unwrap().pool().statistics().node_count, 4);
    pipeline(program())
        .run_durable(&mut ctx, &mut store, &source)
        .unwrap();
    assert_eq!(store.graph().unwrap().pool().statistics().node_count, 6);
}

#[test]
fn unsupported_suffix_stale_foreign_and_cancelled_programs_leave_journal_unchanged() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("graph.journal");
    let mut store = DurableGraph::create(&path, GraphConfig::default(), None).unwrap();
    let source = store.read_view().unwrap();
    let bytes = fs::read(&path).unwrap();
    let mut ctx = Context::new();
    let p = pipeline(json!([
        {"id":"graph.add_nodes","params":{"count":1,"output":"new"}},
        {"id":"core.node_degree","params":{"target":"unsupported"}}
    ]));
    assert!(p
        .run_durable(&mut ctx, &mut store, &source)
        .unwrap_err()
        .to_string()
        .contains("no detached native kernel"));
    assert_eq!(fs::read(&path).unwrap(), bytes);
    let empty = pipeline(json!([]));
    let foreign = Graph::new().read_view().unwrap();
    assert!(empty.run_durable(&mut ctx, &mut store, &foreign).is_err());
    let token = Arc::new(AtomicBool::new(true));
    ctx = Context::with_cancel_token(token.clone());
    assert!(empty.run_durable(&mut ctx, &mut store, &source).is_err());
    assert_eq!(fs::read(&path).unwrap(), bytes);
    token.store(false, std::sync::atomic::Ordering::Relaxed);
    let accepted = empty.run_durable(&mut ctx, &mut store, &source).unwrap();
    assert_eq!(accepted.write_state_id(), source.write_state_id());
    assert_eq!(fs::read(&path).unwrap(), bytes);
    pipeline(json!([{"id":"graph.add_nodes","params":{"count":1,"output":"new"}}]))
        .run_durable(&mut ctx, &mut store, &source)
        .unwrap();
    let bytes = fs::read(&path).unwrap();
    assert!(empty.run_durable(&mut ctx, &mut store, &source).is_err());
    assert_eq!(fs::read(&path).unwrap(), bytes);
}

#[test]
fn preparation_runs_off_owner_thread_and_dropped_or_cancelled_proposals_publish_nothing() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("graph.journal");
    let mut store = DurableGraph::create_from_graph(&path, &fixture(), None).unwrap();
    let source = store.read_view().unwrap();
    let bytes = fs::read(&path).unwrap();
    let p = pipeline(program());
    let plan = p.prepare_durable(&store, &source).unwrap();
    let prepared = std::thread::scope(|threads| {
        threads
            .spawn(move || plan.execute(&mut Context::new()).unwrap())
            .join()
            .unwrap()
    });
    assert_eq!(fs::read(&path).unwrap(), bytes);
    drop(prepared);
    assert_eq!(store.graph().unwrap().pool().statistics().node_count, 4);
    let token = Arc::new(AtomicBool::new(false));
    let mut ctx = Context::with_cancel_token(token.clone());
    let prepared = p
        .prepare_durable(&store, &source)
        .unwrap()
        .execute(&mut ctx)
        .unwrap();
    token.store(true, std::sync::atomic::Ordering::Relaxed);
    assert!(prepared
        .accept(&mut store)
        .unwrap_err()
        .to_string()
        .contains("cancelled"));
    assert_eq!(fs::read(&path).unwrap(), bytes);
}

#[test]
fn acceptance_rechecks_latest_schema_and_sibling_allocation_even_at_original_head() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("graph.journal");
    let mut store = DurableGraph::create_from_graph(&path, &fixture(), None).unwrap();
    let source = store.read_view().unwrap();
    let head = store.graph().unwrap().current_state_id();
    let p = pipeline(json!([
        {"id":"graph.add_nodes","params":{"count":1,"output":"new","attributes":{"labels":["Entity"],"x":1}}}
    ]));
    let prepared = p
        .prepare_durable(&store, &source)
        .unwrap()
        .execute(&mut Context::new())
        .unwrap();
    let schema = groggy::GraphSchema::from_json(
        r#"{"node_labels":{"Entity":{"x":{"type":"text","required":false}}}}"#,
    )
    .unwrap();
    store.set_schema(Some(schema), head, 0).unwrap();
    let bytes = fs::read(&path).unwrap();
    assert!(prepared.accept(&mut store).is_err());
    assert_eq!(fs::read(&path).unwrap(), bytes);
    store.set_schema(None, head, 1).unwrap();
    let prepared = p
        .prepare_durable(&store, &source)
        .unwrap()
        .execute(&mut Context::new())
        .unwrap();
    store.create_branch("sibling".into(), head).unwrap();
    store.checkout_branch("sibling".into(), head).unwrap();
    store
        .apply_batch(
            GraphBatch {
                add_nodes: 1,
                ..Default::default()
            },
            head,
            "sibling".into(),
            "test".into(),
        )
        .unwrap();
    let other = store.graph().unwrap().current_state_id();
    store.checkout_branch("main".into(), other).unwrap();
    assert_eq!(store.graph().unwrap().current_state_id(), head);
    let bytes = fs::read(&path).unwrap();
    assert!(prepared
        .accept(&mut store)
        .unwrap_err()
        .to_string()
        .contains("allocation coordinates"));
    assert_eq!(fs::read(&path).unwrap(), bytes);
    assert!(!store.graph().unwrap().has_node(4));
}

#[test]
fn all_thirteen_captured_kernels_match_memory_on_the_same_pending_selection() {
    let variants = [
        ("bfs", json!({"start":0})),
        ("dfs", json!({"start":0})),
        ("dijkstra", json!({"start":0})),
        ("astar", json!({"start":0,"goal":3})),
        ("pagerank", json!({})),
        ("betweenness", json!({})),
        ("closeness", json!({})),
        ("components", json!({})),
        ("louvain", json!({})),
        ("leiden", json!({"seed":7})),
        ("infomap", json!({"seed":7,"num_trials":2})),
        ("girvan_newman", json!({"num_levels":2})),
        ("lpa", json!({"seed":7})),
    ];
    for (algorithm, mut options) in variants {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("graph.journal");
        let mut graph = Graph::new_publishing();
        graph.insert_nodes(3).unwrap();
        graph.add_edges(&[(0, 1), (1, 2), (1, 2), (2, 2)]).unwrap();
        let mut store = DurableGraph::create_from_graph(&path, &graph, None).unwrap();
        let source = store.read_view().unwrap();
        options["algorithm"] = json!(algorithm);
        options["source"] = json!("pending");
        options["output"] = json!("answer");
        let p = pipeline(json!([
            {"id":"graph.add_nodes","params":{"count":1,"output":"new"}},
            {"id":"graph.add_edges","params":{"edges":[[2,{"added":0}]],"output":"edge"}},
            {"id":"graph.capture_staged","params":{"output":"pending"}},
            {"id":"graph.algorithm_table","params":options},
            {"id":"table.emit","params":{"source":"answer","name":"answer"}}
        ]));
        let mut durable_ctx = Context::new();
        p.run_durable(&mut durable_ctx, &mut store, &source)
            .unwrap_or_else(|e| panic!("{algorithm}: {e}"));
        let owner = graph.into_shared();
        let selected = Subgraph::from_frozen_view(
            owner.clone(),
            owner.borrow().read_view().unwrap(),
            "test".into(),
        );
        let mut memory_ctx = Context::new();
        p.run(&mut memory_ctx, selected).unwrap();
        let durable = durable_ctx.take_outputs();
        let memory = memory_ctx.take_outputs();
        let actual = table(&durable, "answer");
        let expected = table(&memory, "answer");
        assert_eq!(
            actual.column_names(),
            expected.column_names(),
            "{algorithm}"
        );
        for name in actual.column_names() {
            assert_eq!(
                actual.column(name).unwrap().to_vec(),
                expected.column(name).unwrap().to_vec(),
                "{algorithm}:{name}"
            );
        }
    }
}

#[test]
fn captured_computation_without_effects_returns_outputs_without_recording_history() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("graph.journal");
    let mut store = DurableGraph::create_from_graph(&path, &fixture(), None).unwrap();
    let selected = store.read_view().unwrap();
    let bytes = fs::read(&path).unwrap();
    let p = pipeline(json!([
        {"id":"graph.capture_staged","params":{"output":"pending"}},
        {"id":"graph.algorithm_table","params":{"source":"pending","algorithm":"bfs","start":0,"output":"answer"}},
        {"id":"table.emit","params":{"source":"answer","name":"answer"}}
    ]));
    let mut ctx = Context::new();
    let output = p.run_durable(&mut ctx, &mut store, &selected).unwrap();
    assert_eq!(output.write_state_id(), selected.write_state_id());
    assert_eq!(fs::read(&path).unwrap(), bytes);
    assert!(ctx.take_outputs().contains_key("answer"));
}
