//! Explicit pending snapshots are immutable read values, never accepted-state pins.
use groggy::ownership::OwnerCell;
use groggy::{
    algorithms::{
        global_registry,
        pipeline::{Pipeline, PipelineBuilder},
        AlgorithmOutput, Context,
    },
    subgraphs::Subgraph,
    AttrValue, Graph,
};
use serde_json::{json, Value};
use std::sync::Arc;

fn pipeline(steps: Value) -> anyhow::Result<Pipeline> {
    PipelineBuilder::new()
        .with_algorithm("builder.step_pipeline", |p| {
            p.set_text("steps", json!({"atomic":true,"steps":steps}).to_string());
        })
        .build(global_registry())
}
fn scope(graph: &Arc<OwnerCell<Graph>>) -> Subgraph {
    Subgraph::from_frozen_view(
        graph.clone(),
        graph.borrow().read_view().unwrap(),
        "test".into(),
    )
}
fn graph_output(
    outputs: &std::collections::HashMap<String, AlgorithmOutput>,
    name: &str,
) -> groggy::GraphView {
    let AlgorithmOutput::GraphView(view) = &outputs[name] else {
        panic!("graph output")
    };
    view.clone()
}

#[test]
fn captures_before_after_properties_and_removals_keep_distinct_immutable_values() {
    let mut graph = Graph::new_directed();
    graph.add_nodes(4).unwrap();
    graph
        .add_edges(&[(0, 1), (0, 1), (1, 2), (2, 2), (2, 3)])
        .unwrap();
    let graph = Arc::new(OwnerCell::new(graph));
    let head = graph.borrow().current_state_id();
    let steps = json!([
        {"id":"graph.capture_staged","params":{"output":"before"}},
        {"id":"core.init_nodes","params":{"target":"v","value":9007199254740993_i64}},
        {"id":"core.attach_node_attr","params":{"source":"v","attr":"exact"}},
        {"id":"graph.capture_staged","params":{"output":"props"}},
        {"id":"graph.remove_edges","params":{"ids":[0,1]}},
        {"id":"graph.capture_staged","params":{"output":"after"}},
        {"id":"alias","params":{"source":"after","target":"alias"}},
        {"id":"graph.nodes_table","params":{"source":"alias","output":"nodes"}},
        {"id":"table.select_columns","params":{"source":"nodes","columns":["exact"],"output":"values"}},
        {"id":"table.emit","params":{"source":"values","name":"values"}},
        {"id":"graph.edges_table","params":{"source":"after","output":"edges"}},
        {"id":"table.emit","params":{"source":"edges","name":"edges"}},
        {"id":"graph.emit","params":{"source":"before","name":"before"}},
        {"id":"graph.emit","params":{"source":"props","name":"props"}},
        {"id":"graph.emit","params":{"source":"after","name":"after"}}
    ]);
    let mut ctx = Context::new();
    pipeline(steps)
        .unwrap()
        .run(&mut ctx, scope(&graph))
        .unwrap();
    assert_eq!(graph.borrow().current_state_id(), head + 1);
    let outputs = ctx.take_outputs();
    let before = graph_output(&outputs, "before");
    let props = graph_output(&outputs, "props");
    let after = graph_output(&outputs, "after");
    assert_eq!(before.edge_count(), 5);
    assert_eq!(props.edge_count(), 5);
    assert_eq!(after.edge_ids(), vec![2, 3, 4]);
    assert_eq!(before.node_attr(0, &"exact".into()).unwrap(), None);
    assert_eq!(
        props.node_attr(0, &"exact".into()).unwrap(),
        Some(AttrValue::Int(9007199254740993))
    );
    assert_eq!(
        after.node_attr(0, &"exact".into()).unwrap(),
        Some(AttrValue::Int(9007199254740993))
    );
    for view in [&before, &props, &after] {
        assert_eq!(view.write_state_id(), None);
    }
    assert!(before.merge_with(&props).is_err());
    let AlgorithmOutput::Table(edges) = &outputs["edges"] else {
        panic!("table")
    };
    assert_eq!(edges.source_entity_ids(), Some([2, 3, 4].as_slice()));
    graph.borrow_mut().remove_node(2).unwrap();
    drop(outputs);
    drop(graph);
    std::thread::spawn(move || {
        assert_eq!(after.edge_ids(), vec![2, 3, 4]);
        assert_eq!(after.edge_endpoints(3).unwrap(), (2, 2));
    })
    .join()
    .unwrap();
}

#[test]
fn late_capture_or_projection_errors_restore_old_outputs_and_discard_effects() {
    let mut graph = Graph::new();
    graph.add_nodes(3).unwrap();
    let graph = Arc::new(OwnerCell::new(graph));
    let head = graph.borrow().current_state_id();
    for tail in [
        json!({"id":"graph.select_nodes","params":{"source":"pending","ids":[1],"output":"invalid"}}),
        json!({"id":"table.select_rows","params":{"source":"nodes","rows":[9],"output":"invalid"}}),
    ] {
        let mut ctx = Context::new();
        ctx.add_output("pending", AlgorithmOutput::Histogram(vec![7]));
        let mut steps = vec![
            json!({"id":"graph.remove_nodes","params":{"ids":[1]}}),
            json!({"id":"graph.capture_staged","params":{"output":"pending"}}),
            json!({"id":"graph.emit","params":{"source":"pending","name":"pending"}}),
            json!({"id":"graph.nodes_table","params":{"source":"pending","output":"nodes"}}),
        ];
        steps.push(tail);
        assert!(pipeline(json!(steps))
            .unwrap()
            .run(&mut ctx, scope(&graph))
            .is_err());
        assert_eq!(graph.borrow().current_state_id(), head);
        assert!(
            matches!(&ctx.take_outputs()["pending"],AlgorithmOutput::Histogram(v) if v==&vec![7])
        );
        pipeline(json!([{"id":"graph.capture_staged","params":{"output":"empty"}}]))
            .unwrap()
            .run(&mut ctx, scope(&graph))
            .unwrap();
        assert_eq!(graph.borrow().current_state_id(), head);
    }
}

#[test]
fn staged_graph_capture_does_not_enable_unadapted_topology_reads_or_property_effects() {
    for late in [
        json!({"id":"temporal.snapshot_at","params":{"state_id":0,"output":"neighbors"}}),
        json!({"id":"drot.smooth_strokes","params":{}}),
    ] {
        assert!(pipeline(json!([
            {"id":"graph.remove_nodes","params":{"ids":[]}},
            {"id":"graph.capture_staged","params":{"output":"pending"}},late
        ]))
        .is_err());
    }
    let mut graph = Graph::new();
    graph.add_nodes(1).unwrap();
    let graph = Arc::new(OwnerCell::new(graph));
    let pipe =
        pipeline(json!([{"id":"graph.capture_staged","params":{"output":"value"}}])).unwrap();
    let mut ctx = Context::new();
    ctx.set_persist_results(false);
    assert!(pipe.run(&mut ctx, scope(&graph)).is_err());
    assert!(pipe.run_atomic_properties(&mut ctx, scope(&graph)).is_err());
    ctx.set_persist_results(true);
    pipe.run_atomic_properties(&mut ctx, scope(&graph)).unwrap();
    assert_eq!(graph.borrow().current_state_id(), 1);
}
