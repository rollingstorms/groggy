//! Typed captured graph/table values compose without a temporary writable graph.
use groggy::ownership::OwnerCell;
use groggy::{
    algorithms::{
        global_registry,
        pipeline::{Pipeline, PipelineBuilder},
        steps::{
            ensure_core_steps_registered, global_step_registry, StepScope, StepSpec, StepValue,
            StepVariables,
        },
        AlgorithmOutput, AlgorithmParams, Context,
    },
    storage::table::Table,
    subgraphs::Subgraph,
    AttrValue as V, Graph,
};
use serde_json::{json, Value};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

fn pipeline(steps: Value) -> Pipeline {
    PipelineBuilder::new()
        .with_algorithm("builder.step_pipeline", |p| {
            p.set_text("steps", json!({"atomic":true,"steps":steps}).to_string());
        })
        .build(global_registry())
        .unwrap()
}
fn selected(graph: &Arc<OwnerCell<Graph>>) -> Subgraph {
    let view = graph.borrow().read_view().unwrap();
    Subgraph::from_frozen_view(graph.clone(), view, "input".into())
}
fn fixture() -> (Arc<OwnerCell<Graph>>, u64) {
    let mut graph = Graph::new_directed();
    graph.add_nodes(6).unwrap();
    graph.remove_nodes(&[1, 3]).unwrap();
    graph
        .set_node_attr(0, "x".into(), V::Int(9_007_199_254_740_993))
        .unwrap();
    graph.set_node_attr(2, "x".into(), V::Null).unwrap();
    graph
        .set_node_attr(4, "x".into(), V::Float64(1.0000000001))
        .unwrap();
    graph.add_edges(&[(0, 2), (0, 2), (2, 4), (4, 4)]).unwrap();
    graph
        .set_edge_attr(1, "w".into(), V::Float64(2.0000000001))
        .unwrap();
    let state = graph.current_state_id();
    (Arc::new(OwnerCell::new(graph)), state)
}
fn capture(state: u64) -> Vec<Value> {
    vec![
        json!({"id":"temporal.snapshot_at","params":{"state_id":state,"output":"saved"}}),
        json!({"id":"alias","params":{"source":"saved","target":"snapshot_alias"}}),
        json!({"id":"graph.from_snapshot","params":{"source":"snapshot_alias","output":"graph"}}),
    ]
}

#[test]
fn historical_graph_and_table_values_retain_source_rows_and_outlive_owner() {
    let (graph, state) = fixture();
    graph.borrow_mut().remove_node(2).unwrap();
    graph
        .borrow_mut()
        .set_node_attr(0, "x".into(), V::Int(-1))
        .unwrap();
    let head = graph.borrow().current_state_id();
    let mut steps = capture(state);
    steps.extend([
        json!({"id":"graph.select_nodes","params":{"source":"graph","ids":[5,4,0,2,2],"output":"subset"}}),
        json!({"id":"alias","params":{"source":"subset","target":"graph_alias"}}),
        json!({"id":"graph.nodes_table","params":{"source":"graph_alias","output":"nodes"}}),
        json!({"id":"alias","params":{"source":"nodes","target":"table_alias"}}),
        json!({"id":"table.emit","params":{"source":"table_alias","name":"all_nodes"}}),
        json!({"id":"table.select_columns","params":{"source":"nodes","columns":["x"],"output":"nodes"}}),
        json!({"id":"table.select_rows","params":{"source":"nodes","rows":[3,0,0],"output":"rows"}}),
        json!({"id":"table.emit","params":{"source":"rows","name":"rows"}}),
        json!({"id":"graph.select_edges","params":{"source":"graph","ids":[1,3],"output":"edges_graph"}}),
        json!({"id":"graph.edges_table","params":{"source":"edges_graph","output":"edges"}}),
        json!({"id":"table.emit","params":{"source":"edges","name":"edges"}}),
        json!({"id":"graph.emit","params":{"source":"subset","name":"graph"}}),
    ]);
    let mut ctx = Context::new();
    pipeline(json!(steps))
        .run(&mut ctx, selected(&graph))
        .unwrap();
    assert_eq!(graph.borrow().current_state_id(), head);
    let outputs = ctx.take_outputs();
    let AlgorithmOutput::GraphView(view) = &outputs["graph"] else {
        panic!("graph output")
    };
    assert_eq!(view.node_ids(), vec![0, 2, 4, 5]);
    assert_eq!(view.edge_ids(), vec![0, 1, 2, 3]);
    assert_eq!(
        view.node_attr(0, &"x".into()).unwrap(),
        Some(V::Int(9_007_199_254_740_993))
    );
    let AlgorithmOutput::Table(nodes) = &outputs["all_nodes"] else {
        panic!("table output")
    };
    assert_eq!(nodes.source_entity_ids(), Some([0, 2, 4, 5].as_slice()));
    assert_eq!(
        nodes.column("x").unwrap().data(),
        &vec![
            V::Int(9_007_199_254_740_993),
            V::Null,
            V::Float64(1.0000000001),
            V::Null
        ]
    );
    let AlgorithmOutput::Table(rows) = &outputs["rows"] else {
        panic!("row output")
    };
    assert_eq!(rows.source_entity_ids(), Some([5, 0, 0].as_slice()));
    assert_eq!(rows.columns().len(), 1);
    let AlgorithmOutput::Table(edges) = &outputs["edges"] else {
        panic!("edge output")
    };
    assert_eq!(edges.source_entity_ids(), Some([1, 3].as_slice()));
    assert_eq!(
        edges.column("w").unwrap().data(),
        &vec![V::Float64(2.0000000001), V::Null]
    );
    drop(graph);
    assert_eq!(view.edge_endpoints(1).unwrap(), (0, 2));
    assert_eq!(nodes.source_views()[0].node_ids(), vec![0, 2, 4, 5]);
}

#[test]
fn explicit_snapshot_inputs_ignore_inflight_attachments_and_reads_publish_nothing() {
    for outer in [false, true] {
        for persist in [false, true] {
            let (graph, state) = fixture();
            let mut steps = vec![
                json!({"id":"core.init_nodes","params":{"target":"values","value":7}}),
                json!({"id":"core.attach_node_attr","params":{"source":"values","attr":"x"}}),
            ];
            steps.extend(capture(state));
            steps.push(json!({"id":"graph.emit","params":{"source":"graph","name":"captured"}}));
            let pipe = pipeline(json!(steps));
            let mut ctx = Context::new();
            ctx.set_persist_results(persist);
            if outer {
                pipe.run_atomic_properties(&mut ctx, selected(&graph))
            } else {
                pipe.run(&mut ctx, selected(&graph))
            }
            .unwrap();
            assert_eq!(
                graph.borrow().current_state_id(),
                state + u64::from(persist)
            );
            let outputs = ctx.take_outputs();
            let AlgorithmOutput::GraphView(view) = &outputs["captured"] else {
                panic!("graph output")
            };
            assert_eq!(
                view.node_attr(0, &"x".into()).unwrap(),
                Some(V::Int(9_007_199_254_740_993))
            );
            if persist {
                assert_eq!(
                    graph.borrow().get_node_attr(0, &"x".into()).unwrap(),
                    Some(V::Int(7))
                );
            }
        }
    }
}

#[test]
fn late_type_error_rolls_back_local_atomic_outputs_and_properties_and_context_can_be_reused() {
    let (graph, state) = fixture();
    let mut steps = capture(state);
    steps.extend([
        json!({"id":"core.init_nodes","params":{"target":"values","value":7}}),
        json!({"id":"core.attach_node_attr","params":{"source":"values","attr":"prefix"}}),
        json!({"id":"graph.emit","params":{"source":"graph","name":"previous"}}),
        json!({"id":"graph.nodes_table","params":{"source":"saved","output":"bad"}}),
    ]);
    let mut ctx = Context::new();
    ctx.add_output("previous", AlgorithmOutput::Histogram(vec![3]));
    assert!(pipeline(json!(steps))
        .run(&mut ctx, selected(&graph))
        .is_err());
    assert_eq!(graph.borrow().current_state_id(), state);
    assert!(graph
        .borrow()
        .get_node_attr(0, &"prefix".into())
        .unwrap()
        .is_none());
    let outputs = ctx.take_outputs();
    assert!(
        matches!(&outputs["previous"], AlgorithmOutput::Histogram(values) if values == &vec![3])
    );
    assert_eq!(outputs.len(), 1);
    let mut good = capture(state);
    good.push(json!({"id":"graph.emit","params":{"source":"graph","name":"ok"}}));
    pipeline(json!(good))
        .run(&mut ctx, selected(&graph))
        .unwrap();
    assert!(matches!(
        &ctx.take_outputs()["ok"],
        AlgorithmOutput::GraphView(_)
    ));
}

#[test]
fn controls_validate_before_execution_and_cancelled_or_failed_stages_keep_targets() {
    ensure_core_steps_registered();
    for (id, key, value) in [
        ("graph.select_nodes", "ids", json!([-1])),
        ("graph.select_nodes", "ids", json!([true])),
        ("graph.select_edges", "ids", json!([1.5])),
        ("table.select_rows", "rows", json!(["0"])),
        ("table.select_columns", "columns", json!([2])),
    ] {
        let mut params = AlgorithmParams::new();
        params.set_text("source", "input");
        params.set_text("output", "target");
        params.insert(key, groggy::algorithms::AlgorithmParamValue::Json(value));
        assert!(global_step_registry()
            .instantiate(&StepSpec {
                id: id.into(),
                params,
                inputs: vec![],
                outputs: vec![]
            })
            .is_err());
    }
    let (graph, _) = fixture();
    let selection = selected(&graph);
    let mut vars = StepVariables::default();
    vars.set_value("graph", StepValue::GraphView(selection.pin_view().unwrap()));
    vars.set_scalar("target", groggy::algorithms::AlgorithmParamValue::Int(42));
    let mut params = AlgorithmParams::new();
    params.set_text("source", "graph");
    params.set_text("output", "target");
    let step = global_step_registry()
        .instantiate(&StepSpec {
            id: "graph.nodes_table".into(),
            params,
            inputs: vec![],
            outputs: vec![],
        })
        .unwrap();
    let mut ctx = Context::with_cancel_token(Arc::new(AtomicBool::new(true)));
    assert!(step
        .apply(&mut ctx, &mut StepScope::new(&selection, &mut vars))
        .is_err());
    assert_eq!(
        vars.scalar("target").unwrap(),
        &groggy::algorithms::AlgorithmParamValue::Int(42)
    );
}

#[test]
fn local_atomic_output_scopes_move_prior_buffers_on_success_and_failure() {
    let (graph, state) = fixture();
    let mut ctx = Context::new();
    let rows = (0..128).map(|id| (id, id as i64)).collect::<Vec<_>>();
    let pointer = rows.as_ptr();
    ctx.add_output("retained", AlgorithmOutput::NodeIntegers(rows));
    for fail in [false, true] {
        let mut steps = capture(state);
        steps.push(json!({"id":"graph.emit","params":{"source":"graph","name":"new"}}));
        if fail {
            steps
                .push(json!({"id":"graph.nodes_table","params":{"source":"saved","output":"bad"}}));
        }
        let result = pipeline(json!(steps)).run(&mut ctx, selected(&graph));
        assert_eq!(result.is_err(), fail);
        let mut outputs = ctx.take_outputs();
        let AlgorithmOutput::NodeIntegers(rows) = outputs.remove("retained").unwrap() else {
            panic!("retained output")
        };
        assert_eq!(
            rows.as_ptr(),
            pointer,
            "rollback must not clone retained row buffers"
        );
        assert_eq!(rows.len(), 128);
        ctx.add_output("retained", AlgorithmOutput::NodeIntegers(rows));
    }
}
