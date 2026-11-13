//! Native atomic property-builder receipt; no whole-graph clone or speedup claim.
use groggy::{
    algorithms::{
        builder::register_algorithms, AlgorithmParamValue, AlgorithmParams, AlgorithmSpec, Context,
        Registry,
    },
    subgraphs::Subgraph,
    AttrValue, Graph,
};
use serde_json::json;
use std::{cell::RefCell, rc::Rc, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::args().any(|arg| arg == "--captured-degrees") {
        return captured_degrees_receipt();
    }
    if std::env::args().any(|arg| arg == "--endpoint-revisions") {
        return endpoint_revisions_receipt();
    }
    if std::env::args().any(|arg| arg == "--terminal-removals" || arg == "--staged-capture") {
        return terminal_removals_receipt();
    }
    if std::env::args().any(|arg| arg == "--typed-values") {
        return typed_values_receipt();
    }
    let registry = Registry::default();
    register_algorithms(&registry)?;
    let mut cases = Vec::new();
    for count in [1_000, 10_000] {
        for columns in [1, 4] {
            let mut steps =
                vec![json!({"id":"core.init_nodes","params":{"target":"values","value":0.125}})];
            for column in 0..columns {
                steps.push(json!({"id":"core.attach_node_attr","params":{"source":"values","attr":format!("value{column}")}}));
                steps.push(json!({"id":"core.load_node_attr","params":{"target":format!("copy{column}"),"attr":format!("value{column}")}}));
                steps.push(json!({"id":"core.attach_node_attr","params":{"source":format!("copy{column}"),"attr":format!("mirror{column}")}}));
            }
            let mut params = AlgorithmParams::new();
            params.insert(
                "steps",
                AlgorithmParamValue::Json(json!({"atomic":true,"steps":steps})),
            );
            let algorithm = registry.instantiate(&AlgorithmSpec {
                id: "builder.step_pipeline".into(),
                params,
            })?;
            let mut samples = Vec::new();
            for _ in 0..5 {
                let mut graph = Graph::new_publishing();
                graph.insert_nodes(count)?;
                let shared = Rc::new(RefCell::new(graph));
                let selection =
                    Subgraph::from_nodes(shared.clone(), (0..count).collect(), "bench".into())?;
                let mut context = Context::new();
                let started = Instant::now();
                algorithm.execute(&mut context, selection)?;
                let seconds = started.elapsed().as_secs_f64();
                let graph = shared.borrow();
                if graph.current_state_id() != 2
                    || graph.has_uncommitted_changes()
                    || graph.get_node_attr(count - 1, &format!("mirror{}", columns - 1))?
                        != Some(AttrValue::Float64(0.125))
                {
                    return Err(
                        "Invalid atomic property publication/read-after-write result".into(),
                    );
                }
                samples.push(json!({"seconds":seconds,"state_id":graph.current_state_id()}));
            }
            cases.push(json!({"nodes":count,"attachment_columns":columns*2,"samples":samples}));
        }
    }
    let path = std::env::var("GROGGY_ATOMIC_BUILDER_PROFILE")
        .unwrap_or_else(|_| "atomic_builder_properties.json".into());
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(
            &json!({"scope":"Native release property-only atomic builder runs, five observations. Fixture/algorithm construction excluded; includes staging, read-after-write, validation and one accepted version. No before/after speedup, FFI, RSS/allocation or production budget claim.","cases":cases}),
        )?,
    )?;
    println!("Wrote atomic builder receipt to {path}");
    Ok(())
}

fn typed_values_receipt() -> Result<(), Box<dyn std::error::Error>> {
    use groggy::{algorithms::AlgorithmOutput, storage::table::Table, BatchNodeRef, GraphBatch};
    let registry = Registry::default();
    register_algorithms(&registry)?;
    let mut cases = Vec::new();
    for count in [1_000, 100_000] {
        let mut graph = Graph::new();
        graph.apply_batch(
            GraphBatch {
                add_nodes: count,
                node_attrs: [(
                    "score".into(),
                    (0..count)
                        .map(|id| (BatchNodeRef::Added(id), AttrValue::Int(id as i64)))
                        .collect(),
                )]
                .into(),
                ..Default::default()
            },
            0,
            "fixture".into(),
            "bench".into(),
        )?;
        let shared = Rc::new(RefCell::new(graph));
        let selected = Subgraph::from_nodes(shared.clone(), (0..count).collect(), "bench".into())?;
        let mut params = AlgorithmParams::new();
        params.set_text("steps", json!({"atomic":true,"steps":[
            {"id":"temporal.snapshot_at","params":{"state_id":1,"output":"saved"}},
            {"id":"graph.from_snapshot","params":{"source":"saved","output":"graph"}},
            {"id":"alias","params":{"source":"graph","target":"alias"}},
            {"id":"graph.nodes_table","params":{"source":"alias","output":"nodes"}},
            {"id":"table.select_columns","params":{"source":"nodes","columns":["score"],"output":"projected"}},
            {"id":"table.emit","params":{"source":"projected","name":"table"}}
        ]}).to_string());
        let algorithm = registry.instantiate(&AlgorithmSpec {
            id: "builder.step_pipeline".into(),
            params,
        })?;
        let mut ctx = Context::new();
        ctx.set_persist_results(false);
        let mut samples = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            algorithm.execute(&mut ctx, selected.clone())?;
            samples.push(start.elapsed().as_secs_f64());
            let outputs = ctx.take_outputs();
            let AlgorithmOutput::Table(table) = &outputs["table"] else {
                return Err("typed table missing".into());
            };
            assert_eq!(table.column("score").unwrap().len(), count);
            assert_eq!(
                table.column("score").unwrap().data()[count - 1],
                AttrValue::Int((count - 1) as i64)
            );
            assert_eq!(table.source_entity_ids().unwrap().len(), count);
            assert_eq!(shared.borrow().current_state_id(), 1);
        }
        cases.push(json!({"nodes":count,"retained_snapshot_to_table_seconds":samples}));
    }
    let mut graph = Graph::new();
    graph.add_node()?;
    let shared = Rc::new(RefCell::new(graph));
    let selected = Subgraph::from_nodes(shared.clone(), [0].into_iter().collect(), "bench".into())?;
    let mut params = AlgorithmParams::new();
    params.set_text("steps", json!({"atomic":true,"steps":[]}).to_string());
    let algorithm = registry.instantiate(&AlgorithmSpec {
        id: "builder.step_pipeline".into(),
        params,
    })?;
    let mut rollback = Vec::new();
    for rows in [0, 1_000_000] {
        let mut ctx = Context::new();
        let values = (0..rows).map(|id| (id, id as i64)).collect::<Vec<_>>();
        let pointer = values.as_ptr();
        ctx.add_output("retained", AlgorithmOutput::NodeIntegers(values));
        let mut samples = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            algorithm.execute(&mut ctx, selected.clone())?;
            samples.push(start.elapsed().as_secs_f64());
            let mut outputs = ctx.take_outputs();
            let AlgorithmOutput::NodeIntegers(values) = outputs.remove("retained").unwrap() else {
                return Err("retained output missing".into());
            };
            assert_eq!(values.as_ptr(), pointer);
            ctx.add_output("retained", AlgorithmOutput::NodeIntegers(values));
        }
        rollback.push(json!({"retained_output_rows":rows,"empty_atomic_run_seconds":samples,"original_buffer_pointer_preserved":true}));
    }
    let path = std::env::var("GROGGY_ATOMIC_BUILDER_PROFILE")
        .unwrap_or_else(|_| "typed_pipeline_values.json".into());
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&json!({
            "scope":"Native release, five samples. Reconstruction/projection runs include retained snapshot reconstruction, graph alias, node table projection, column selection and explicit emission; construction/compilation and output validation excluded. Output-scope cases exclude retained buffer construction and assert pointer identity after every run. No before/after speedup, allocation/RSS, GIL or throughput guarantee.", "typed_reads":cases,"output_scopes":rollback
        }))?,
    )?;
    println!("Wrote typed pipeline receipt to {path}");
    Ok(())
}

fn terminal_removals_receipt() -> Result<(), Box<dyn std::error::Error>> {
    use groggy::{BatchNodeRef as N, GraphBatch};
    let registry = Registry::default();
    register_algorithms(&registry)?;
    let mut cases = Vec::new();
    let with_capture = std::env::args().any(|arg| arg == "--staged-capture");
    for count in [1_000, 100_000] {
        let mut steps = vec![
            json!({"id":"graph.remove_edges","params":{"ids":[0]}}),
            json!({"id":"graph.remove_nodes","params":{"ids":[count-1]}}),
        ];
        if with_capture {
            steps.extend([
                json!({"id":"graph.capture_staged","params":{"output":"pending"}}),
                json!({"id":"graph.nodes_table","params":{"source":"pending","output":"nodes"}}),
                json!({"id":"table.emit","params":{"source":"nodes","name":"nodes"}}),
                json!({"id":"graph.emit","params":{"source":"pending","name":"pending"}}),
            ]);
        }
        let mut params = AlgorithmParams::new();
        params.set_text("steps", json!({"atomic":true,"steps":steps}).to_string());
        let algorithm = registry.instantiate(&AlgorithmSpec {
            id: "builder.step_pipeline".into(),
            params,
        })?;
        let mut samples = Vec::new();
        for _ in 0..5 {
            let mut graph = Graph::new();
            graph.apply_batch(
                GraphBatch {
                    add_nodes: count,
                    add_edges: (1..count)
                        .map(|id| (N::Added(id - 1), N::Added(id)))
                        .collect(),
                    ..Default::default()
                },
                0,
                "fixture".into(),
                "bench".into(),
            )?;
            let graph = Rc::new(RefCell::new(graph));
            let view = graph.borrow().read_view()?;
            let selection = Subgraph::from_frozen_view(graph.clone(), view, "bench".into());
            let mut ctx = Context::new();
            let started = Instant::now();
            let result = algorithm.execute(&mut ctx, selection)?;
            let seconds = started.elapsed().as_secs_f64();
            assert_eq!(graph.borrow().current_state_id(), 2);
            assert_eq!(graph.borrow().node_count(), count - 1);
            assert_eq!(graph.borrow().edge_count(), count - 3);
            assert_eq!(result.node_count(), count - 1);
            assert_eq!(result.edge_count(), count - 3);
            if with_capture {
                let outputs = ctx.take_outputs();
                let groggy::algorithms::AlgorithmOutput::GraphView(view) = &outputs["pending"]
                else {
                    return Err("missing pending graph output".into());
                };
                assert_eq!(view.write_state_id(), None);
                assert_eq!(view.node_count(), count - 1);
                assert_eq!(view.edge_count(), count - 3);
                let groggy::algorithms::AlgorithmOutput::Table(table) = &outputs["nodes"] else {
                    return Err("missing pending node table".into());
                };
                use groggy::storage::table::Table;
                assert_eq!(table.shape().0, count - 1);
            }
            samples.push(seconds);
        }
        cases.push(json!({"nodes":count,"initial_edges":count-1,"seconds":samples}));
    }
    let path = std::env::var("GROGGY_ATOMIC_BUILDER_PROFILE")
        .unwrap_or_else(|_| "terminal_topology_pipeline.json".into());
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&json!({
            "scope":"Native release, five samples. Two removal stages on a captured full path graph, includes execution-scope validation, staging, one checked publication and surviving exact membership reconstruction. When staged_capture is true, also materializes an explicit pending graph, projects its node table and emits both values before publication. Fixture, initial capture, compilation and result assertions excluded. No FFI, allocation/RSS, before/after speedup or large-scale guarantee. Membership reconstruction and explicit capture/projection are linear in selected data, not mutation count.",
            "staged_capture": with_capture,
            "cases":cases
        }))?,
    )?;
    println!("Wrote terminal topology receipt to {path}");
    Ok(())
}

fn endpoint_revisions_receipt() -> Result<(), Box<dyn std::error::Error>> {
    use groggy::{BatchNodeRef as N, GraphBatch};
    let registry = Registry::default();
    register_algorithms(&registry)?;
    let mut cases = Vec::new();
    for count in [1_000, 100_000] {
        let mut params = AlgorithmParams::new();
        params.set_text(
            "steps",
            json!({"atomic":true,"steps":[
                {"id":"graph.set_edge_endpoints","params":{"revisions":[[0,0,count-1]]}},
                {"id":"graph.remove_nodes","params":{"ids":[1]}},
                {"id":"graph.capture_staged","params":{"output":"pending"}},
                {"id":"graph.emit","params":{"source":"pending","name":"pending"}}
            ]})
            .to_string(),
        );
        let algorithm = registry.instantiate(&AlgorithmSpec {
            id: "builder.step_pipeline".into(),
            params,
        })?;
        let mut samples = Vec::new();
        for _ in 0..5 {
            let mut graph = Graph::new_directed();
            graph.apply_batch(
                GraphBatch {
                    add_nodes: count,
                    add_edges: (1..count)
                        .map(|id| (N::Added(id - 1), N::Added(id)))
                        .collect(),
                    ..Default::default()
                },
                0,
                "fixture".into(),
                "bench".into(),
            )?;
            let graph = Rc::new(RefCell::new(graph));
            let view = graph.borrow().read_view()?;
            let selected = Subgraph::from_frozen_view(graph.clone(), view, "bench".into());
            let mut ctx = Context::new();
            let started = Instant::now();
            let result = algorithm.execute(&mut ctx, selected)?;
            let seconds = started.elapsed().as_secs_f64();
            assert_eq!(graph.borrow().current_state_id(), 2);
            assert_eq!(result.node_count(), count - 1);
            assert_eq!(result.edge_count(), count - 2);
            let outputs = ctx.take_outputs();
            let groggy::algorithms::AlgorithmOutput::GraphView(view) = &outputs["pending"] else {
                return Err("missing pending graph".into());
            };
            assert_eq!(view.edge_endpoints(0)?, (0, count - 1));
            assert_eq!(view.neighbors(0)?, vec![count - 1]);
            samples.push(seconds);
        }
        cases.push(json!({"nodes":count,"initial_edges":count-1,"seconds":samples}));
    }
    let path = std::env::var("GROGGY_ATOMIC_BUILDER_PROFILE")
        .unwrap_or_else(|_| "pipeline_endpoint_revisions.json".into());
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&json!({
            "scope":"Native release, five samples. Existing-edge revision, old-endpoint removal, explicit pending graph capture/emission, one publication and surviving selection on a directed path. Fixture, initial capture, compilation and assertions excluded. No before/after speedup, FFI, allocation/RSS or scale budget claim.","cases":cases
        }))?,
    )?;
    println!("Wrote endpoint revision receipt to {path}");
    Ok(())
}

fn captured_degrees_receipt() -> Result<(), Box<dyn std::error::Error>> {
    use groggy::{
        algorithms::steps::{
            ensure_core_steps_registered, global_step_registry, StepScope, StepSpec, StepValue,
            StepVariables,
        },
        storage::table::Table,
        BatchNodeRef as N, GraphBatch,
    };
    ensure_core_steps_registered();
    let mut params = AlgorithmParams::new();
    params.set_text("source", "input");
    params.set_text("output", "metrics");
    let step = global_step_registry().instantiate(&StepSpec {
        id: "graph.degrees_table".into(),
        params,
        inputs: vec![],
        outputs: vec![],
    })?;
    let mut cases = Vec::new();
    for count in [1_000, 100_000] {
        let mut g = Graph::new_directed();
        let mut edges: Vec<_> = (1..count)
            .map(|id| (N::Added(id - 1), N::Added(id)))
            .collect();
        edges.extend([
            (N::Added(0), N::Added(1)),
            (N::Added(count - 1), N::Added(count - 1)),
        ]);
        g.apply_batch(
            GraphBatch {
                add_nodes: count,
                add_edges: edges,
                ..Default::default()
            },
            0,
            "fixture".into(),
            "bench".into(),
        )?;
        let g = Rc::new(RefCell::new(g));
        let view = g.borrow().read_view()?;
        let selected = Subgraph::from_frozen_view(g.clone(), view.clone(), "bench".into());
        let mut samples = Vec::new();
        for _ in 0..5 {
            let mut vars = StepVariables::default();
            vars.set_value("input", StepValue::GraphView(view.clone()));
            let mut ctx = Context::new();
            let start = Instant::now();
            step.apply(&mut ctx, &mut StepScope::new(&selected, &mut vars))?;
            let seconds = start.elapsed().as_secs_f64();
            let table = vars.table("metrics")?;
            assert_eq!(table.shape(), (count, 4));
            assert_eq!(
                table
                    .column("out_degree")
                    .and_then(|column| column.data().first()),
                Some(&AttrValue::Int(2))
            );
            assert_eq!(
                table
                    .column("in_degree")
                    .and_then(|column| column.data().last()),
                Some(&AttrValue::Int(2))
            );
            assert_eq!(g.borrow().current_state_id(), 1);
            samples.push(seconds);
        }
        cases.push(json!({"nodes":count,"edges":count+1,"seconds":samples}));
    }
    let path = std::env::var("GROGGY_ATOMIC_BUILDER_PROFILE")
        .unwrap_or_else(|_| "captured_pipeline_degrees.json".into());
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&json!({
            "scope":"Native release, five samples, directed captured path plus one parallel and one loop edge. Measures only the degree-table stage including selected ID ordering, three canonical degree computations, four owned columns and provenance binding. Fixture, source capture, scope/variable construction, compilation, assertions and result destruction excluded. No FFI, allocation/RSS, before/after speedup or scale budget claim.","cases":cases
        }))?,
    )?;
    println!("Wrote captured degree receipt to {path}");
    Ok(())
}
