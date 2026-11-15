//! Native atomic property-builder receipt; no whole-graph clone or speedup claim.
use groggy::ownership::OwnerCell;
use groggy::{
    algorithms::{
        builder::register_algorithms, AlgorithmParamValue, AlgorithmParams, AlgorithmSpec, Context,
        Registry,
    },
    subgraphs::Subgraph,
    AttrValue, Graph,
};
use serde_json::json;
use std::sync::Arc;
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::args().any(|arg| arg == "--child-frames") {
        return child_frames_receipt();
    }
    if std::env::args().any(|arg| arg == "--pending-decay") {
        return pending_core_receipt(PendingCompute::Decay);
    }
    if std::env::args().any(|arg| arg == "--checked-flow") {
        return pending_core_receipt(PendingCompute::Flow);
    }
    if std::env::args().any(|arg| arg == "--pending-core") {
        return pending_core_receipt(PendingCompute::Core);
    }
    if std::env::args().any(|arg| arg == "--nested-pending" || arg == "--capture-churn") {
        return nested_pending_receipt();
    }
    if std::env::args().any(|arg| arg == "--captured-attachment") {
        return captured_attachment_receipt();
    }
    if std::env::args().any(|arg| arg == "--pending-readers") {
        return pending_readers_receipt();
    }
    if std::env::args().any(|arg| arg == "--pending-properties") {
        return pending_properties_receipt();
    }
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
                let shared = Arc::new(OwnerCell::new(graph));
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
        let shared = Arc::new(OwnerCell::new(graph));
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
    let shared = Arc::new(OwnerCell::new(graph));
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
            let graph = Arc::new(OwnerCell::new(graph));
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
            let graph = Arc::new(OwnerCell::new(graph));
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
        let g = Arc::new(OwnerCell::new(g));
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

fn pending_properties_receipt() -> Result<(), Box<dyn std::error::Error>> {
    let registry = Registry::default();
    register_algorithms(&registry)?;
    let mut cases = Vec::new();
    for count in [1_000, 10_000] {
        for pending_topology in [false, true] {
            let steps = json!([
                {"id":"graph.add_nodes","params":{"count":usize::from(pending_topology),"output":"new"}},
                {"id":"core.init_nodes","params":{"target":"values","value":0.125}},
                {"id":"core.graph_node_count","params":{"target":"count"}},
                {"id":"core.init_nodes_with_index","params":{"target":"positions"}},
                {"id":"core.attach_node_attr","params":{"source":"values","attr":"value"}},
                {"id":"core.load_node_attr","params":{"target":"copy","attr":"value"}},
                {"id":"core.attach_node_attr","params":{"source":"copy","attr":"mirror"}}
            ]);
            let mut params = AlgorithmParams::new();
            params.set_text("steps", json!({"atomic":true,"steps":steps}).to_string());
            let algorithm = registry.instantiate(&AlgorithmSpec {
                id: "builder.step_pipeline".into(),
                params,
            })?;
            let mut samples = Vec::new();
            for _ in 0..5 {
                let mut graph = Graph::new_publishing();
                graph.insert_nodes(count)?;
                let graph = Arc::new(OwnerCell::new(graph));
                let view = graph.borrow().read_view()?;
                let selected = Subgraph::from_frozen_view(graph.clone(), view, "bench".into());
                let mut ctx = Context::new();
                let started = Instant::now();
                let result = algorithm.execute(&mut ctx, selected)?;
                samples.push(started.elapsed().as_secs_f64());
                let total = count + usize::from(pending_topology);
                assert_eq!(result.node_count(), total);
                assert_eq!(graph.borrow().current_state_id(), 2);
                assert_eq!(
                    graph.borrow().get_node_attr(total - 1, &"mirror".into())?,
                    Some(AttrValue::Float64(0.125))
                );
            }
            cases.push(json!({"initial_nodes":count,"pending_topology":pending_topology,"seconds":samples}));
        }
    }
    let path = std::env::var("GROGGY_ATOMIC_BUILDER_PROFILE")
        .unwrap_or_else(|_| "pending_topology_properties.json".into());
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&json!({
            "scope":"Native release, five samples per case. Initialization/count/index reads, one property attachment, reload and second attachment, with zero or one staged new node. Includes scope resolution, pending materialization/reuse, validation and one publication. Excludes fixture/initial capture/algorithm construction, compilation and assertions. No before/after optimization, FFI, allocation/RSS or throughput guarantee. The topology case has one extra final node and a canonical creation operation; these are bounded operation-cost observations.",
            "cases":cases
        }))?,
    )?;
    println!("Wrote pending properties receipt to {path}");
    Ok(())
}

fn pending_readers_receipt() -> Result<(), Box<dyn std::error::Error>> {
    use groggy::{BatchNodeRef as N, GraphBatch};
    let registry = Registry::default();
    register_algorithms(&registry)?;
    let mut cases = Vec::new();
    for count in [1_000, 10_000] {
        let mut params = AlgorithmParams::new();
        params.set_text("steps", json!({"atomic":true,"steps":[
            {"id":"graph.add_nodes","params":{"count":1,"output":"new"}},
            {"id":"graph.add_edges","params":{"edges":[[count-1,{"added":0}],[{"added":0},0]],"output":"new_edges","attributes":{"weight":2.0}}},
            {"id":"core.init_nodes","params":{"target":"values","value":1}},
            {"id":"core.node_degree","params":{"target":"degrees"}},
            {"id":"core.neighbor_agg","params":{"source":"values","target":"neighbors","agg":"sum","direction":"in"}},
            {"id":"core.init_scalar","params":{"target":"source","value":0}},
            {"id":"core.shortest_path_map","params":{"source":"source","weight_attr":"weight","output":"paths"}},
            {"id":"core.attach_node_attr","params":{"source":"paths","attr":"distance"}}
        ]}).to_string());
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
                    add_edges: (0..count)
                        .map(|node| (N::Added(node), N::Added((node + 1) % count)))
                        .collect(),
                    ..Default::default()
                },
                0,
                "fixture".into(),
                "bench".into(),
            )?;
            let graph = Arc::new(OwnerCell::new(graph));
            let selected = Subgraph::from_frozen_view(
                graph.clone(),
                graph.borrow().read_view()?,
                "bench".into(),
            );
            let mut ctx = Context::new();
            let started = Instant::now();
            let result = algorithm.execute(&mut ctx, selected)?;
            samples.push(started.elapsed().as_secs_f64());
            assert_eq!(result.node_count(), count + 1);
            assert_eq!(graph.borrow().current_state_id(), 2);
            assert_eq!(
                graph.borrow().get_node_attr(count, &"distance".into())?,
                Some(AttrValue::Float64((count + 1) as f64))
            );
        }
        cases.push(json!({"initial_nodes":count,"initial_edges":count,"seconds":samples}));
    }
    let path = std::env::var("GROGGY_ATOMIC_BUILDER_PROFILE")
        .unwrap_or_else(|_| "pending_topology_readers.json".into());
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&json!({
            "scope":"Native release, five observations. Directed ring with one staged node and two staged weighted edges; initialization, degree, incoming aggregation, weighted shortest paths, property attachment and one publication. Includes pending capture/reuse and kernels; fixture, initial capture, compilation, algorithm construction and assertions excluded. No before/after speedup, FFI, memory or scale guarantee.",
            "cases":cases
        }))?,
    )?;
    println!("Wrote pending reader receipt to {path}");
    Ok(())
}

fn captured_attachment_receipt() -> Result<(), Box<dyn std::error::Error>> {
    use groggy::{BatchNodeRef as N, GraphBatch};
    let registry = Registry::default();
    register_algorithms(&registry)?;
    let mut cases = Vec::new();
    for count in [1_000, 10_000] {
        let mut params = AlgorithmParams::new();
        params.set_text("steps", json!({"atomic":true,"steps":[
            {"id":"graph.add_nodes","params":{"count":1,"output":"new"}},
            {"id":"graph.add_edges","params":{"edges":[[count-1,{"added":0}],[{"added":0},0]],"output":"new_edges"}},
            {"id":"graph.capture_staged","params":{"output":"pending"}},
            {"id":"graph.degrees_table","params":{"source":"pending","output":"metrics"}},
            {"id":"table.attach","params":{"source":"metrics","columns":{"degree":"degree","out_degree":"outdegree"}}}
        ]}).to_string());
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
                    add_edges: (0..count)
                        .map(|node| (N::Added(node), N::Added((node + 1) % count)))
                        .collect(),
                    ..Default::default()
                },
                0,
                "fixture".into(),
                "bench".into(),
            )?;
            let graph = Arc::new(OwnerCell::new(graph));
            let selected = Subgraph::from_frozen_view(
                graph.clone(),
                graph.borrow().read_view()?,
                "bench".into(),
            );
            let mut ctx = Context::new();
            let started = Instant::now();
            let result = algorithm.execute(&mut ctx, selected)?;
            samples.push(started.elapsed().as_secs_f64());
            assert_eq!(result.node_count(), count + 1);
            assert_eq!(graph.borrow().current_state_id(), 2);
            assert_eq!(
                graph.borrow().get_node_attr(count, &"degree".into())?,
                Some(AttrValue::Int(2))
            );
            assert_eq!(
                graph.borrow().get_node_attr(count, &"outdegree".into())?,
                Some(AttrValue::Int(1))
            );
        }
        cases.push(json!({"initial_nodes":count,"initial_edges":count,"seconds":samples}));
    }
    let path = std::env::var("GROGGY_ATOMIC_BUILDER_PROFILE")
        .unwrap_or_else(|_| "captured_table_attachment.json".into());
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&json!({
            "scope":"Native release, five observations. Directed ring with one staged node/two edges, explicit pending capture, owned degree table, aliased attachment of two row-bound columns and one publication. Includes source authority, current pending selection materialization, canonical validation and result projection; fixture, initial capture, compilation, algorithm construction and assertions excluded. No before/after speedup, FFI, allocation/RSS or scale guarantee.",
            "cases":cases
        }))?,
    )?;
    println!("Wrote captured attachment receipt to {path}");
    Ok(())
}

fn nested_pending_receipt() -> Result<(), Box<dyn std::error::Error>> {
    let churn = std::env::args().any(|arg| arg == "--capture-churn");
    let iterations = if churn { 1000 } else { 3 };
    let sizes = if churn { [10, 100] } else { [1000, 10000] };
    let registry = Registry::default();
    register_algorithms(&registry)?;
    let increment = vec![
        json!({"id":"core.load_node_attr","params":{"attr":"counter","target":"counter","default":0}}),
        json!({"id":"core.init_scalar","params":{"target":"one","value":1}}),
        json!({"id":"core.add","params":{"left":"counter","right":"one","target":"next"}}),
        json!({"id":"core.attach_node_attr","params":{"source":"next","attr":"counter"}}),
    ];
    let mut cases = Vec::new();
    for count in sizes {
        for mode in ["flat", "loop", "nested"] {
            let mut steps =
                vec![json!({"id":"graph.add_nodes","params":{"count":1,"output":"new"}})];
            match mode {
                "flat" => {
                    for _ in 0..iterations {
                        steps.extend(increment.clone());
                    }
                }
                "loop" => steps.push(
                    json!({"id":"iter.loop","params":{"iterations":iterations,"body":increment}}),
                ),
                _ => steps.push(
                    json!({"id":"iter.loop","params":{"iterations":iterations,"body":[
                        {"id":"iter.loop","params":{"iterations":1,"body":increment}}
                    ]}}),
                ),
            }
            let mut params = AlgorithmParams::new();
            params.set_text("steps", json!({"atomic":true,"steps":steps}).to_string());
            let algorithm = registry.instantiate(&AlgorithmSpec {
                id: "builder.step_pipeline".into(),
                params,
            })?;
            let mut samples = Vec::new();
            for _ in 0..5 {
                let mut graph = Graph::new_publishing();
                graph.insert_nodes(count)?;
                let graph = Arc::new(OwnerCell::new(graph));
                let selected = Subgraph::from_frozen_view(
                    graph.clone(),
                    graph.borrow().read_view()?,
                    "bench".into(),
                );
                let mut ctx = Context::new();
                let started = Instant::now();
                let result = algorithm.execute(&mut ctx, selected)?;
                samples.push(started.elapsed().as_secs_f64());
                assert_eq!(result.node_count(), count + 1);
                assert_eq!(graph.borrow().current_state_id(), 2);
                assert_eq!(
                    graph.borrow().get_node_attr(count, &"counter".into())?,
                    Some(AttrValue::Int(iterations as i64))
                );
            }
            cases.push(json!({"initial_nodes":count,"mode":mode,"iterations":iterations,"seconds":samples}));
        }
    }
    let path = std::env::var("GROGGY_ATOMIC_BUILDER_PROFILE").unwrap_or_else(|_| {
        if churn {
            "capture_churn.json".into()
        } else {
            "nested_pending_execution.json".into()
        }
    });
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&json!({
            "scope":"Native release, five observations each. Node-only graph, one staged new node, three identical load/increment/attach operations, either flat, one loop or nested loops; one final publication. Includes pending revision capture/reuse and execution frames. Fixture, initial capture, algorithm construction, compilation and assertions excluded. Compares equivalent execution shapes, not a before/after engine version. No memory, FFI or general overhead guarantee.",
            "cases":cases
        }))?,
    )?;
    println!("Wrote nested pending receipt to {path}");
    Ok(())
}

#[derive(Clone, Copy)]
enum PendingCompute {
    Core,
    Flow,
    Decay,
}

fn pending_core_receipt(workload: PendingCompute) -> Result<(), Box<dyn std::error::Error>> {
    use groggy::{BatchNodeRef as N, GraphBatch};
    let flow = matches!(workload, PendingCompute::Flow);
    let decay = matches!(workload, PendingCompute::Decay);
    let registry = Registry::default();
    register_algorithms(&registry)?;
    let mut cases = Vec::new();
    for count in [1_000, 10_000] {
        let computation = if flow {
            json!([
                {"id":"core.init_edges","params":{"target":"flow","value":3}},
                {"id":"core.init_scalar","params":{"target":"delta","value":2}},
                {"id":"core.flow_update","params":{"flow":"flow","delta":"delta","target":"updated"}},
                {"id":"core.init_edges","params":{"target":"capacity","value":9}},
                {"id":"core.residual_capacity","params":{"capacity":"capacity","flow":"updated","target":"residual"}},
                {"id":"core.attach_edge_attr","params":{"source":"updated","attr":"flow"}},
                {"id":"core.attach_edge_attr","params":{"source":"residual","attr":"residual"}}
            ])
        } else if decay {
            json!([
                {"id":"core.load_node_attr","params":{"attr":"node_id","target":"values"}},
                {"id":"temporal.decay","params":{"attr":"values","half_life":2.0,"time_delta":2.0,"output":"result"}},
                {"id":"core.attach_node_attr","params":{"source":"result","attr":"decayed"}}
            ])
        } else {
            json!([
                {"id":"core.load_node_attr","params":{"attr":"node_id","target":"ids"}},
                {"id":"core.normalize_values","params":{"source":"ids","target":"normalized","method":"sum"}},
                {"id":"core.init_nodes","params":{"target":"two","value":2}},
                {"id":"graph.fused_neighbor_mul_agg","params":{"values":"normalized","scalars":"two","target":"fused","direction":"out"}},
                {"id":"core.top_k","params":{"source":"ids","target":"top","k":10,"order":"desc"}},
                {"id":"core.sample_nodes","params":{"target":"sample","count":10,"seed":43}},
                {"id":"core.attach_node_attr","params":{"source":"normalized","attr":"normalized"}},
                {"id":"core.attach_node_attr","params":{"source":"fused","attr":"fused"}},
                {"id":"core.attach_node_attr","params":{"source":"top","attr":"rank"}},
                {"id":"core.attach_node_attr","params":{"source":"sample","attr":"sample"}}
            ])
        };
        let mut params = AlgorithmParams::new();
        params.set_text("steps", json!({"atomic":true,"steps":[
            {"id":"graph.add_nodes","params":{"count":1,"output":"new"}},
            {"id":"graph.add_edges","params":{"edges":[[count-1,{"added":0}],[{"added":0},0]],"output":"new_edges","attributes":{"weight":2.0}}},
            {"id":"iter.loop","params":{"iterations":1,"body":computation}}
        ]}).to_string());
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
                    add_edges: (0..count)
                        .map(|node| (N::Added(node), N::Added((node + 1) % count)))
                        .collect(),
                    ..Default::default()
                },
                0,
                "fixture".into(),
                "bench".into(),
            )?;
            let graph = Arc::new(OwnerCell::new(graph));
            let selected = Subgraph::from_frozen_view(
                graph.clone(),
                graph.borrow().read_view()?,
                "bench".into(),
            );
            let mut ctx = Context::new();
            let started = Instant::now();
            let result = algorithm.execute(&mut ctx, selected)?;
            samples.push(started.elapsed().as_secs_f64());
            assert_eq!(result.node_count(), count + 1);
            assert_eq!(graph.borrow().current_state_id(), 2);
            if flow {
                assert_eq!(
                    graph.borrow().get_edge_attr(count, &"residual".into())?,
                    Some(AttrValue::Int(4))
                );
            } else if decay {
                assert_eq!(
                    graph.borrow().get_node_attr(count, &"decayed".into())?,
                    Some(AttrValue::Float64(count as f64 / 2.0))
                );
            } else {
                assert_eq!(
                    graph.borrow().get_node_attr(count, &"rank".into())?,
                    Some(AttrValue::Int(0))
                );
            }
        }
        cases.push(json!({"initial_nodes":count,"initial_edges":count,"seconds":samples}));
    }
    let path = std::env::var("GROGGY_ATOMIC_BUILDER_PROFILE").unwrap_or_else(|_| {
        if flow {
            "checked_flow_compute.json".into()
        } else if decay {
            "pending_decay_compute.json".into()
        } else {
            "pending_core_compute.json".into()
        }
    });
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&json!({
            "scope":"Native release, five observations. Directed ring with one staged node and two staged weighted edges; one compute loop with ID load, sum normalization, fused outgoing aggregation, top-k, seeded mask, four attachments and one publication. Includes pending capture/reuse and kernels; fixture, initial capture, compilation, algorithm construction and assertions excluded. No before/after speedup, FFI, memory or scale guarantee.",
            "workload":if flow {"checked flow update/residual, two edge attachments"} else if decay {"node ID load, temporal decay, one node attachment"} else {"normalization/fused/top-k/sampling, four node attachments"},
            "cases":cases
        }))?,
    )?;
    println!("Wrote pending core receipt to {path}");
    Ok(())
}

fn child_frames_receipt() -> Result<(), Box<dyn std::error::Error>> {
    let registry = Registry::default();
    register_algorithms(&registry)?;
    let mut cases = Vec::new();
    for count in [100, 1000] {
        for topology in [false, true] {
            let mut steps = Vec::new();
            if topology {
                steps.push(json!({"id":"graph.add_nodes","params":{"count":1,"output":"added"}}));
            }
            steps.push(json!({"id":"sample.iterate_nodes","params":{"target":"seeds"}}));
            steps.push(json!({"id":"sample.for_each","params":{"source":"seeds","target":"mapped","body":[
                {"id":"core.load_node_attr","params":{"attr":"counter","target":"values","default":0}},
                {"id":"core.init_scalar","params":{"value":1,"target":"one"}},
                {"id":"core.add","params":{"left":"values","right":"one","target":"next"}},
                {"id":"core.attach_node_attr","params":{"source":"next","attr":"counter"}}
            ]}}));
            let mut params = AlgorithmParams::new();
            params.set_text("steps", json!({"atomic":true,"steps":steps}).to_string());
            let algorithm = registry.instantiate(&AlgorithmSpec {
                id: "builder.step_pipeline".into(),
                params,
            })?;
            let mut samples = Vec::new();
            for _ in 0..5 {
                let mut graph = Graph::new_publishing();
                graph.insert_nodes(count)?;
                let graph = Arc::new(OwnerCell::new(graph));
                let selected = Subgraph::from_frozen_view(
                    graph.clone(),
                    graph.borrow().read_view()?,
                    "bench".into(),
                );
                let mut ctx = Context::new();
                let started = Instant::now();
                let result = algorithm.execute(&mut ctx, selected)?;
                samples.push(started.elapsed().as_secs_f64());
                let expected = count + usize::from(topology);
                assert_eq!(result.node_count(), expected);
                assert_eq!(graph.borrow().current_state_id(), 2);
                for node in 0..expected {
                    assert_eq!(
                        graph.borrow().get_node_attr(node, &"counter".into())?,
                        Some(AttrValue::Int(1))
                    );
                }
            }
            cases
                .push(json!({"initial_nodes":count,"pending_topology":topology,"seconds":samples}));
        }
    }
    let path = std::env::var("GROGGY_ATOMIC_BUILDER_PROFILE")
        .unwrap_or_else(|_| "child_frames.json".into());
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(
            &json!({"scope":"Native release, five observations each. Node-only graph, optional staged node, immutable singleton production and per-child load/increment/attach, one publication. Includes frame setup, selected pending materialization, validation and execution; excludes fixture/initial capture, compilation, algorithm construction and assertions. No before/after speedup, FFI, RSS or general scale guarantee.","cases":cases}),
        )?,
    )?;
    println!("Wrote child frame receipt to {path}");
    Ok(())
}
