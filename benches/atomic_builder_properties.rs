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
