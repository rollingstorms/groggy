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
