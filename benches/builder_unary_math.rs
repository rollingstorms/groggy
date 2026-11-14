//! Bounded registered unary-math receipts with checked complete outputs.
use groggy::ownership::OwnerCell;
use groggy::{
    algorithms::{
        steps::{
            ensure_core_steps_registered, global_step_registry, StepScope, StepSpec, StepVariables,
        },
        AlgorithmParamValue as V, AlgorithmParams, Context,
    },
    subgraphs::Subgraph,
    Graph,
};
use serde_json::json;
use std::collections::HashSet;
use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    ensure_core_steps_registered();
    let mut cases = Vec::new();
    for count in [1_000usize, 100_000] {
        let mut graph = Graph::new();
        graph.add_nodes(count)?;
        let selected = Subgraph::new(
            Arc::new(OwnerCell::new(graph)),
            (0..count).collect(),
            HashSet::new(),
            "benchmark".into(),
        );
        for (id, value, expected) in [
            ("core.abs", V::Int(-3), V::Int(3)),
            ("core.sqrt", V::Int(9), V::Float(3.)),
            ("core.exp", V::Int(0), V::Float(1.)),
            ("core.log", V::Int(16), V::Float(2.)),
        ] {
            let mut params = AlgorithmParams::new();
            params.set_text("source", "in");
            params.set_text("target", "out");
            if id == "core.log" {
                params.set_int("base", 4);
            }
            let step = global_step_registry().instantiate(&StepSpec {
                id: id.into(),
                params,
                inputs: vec![],
                outputs: vec![],
            })?;
            let mut vars = StepVariables::default();
            vars.set_node_map("in", (0..count).map(|key| (key, value.clone())).collect());
            let mut samples = Vec::new();
            for _ in 0..5 {
                let start = Instant::now();
                step.apply(
                    &mut Context::new(),
                    &mut StepScope::new(&selected, &mut vars),
                )?;
                samples.push(start.elapsed().as_secs_f64());
                let out = vars.node_map("out")?;
                assert_eq!(out.len(), count);
                assert!(out.values().all(|value| *value == expected));
                black_box(out);
            }
            cases.push(json!({"nodes":count,"operation":id,"samples_seconds":samples}));
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"profile":"release","fixture":"registered checked unary math over node maps","cases":cases})
        )?
    );
    Ok(())
}
