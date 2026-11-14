//! Bounded native column-power receipts, separate from graph construction/history.
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
    let mut params = AlgorithmParams::new();
    params.set_text("left", "base");
    params.set_text("right", "exponent");
    params.set_text("target", "out");
    let step = global_step_registry().instantiate(&StepSpec {
        id: "core.pow".into(),
        params,
        inputs: vec![],
        outputs: vec![],
    })?;
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
        let mut vars = StepVariables::default();
        vars.set_node_map("base", (0..count).map(|id| (id, V::Int(3))).collect());
        vars.set_scalar("exponent", V::Int(9));
        let mut samples = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            step.apply(
                &mut Context::new(),
                &mut StepScope::new(&selected, &mut vars),
            )?;
            samples.push(start.elapsed().as_secs_f64());
            let result = vars.node_map("out")?;
            assert_eq!(result.len(), count);
            assert!(result.values().all(|value| *value == V::Int(19_683)));
            black_box(result);
        }
        cases.push(json!({"nodes":count,"samples_seconds":samples}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"profile":"release", "fixture":"checked integer cube-of-cube column power, scalar exponent 9", "cases":cases})
        )?
    );
    Ok(())
}
