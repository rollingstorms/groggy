//! Descriptive checked-loop comparison; no service latency or memory claim.
use groggy::{
    algorithms::{global_registry, Context, Pipeline, PipelineBuilder},
    subgraphs::Subgraph,
    AttrValue, DurableGraph, Graph, GraphBatch,
};
use serde_json::json;
use std::{fs, time::Instant};
fn program(lowered: bool, iterations: usize) -> anyhow::Result<Pipeline> {
    let loop_step = if lowered {
        json!({"id":"iter.loop","params":{"body":[],"iterations":iterations,"batch_plan":{
        "slot_count":2,"carried_slots":[],"instructions":[
            {"type":"load_node_prop","dst":0,"var_name":"x"},
            {"type":"load_scalar","dst":1,"value":1.01},
            {"type":"mul","dst":0,"lhs":0,"rhs":1},
            {"type":"store_node_prop","src":0,"var_name":"x"}
        ]}}})
    } else {
        json!({"id":"iter.loop","params":{"iterations":iterations,"body":[
            {"id":"core.init_scalar","params":{"target":"factor","value":1.01}},
            {"id":"core.mul","params":{"left":"x","right":"factor","target":"x"}}
        ]}})
    };
    PipelineBuilder::new()
        .with_algorithm("builder.step_pipeline", |p| {
            p.set_text("steps", json!({"atomic":true,"steps":[
            {"id":"core.init_nodes","params":{"target":"x","value":1.}},loop_step,
            {"id":"core.attach_node_attr","params":{"source":"x","attr":"answer","effective_at":10}}
        ]}).to_string());
        })
        .build(global_registry())
}
fn fixture(count: usize) -> anyhow::Result<Graph> {
    let mut graph = Graph::new();
    graph.apply_batch(
        GraphBatch {
            add_nodes: count,
            effective_at: Some(0),
            ..Default::default()
        },
        0,
        "fixture".into(),
        "bench".into(),
    )?;
    Ok(graph)
}
fn memory(program: &Pipeline, count: usize, expected: f64) -> anyhow::Result<f64> {
    let owner = fixture(count)?.into_shared();
    let selected =
        Subgraph::from_frozen_view(owner.clone(), owner.borrow().read_view()?, "bench".into());
    let started = Instant::now();
    program.run(&mut Context::new(), selected)?;
    let elapsed = started.elapsed().as_secs_f64();
    assert_eq!(
        owner.borrow().get_node_attr(count - 1, &"answer".into())?,
        Some(AttrValue::Float64(expected))
    );
    Ok(elapsed)
}
fn main() -> anyhow::Result<()> {
    let mut cases = Vec::new();
    for iterations in [10, 100] {
        let native = program(false, iterations)?;
        let lowered = program(true, iterations)?;
        let expected = (0..iterations).fold(1., |x, _| x * 1.01);
        // Populate both lazy native-loop caches before timed observations.
        memory(&native, 1, expected)?;
        memory(&lowered, 1, expected)?;
        for count in [1, 1000, 10000] {
            let mut samples = Vec::new();
            for sample in 0..7 {
                // Alternate ordering to reduce a consistent first-run bias.
                let (native_seconds, lowered_seconds) = if sample % 2 == 0 {
                    (
                        memory(&native, count, expected)?,
                        memory(&lowered, count, expected)?,
                    )
                } else {
                    let l = memory(&lowered, count, expected)?;
                    (memory(&native, count, expected)?, l)
                };
                let folder = tempfile::tempdir()?;
                let path = folder.path().join("graph.journal");
                let mut store = DurableGraph::create_from_graph(&path, &fixture(count)?, None)?;
                let source = store.read_view()?;
                let mut ctx = Context::new();
                let started = Instant::now();
                let prepared = lowered
                    .prepare_durable(&store, &source)?
                    .execute(&mut ctx)?;
                let prepare_seconds = started.elapsed().as_secs_f64();
                let started = Instant::now();
                let accepted = prepared.accept(&mut store)?;
                let accept_seconds = started.elapsed().as_secs_f64();
                assert_eq!(
                    accepted.node_attr(count - 1, &"answer".into())?,
                    Some(AttrValue::Float64(expected))
                );
                samples.push(json!({"native_step_loop_seconds":native_seconds,"lowered_plan_seconds":lowered_seconds,"durable_prepare_seconds":prepare_seconds,"durable_accept_seconds":accept_seconds,"answer":expected}));
            }
            cases.push(json!({"nodes":count,"iterations":iterations,"samples":samples}));
        }
    }
    let report = json!({"scope":"Same f64 recurrence through native step loop and checked numeric register-plan frontend; seven observations, warmed plan caches, graph/journal setup excluded. Native/lowered memory times include final atomic publication; durable preparation and synced acceptance timed separately. No old vector-interpreter, RSS, service budget or speedup claim.","cases":cases});
    let text = serde_json::to_string_pretty(&report)?;
    if let Ok(path) = std::env::var("GROGGY_CHECKED_BATCH_PROFILE") {
        fs::write(path, &text)?;
    }
    println!("{text}");
    Ok(())
}
