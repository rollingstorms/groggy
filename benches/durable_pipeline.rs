//! Phase-separated local baseline for a complete supported durable program.
//! Setup/build excluded; acceptance includes proposed-state preparation and fsync.
use groggy::{
    algorithms::{global_registry, Context, Pipeline, PipelineBuilder},
    subgraphs::Subgraph,
    BatchNodeRef as N, DurableGraph, Graph, GraphBatch,
};
use serde_json::json;
use std::time::Instant;

fn fixture(count: usize) -> anyhow::Result<Graph> {
    let mut graph = Graph::new_publishing();
    graph.apply_batch(
        GraphBatch {
            add_nodes: count,
            add_edges: (1..count)
                .map(|id| (N::Added(id - 1), N::Added(id)))
                .collect(),
            effective_at: Some(0),
            ..Default::default()
        },
        0,
        "fixture".into(),
        "benchmark".into(),
    )?;
    Ok(graph)
}
fn program() -> anyhow::Result<Pipeline> {
    PipelineBuilder::new().with_algorithm("builder.step_pipeline", |p| {
        p.set_text("steps", json!({"atomic":true,"steps":[
            {"id":"graph.add_nodes","params":{"count":1,"output":"new","effective_at":10}},
            {"id":"graph.add_edges","params":{"edges":[[0,{"added":0}]],"output":"edge","effective_at":20}},
            {"id":"core.init_nodes","params":{"target":"value","value":7}},
            {"id":"core.attach_node_attr","params":{"source":"value","attr":"score"}},
            {"id":"graph.capture_staged","params":{"output":"pending"}},
            {"id":"graph.algorithm_table","params":{"source":"pending","algorithm":"bfs","start":0,"output":"distance"}},
            {"id":"table.emit","params":{"source":"distance","name":"distance"}}
        ]}).to_string());
    }).build(global_registry())
}
fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}
fn main() -> anyhow::Result<()> {
    let p = program()?;
    let mut cases = Vec::new();
    for count in [1, 1000, 10000] {
        let mut observations = Vec::new();
        for _ in 0..7 {
            let owner = fixture(count)?.into_shared();
            let selected = Subgraph::from_frozen_view(
                owner.clone(),
                owner.borrow().read_view()?,
                "benchmark".into(),
            );
            let mut context = Context::new();
            let started = Instant::now();
            let result = p.run(&mut context, selected)?;
            let memory = started.elapsed().as_secs_f64();
            assert_eq!(result.nodes().len(), count + 1);
            drop(result);
            drop(context);
            drop(owner);
            let directory = tempfile::tempdir()?;
            let path = directory.path().join("graph.journal");
            let mut store = DurableGraph::create_from_graph(&path, &fixture(count)?, None)?;
            let selected = store.read_view()?;
            let head = store.graph()?.current_state_id();
            let mut context = Context::new();
            let started = Instant::now();
            let prepared = p
                .prepare_durable(&store, &selected)?
                .execute(&mut context)?;
            let prepare = started.elapsed().as_secs_f64();
            let started = Instant::now();
            let accepted = prepared.accept(&mut store)?;
            let accept = started.elapsed().as_secs_f64();
            assert_eq!(accepted.nodes().len(), count + 1);
            assert_eq!(store.graph()?.current_state_id(), head + 1);
            let bytes = std::fs::metadata(&path)?.len();
            drop(store);
            let started = Instant::now();
            let reopened = DurableGraph::open(&path)?;
            let reopen = started.elapsed().as_secs_f64();
            assert_eq!(reopened.graph()?.current_state_id(), head + 1);
            assert_eq!(reopened.read_view()?.node_ids(), accepted.node_ids());
            observations.push(json!({"memory_atomic_seconds":memory,"durable_prepare_seconds":prepare,
                "durable_accept_seconds":accept,"durable_reopen_seconds":reopen,"journal_bytes":bytes}));
        }
        let mut medians = serde_json::Map::new();
        for field in [
            "memory_atomic_seconds",
            "durable_prepare_seconds",
            "durable_accept_seconds",
            "durable_reopen_seconds",
        ] {
            let mut values: Vec<_> = observations
                .iter()
                .map(|row| row[field].as_f64().expect("numeric measurement"))
                .collect();
            medians.insert(field.into(), json!(median(&mut values)));
        }
        cases.push(json!({"accepted_base_nodes":count,"base_edges":count-1,"observations":observations,"medians":medians}));
    }
    let receipt = json!({"scope":"One supported program: create node/edge, initialize/attach, capture, BFS table. Local baseline; not a performance budget or historical speedup.",
        "measurement":"Instant wall time, seven observations per size. Fixture, journal initialization and program compilation excluded. Durable accept includes complete proposed snapshot, journal encoding, append and fsync; reopen includes replay. Memory returns its usual Subgraph continuation; durable returns an accepted immutable view. No interchangeable-operation speedup, RSS/allocation or cross-platform claim.",
        "cases":cases});
    let path = std::env::var("GROGGY_DURABLE_PROFILE")
        .unwrap_or_else(|_| "durable-pipeline-baseline.json".into());
    std::fs::write(&path, serde_json::to_vec_pretty(&receipt)?)?;
    println!("Wrote {path}");
    Ok(())
}
