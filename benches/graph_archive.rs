//! Explicit full-retention archive cost, separate from ordinary mutation throughput.
use groggy::{AttrValue, BatchNodeRef, Graph, GraphBatch};
use serde_json::json;
use std::{collections::HashMap, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for nodes in [1_000usize, 10_000, 100_000] {
        let mut graph = Graph::new_publishing();
        graph.apply_batch(
            GraphBatch {
                add_nodes: nodes,
                node_attrs: HashMap::from([(
                    "score".into(),
                    (0..nodes)
                        .map(|i| (BatchNodeRef::Added(i), AttrValue::Float64(i as f64)))
                        .collect(),
                )]),
                effective_at: Some(0),
                ..Default::default()
            },
            0,
            "root".into(),
            "bench".into(),
        )?;
        graph.create_branch("root".into())?;
        graph
            .writer("bench")
            .with_effective_at(Some(10))
            .set_node_attrs(HashMap::from([(
                "score".into(),
                (0..nodes)
                    .map(|i| (i, AttrValue::Float64(i as f64 + 0.5)))
                    .collect(),
            )]))?;
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("graph.gga");
        let start = Instant::now();
        graph.save_archive(&path)?;
        let save = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let mut restored = Graph::load_archive(&path)?;
        let load = start.elapsed().as_secs_f64();
        assert_eq!(restored.node_count(), nodes);
        assert_eq!(
            restored.get_node_attr(nodes - 1, &"score".into())?,
            Some(AttrValue::Float64(nodes as f64 - 0.5))
        );
        let count = restored
            .pool()
            .get_node_attribute_column(&"score".into())
            .map_or(0, |c| c.len());
        assert_eq!(count, 2 * nodes);
        restored.checkout_branch("root".into())?;
        assert_eq!(
            restored.get_node_attr(nodes - 1, &"score".into())?,
            Some(AttrValue::Float64((nodes - 1) as f64))
        );
        cases.push(json!({"nodes":nodes,"versions":2,"retained_score_values":count,"archive_bytes":std::fs::metadata(&path)?.len(),"save_seconds_including_validation_and_sync":save,"load_seconds":load}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({"benchmark":"graph_archive","cases":cases}))?
    );
    Ok(())
}
