//! Retained baseline size, open time, and isolated-process peak RSS.
use groggy::{AttrValue, DurableGraph, Graph};
use serde_json::{json, Value};
use std::{process::Command, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().filter(|arg| arg != "--bench").collect();
    if args.get(1).is_some_and(|v| v == "--open") {
        let start = Instant::now();
        let store = DurableGraph::open(std::path::Path::new(&args[2]))?;
        let elapsed = start.elapsed().as_secs_f64();
        let nodes: usize = args[3].parse()?;
        assert_eq!(store.graph()?.node_count(), nodes);
        assert_eq!(store.graph()?.current_state_id(), 2);
        let pool = store.graph()?.pool();
        let rows = pool
            .get_node_attribute_column(&"payload".into())
            .ok_or("missing values")?;
        assert_eq!(rows.len(), nodes);
        println!(
            "{}",
            json!({"open_seconds":elapsed,"nodes":nodes,"raw_values":rows.len()})
        );
        return Ok(());
    }
    let sizes: Vec<usize> = args
        .get(1)
        .map(|s| s.split(',').map(str::parse).collect())
        .transpose()?
        .unwrap_or_else(|| vec![1, 16, 270]);
    let mut cases = Vec::new();
    for mib in sizes {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("large.ggj");
        let mut graph = Graph::new();
        graph.add_nodes(mib)?;
        let rows = (0..mib)
            .map(|node| {
                let mut text = "x".repeat(1024 * 1024);
                text.replace_range(0..8, &format!("{node:08}"));
                (node, AttrValue::Text(text))
            })
            .collect();
        graph.set_node_attr_column("payload".into(), rows)?;
        let start = Instant::now();
        let mut store = DurableGraph::create_from_graph(&path, &graph, None)?;
        let create = start.elapsed().as_secs_f64();
        drop(graph);
        let state = store.graph()?.current_state_id();
        let start = Instant::now();
        let report = store.checkpoint_storage(state)?;
        let checkpoint = start.elapsed().as_secs_f64();
        drop(store);
        // On macOS /usr/bin/time -l reports maximum resident set size in bytes.
        #[cfg(target_os = "macos")]
        let output = Command::new("/usr/bin/time")
            .arg("-l")
            .arg(std::env::current_exe()?)
            .arg("--open")
            .arg(&path)
            .arg(mib.to_string())
            .output()?;
        #[cfg(not(target_os = "macos"))]
        let output = Command::new(std::env::current_exe()?)
            .arg("--open")
            .arg(&path)
            .arg(mib.to_string())
            .output()?;
        if !output.status.success() {
            return Err(String::from_utf8_lossy(&output.stderr).into_owned().into());
        }
        let open: Value = serde_json::from_slice(&output.stdout)?;
        let stderr = String::from_utf8_lossy(&output.stderr);
        let rss = stderr
            .lines()
            .find(|l| l.contains("maximum resident set size"))
            .and_then(|l| l.split_whitespace().next())
            .and_then(|s| s.parse::<u64>().ok());
        cases.push(json!({"text_payload_mib":mib,"baseline_bytes":report.current_bytes,"create_from_graph_seconds":create,"checkpoint_seconds_including_sync_and_verification":checkpoint,"isolated_open":open,"isolated_open_peak_rss_bytes_macos":rss}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"benchmark":"large_storage_baseline","chunk_payload_bytes":8*1024*1024,"scope":"aggregate retained text, locked local journal, warm filesystem cache; decoded graph/archive allocations remain; no automatic scheduling or SLA","cases":cases})
        )?
    );
    Ok(())
}
