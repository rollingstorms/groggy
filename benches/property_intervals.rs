//! Bounded property/lifecycle posting lookup versus an independent linear timeline.
use groggy::{AttrValue as V, BatchNodeRef as N, Graph, GraphBatch};
use serde_json::json;
use std::{hint::black_box, time::Instant};

type Row = (i64, Option<i64>, Option<V>);

// Independent reference for this monotonic, continuously active fixture.
fn scan(history: &[(i64, &str, i64)]) -> Vec<Row> {
    let mut rows: Vec<Row> = Vec::new();
    for &(at, name, value) in history {
        if name == "x" {
            if let Some(last) = rows.last_mut() {
                last.1 = Some(at);
            }
            rows.push((at, None, Some(V::Int(value))));
        }
    }
    rows
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for count in [100usize, 10_000] {
        let mut graph = Graph::new();
        graph.apply_batch(
            GraphBatch {
                add_nodes: 1,
                effective_at: Some(0),
                node_attrs: [("x".into(), vec![(N::Added(0), V::Int(1))])].into(),
                ..Default::default()
            },
            0,
            "root".into(),
            "bench".into(),
        )?;
        let mut reference = vec![(0, "x", 1)];
        for i in 1..=count + 1 {
            let name = if i == count + 1 { "x" } else { "noise" };
            let value = if name == "x" { 2 } else { i as i64 };
            graph.apply_batch(
                GraphBatch {
                    node_attrs: [(name.into(), vec![(N::Existing(0), V::Int(value))])].into(),
                    effective_at: Some(i as i64),
                    ..Default::default()
                },
                graph.current_state_id(),
                "update".into(),
                "bench".into(),
            )?;
            reference.push((i as i64, name, value));
        }
        let start = Instant::now();
        let index = graph.build_effective_index()?;
        let build = start.elapsed().as_secs_f64();
        let expected = scan(&reference);
        let indexed = || -> Result<Vec<Row>, groggy::GraphError> {
            Ok(index
                .node_property_intervals(0, "x", None)?
                .into_iter()
                .map(|r| (r.start, r.end, r.value))
                .collect())
        };
        assert_eq!(indexed()?, expected);
        let mut samples = Vec::new();
        for iteration in 0..5 {
            let mut times = [0.; 2];
            for kind in if iteration % 2 == 0 { [0, 1] } else { [1, 0] } {
                let start = Instant::now();
                for _ in 0..100 {
                    let rows = if kind == 0 {
                        indexed()?
                    } else {
                        scan(black_box(&reference))
                    };
                    assert_eq!(rows, expected);
                    black_box(rows);
                }
                times[kind] = start.elapsed().as_secs_f64() / 100.;
            }
            samples.push(
                json!({"indexed_interval_seconds":times[0], "linear_timeline_seconds":times[1]}),
            );
        }
        cases.push(json!({"unrelated_property_events":count,"total_events":index.event_count(),"intervals":expected.len(),"build_seconds":build,"samples":samples}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"profile":"release","fixture":"one continuously active node; two x assignments and 100/10000 intervening noise assignments","cases":cases})
        )?
    );
    Ok(())
}
