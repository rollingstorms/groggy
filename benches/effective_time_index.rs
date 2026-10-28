//! Bounded explicit effective-time index receipts and independent metadata scans.
use groggy::{
    query::EffectiveTimeEvent, AttrValue as V, BatchNodeRef as N, ChangeKey, Graph, GraphBatch,
};
use serde_json::json;
use std::{hint::black_box, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for count in [100usize, 10_000] {
        let mut graph = Graph::new();
        graph.apply_batch(
            GraphBatch {
                add_nodes: 1,
                effective_at: Some(0),
                ..Default::default()
            },
            0,
            "root".into(),
            "bench".into(),
        )?;
        let mut reference = Vec::new();
        for i in 1..=count {
            let at = i as i64;
            let state = graph
                .apply_batch(
                    GraphBatch {
                        node_attrs: [("x".into(), vec![(N::Existing(0), V::Int(at))])].into(),
                        effective_at: Some(at),
                        ..Default::default()
                    },
                    graph.current_state_id(),
                    "update".into(),
                    "bench".into(),
                )?
                .state_id;
            let metadata = graph
                .publication_metadata(state)?
                .ok_or("Missing metadata")?;
            reference.push(EffectiveTimeEvent {
                state_id: state,
                effective_at: at,
                recorded_at: metadata.recorded_at,
                change: ChangeKey::NodeAttribute(0, "x".into()),
            });
        }
        let start = Instant::now();
        let index = graph.build_effective_index()?;
        let build = start.elapsed().as_secs_f64();
        let lower = count as i64 - 9;
        let upper = count as i64;
        let expected: Vec<_> = reference
            .iter()
            .filter(|event| event.effective_at >= lower && event.effective_at <= upper)
            .cloned()
            .collect();
        assert_eq!(index.events_between(lower, upper, None)?, expected);
        assert_eq!(
            index.read_view_at(upper, None)?.node_attr(0, &"x".into())?,
            Some(V::Int(upper))
        );
        let mut samples = Vec::new();
        for iteration in 0..5 {
            let mut indexed = 0.;
            let mut scanned = 0.;
            for kind in if iteration % 2 == 0 { [0, 1] } else { [1, 0] } {
                let start = Instant::now();
                for _ in 0..100 {
                    let rows = if kind == 0 {
                        index.events_between(lower, upper, None)?
                    } else {
                        reference
                            .iter()
                            .filter(|event| {
                                event.effective_at >= lower && event.effective_at <= upper
                            })
                            .cloned()
                            .collect()
                    };
                    assert_eq!(rows, expected);
                    black_box(rows);
                }
                let seconds = start.elapsed().as_secs_f64() / 100.;
                if kind == 0 {
                    indexed = seconds;
                } else {
                    scanned = seconds;
                }
            }
            let start = Instant::now();
            let view = index.read_view_at(upper, None)?;
            let point = start.elapsed().as_secs_f64();
            assert_eq!(view.node_attr(0, &"x".into())?, Some(V::Int(upper)));
            black_box(view);
            samples.push(json!({"indexed_window_seconds":indexed,"linear_window_seconds":scanned,"full_prefix_read_seconds":point}));
        }
        cases.push(json!({"property_events":count,"total_events":index.event_count(),"build_seconds":build,"window_hits":expected.len(),"samples":samples}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"profile":"release","fixture":"one entity with monotonic effective property events; final ten-event window","cases":cases})
        )?
    );
    Ok(())
}
