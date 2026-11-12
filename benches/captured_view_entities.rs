//! Scalar ID reads stay independent of collection ordering; bulk columns allocate explicitly.
use groggy::{AttrValue as V, BatchNodeRef as N, Graph, GraphBatch};
use rayon::prelude::*;
use serde_json::json;
use std::{hint::black_box, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for count in [1_000usize, 100_000] {
        let mut graph = Graph::new();
        graph.apply_batch(
            GraphBatch {
                add_nodes: count,
                node_attrs: [(
                    "score".into(),
                    (0..count)
                        .filter(|i| i % 3 != 2)
                        .map(|i| {
                            (
                                N::Added(i),
                                if i % 3 == 0 {
                                    V::Int(i as i64)
                                } else {
                                    V::Null
                                },
                            )
                        })
                        .collect(),
                )]
                .into(),
                ..Default::default()
            },
            0,
            "fixture".into(),
            "bench".into(),
        )?;
        let view = graph.read_view()?;
        let nodes = view.node_accessor();
        let name = "score".to_owned();
        let target = count - 1;
        let mut direct = Vec::new();
        let mut fresh = Vec::new();
        let mut reused = Vec::new();
        for _ in 0..5 {
            for (kind, times) in [(0, &mut direct), (1, &mut fresh), (2, &mut reused)] {
                let start = Instant::now();
                for _ in 0..10_000 {
                    let value = match kind {
                        0 => view.node_attr(target, &name)?,
                        1 => view.node_accessor().by_id(target)?.get(&name),
                        _ => nodes.by_id(target)?.get(&name),
                    };
                    assert_eq!(value, Some(V::Int(target as i64)));
                    black_box(value);
                }
                times.push(start.elapsed().as_secs_f64() / 10_000.);
            }
        }
        let start = Instant::now();
        assert_eq!(nodes.ids().len(), count);
        let ids = start.elapsed().as_secs_f64();
        let mut column = Vec::new();
        let mut presence = Vec::new();
        let mut masking = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            assert_eq!(nodes.column(&name).len(), count);
            column.push(start.elapsed().as_secs_f64());
            let start = Instant::now();
            let mask = nodes.presence(&name);
            presence.push(start.elapsed().as_secs_f64());
            let start = Instant::now();
            assert_eq!(nodes.select_mask(&mask)?.node_count(), count - count / 3);
            masking.push(start.elapsed().as_secs_f64());
        }
        cases.push(json!({"nodes":count,"first_sorted_ids_seconds":ids,"direct_scalar_seconds":direct,"fresh_accessor_scalar_seconds":fresh,"reused_accessor_scalar_seconds":reused,"owned_column_seconds":column,"presence_seconds":presence,"mask_selection_seconds":masking}));
    }
    let grouped = grouped_projection()?;
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"profile":"release","fixture":"accepted isolates; one typed integer/null/absent column; scalar reads run before any ID-order materialization; setup excluded; five samples; 10000 scalar calls/sample","cases":cases,"grouped_projection":grouped})
        )?
    );
    Ok(())
}

fn grouped_projection() -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let count = 100_000;
    let mut graph = Graph::new_manual();
    graph.add_nodes(count)?;
    for node in 0..count {
        graph.set_node_attr(node, "x".into(), V::Int(node as i64))?;
    }
    let root = graph.read_view()?;
    let mut cases = Vec::new();
    for groups in [1, 100] {
        let views = (0..groups)
            .map(|group| root.select_nodes(&(group..count).step_by(groups).collect::<Vec<_>>()))
            .collect::<Result<Vec<_>, _>>()?;
        let name = "x".to_owned();
        let mut sorting = Vec::new();
        let mut sequential = Vec::new();
        let mut parallel = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            black_box(views.iter().map(|view| view.node_ids()).collect::<Vec<_>>());
            sorting.push(start.elapsed().as_secs_f64());
            let start = Instant::now();
            let values = views
                .iter()
                .map(|view| view.node_accessor().present_column(&name))
                .collect::<Vec<_>>();
            sequential.push(start.elapsed().as_secs_f64());
            assert_eq!(
                values.iter().map(|column| column.len()).sum::<usize>(),
                count
            );
            black_box(values);
            let start = Instant::now();
            let values = views
                .par_iter()
                .map(|view| view.node_accessor().present_column(&name))
                .collect::<Vec<_>>();
            parallel.push(start.elapsed().as_secs_f64());
            assert_eq!(
                values.iter().map(|column| column.len()).sum::<usize>(),
                count
            );
            black_box(values);
        }
        cases.push(json!({"nodes":count,"groups":groups,"sort_only_seconds":sorting,"sequential_group_observations_seconds":sequential,"parallel_group_observations_seconds":parallel}));
    }
    Ok(
        json!({"threads":rayon::current_num_threads(),"scope":"fresh entity accessors each sample; capture/grouping and output validation excluded; per-column reader unchanged; parallel comparison changes only group scheduling; no Python/GIL transitions", "cases":cases}),
    )
}
