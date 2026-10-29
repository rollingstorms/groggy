//! Scoped immutable write publication, including complete prepared output construction.
use groggy::{AttrValue as V, BatchNodeRef as N, Graph, GraphBatch};
use serde_json::json;
use std::{collections::HashMap, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for count in [1_000usize, 100_000] {
        let mut graph = Graph::new();
        graph.apply_batch(
            GraphBatch {
                add_nodes: count,
                add_edges: (1..count)
                    .map(|id| (N::Added(id - 1), N::Added(id)))
                    .collect(),
                node_attrs: HashMap::from([
                    (
                        "value".into(),
                        (0..count)
                            .map(|id| (N::Added(id), V::Float64(id as f64)))
                            .collect(),
                    ),
                    (
                        "embedding".into(),
                        (0..count)
                            .map(|id| (N::Added(id), V::Float64Vec(vec![id as f64; 16])))
                            .collect(),
                    ),
                ]),
                ..Default::default()
            },
            0,
            "fixture".into(),
            "bench".into(),
        )?;
        let initial = graph.read_view()?.select_nodes(&[0, 1])?;
        let mut view = initial.clone();
        let mut samples = Vec::new();
        for iteration in 0..5 {
            let score = 1000. + iteration as f64;
            let batch = GraphBatch {
                node_attrs: HashMap::from([(
                    "value".into(),
                    vec![(N::Existing(0), V::Float64(score))],
                )]),
                ..Default::default()
            };
            let before = graph.current_state_id();
            let start = Instant::now();
            let result = graph.apply_view_batch(&view, batch, "score".into(), "bench".into())?;
            samples.push(start.elapsed().as_secs_f64());
            assert_eq!(result.batch.state_id, before + 1);
            assert_eq!(
                result.view.node_attr(0, &"value".into())?,
                Some(V::Float64(score))
            );
            assert_eq!(initial.node_attr(0, &"value".into())?, Some(V::Float64(0.)));
            assert_eq!(result.view.node_ids(), vec![0, 1]);
            assert_eq!(result.view.edge_ids(), vec![0]);
            view = result.view;
        }
        cases.push(json!({"nodes":count,"edges":count-1,"vector_dimensions":16,"selected_nodes":2,"selected_edges":1,"publication_seconds":samples,"final_state":graph.current_state_id()}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"profile":"release","fixture":"chain, f64 scalar and 16-element f64 vector per node; two-node immutable scope; five sequential score writes with original capture retained; includes publication and refreshed payload construction","cases":cases})
        )?
    );
    Ok(())
}
