//! Current costs: a fresh source capture versus reuse of one existing immutable view.
use groggy::{storage::table::Table, subgraphs::Subgraph, AttrValue as V, Graph};
use serde_json::json;
use std::{
    collections::{HashMap, HashSet},
    time::Instant,
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for n in [1_000, 10_000, 100_000] {
        let mut g = Graph::new();
        g.add_nodes(n);
        g.add_edges(&(0..n - 1).map(|r| (r, r + 1)).collect::<Vec<_>>())?;
        g.set_node_attrs(HashMap::from([(
            "x".into(),
            (0..n).map(|r| (r, V::Float64(r as f64 + 0.125))).collect(),
        )]))?;
        g.commit("fixture".into(), "benchmark".into())?;
        let graph = g.into_shared();
        let view = graph.borrow().read_view()?;
        let selection = Subgraph::from_nodes(
            graph.clone(),
            (0..n / 100).collect::<HashSet<_>>(),
            "selected".into(),
        )?;
        for operation in [
            "fresh_nodes",
            "fresh_bundle",
            "reused_nodes",
            "reused_bundle",
            "selected_bundle_fresh_capture",
        ] {
            let mut seconds = Vec::new();
            for _ in 0..3 {
                let start = Instant::now();
                match operation {
                    "fresh_nodes" => {
                        let output = graph.borrow().nodes_table()?;
                        seconds.push(start.elapsed().as_secs_f64());
                        assert_eq!(output.nrows(), n);
                        std::hint::black_box(output);
                    }
                    "fresh_bundle" => {
                        let output = graph.borrow().table()?;
                        seconds.push(start.elapsed().as_secs_f64());
                        assert_eq!(output.nodes().nrows(), n);
                        std::hint::black_box(output);
                    }
                    "reused_nodes" => {
                        let output = view.nodes_table()?;
                        seconds.push(start.elapsed().as_secs_f64());
                        assert_eq!(output.nrows(), n);
                        std::hint::black_box(output);
                    }
                    "reused_bundle" => {
                        let output = view.table()?;
                        seconds.push(start.elapsed().as_secs_f64());
                        assert_eq!(output.nodes().nrows(), n);
                        std::hint::black_box(output);
                    }
                    _ => {
                        let output = selection.table()?;
                        seconds.push(start.elapsed().as_secs_f64());
                        assert_eq!(output.nodes().nrows(), n / 100);
                        std::hint::black_box(output);
                    }
                }
            }
            cases.push(json!({"parent_nodes":n,"parent_edges":n-1,"operation":operation,"seconds":seconds}));
        }
    }
    let path = std::env::var("GROGGY_TABLE_EXPORT_CAPTURE_PROFILE")
        .unwrap_or_else(|_| "table_export_capture.json".into());
    std::fs::write(
        path,
        serde_json::to_vec_pretty(
            &json!({"scope":"Native optimized current-cost comparison of fresh immutable graph captures and reuse of a pre-captured GraphView, three observations. Selected bundle uses 1% of parent nodes. Complete capture/selection/projection/source binding timed as applicable; fixtures, shared view creation for reused cases, assertions and output destruction excluded. Fresh capture copies parent read state, not just selected columns. No matched old-version speedup, RSS, COW or FFI-budget claim.","cases":cases}),
        )?,
    )?;
    Ok(())
}
