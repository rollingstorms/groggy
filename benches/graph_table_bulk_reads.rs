//! Native current costs for checked bulk identity and external-neighbor reads.
use groggy::storage::table::{EdgesTable, GraphTable, NodesTable};
use serde_json::json;
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for n in [1_000, 10_000, 100_000] {
        let table = GraphTable::new(
            NodesTable::new((0..n).collect()),
            EdgesTable::new((0..n - 1).map(|r| (r, r, r + 1)).collect()),
        );
        let ids = (0..n).step_by(n / 100).collect::<Vec<_>>();
        for op in [
            "node_lookup_100",
            "edge_lookup_100",
            "external_neighbors_100",
        ] {
            let mut seconds = Vec::new();
            for _ in 0..5 {
                let start = Instant::now();
                match op {
                    "node_lookup_100" => {
                        let result = table.get_nodes(&ids)?;
                        seconds.push(start.elapsed().as_secs_f64());
                        assert_eq!(result.len(), 100);
                        std::hint::black_box(result);
                    }
                    "edge_lookup_100" => {
                        let result = table.get_edges(&ids)?;
                        seconds.push(start.elapsed().as_secs_f64());
                        assert_eq!(result.len(), 100);
                        std::hint::black_box(result);
                    }
                    _ => {
                        let result = table.neighbors(&ids)?;
                        seconds.push(start.elapsed().as_secs_f64());
                        assert!(!result.is_empty());
                        assert!(!result.contains(&0));
                        std::hint::black_box(result);
                    }
                }
            }
            cases.push(json!({"nodes":n,"edges":n-1,"selected":ids.len(),"operation":op,"seconds":seconds}));
        }
    }
    let path = std::env::var("GROGGY_GRAPH_TABLE_BULK_READS_PROFILE")
        .unwrap_or_else(|_| "graph_table_bulk_reads.json".into());
    std::fs::write(
        path,
        serde_json::to_vec_pretty(
            &json!({"scope":"Native optimized checked reads on chain tables, 100 requested IDs, five observations. Complete operation validation and output allocation timed; fixtures, input selection, assertions and output destruction excluded. Current costs only; no matched speedup, RSS or FFI claim.","cases":cases}),
        )?,
    )?;
    Ok(())
}
