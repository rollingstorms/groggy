//! Native current costs for the complete typed table bundle boundary.
use groggy::{
    storage::table::{EdgesTable, GraphTable, NodesTable},
    AttrValue as V,
};
use serde_json::json;
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for n in [1_000, 10_000, 100_000] {
        let nodes = NodesTable::new((0..n).collect())
            .with_attributes(
                "x".into(),
                (0..n).map(|r| (r, V::Float64(r as f64 + 0.125))).collect(),
            )?
            .with_attributes(
                "embedding".into(),
                (0..n)
                    .map(|r| (r, V::Float64Vec(vec![1.0, 2.0, 3.0, 4.0])))
                    .collect(),
            )?;
        let edges = EdgesTable::new((0..n - 1).map(|r| (r, r, r + 1)).collect()).with_attributes(
            "weight".into(),
            (0..n - 1).map(|r| (r, V::Float64(1.25))).collect(),
        )?;
        let table = GraphTable::new(nodes, edges);
        let dir = tempfile::tempdir()?;
        table.save_bundle(dir.path())?;
        let bytes = std::fs::metadata(dir.path().join("bundle.cbor"))?.len();
        for operation in [
            "save_replace_sync",
            "load_verify_decode",
            "verify_decode_metadata",
        ] {
            let mut seconds = Vec::new();
            for _ in 0..3 {
                let start = Instant::now();
                match operation {
                    "save_replace_sync" => {
                        table.save_bundle(dir.path())?;
                    }
                    "load_verify_decode" => {
                        std::hint::black_box(GraphTable::load_bundle(dir.path())?);
                    }
                    _ => {
                        std::hint::black_box(GraphTable::validate_bundle(dir.path())?);
                    }
                }
                seconds.push(start.elapsed().as_secs_f64());
            }
            let metadata = GraphTable::validate_bundle(dir.path())?;
            assert_eq!(metadata.node_count, n);
            assert_eq!(metadata.edge_count, n - 1);
            cases.push(json!({"nodes":n,"edges":n-1,"bytes":bytes,"operation":operation,"seconds":seconds}));
        }
    }
    let path = std::env::var("GROGGY_GRAPH_TABLE_BUNDLES_PROFILE")
        .unwrap_or_else(|_| "graph_table_bundles.json".into());
    std::fs::write(
        path,
        serde_json::to_vec_pretty(
            &json!({"scope":"Native optimized complete v3 bundle costs. Three observations; node scalar/four-f64 vector plus chain edge weight. Fixtures excluded. Save includes encode, checksum, temp write, file sync, replacement and directory sync. Load/verify include bytes, checksum, schema/value decoding and metadata construction; output destruction is inside these timers. Warm OS caches; no cold-open, speedup, RSS, streaming or FFI claim.","cases":cases}),
        )?,
    )?;
    Ok(())
}
