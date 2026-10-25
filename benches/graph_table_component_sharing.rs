//! Current costs for shared composite components, induced grouping and first mutable access.
use groggy::storage::table::Table;
use groggy::{
    storage::table::{EdgesTable, GraphTable, NodesTable},
    AttrValue as V, GraphError,
};
use serde_json::json;
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for n in [1_000, 10_000, 100_000] {
        let nodes = NodesTable::new((0..n).collect()).with_attributes(
            "g".into(),
            (0..n).map(|r| (r, V::Int((r % 100) as i64))).collect(),
        )?;
        let table = GraphTable::new(
            nodes,
            EdgesTable::new((0..n - 1).map(|r| (r, r, r + 1)).collect()),
        );
        for op in [
            "clone_drop_shared_components",
            "group_nodes_100",
            "first_node_mutable_access",
        ] {
            let mut seconds = Vec::new();
            for _ in 0..5 {
                let mut prepared = table.clone();
                let start = Instant::now();
                match op {
                    "clone_drop_shared_components" => {
                        for _ in 0..10_000 {
                            std::hint::black_box(table.clone());
                        }
                        seconds.push(start.elapsed().as_secs_f64() / 10_000.0);
                        assert!(std::ptr::eq(prepared.nodes(), table.nodes()));
                        assert!(std::ptr::eq(prepared.edges(), table.edges()));
                    }
                    "group_nodes_100" => {
                        let groups = table.group_by(&["g".into()])?;
                        seconds.push(start.elapsed().as_secs_f64());
                        if groups.len() != 100 || groups.iter().any(|g| g.edges().nrows() != 0) {
                            return Err(GraphError::InvalidInput(
                                "induced grouping mismatch".into(),
                            )
                            .into());
                        }
                        std::hint::black_box(groups);
                    }
                    _ => {
                        std::hint::black_box(prepared.nodes_mut());
                        seconds.push(start.elapsed().as_secs_f64());
                        assert!(!std::ptr::eq(prepared.nodes(), table.nodes()));
                        assert!(std::ptr::eq(prepared.edges(), table.edges()));
                        std::hint::black_box(prepared);
                    }
                }
            }
            cases.push(json!({"nodes":n,"edges":n-1,"operation":op,"iterations_per_observation":if op == "clone_drop_shared_components" {10_000} else {1},"seconds":seconds}));
        }
    }
    let path = std::env::var("GROGGY_COMPONENT_SHARING_PROFILE")
        .unwrap_or_else(|_| "graph_table_component_sharing.json".into());
    std::fs::write(
        path,
        serde_json::to_vec_pretty(
            &json!({"scope":"Native optimized current costs, five observations. Clone shares component table/schema objects above already-shared BaseArray value buffers; node grouping validates identity/topology and creates 100 induced groups by dispatching edges once; the chain fixture uses r modulo 100, so every edge crosses groups and all output edge tables are empty; first mutable access refreshes presentation identity and detaches node table/schema metadata while column buffers remain shared. Clone/drop is amortized over 10000 iterations per observation to exceed clock resolution; its destruction is included. Other operations time one output and exclude output destruction. Fixtures, prepared clone creation and assertions excluded; complete operation/allocation timed. Pointer checks prove shared untouched components, not an RSS estimate. The older bundle-retention receipt has different grouping semantics and is not a matched speedup comparison. No whole-graph capture COW, RSS, Python getter or FFI-budget claim.","cases":cases}),
        )?,
    )?;
    Ok(())
}
