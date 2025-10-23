mod betweenness;
mod closeness;
mod pagerank;
mod traversal_utils;

pub use betweenness::BetweennessCentrality;
pub use closeness::ClosenessCentrality;
pub use pagerank::PageRank;

use crate::algorithms::registry::Registry;

pub fn register_algorithms(registry: &Registry) -> anyhow::Result<()> {
    pagerank::register(registry)?;
    betweenness::register(registry)?;
    closeness::register(registry)?;
    Ok(())
}

// Future centrality work: weighted betweenness/closeness will reuse
// traversal_utils for Brandes-style multi-source traversals. Extend this module
// with additional `mod` declarations and update register_algorithms when those
// land.
