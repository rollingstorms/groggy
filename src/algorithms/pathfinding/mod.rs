mod astar;
mod bfs_dfs;
mod dijkstra;
pub mod utils;

pub use astar::AStarPathfinding;
pub use bfs_dfs::{BfsTraversal, DfsTraversal};
pub use dijkstra::DijkstraShortestPath;

use crate::algorithms::registry::Registry;

pub fn register_algorithms(registry: &Registry) -> anyhow::Result<()> {
    bfs_dfs::register(registry)?;
    dijkstra::register(registry)?;
    astar::register(registry)?;
    Ok(())
}

// A* currently supports heuristic lookup via node attributes – additional heuristic
// providers can hook into the existing registration in future iterations.
