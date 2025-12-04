//! Neighbor direction specification for graph traversal operations.

use serde::{Deserialize, Serialize};

/// Direction for neighbor aggregation and traversal operations.
///
/// This enum makes edge direction explicit in algorithm specifications,
/// preventing silent failures from direction mismatches.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NeighborDirection {
    /// Incoming edges: for node N, iterate over nodes M where M→N exists.
    /// Used by: PageRank (aggregate from incoming), LPA (incoming labels).
    In,

    /// Outgoing edges: for node N, iterate over nodes M where N→M exists.
    /// Used by: BFS/DFS (traverse outward), shortest paths.
    Out,

    /// Undirected: for node N, iterate over all connected nodes (both directions).
    /// Used by: undirected graphs, community detection on symmetric networks.
    Undirected,
}

impl Default for NeighborDirection {
    /// Default to incoming edges for backward compatibility with PageRank/LPA patterns.
    fn default() -> Self {
        NeighborDirection::In
    }
}

impl NeighborDirection {
    /// Parse from string representation.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "in" | "incoming" => Some(NeighborDirection::In),
            "out" | "outgoing" => Some(NeighborDirection::Out),
            "undirected" | "both" => Some(NeighborDirection::Undirected),
            _ => None,
        }
    }

    /// Convert to string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            NeighborDirection::In => "in",
            NeighborDirection::Out => "out",
            NeighborDirection::Undirected => "undirected",
        }
    }

    /// Returns true if reverse edges should be added when building CSR.
    pub fn add_reverse_edges(&self) -> bool {
        matches!(self, NeighborDirection::Undirected)
    }

    /// Returns the edge mapping function for CSR construction.
    ///
    /// Given an edge (source, target):
    /// - In: Map to (target, source) for incoming aggregation
    /// - Out: Map to (source, target) for outgoing traversal
    /// - Undirected: Map to (source, target) with add_reverse_edges=true
    pub fn edge_mapping(&self) -> fn((usize, usize)) -> (usize, usize) {
        match self {
            NeighborDirection::In => |(source, target)| (target, source),
            NeighborDirection::Out | NeighborDirection::Undirected => {
                |(source, target)| (source, target)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_direction_parsing() {
        assert_eq!(
            NeighborDirection::from_str("in"),
            Some(NeighborDirection::In)
        );
        assert_eq!(
            NeighborDirection::from_str("IN"),
            Some(NeighborDirection::In)
        );
        assert_eq!(
            NeighborDirection::from_str("incoming"),
            Some(NeighborDirection::In)
        );

        assert_eq!(
            NeighborDirection::from_str("out"),
            Some(NeighborDirection::Out)
        );
        assert_eq!(
            NeighborDirection::from_str("outgoing"),
            Some(NeighborDirection::Out)
        );

        assert_eq!(
            NeighborDirection::from_str("undirected"),
            Some(NeighborDirection::Undirected)
        );
        assert_eq!(
            NeighborDirection::from_str("both"),
            Some(NeighborDirection::Undirected)
        );

        assert_eq!(NeighborDirection::from_str("invalid"), None);
    }

    #[test]
    fn test_default_direction() {
        assert_eq!(NeighborDirection::default(), NeighborDirection::In);
    }

    #[test]
    fn test_reverse_edges() {
        assert!(!NeighborDirection::In.add_reverse_edges());
        assert!(!NeighborDirection::Out.add_reverse_edges());
        assert!(NeighborDirection::Undirected.add_reverse_edges());
    }

    #[test]
    fn test_edge_mapping() {
        let edge = (1, 2); // source=1, target=2

        // Incoming: swap to (target, source)
        assert_eq!(NeighborDirection::In.edge_mapping()(edge), (2, 1));

        // Outgoing: keep as (source, target)
        assert_eq!(NeighborDirection::Out.edge_mapping()(edge), (1, 2));

        // Undirected: keep as (source, target), will add reverse via flag
        assert_eq!(NeighborDirection::Undirected.edge_mapping()(edge), (1, 2));
    }

    #[test]
    fn test_serialization() {
        let dir = NeighborDirection::In;
        let json = serde_json::to_string(&dir).unwrap();
        assert_eq!(json, r#""in""#);

        let deserialized: NeighborDirection = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, NeighborDirection::In);
    }
}
