//! Membership algebra and grouping over one immutable capture.
use super::{GraphView, SimilarityMetric};
use crate::{AttrName, AttrValue, EdgeId, GraphError, GraphResult, NodeId};
use std::collections::{BTreeMap, HashSet};

impl GraphView {
    fn require_common_capture(&self, other: &Self) -> GraphResult<()> {
        if !self.read_state().shares_capture(other.read_state()) {
            return Err(GraphError::InvalidInput(
                "View algebra requires selections derived from the same capture".into(),
            ));
        }
        Ok(())
    }

    /// Union exact node/edge membership from one capture; never induce extra edges.
    pub fn merge_with(&self, other: &Self) -> GraphResult<Self> {
        self.require_common_capture(other)?;
        Self::from_membership(
            self.read_state().clone(),
            self.nodes().union(other.nodes()).copied().collect(),
            self.edges().union(other.edges()).copied().collect(),
        )
    }

    /// Intersection of exact node/edge membership from one capture.
    pub fn intersect_with(&self, other: &Self) -> GraphResult<Self> {
        self.require_common_capture(other)?;
        Self::from_membership(
            self.read_state().clone(),
            self.nodes().intersection(other.nodes()).copied().collect(),
            self.edges().intersection(other.edges()).copied().collect(),
        )
    }

    /// Remove the other's nodes and edges, including all edges incident to removed nodes.
    pub fn subtract_from(&self, other: &Self) -> GraphResult<Self> {
        self.require_common_capture(other)?;
        let nodes: HashSet<_> = self.nodes().difference(other.nodes()).copied().collect();
        let mut edges = HashSet::new();
        for &edge in self.edges().difference(other.edges()) {
            let (source, target) = self.edge_endpoints(edge)?;
            if nodes.contains(&source) && nodes.contains(&target) {
                edges.insert(edge);
            }
        }
        Self::from_membership(self.read_state().clone(), nodes, edges)
    }

    /// Node-membership similarity; IDs from unrelated captures cannot be compared.
    pub fn calculate_similarity(&self, other: &Self, metric: SimilarityMetric) -> GraphResult<f64> {
        self.require_common_capture(other)?;
        Ok(membership_similarity(self.nodes(), other.nodes(), metric))
    }

    /// Nonempty intersections, preserving input order and the original capture.
    pub fn find_overlaps(&self, others: &[Self]) -> GraphResult<Vec<Self>> {
        for other in others {
            self.require_common_capture(other)?;
        }
        others
            .iter()
            .map(|other| self.intersect_with(other))
            .filter_map(|result| match result {
                Ok(view) if view.node_count() == 0 => None,
                other => Some(other),
            })
            .collect()
    }

    /// Group present captured node values in AttrValue's canonical order; null is a value.
    /// Missing attributes are omitted. Each child preserves the parent's selected edge scope.
    pub fn group_by_nodes(&self, attr: &AttrName) -> GraphResult<Vec<Self>> {
        Ok(self
            .grouped_nodes(attr)?
            .into_iter()
            .map(|(_, view)| view)
            .collect())
    }

    pub(crate) fn grouped_nodes(&self, attr: &AttrName) -> GraphResult<Vec<(AttrValue, Self)>> {
        let mut groups: BTreeMap<AttrValue, Vec<NodeId>> = BTreeMap::new();
        for node in self.node_ids() {
            if let Some(value) = self.node_attr(node, attr)? {
                groups.entry(value).or_default().push(node);
            }
        }
        groups
            .into_iter()
            .map(|(value, nodes)| Ok((value, self.select_nodes(&nodes)?)))
            .collect()
    }

    /// Group present captured edge values; children contain exact instances and endpoints.
    pub fn group_by_edges(&self, attr: &AttrName) -> GraphResult<Vec<Self>> {
        Ok(self
            .grouped_edges(attr)?
            .into_iter()
            .map(|(_, view)| view)
            .collect())
    }

    pub(crate) fn grouped_edges(&self, attr: &AttrName) -> GraphResult<Vec<(AttrValue, Self)>> {
        let mut groups: BTreeMap<AttrValue, Vec<EdgeId>> = BTreeMap::new();
        for edge in self.edge_ids() {
            if let Some(value) = self.edge_attr(edge, attr)? {
                groups.entry(value).or_default().push(edge);
            }
        }
        groups
            .into_iter()
            .map(|(value, edges)| Ok((value, self.select_edges(&edges)?)))
            .collect()
    }
}

pub(crate) fn membership_similarity(
    left: &HashSet<NodeId>,
    right: &HashSet<NodeId>,
    metric: SimilarityMetric,
) -> f64 {
    let intersection = left.intersection(right).count() as f64;
    let denominator = match metric {
        SimilarityMetric::Jaccard => left.union(right).count() as f64,
        SimilarityMetric::Dice => (left.len() as f64 + right.len() as f64) / 2.,
        SimilarityMetric::Cosine => (left.len() as f64).sqrt() * (right.len() as f64).sqrt(),
        SimilarityMetric::Overlap => left.len().min(right.len()) as f64,
    };
    if denominator == 0. {
        0.
    } else {
        intersection / denominator
    }
}

impl std::str::FromStr for SimilarityMetric {
    type Err = GraphError;
    fn from_str(value: &str) -> GraphResult<Self> {
        match value {
            "jaccard" => Ok(Self::Jaccard),
            "dice" => Ok(Self::Dice),
            "cosine" => Ok(Self::Cosine),
            "overlap" => Ok(Self::Overlap),
            _ => Err(GraphError::InvalidInput(
                "similarity metric must be jaccard, dice, cosine or overlap".into(),
            )),
        }
    }
}
