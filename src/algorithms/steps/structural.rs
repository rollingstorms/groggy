//! Structural graph operation step primitives.

use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, Result};

use crate::traits::SubgraphOperations;
use crate::types::{AttrName, NodeId};

use super::super::{AlgorithmParamValue, Context, CostHint};
use super::core::{Step, StepMetadata, StepScope};

/// Compute node degrees and store them as a node map.
pub struct NodeDegreeStep {
    source: Option<String>,
    target: String,
}

impl NodeDegreeStep {
    pub fn new(target: impl Into<String>, source: Option<String>) -> Self {
        Self {
            source,
            target: target.into(),
        }
    }
}

impl Step for NodeDegreeStep {
    fn id(&self) -> &'static str {
        "core.node_degree"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Compute per-node degree within the subgraph".to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, _ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        let subgraph = scope.subgraph();

        // Determine which nodes to evaluate, preserving deterministic ordering when possible.
        let nodes: Vec<NodeId> = if let Some(ref source_name) = self.source {
            if let Ok(column) = scope.variables().node_column(source_name) {
                column.nodes().to_vec()
            } else if let Ok(map) = scope.variables().node_map(source_name) {
                let mut keys: Vec<NodeId> = map.keys().copied().collect();
                keys.sort_unstable();
                keys
            } else {
                subgraph.ordered_nodes().as_ref().to_vec()
            }
        } else {
            subgraph.ordered_nodes().as_ref().to_vec()
        };

        let mut map = HashMap::with_capacity(nodes.len());

        let is_directed = {
            let graph = subgraph.graph();
            let graph_ref = graph.borrow();
            graph_ref.is_directed()
        };

        if is_directed {
            let mut out_counts: HashMap<NodeId, usize> = HashMap::new();
            for &edge_id in scope.edge_ids() {
                if let Ok((source, _)) = subgraph.edge_endpoints(edge_id) {
                    *out_counts.entry(source).or_insert(0) += 1;
                }
            }

            for node in nodes {
                let degree = out_counts.get(&node).copied().unwrap_or(0);
                map.insert(node, AlgorithmParamValue::Int(degree as i64));
            }
        } else {
            for node in nodes {
                let degree = subgraph.degree(node).unwrap_or(0);
                map.insert(node, AlgorithmParamValue::Int(degree as i64));
            }
        }

        scope.variables_mut().set_node_map(self.target.clone(), map);
        Ok(())
    }
}

/// Compute weighted degree using edge attribute weights.
pub struct WeightedDegreeStep {
    weight_attr: AttrName,
    target: String,
}

impl WeightedDegreeStep {
    pub fn new(weight_attr: AttrName, target: impl Into<String>) -> Self {
        Self {
            weight_attr,
            target: target.into(),
        }
    }
}

impl Step for WeightedDegreeStep {
    fn id(&self) -> &'static str {
        "core.weighted_degree"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Compute weighted degree using edge attribute".to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        let mut map = HashMap::with_capacity(scope.subgraph().node_set().len());

        // Initialize all nodes with 0.0
        for &node in scope.node_ids() {
            map.insert(node, AlgorithmParamValue::Float(0.0));
        }

        // Sum edge weights for each node
        for &edge in scope.edge_ids() {
            if ctx.is_cancelled() {
                return Err(anyhow!("weighted_degree cancelled"));
            }

            let (source, target) = scope.subgraph().edge_endpoints(edge)?;
            let weight = scope
                .subgraph()
                .get_edge_attribute(edge, &self.weight_attr)?
                .and_then(AlgorithmParamValue::from_attr_value)
                .unwrap_or(AlgorithmParamValue::Float(1.0));

            let weight_f64 = match weight {
                AlgorithmParamValue::Float(v) => v,
                AlgorithmParamValue::Int(v) => v as f64,
                _ => 1.0,
            };

            // Add weight to both endpoints (undirected)
            if let Some(AlgorithmParamValue::Float(val)) = map.get_mut(&source) {
                *val += weight_f64;
            }
            if let Some(AlgorithmParamValue::Float(val)) = map.get_mut(&target) {
                *val += weight_f64;
            }
        }

        scope.variables_mut().set_node_map(self.target.clone(), map);
        Ok(())
    }
}

/// Mark nodes that belong to a k-core.
pub struct KCoreMarkStep {
    k: usize,
    target: String,
}

impl KCoreMarkStep {
    pub fn new(k: usize, target: impl Into<String>) -> Self {
        Self {
            k,
            target: target.into(),
        }
    }
}

impl Step for KCoreMarkStep {
    fn id(&self) -> &'static str {
        "core.k_core_mark"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: format!("Mark nodes in {}-core", self.k),
            cost_hint: CostHint::Quadratic,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        // Start with all nodes
        let mut remaining: HashSet<NodeId> = scope.node_ids().copied().collect();
        let mut degrees: HashMap<NodeId, usize> = HashMap::new();

        // Compute initial degrees within remaining set
        for &node in &remaining {
            let neighbors = scope.subgraph().neighbors(node)?;
            let degree = neighbors.iter().filter(|n| remaining.contains(n)).count();
            degrees.insert(node, degree);
        }

        // Iteratively remove nodes with degree < k
        let mut changed = true;
        while changed && !remaining.is_empty() {
            if ctx.is_cancelled() {
                return Err(anyhow!("k_core_mark cancelled"));
            }

            changed = false;
            let to_remove: Vec<NodeId> = remaining
                .iter()
                .filter(|&&node| degrees.get(&node).copied().unwrap_or(0) < self.k)
                .copied()
                .collect();

            if !to_remove.is_empty() {
                changed = true;
                for node in to_remove {
                    remaining.remove(&node);
                    degrees.remove(&node);

                    // Update neighbors' degrees
                    let neighbors = scope.subgraph().neighbors(node)?;
                    for neighbor in neighbors {
                        if let Some(deg) = degrees.get_mut(&neighbor) {
                            *deg = deg.saturating_sub(1);
                        }
                    }
                }
            }
        }

        // Create result map: 1 for nodes in k-core, 0 otherwise
        let mut map = HashMap::with_capacity(scope.subgraph().node_set().len());
        for &node in scope.node_ids() {
            let in_core = remaining.contains(&node);
            map.insert(node, AlgorithmParamValue::Int(if in_core { 1 } else { 0 }));
        }

        scope.variables_mut().set_node_map(self.target.clone(), map);
        Ok(())
    }
}

/// Count triangles per node (number of triangles each node participates in).
pub struct TriangleCountStep {
    target: String,
}

impl TriangleCountStep {
    pub fn new(target: impl Into<String>) -> Self {
        Self {
            target: target.into(),
        }
    }
}

impl Step for TriangleCountStep {
    fn id(&self) -> &'static str {
        "core.triangle_count"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Count triangles each node participates in".to_string(),
            cost_hint: CostHint::Quadratic,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        let mut counts: HashMap<NodeId, i64> = HashMap::new();

        // Initialize all nodes with 0
        for &node in scope.node_ids() {
            counts.insert(node, 0);
        }

        let nodes: Vec<NodeId> = scope.node_ids().copied().collect();

        // For each node, check pairs of neighbors for edges
        for &node in &nodes {
            if ctx.is_cancelled() {
                return Err(anyhow!("triangle_count cancelled"));
            }

            let neighbors = scope.subgraph().neighbors(node)?;

            // Check each pair of neighbors
            for i in 0..neighbors.len() {
                for j in (i + 1)..neighbors.len() {
                    let n1 = neighbors[i];
                    let n2 = neighbors[j];

                    // If n1 and n2 are connected, we have a triangle with node
                    if scope.subgraph().has_edge_between(n1, n2)? {
                        // Increment count only for the current node
                        if let Some(count) = counts.get_mut(&node) {
                            *count += 1;
                        }
                    }
                }
            }
        }

        let map: HashMap<NodeId, AlgorithmParamValue> = counts
            .into_iter()
            .map(|(k, v)| (k, AlgorithmParamValue::Int(v)))
            .collect();

        scope.variables_mut().set_node_map(self.target.clone(), map);
        Ok(())
    }
}

/// Sum edge weights incident to each node.
pub struct EdgeWeightSumStep {
    weight_attr: AttrName,
    target: String,
}

impl EdgeWeightSumStep {
    pub fn new(weight_attr: AttrName, target: impl Into<String>) -> Self {
        Self {
            weight_attr,
            target: target.into(),
        }
    }
}

impl Step for EdgeWeightSumStep {
    fn id(&self) -> &'static str {
        "core.edge_weight_sum"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Sum edge weights incident to each node".to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        let mut map: HashMap<NodeId, f64> = HashMap::new();

        // Initialize all nodes with 0.0
        for &node in scope.node_ids() {
            map.insert(node, 0.0);
        }

        // Sum edge weights for each node
        for &edge in scope.edge_ids() {
            if ctx.is_cancelled() {
                return Err(anyhow!("edge_weight_sum cancelled"));
            }

            let (source, target) = scope.subgraph().edge_endpoints(edge)?;
            let weight = scope
                .subgraph()
                .get_edge_attribute(edge, &self.weight_attr)?
                .and_then(AlgorithmParamValue::from_attr_value)
                .unwrap_or(AlgorithmParamValue::Float(1.0));

            let weight_f64 = match weight {
                AlgorithmParamValue::Float(v) => v,
                AlgorithmParamValue::Int(v) => v as f64,
                _ => 1.0,
            };

            // Add weight to both endpoints
            if let Some(val) = map.get_mut(&source) {
                *val += weight_f64;
            }
            if let Some(val) = map.get_mut(&target) {
                *val += weight_f64;
            }
        }

        let result: HashMap<NodeId, AlgorithmParamValue> = map
            .into_iter()
            .map(|(k, v)| (k, AlgorithmParamValue::Float(v)))
            .collect();

        scope
            .variables_mut()
            .set_node_map(self.target.clone(), result);
        Ok(())
    }
}
