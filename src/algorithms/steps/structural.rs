//! Structural graph operation step primitives.

use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, Result};

use crate::traits::SubgraphOperations;
use crate::types::{AttrName, AttrValue, NodeId};

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
    fn supports_atomic_properties(&self) -> bool {
        true
    }

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

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        if ctx.is_cancelled() {
            return Err(anyhow!("node_degree cancelled"));
        }
        let subgraph = scope.subgraph();
        let nodes = if let Some(source) = &self.source {
            if let Ok(column) = scope.variables().node_column(source) {
                column.nodes().to_vec()
            } else {
                let mut nodes: Vec<_> = scope
                    .variables()
                    .node_map(source)?
                    .keys()
                    .copied()
                    .collect();
                nodes.sort_unstable();
                nodes
            }
        } else {
            subgraph.ordered_nodes().to_vec()
        };
        let graph = subgraph.graph();
        let graph = graph.borrow();
        let mut counts = HashMap::with_capacity(nodes.len());
        for node in nodes {
            if !subgraph.has_node(node) || !graph.has_node(node) {
                return Err(anyhow!(
                    "node_degree source node {} is outside the active selection",
                    node
                ));
            }
            counts.insert(node, 0i64);
        }
        for &edge in subgraph.ordered_edges().iter() {
            if ctx.is_cancelled() {
                return Err(anyhow!("node_degree cancelled"));
            }
            let (source, target) = subgraph.edge_endpoints(edge)?;
            for endpoint in [
                Some(source),
                (!graph.is_directed() && target != source).then_some(target),
            ]
            .into_iter()
            .flatten()
            {
                if let Some(count) = counts.get_mut(&endpoint) {
                    *count = count
                        .checked_add(1)
                        .ok_or_else(|| anyhow!("node degree exceeds i64"))?;
                }
            }
        }
        drop(graph);
        scope.variables_mut().set_node_map(
            self.target.clone(),
            counts
                .into_iter()
                .map(|(node, count)| (node, AlgorithmParamValue::Int(count)))
                .collect(),
        );
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
    fn supports_atomic_properties(&self) -> bool {
        true
    }

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
        let result = incident_weight_sums(ctx, scope, &self.weight_attr)?;
        scope
            .variables_mut()
            .set_node_map(self.target.clone(), result);
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
    fn supports_atomic_properties(&self) -> bool {
        true
    }

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
        let result = incident_weight_sums(ctx, scope, &self.weight_attr)?;
        scope
            .variables_mut()
            .set_node_map(self.target.clone(), result);
        Ok(())
    }
}

/// Total incidence weight: both endpoints contribute, even on directed graphs;
/// self-loops contribute twice. Missing weights default to one; present values
/// must be finite numeric scalars. Signed weights are valid for this sum.
fn incident_weight_sums(
    ctx: &Context,
    scope: &StepScope<'_>,
    weight_attr: &AttrName,
) -> Result<HashMap<NodeId, AlgorithmParamValue>> {
    if ctx.is_cancelled() {
        return Err(anyhow!("incident weight sum cancelled"));
    }
    let subgraph = scope.subgraph();
    let mut sums = HashMap::with_capacity(subgraph.node_count());
    {
        let graph = subgraph.graph();
        let graph = graph.borrow();
        for &node in subgraph.ordered_nodes().iter() {
            if !graph.has_node(node) {
                return Err(anyhow!(
                    "incident weight node {} is outside the active graph",
                    node
                ));
            }
            sums.insert(node, 0.0f64);
        }
    }
    for &edge in subgraph.ordered_edges().iter() {
        if ctx.is_cancelled() {
            return Err(anyhow!("incident weight sum cancelled"));
        }
        let (source, target) = subgraph.edge_endpoints(edge)?;
        let weight = match scope.edge_attribute(ctx, edge, weight_attr)? {
            None => 1.0,
            Some(AttrValue::Int(value)) => value as f64,
            Some(AttrValue::SmallInt(value)) => f64::from(value),
            Some(AttrValue::Float(value)) => f64::from(value),
            Some(AttrValue::Float64(value)) => value,
            Some(_) => {
                return Err(anyhow!(
                    "edge {} weight '{}' must be numeric",
                    edge,
                    weight_attr
                ))
            }
        };
        if !weight.is_finite() {
            return Err(anyhow!(
                "edge {} weight '{}' must be finite",
                edge,
                weight_attr
            ));
        }
        for node in [source, target] {
            let sum = sums.get_mut(&node).ok_or_else(|| {
                anyhow!(
                    "edge {} endpoint {} is outside the selected graph",
                    edge,
                    node
                )
            })?;
            *sum += weight;
            if !sum.is_finite() {
                return Err(anyhow!(
                    "incident weight sum at node {} is not finite",
                    node
                ));
            }
        }
    }
    Ok(sums
        .into_iter()
        .map(|(node, value)| (node, AlgorithmParamValue::Float(value)))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        algorithms::steps::{core::NodeColumn, StepVariables},
        api::graph::Graph,
        subgraphs::Subgraph,
    };
    use std::{cell::RefCell, rc::Rc};

    #[test]
    fn degree_uses_selected_edges_and_rejects_foreign_source_nodes() {
        let mut graph = Graph::new_directed();
        graph.add_nodes(3);
        for (u, v) in [(0, 1), (0, 1), (0, 0), (2, 0)] {
            graph.add_edge(u, v).expect("edge");
        }
        let selection = Subgraph::from_nodes(
            Rc::new(RefCell::new(graph)),
            [0, 1].into_iter().collect(),
            "selected".into(),
        )
        .expect("selection");
        let mut vars = StepVariables::default();
        let mut scope = StepScope::new(&selection, &mut vars);
        NodeDegreeStep::new("degrees", None)
            .apply(&mut Context::new(), &mut scope)
            .expect("degrees");
        assert_eq!(
            scope.variables().node_map("degrees").expect("map").get(&0),
            Some(&AlgorithmParamValue::Int(3))
        );
        scope.variables_mut().set_node_column(
            "foreign",
            NodeColumn::new(vec![2], vec![AlgorithmParamValue::Int(0)]),
        );
        assert!(NodeDegreeStep::new("degrees", Some("foreign".into()))
            .apply(&mut Context::new(), &mut scope)
            .is_err());
        assert_eq!(
            scope
                .variables()
                .node_map("degrees")
                .expect("unchanged")
                .len(),
            2
        );
    }

    #[test]
    fn cancelled_degree_leaves_existing_result_intact() {
        use std::sync::{atomic::AtomicBool, Arc};
        let selection = Subgraph::from_nodes(
            Rc::new(RefCell::new(Graph::new())),
            Default::default(),
            "empty".into(),
        )
        .expect("selection");
        let mut vars = StepVariables::default();
        vars.set_node_map(
            "result",
            [(99, AlgorithmParamValue::Int(7))].into_iter().collect(),
        );
        let mut scope = StepScope::new(&selection, &mut vars);
        let mut ctx = Context::with_cancel_token(Arc::new(AtomicBool::new(true)));
        assert!(NodeDegreeStep::new("result", None)
            .apply(&mut ctx, &mut scope)
            .is_err());
        assert!(scope
            .variables()
            .node_map("result")
            .expect("original")
            .contains_key(&99));
    }
}
