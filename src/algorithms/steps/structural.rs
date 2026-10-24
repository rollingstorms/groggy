//! Structural graph operation step primitives.

use std::collections::{HashMap, VecDeque};

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
    fn supports_atomic_properties(&self) -> bool {
        true
    }

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
        let topology = undirected_topology(ctx, scope)?;
        let mut degrees: Vec<_> = topology.neighbors.iter().map(Vec::len).collect();
        let mut retained = vec![true; degrees.len()];
        let mut queued = vec![false; degrees.len()];
        let mut queue = VecDeque::new();
        for (node, &degree) in degrees.iter().enumerate() {
            if degree < self.k {
                queued[node] = true;
                queue.push_back(node);
            }
        }
        while let Some(node) = queue.pop_front() {
            if ctx.is_cancelled() {
                return Err(anyhow!("k_core_mark cancelled"));
            }
            retained[node] = false;
            for &neighbor in &topology.neighbors[node] {
                if ctx.is_cancelled() {
                    return Err(anyhow!("k_core_mark cancelled"));
                }
                if retained[neighbor] {
                    degrees[neighbor] = degrees[neighbor]
                        .checked_sub(1)
                        .ok_or_else(|| anyhow!("Invalid k-core degree accounting"))?;
                    if degrees[neighbor] < self.k && !queued[neighbor] {
                        queued[neighbor] = true;
                        queue.push_back(neighbor);
                    }
                }
            }
        }
        if ctx.is_cancelled() {
            return Err(anyhow!("k_core_mark cancelled"));
        }
        let result = topology
            .nodes
            .into_iter()
            .enumerate()
            .map(|(index, node)| (node, AlgorithmParamValue::Int(i64::from(retained[index]))))
            .collect();
        scope
            .variables_mut()
            .set_node_map(self.target.clone(), result);
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
    fn supports_atomic_properties(&self) -> bool {
        true
    }

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
        let mut topology = undirected_topology(ctx, scope)?;
        // Orient the simple projection by node ID: every triangle is visited once.
        for (node, neighbors) in topology.neighbors.iter_mut().enumerate() {
            neighbors.dedup();
            neighbors.retain(|&neighbor| neighbor > node);
        }
        let mut counts = vec![0i64; topology.nodes.len()];
        for (u, neighbors) in topology.neighbors.iter().enumerate() {
            if ctx.is_cancelled() {
                return Err(anyhow!("triangle_count cancelled"));
            }
            for (position, &v) in neighbors.iter().enumerate() {
                let mut left = position + 1;
                let mut right = 0;
                let other = &topology.neighbors[v];
                while left < neighbors.len() && right < other.len() {
                    if ctx.is_cancelled() {
                        return Err(anyhow!("triangle_count cancelled"));
                    }
                    match neighbors[left].cmp(&other[right]) {
                        std::cmp::Ordering::Less => left += 1,
                        std::cmp::Ordering::Greater => right += 1,
                        std::cmp::Ordering::Equal => {
                            for node in [u, v, neighbors[left]] {
                                counts[node] = counts[node]
                                    .checked_add(1)
                                    .ok_or_else(|| anyhow!("Triangle count exceeds i64"))?;
                            }
                            left += 1;
                            right += 1;
                        }
                    }
                }
            }
        }
        if ctx.is_cancelled() {
            return Err(anyhow!("triangle_count cancelled"));
        }
        let result = topology
            .nodes
            .into_iter()
            .zip(counts)
            .map(|(node, count)| (node, AlgorithmParamValue::Int(count)))
            .collect();
        scope
            .variables_mut()
            .set_node_map(self.target.clone(), result);
        Ok(())
    }
}

struct UndirectedTopology {
    nodes: Vec<NodeId>,
    neighbors: Vec<Vec<usize>>,
}

/// Preserve selected multiedges and once-per-loop incidence. Directed variants
/// need separate definitions; never infer one from endpoint storage order.
fn undirected_topology(ctx: &Context, scope: &StepScope<'_>) -> Result<UndirectedTopology> {
    if ctx.is_cancelled() {
        return Err(anyhow!("structural topology cancelled"));
    }
    let selection = scope.subgraph();
    let graph = selection.graph();
    let graph = graph.borrow();
    if graph.is_directed() {
        return Err(anyhow!("Triangle and k-core builder steps require an undirected graph; directed definitions are not implemented"));
    }
    let nodes = selection.ordered_nodes().to_vec();
    let positions: HashMap<_, _> = nodes
        .iter()
        .enumerate()
        .map(|(index, &node)| (node, index))
        .collect();
    for &node in &nodes {
        if !graph.has_node(node) {
            return Err(anyhow!("Structural node {} is inactive", node));
        }
    }
    let mut neighbors = vec![Vec::new(); nodes.len()];
    for &edge in selection.ordered_edges().iter() {
        if ctx.is_cancelled() {
            return Err(anyhow!("structural topology cancelled"));
        }
        if !graph.has_edge(edge) {
            return Err(anyhow!("Structural edge {} is inactive", edge));
        }
        let (source, target) = graph.edge_endpoints(edge)?;
        let source = *positions
            .get(&source)
            .ok_or_else(|| anyhow!("Edge {} source is outside selection", edge))?;
        let target = *positions
            .get(&target)
            .ok_or_else(|| anyhow!("Edge {} target is outside selection", edge))?;
        neighbors[source].push(target);
        if source != target {
            neighbors[target].push(source);
        }
    }
    for row in &mut neighbors {
        if ctx.is_cancelled() {
            return Err(anyhow!("structural topology cancelled"));
        }
        row.sort_unstable();
    }
    Ok(UndirectedTopology { nodes, neighbors })
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
    fn structural_steps_match_exhaustive_five_node_simple_graph_references() {
        let pairs: Vec<_> = (0..5)
            .flat_map(|u| (u + 1..5).map(move |v| (u, v)))
            .collect();
        for mask in 0..(1usize << pairs.len()) {
            let mut graph = Graph::new_undirected();
            graph.add_nodes(5);
            let mut adjacency = [[false; 5]; 5];
            for (bit, &(u, v)) in pairs.iter().enumerate() {
                if mask & (1 << bit) != 0 {
                    adjacency[u][v] = true;
                    adjacency[v][u] = true;
                    graph.add_edge(u, v).expect("edge");
                }
            }
            let selection = Subgraph::from_nodes(
                Rc::new(RefCell::new(graph)),
                (0..5).collect(),
                "exhaustive".into(),
            )
            .expect("selection");
            let mut vars = StepVariables::default();
            let mut scope = StepScope::new(&selection, &mut vars);
            TriangleCountStep::new("triangles")
                .apply(&mut Context::new(), &mut scope)
                .expect("triangles");
            for node in 0..5 {
                let mut expected = 0;
                for u in 0..5 {
                    for v in u + 1..5 {
                        if adjacency[node][u] && adjacency[node][v] && adjacency[u][v] {
                            expected += 1;
                        }
                    }
                }
                assert_eq!(
                    scope.variables().node_map("triangles").expect("map")[&node],
                    AlgorithmParamValue::Int(expected),
                    "graph {} node {}",
                    mask,
                    node
                );
            }
            for k in 0..=5 {
                let mut expected = [true; 5];
                loop {
                    let next = std::array::from_fn(|node| {
                        expected[node]
                            && (0..5)
                                .filter(|&other| expected[other] && adjacency[node][other])
                                .count()
                                >= k
                    });
                    if next == expected {
                        break;
                    }
                    expected = next;
                }
                KCoreMarkStep::new(k, "core")
                    .apply(&mut Context::new(), &mut scope)
                    .expect("core");
                for (node, value) in expected.into_iter().enumerate() {
                    assert_eq!(
                        scope.variables().node_map("core").expect("map")[&node],
                        AlgorithmParamValue::Int(i64::from(value)),
                        "graph {} k {} node {}",
                        mask,
                        k,
                        node
                    );
                }
            }
        }
    }

    #[test]
    fn structural_steps_keep_selected_noncontiguous_ids_and_reject_cancellation() {
        use std::sync::{atomic::AtomicBool, Arc};
        let mut graph = Graph::new_undirected();
        graph.add_nodes(5);
        for (u, v) in [(1, 2), (2, 3), (3, 1), (3, 4)] {
            graph.add_edge(u, v).expect("edge");
        }
        let selection = Subgraph::from_nodes(
            Rc::new(RefCell::new(graph)),
            [1, 2, 3].into_iter().collect(),
            "selected".into(),
        )
        .expect("selection");
        let mut vars = StepVariables::default();
        let mut scope = StepScope::new(&selection, &mut vars);
        for step in [
            Box::new(TriangleCountStep::new("result")) as Box<dyn Step>,
            Box::new(KCoreMarkStep::new(2, "result")),
        ] {
            step.apply(&mut Context::new(), &mut scope)
                .expect("selected calculation");
            assert_eq!(scope.variables().node_map("result").expect("map").len(), 3);
            assert!(scope
                .variables()
                .node_map("result")
                .expect("map")
                .values()
                .all(|value| *value == AlgorithmParamValue::Int(1)));
            let mut ctx = Context::with_cancel_token(Arc::new(AtomicBool::new(true)));
            assert!(step.apply(&mut ctx, &mut scope).is_err());
            assert!(scope
                .variables()
                .node_map("result")
                .expect("unchanged")
                .values()
                .all(|value| *value == AlgorithmParamValue::Int(1)));
        }
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
