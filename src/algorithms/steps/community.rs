//! Community detection helper step primitives.

use std::collections::HashMap;

use anyhow::{anyhow, Result};

use crate::traits::SubgraphOperations;
use crate::types::NodeId;

use super::super::{AlgorithmParamValue, Context, CostHint};
use super::core::{NodeColumn, Step, StepMetadata, StepScope};

/// Community seeding strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeedStrategy {
    /// Each node starts in its own community (node ID as label).
    Singleton,
    /// Assign communities based on degree: high-degree nodes get unique labels,
    /// low-degree nodes share with neighbors.
    DegreeBased,
    /// Random assignment of k community labels.
    Random,
}

impl SeedStrategy {
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "singleton" => Ok(SeedStrategy::Singleton),
            "degree" | "degree_based" => Ok(SeedStrategy::DegreeBased),
            "random" => Ok(SeedStrategy::Random),
            _ => Err(anyhow!(
                "unknown seed strategy '{}'; use 'singleton', 'degree', or 'random'",
                s
            )),
        }
    }
}

/// Initialize community labels for nodes based on a seeding strategy.
///
/// Parameters:
/// - `strategy`: SeedStrategy (singleton, degree_based, random)
/// - `k`: Number of communities (used for random strategy)
/// - `seed`: Optional RNG seed for reproducibility (used for random strategy)
/// - `target`: Output variable name for node map of community labels
pub struct CommunitySeedStep {
    strategy: SeedStrategy,
    k: Option<usize>,
    seed: Option<u64>,
    target: String,
}

impl CommunitySeedStep {
    pub fn new(
        strategy: SeedStrategy,
        target: impl Into<String>,
        k: Option<usize>,
        seed: Option<u64>,
    ) -> Self {
        Self {
            strategy,
            k,
            seed,
            target: target.into(),
        }
    }
}

impl Step for CommunitySeedStep {
    fn id(&self) -> &'static str {
        "core.community_seed"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Initialize community labels using a seeding strategy".to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        let nodes: Vec<NodeId> = scope.node_ids().copied().collect();
        let mut values = Vec::with_capacity(nodes.len());

        match self.strategy {
            SeedStrategy::Singleton => {
                for &node in &nodes {
                    values.push(AlgorithmParamValue::Int(node as i64));
                }
            }
            SeedStrategy::DegreeBased => {
                // Sort nodes by degree (descending)
                let mut degree_pairs: Vec<(NodeId, usize)> = nodes
                    .iter()
                    .map(|&n| {
                        let deg = scope.subgraph().degree(n).unwrap_or(0);
                        (n, deg)
                    })
                    .collect();
                degree_pairs.sort_by(|a, b| b.1.cmp(&a.1));

                // High-degree nodes (top 10% or at least 1) get unique labels
                let num_hubs = ((nodes.len() as f64 * 0.1).ceil() as usize).max(1);
                let mut assigned: HashMap<NodeId, i64> = HashMap::new();

                // Assign unique labels to hubs
                for i in 0..num_hubs.min(degree_pairs.len()) {
                    let node = degree_pairs[i].0;
                    assigned.insert(node, node as i64);
                }

                // Assign remaining nodes to a neighbor's community if possible
                for &node in &nodes {
                    if assigned.contains_key(&node) {
                        continue;
                    }

                    if ctx.is_cancelled() {
                        return Err(anyhow!("community_seed cancelled"));
                    }

                    let neighbors = scope.subgraph().neighbors(node)?;
                    let neighbor_label = neighbors
                        .iter()
                        .find_map(|&n| assigned.get(&n).copied())
                        .unwrap_or(node as i64);
                    assigned.insert(node, neighbor_label);
                }

                for &node in &nodes {
                    let label = assigned.get(&node).copied().unwrap_or(node as i64);
                    values.push(AlgorithmParamValue::Int(label));
                }
            }
            SeedStrategy::Random => {
                let k = self
                    .k
                    .ok_or_else(|| anyhow!("random strategy requires 'k' parameter"))?;
                if k == 0 {
                    return Err(anyhow!("k must be greater than zero"));
                }

                // Use seed if provided, otherwise system randomness
                if let Some(seed) = self.seed {
                    fastrand::seed(seed);
                }

                // Randomly assign each node to one of k communities
                for _node in &nodes {
                    let label = fastrand::usize(0..k) as i64;
                    values.push(AlgorithmParamValue::Int(label));
                }
            }
        }

        scope
            .variables_mut()
            .set_node_column(self.target.clone(), NodeColumn::new(nodes, values));
        Ok(())
    }
}

/// Compute the change in modularity for each node if it were to move to a different community.
///
/// This step computes the modularity gain/loss for each node relative to its current assignment,
/// useful for iterative community refinement algorithms like Louvain.
///
/// Parameters:
/// - `partition`: Input variable (node map) with current community labels
/// - `target`: Output variable (node map) with modularity gain values
pub struct ModularityGainStep {
    partition: String,
    target: String,
}

impl ModularityGainStep {
    pub fn new(partition: impl Into<String>, target: impl Into<String>) -> Self {
        Self {
            partition: partition.into(),
            target: target.into(),
        }
    }
}

impl Step for ModularityGainStep {
    fn id(&self) -> &'static str {
        "core.modularity_gain"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Compute modularity change for each node relative to current partition"
                .to_string(),
            cost_hint: CostHint::Quadratic, // Needs neighbor scanning
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        let partition = scope.variables().node_column(&self.partition)?.clone();
        let nodes: Vec<NodeId> = partition.nodes().to_vec();
        let total_edges = scope.edge_ids().count() as f64;
        let cache = scope.neighbor_cache()?;

        // Compute total edge weight (2m)
        if total_edges == 0.0 {
            let zeros = vec![AlgorithmParamValue::Float(0.0); nodes.len()];
            scope
                .variables_mut()
                .set_node_column(self.target.clone(), NodeColumn::new(nodes, zeros));
            return Ok(());
        }

        let two_m = total_edges * 2.0;

        let mut gains = Vec::with_capacity(nodes.len());

        for &node in &nodes {
            if ctx.is_cancelled() {
                return Err(anyhow!("modularity_gain cancelled"));
            }

            let current_comm = partition
                .get(node)
                .and_then(|v| match v {
                    AlgorithmParamValue::Int(i) => Some(*i),
                    _ => None,
                })
                .unwrap_or(node as i64);

            let neighbors = cache
                .neighbors(node)
                .ok_or_else(|| anyhow!("node {node} missing from neighbor cache"))?;

            // Count edges to communities and find best alternative
            let mut comm_edges: HashMap<i64, f64> = HashMap::new();
            for &neighbor in neighbors {
                let neighbor_comm = partition
                    .get(neighbor)
                    .and_then(|v| match v {
                        AlgorithmParamValue::Int(i) => Some(*i),
                        _ => None,
                    })
                    .unwrap_or(neighbor as i64);
                *comm_edges.entry(neighbor_comm).or_insert(0.0) += 1.0;
            }

            // Compute gain: deltaQ = (k_i_in - k_i * Sigma_tot / (2m)) / m
            // Simplified: we compute the relative benefit of moving to the best community
            let mut best_gain = 0.0;

            for (&comm, &edges_to_comm) in &comm_edges {
                if comm == current_comm {
                    continue; // Already in this community
                }

                // Approximate gain: (edges_to_comm / two_m) - (node_degree * comm_degree / (two_m * two_m))
                // Simplified: just count the edge fraction as a proxy
                let gain = edges_to_comm / two_m;
                if gain > best_gain {
                    best_gain = gain;
                }
            }

            gains.push(AlgorithmParamValue::Float(best_gain));
        }

        scope
            .variables_mut()
            .set_node_column(self.target.clone(), NodeColumn::new(nodes, gains));
        Ok(())
    }
}

/// Perform a single iteration of label propagation: each node adopts the most common label
/// among its neighbors.
///
/// This is the core operation of the Label Propagation Algorithm (LPA). Run this step
/// iteratively until convergence (labels stop changing).
///
/// Parameters:
/// - `labels`: Input variable (node map) with current community labels
/// - `target`: Output variable (node map) with updated labels
pub struct LabelPropagateStep {
    labels: String,
    target: String,
}

impl LabelPropagateStep {
    pub fn new(labels: impl Into<String>, target: impl Into<String>) -> Self {
        Self {
            labels: labels.into(),
            target: target.into(),
        }
    }
}

impl Step for LabelPropagateStep {
    fn id(&self) -> &'static str {
        "core.label_propagate_step"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Single iteration of label propagation: adopt most common neighbor label"
                .to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        let current_labels = scope.variables().node_column(&self.labels)?.clone();
        let cache = scope.neighbor_cache()?;
        let nodes: Vec<NodeId> = current_labels.nodes().to_vec();
        let mut new_values = Vec::with_capacity(nodes.len());

        for &node in &nodes {
            if ctx.is_cancelled() {
                return Err(anyhow!("label_propagate_step cancelled"));
            }

            let current = current_labels
                .get(node)
                .cloned()
                .unwrap_or(AlgorithmParamValue::Int(node as i64));

            let neighbors = cache
                .neighbors(node)
                .ok_or_else(|| anyhow!("node {node} missing from neighbor cache"))?;
            if neighbors.is_empty() {
                // No neighbors: keep current label
                new_values.push(current);
                continue;
            }

            // Count neighbor labels (including current node's label with weight 1)
            let mut counts: HashMap<String, (AlgorithmParamValue, usize)> = HashMap::new();
            let current_repr = format!("{:?}", current);
            counts.insert(current_repr.clone(), (current.clone(), 1));

            for &neighbor in neighbors {
                let neighbor_label = current_labels
                    .get(neighbor)
                    .cloned()
                    .unwrap_or(AlgorithmParamValue::Int(neighbor as i64));
                let repr = format!("{:?}", neighbor_label);
                let entry = counts.entry(repr).or_insert((neighbor_label, 0));
                entry.1 += 1;
            }

            // Find the most common label (ties broken by lexicographic order for determinism)
            let mut best_repr = current_repr.clone();
            let mut best_count = 1usize;
            for (repr, (_, count)) in &counts {
                if *count > best_count || (*count == best_count && repr < &best_repr) {
                    best_repr = repr.clone();
                    best_count = *count;
                }
            }

            // Adopt the best label (re-parse from representation)
            let best_label = counts
                .get(&best_repr)
                .map(|(value, _)| value.clone())
                .unwrap_or(current);

            new_values.push(best_label);
        }

        scope
            .variables_mut()
            .set_node_column(self.target.clone(), NodeColumn::new(nodes, new_values));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::steps::StepVariables;
    use crate::algorithms::Context;
    use crate::api::graph::Graph;
    use crate::subgraphs::Subgraph;
    use std::cell::RefCell;
    use std::collections::HashSet;
    use std::rc::Rc;

    fn make_test_graph() -> (Graph, Vec<NodeId>) {
        let mut graph = Graph::new();
        let nodes: Vec<NodeId> = (0..6).map(|_| graph.add_node()).collect();

        // Create two clusters: 0-1-2 and 3-4-5
        graph.add_edge(nodes[0], nodes[1]).unwrap();
        graph.add_edge(nodes[1], nodes[0]).unwrap();
        graph.add_edge(nodes[1], nodes[2]).unwrap();
        graph.add_edge(nodes[2], nodes[1]).unwrap();

        graph.add_edge(nodes[3], nodes[4]).unwrap();
        graph.add_edge(nodes[4], nodes[3]).unwrap();
        graph.add_edge(nodes[4], nodes[5]).unwrap();
        graph.add_edge(nodes[5], nodes[4]).unwrap();

        (graph, nodes)
    }

    #[test]
    fn test_community_seed_singleton() {
        let (graph, nodes) = make_test_graph();
        let node_set: HashSet<NodeId> = nodes.iter().copied().collect();
        let subgraph =
            Subgraph::from_nodes(Rc::new(RefCell::new(graph)), node_set, "test".into()).unwrap();

        let step = CommunitySeedStep::new(SeedStrategy::Singleton, "labels", None, None);
        let mut ctx = Context::new();
        let mut vars = StepVariables::default();
        let mut scope = StepScope::new(&subgraph, &mut vars);

        step.apply(&mut ctx, &mut scope).unwrap();

        let column = scope.variables().node_column("labels").unwrap();
        for (node, value) in column.iter() {
            assert_eq!(
                value,
                &AlgorithmParamValue::Int(node as i64),
                "Node {} should have label {}",
                node,
                node
            );
        }
    }

    #[test]
    fn test_community_seed_random() {
        let (graph, nodes) = make_test_graph();
        let node_set: HashSet<NodeId> = nodes.iter().copied().collect();
        let subgraph =
            Subgraph::from_nodes(Rc::new(RefCell::new(graph)), node_set, "test".into()).unwrap();

        let step = CommunitySeedStep::new(SeedStrategy::Random, "labels", Some(3), Some(42));
        let mut ctx = Context::new();
        let mut vars = StepVariables::default();
        let mut scope = StepScope::new(&subgraph, &mut vars);

        step.apply(&mut ctx, &mut scope).unwrap();

        let column = scope.variables().node_column("labels").unwrap();
        for (_, value) in column.iter() {
            if let AlgorithmParamValue::Int(i) = value {
                assert!(*i >= 0 && *i < 3, "Label {} should be in [0, 3)", i);
            } else {
                panic!("Expected Int label");
            }
        }
    }

    #[test]
    fn test_label_propagate_step() {
        let (graph, nodes) = make_test_graph();
        let node_set: HashSet<NodeId> = nodes.iter().copied().collect();
        let subgraph =
            Subgraph::from_nodes(Rc::new(RefCell::new(graph)), node_set, "test".into()).unwrap();

        // Initialize with singleton labels
        let seed_step = CommunitySeedStep::new(SeedStrategy::Singleton, "labels", None, None);
        let mut ctx = Context::new();
        let mut vars = StepVariables::default();
        let mut scope = StepScope::new(&subgraph, &mut vars);
        seed_step.apply(&mut ctx, &mut scope).unwrap();

        // Run one label propagation step
        let lpa_step = LabelPropagateStep::new("labels", "new_labels");
        lpa_step.apply(&mut ctx, &mut scope).unwrap();

        let new_labels = scope.variables().node_column("new_labels").unwrap();
        // Node 1 should adopt label from node 0 or 2 (its neighbors)
        let label_1 = new_labels.get(nodes[1]).unwrap();
        assert!(
            *label_1 == AlgorithmParamValue::Int(nodes[0] as i64)
                || *label_1 == AlgorithmParamValue::Int(nodes[2] as i64)
                || *label_1 == AlgorithmParamValue::Int(nodes[1] as i64)
        );
    }

    #[test]
    fn test_modularity_gain_empty_graph() {
        let graph = Graph::new();
        let subgraph =
            Subgraph::from_nodes(Rc::new(RefCell::new(graph)), HashSet::new(), "empty".into())
                .unwrap();

        let step = ModularityGainStep::new("labels", "gains");
        let mut ctx = Context::new();
        let mut vars = StepVariables::default();
        let mut scope = StepScope::new(&subgraph, &mut vars);

        // Add empty partition
        scope.variables_mut().set_node_column(
            "labels".to_string(),
            NodeColumn::new(Vec::new(), Vec::new()),
        );

        step.apply(&mut ctx, &mut scope).unwrap();
        let gains = scope.variables().node_column("gains").unwrap();
        assert_eq!(gains.nodes().len(), 0);
    }

    #[test]
    fn test_modularity_gain_with_partition() {
        let (graph, nodes) = make_test_graph();
        let node_set: HashSet<NodeId> = nodes.iter().copied().collect();
        let subgraph =
            Subgraph::from_nodes(Rc::new(RefCell::new(graph)), node_set, "test".into()).unwrap();

        let mut ctx = Context::new();
        let mut vars = StepVariables::default();
        let mut scope = StepScope::new(&subgraph, &mut vars);
        scope.variables_mut().set_node_column(
            "labels".to_string(),
            NodeColumn::new(
                nodes.clone(),
                nodes
                    .iter()
                    .map(|&n| {
                        if nodes[..3].contains(&n) {
                            AlgorithmParamValue::Int(0)
                        } else {
                            AlgorithmParamValue::Int(1)
                        }
                    })
                    .collect(),
            ),
        );

        let step = ModularityGainStep::new("labels", "gains");
        step.apply(&mut ctx, &mut scope).unwrap();

        let gains = scope.variables().node_column("gains").unwrap();
        assert_eq!(gains.nodes().len(), nodes.len());
    }
}
