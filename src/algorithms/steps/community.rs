//! Community detection helper step primitives.

use std::collections::HashMap;

use anyhow::{anyhow, Result};

use crate::types::NodeId;

use super::super::{AlgorithmParamValue, Context, CostHint};
use super::core::{NodeColumn, Step, StepMetadata, StepScope};
use super::transformations::{selected_neighbors, selected_nodes};

fn check_cancelled(ctx: &Context) -> Result<()> {
    if ctx.is_cancelled() {
        return Err(anyhow!("community step cancelled"));
    }
    Ok(())
}

fn node_label(node: NodeId) -> Result<AlgorithmParamValue> {
    Ok(AlgorithmParamValue::Int(
        i64::try_from(node).map_err(|_| anyhow!("Community node ID exceeds i64"))?,
    ))
}

/// Preserve explicit column order; maps use stable ID order. Accept both layouts.
fn checked_input(
    scope: &StepScope<'_>,
    name: &str,
    selected: &[NodeId],
) -> Result<(Vec<NodeId>, HashMap<NodeId, AlgorithmParamValue>)> {
    let (nodes, values) = if let Ok(column) = scope.variables().node_column(name) {
        let nodes = column.nodes().to_vec();
        let values: HashMap<_, _> = column
            .iter()
            .map(|(node, value)| (node, value.clone()))
            .collect();
        (nodes, values)
    } else {
        let values = scope.variables().node_map(name)?.clone();
        let mut nodes: Vec<_> = values.keys().copied().collect();
        nodes.sort_unstable();
        (nodes, values)
    };
    if nodes.len() != values.len()
        || nodes
            .iter()
            .any(|node| selected.binary_search(node).is_err())
    {
        return Err(anyhow!(
            "Community input requires unique active selected node IDs"
        ));
    }
    for value in values.values() {
        super::expression::validate_value(value)?;
    }
    Ok((nodes, values))
}

fn install(
    ctx: &Context,
    scope: &mut StepScope<'_>,
    target: &str,
    nodes: Vec<NodeId>,
    values: Vec<AlgorithmParamValue>,
) -> Result<()> {
    check_cancelled(ctx)?;
    scope
        .variables_mut()
        .set_node_column(target, NodeColumn::new(nodes, values));
    Ok(())
}

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
    fn supports_atomic_properties(&self) -> bool {
        true
    }

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
        check_cancelled(ctx)?;
        let k = if self.strategy == SeedStrategy::Random {
            let k = self
                .k
                .ok_or_else(|| anyhow!("random strategy requires k"))?;
            if k == 0 || i64::try_from(k - 1).is_err() {
                return Err(anyhow!("k must be positive with labels fitting i64"));
            }
            Some(k)
        } else {
            None
        };
        let nodes = selected_nodes(scope.subgraph(), ctx)?;
        let mut values = Vec::with_capacity(nodes.len());
        match self.strategy {
            SeedStrategy::Singleton => {
                for &node in &nodes {
                    check_cancelled(ctx)?;
                    values.push(node_label(node)?);
                }
            }
            SeedStrategy::DegreeBased => {
                let (_, neighbors) = selected_neighbors(scope.subgraph(), ctx)?;
                let mut hubs = nodes.clone();
                hubs.sort_by_key(|node| (std::cmp::Reverse(neighbors[node].len()), *node));
                let hub_count = nodes.len().div_ceil(10).max(1).min(nodes.len());
                let mut assigned = HashMap::new();
                for &node in &hubs[..hub_count] {
                    assigned.insert(node, node_label(node)?);
                }
                for &node in &nodes {
                    check_cancelled(ctx)?;
                    if !assigned.contains_key(&node) {
                        let value = neighbors[&node]
                            .iter()
                            .find_map(|neighbor| assigned.get(neighbor))
                            .cloned()
                            .unwrap_or(node_label(node)?);
                        assigned.insert(node, value);
                    }
                    values.push(assigned[&node].clone());
                }
            }
            SeedStrategy::Random => {
                let mut rng = self.seed.map(fastrand::Rng::with_seed).unwrap_or_default();
                let k = k.ok_or_else(|| anyhow!("random strategy requires k"))?;
                for _ in &nodes {
                    check_cancelled(ctx)?;
                    values.push(AlgorithmParamValue::Int(i64::try_from(rng.usize(0..k))?));
                }
            }
        }
        install(ctx, scope, &self.target, nodes, values)
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
    fn supports_atomic_properties(&self) -> bool {
        true
    }

    fn id(&self) -> &'static str {
        "core.modularity_gain"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Compute modularity change for each node relative to current partition"
                .to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        let (selected, neighbors) = selected_neighbors(scope.subgraph(), ctx)?;
        let (nodes, input) = checked_input(scope, &self.partition, &selected)?;
        let mut labels = HashMap::new();
        for &node in &selected {
            check_cancelled(ctx)?;
            let value = input.get(&node).cloned().unwrap_or(node_label(node)?);
            let AlgorithmParamValue::Int(label) = value else {
                return Err(anyhow!("Modularity partition labels must be integers"));
            };
            labels.insert(node, label);
        }
        let directed = scope.subgraph().graph().borrow().is_directed();
        let m = scope.subgraph().edges().len() as f64;
        let mut incoming: HashMap<NodeId, Vec<NodeId>> =
            selected.iter().map(|&node| (node, Vec::new())).collect();
        let mut degrees = HashMap::new();
        let mut totals: HashMap<i64, (f64, f64)> = HashMap::new();
        for &node in &selected {
            check_cancelled(ctx)?;
            for &neighbor in &neighbors[&node] {
                incoming
                    .get_mut(&neighbor)
                    .ok_or_else(|| anyhow!("Missing selected neighbor"))?
                    .push(node);
            }
        }
        for &node in &selected {
            let out = neighbors[&node].len() as f64;
            let ins = incoming[&node].len() as f64;
            // Undirected stored loops count twice in modularity degree.
            let degree = out
                + neighbors[&node]
                    .iter()
                    .filter(|&&neighbor| neighbor == node)
                    .count() as f64;
            let pair = if directed {
                (out, ins)
            } else {
                (degree, degree)
            };
            degrees.insert(node, pair);
            let total = totals.entry(labels[&node]).or_default();
            total.0 += pair.0;
            total.1 += pair.1;
        }
        let mut gains = Vec::with_capacity(nodes.len());
        for &node in &nodes {
            check_cancelled(ctx)?;
            let own = labels[&node];
            let mut contacts: HashMap<i64, f64> = HashMap::new();
            for &neighbor in &neighbors[&node] {
                if neighbor != node {
                    *contacts.entry(labels[&neighbor]).or_default() += 1.0;
                }
            }
            if directed {
                for &neighbor in &incoming[&node] {
                    if neighbor != node {
                        *contacts.entry(labels[&neighbor]).or_default() += 1.0;
                    }
                }
            }
            let own_contacts = contacts.get(&own).copied().unwrap_or(0.0);
            let (out, ins) = degrees[&node];
            let (own_out, own_in) = totals[&own];
            let mut best = 0.0_f64;
            if m > 0.0 {
                for (&community, &links) in &contacts {
                    if community == own {
                        continue;
                    }
                    let (other_out, other_in) = totals[&community];
                    // Q=sum(L/m-Kout*Kin/m^2), or L/m-(K/2m)^2.
                    // Remove from the old community and insert in the candidate;
                    // stored self-loop terms cancel from internal-edge changes.
                    let null_change = if directed {
                        (out / m) * ((own_in - other_in - ins) / m)
                            + (ins / m) * ((own_out - other_out - out) / m)
                    } else {
                        (out / m) * ((own_out - other_out - out) / (2.0 * m))
                    };
                    let gain = (links - own_contacts) / m + null_change;
                    if !gain.is_finite() {
                        return Err(anyhow!("Nonfinite modularity gain"));
                    }
                    best = best.max(gain);
                }
            }
            gains.push(AlgorithmParamValue::Float(best));
        }
        install(ctx, scope, &self.target, nodes, gains)
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
    fn supports_atomic_properties(&self) -> bool {
        true
    }

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
        let (selected, neighbors) = selected_neighbors(scope.subgraph(), ctx)?;
        let (nodes, labels) = checked_input(scope, &self.labels, &selected)?;
        let mut new_values = Vec::with_capacity(nodes.len());
        for &node in &nodes {
            check_cancelled(ctx)?;
            let current = labels.get(&node).cloned().unwrap_or(node_label(node)?);
            let mut counts: std::collections::BTreeMap<String, (AlgorithmParamValue, usize)> =
                std::collections::BTreeMap::new();
            // An explicit self vote, in addition to a stored self-loop incidence.
            for neighbor in std::iter::once(node).chain(neighbors[&node].iter().copied()) {
                check_cancelled(ctx)?;
                let label = labels
                    .get(&neighbor)
                    .cloned()
                    .unwrap_or(node_label(neighbor)?);
                let key = serde_json::to_string(&label)?;
                let entry = counts.entry(key).or_insert((label, 0));
                entry.1 += 1;
            }
            // Canonical typed-key order breaks equal-frequency ties.
            let mut best = current;
            let mut frequency = 0;
            for (_, (label, count)) in counts {
                if count > frequency {
                    best = label;
                    frequency = count;
                }
            }
            new_values.push(best);
        }
        install(ctx, scope, &self.target, nodes, new_values)
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
        let mut graph = Graph::new_manual();
        let nodes: Vec<NodeId> = (0..6).map(|_| graph.add_node().unwrap()).collect();

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
        let graph = Graph::new_manual();
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

#[cfg(test)]
mod checked_tests {
    use super::*;
    use crate::{algorithms::steps::StepVariables, subgraphs::Subgraph, Graph};
    use std::{cell::RefCell, rc::Rc};

    /// Independent full objective recomputation, not the local move equation.
    fn objective(edges: &[(usize, usize)], labels: &[i64], directed: bool) -> f64 {
        if edges.is_empty() {
            return 0.0;
        }
        let m = edges.len() as f64;
        let mut degree = vec![(0.0, 0.0); labels.len()];
        let mut internal = 0.0;
        for &(source, target) in edges {
            if labels[source] == labels[target] {
                internal += 1.0;
            }
            if directed {
                degree[source].0 += 1.0;
                degree[target].1 += 1.0;
            } else {
                degree[source].0 += 1.0;
                degree[target].0 += 1.0;
            }
        }
        let mut totals: std::collections::BTreeMap<i64, (f64, f64)> =
            std::collections::BTreeMap::new();
        for (index, &label) in labels.iter().enumerate() {
            let total = totals.entry(label).or_default();
            total.0 += degree[index].0;
            total.1 += degree[index].1;
        }
        internal / m
            - totals
                .values()
                .map(|&(out, ins)| {
                    if directed {
                        out * ins / (m * m)
                    } else {
                        (out / (2.0 * m)).powi(2)
                    }
                })
                .sum::<f64>()
    }

    #[test]
    fn modularity_gains_match_full_recomputation_for_all_small_graphs_and_binary_partitions() {
        for directed in [false, true] {
            let count = if directed { 3 } else { 4 };
            let pairs: Vec<_> = (0..count)
                .flat_map(|source| {
                    (0..count)
                        .filter(move |&target| directed || source < target)
                        .map(move |target| (source, target))
                })
                .collect();
            for mask in 0..1usize << pairs.len() {
                let mut edges: Vec<_> = pairs
                    .iter()
                    .enumerate()
                    .filter(|(bit, _)| mask & (1 << bit) != 0)
                    .map(|(_, &edge)| edge)
                    .collect();
                // Extra cases include multiedges and undirected self-loops.
                if mask == 0 {
                    edges.extend([(0, 0), (0, 1), (0, 1), (1, 2)]);
                }
                let mut graph = if directed {
                    Graph::new_directed_manual()
                } else {
                    Graph::new_undirected_manual()
                };
                graph.add_nodes(count).unwrap();
                for &(source, target) in &edges {
                    graph.add_edge(source, target).expect("edge");
                }
                let selected = Subgraph::from_nodes(
                    Rc::new(RefCell::new(graph)),
                    (0..count).collect(),
                    "reference".into(),
                )
                .expect("selection");
                for bits in 0..1usize << count {
                    let labels: Vec<_> =
                        (0..count).map(|node| ((bits >> node) & 1) as i64).collect();
                    let mut vars = StepVariables::default();
                    vars.set_node_column(
                        "labels",
                        NodeColumn::new(
                            (0..count).collect(),
                            labels
                                .iter()
                                .copied()
                                .map(AlgorithmParamValue::Int)
                                .collect(),
                        ),
                    );
                    let mut scope = StepScope::new(&selected, &mut vars);
                    ModularityGainStep::new("labels", "gain")
                        .apply(&mut Context::new(), &mut scope)
                        .expect("gain");
                    let baseline = objective(&edges, &labels, directed);
                    for node in 0..count {
                        let candidates: std::collections::BTreeSet<_> = edges
                            .iter()
                            .filter_map(|&(source, target)| {
                                if source == node && target != node {
                                    Some(labels[target])
                                } else if target == node && source != node {
                                    Some(labels[source])
                                } else {
                                    None
                                }
                            })
                            .collect();
                        let mut expected = 0.0_f64;
                        for candidate in candidates {
                            let mut moved = labels.clone();
                            moved[node] = candidate;
                            expected = expected.max(objective(&edges, &moved, directed) - baseline);
                        }
                        let AlgorithmParamValue::Float(actual) = scope
                            .variables()
                            .node_column("gain")
                            .expect("gain")
                            .get(node)
                            .expect("node")
                        else {
                            panic!("float");
                        };
                        assert!((actual - expected).abs() < 1e-12, "directed={directed}, edges={edges:?}, labels={labels:?}, node={node}, actual={actual}, expected={expected}");
                    }
                }
            }
        }
    }

    #[test]
    fn random_seeds_are_isolated_and_degree_hub_ties_are_stable() {
        let mut graph = Graph::new_undirected_manual();
        graph.add_nodes(3).unwrap();
        graph.add_edge(0, 1).expect("edge");
        graph.add_edge(1, 2).expect("edge");
        graph.add_edge(0, 2).expect("edge");
        let selected = Subgraph::from_nodes(
            Rc::new(RefCell::new(graph)),
            (0..3).collect(),
            "triangle".into(),
        )
        .expect("selection");
        let mut vars = StepVariables::default();
        let mut scope = StepScope::new(&selected, &mut vars);
        fastrand::seed(17);
        let expected = fastrand::u64(..);
        fastrand::seed(17);
        let random = CommunitySeedStep::new(SeedStrategy::Random, "labels", Some(3), Some(91));
        random.apply(&mut Context::new(), &mut scope).expect("seed");
        assert_eq!(fastrand::u64(..), expected);
        let first = scope
            .variables()
            .node_column("labels")
            .expect("column")
            .clone()
            .into_pairs();
        random
            .apply(&mut Context::new(), &mut scope)
            .expect("repeat");
        assert_eq!(
            scope
                .variables()
                .node_column("labels")
                .expect("column")
                .clone()
                .into_pairs(),
            first
        );
        CommunitySeedStep::new(SeedStrategy::DegreeBased, "labels", None, None)
            .apply(&mut Context::new(), &mut scope)
            .expect("degree");
        assert!(scope
            .variables()
            .node_column("labels")
            .expect("column")
            .iter()
            .all(|(_, value)| value == &AlgorithmParamValue::Int(0)));
    }

    #[test]
    fn sparse_columns_preserve_order_and_foreign_or_cancelled_inputs_preserve_outputs() {
        let mut graph = Graph::new_undirected_manual();
        graph.add_nodes(4).unwrap();
        graph.add_edge(1, 2).expect("edge");
        graph.add_edge(2, 3).expect("edge");
        let selected = Subgraph::from_nodes(
            Rc::new(RefCell::new(graph)),
            [1, 2, 3].into_iter().collect(),
            "selected".into(),
        )
        .expect("selection");
        let mut vars = StepVariables::default();
        vars.set_node_column(
            "labels",
            NodeColumn::new(
                vec![3, 1],
                vec![AlgorithmParamValue::Int(3), AlgorithmParamValue::Int(1)],
            ),
        );
        let mut scope = StepScope::new(&selected, &mut vars);
        let step = LabelPropagateStep::new("labels", "result");
        step.apply(&mut Context::new(), &mut scope).expect("sparse");
        let original = scope
            .variables()
            .node_column("result")
            .expect("column")
            .clone()
            .into_pairs();
        assert_eq!(
            original.iter().map(|(node, _)| *node).collect::<Vec<_>>(),
            [3, 1]
        );
        scope
            .variables_mut()
            .set_node_map("labels", HashMap::from([(0, AlgorithmParamValue::Int(1))]));
        assert!(step.apply(&mut Context::new(), &mut scope).is_err());
        let mut ctx = Context::with_cancel_token(std::sync::Arc::new(
            std::sync::atomic::AtomicBool::new(true),
        ));
        assert!(
            CommunitySeedStep::new(SeedStrategy::Singleton, "result", None, None)
                .apply(&mut ctx, &mut scope)
                .is_err()
        );
        assert_eq!(
            scope
                .variables()
                .node_column("result")
                .expect("unchanged")
                .clone()
                .into_pairs(),
            original
        );
    }
}
