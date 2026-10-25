//! Checked selected path primitives sharing one staged-weight topology/kernel.
use super::{
    core::{Step, StepMetadata, StepScope},
    transformations::selected_nodes,
};
use crate::{
    algorithms::{AlgorithmParamValue, Context, CostHint},
    storage::numeric::scalar::attribute_number,
    types::{AttrName, NodeId},
};
use anyhow::{anyhow, bail, Result};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

fn check(ctx: &Context) -> Result<()> {
    if ctx.is_cancelled() {
        bail!("path step cancelled");
    }
    Ok(())
}
fn resolve(scope: &StepScope<'_>, name: &str) -> Result<NodeId> {
    match scope.variables().scalar(name)? {
        AlgorithmParamValue::Int(value) => usize::try_from(*value)
            .map_err(|_| anyhow!("node ID must be non-negative and fit usize")),
        AlgorithmParamValue::Text(value) => value
            .parse::<usize>()
            .map_err(|_| anyhow!("invalid node ID text: {value}")),
        _ => bail!("node ID variable '{name}' must be integer or text"),
    }
}
#[derive(Clone, Copy)]
struct ArcStep {
    target: NodeId,
    edge: usize,
    weight: f64,
}
struct PathTopology {
    rows: HashMap<NodeId, Vec<ArcStep>>,
}
impl PathTopology {
    fn prepare(ctx: &Context, scope: &StepScope<'_>, attr: Option<&AttrName>) -> Result<Self> {
        let nodes = selected_nodes(scope.subgraph(), ctx)?;
        let mut rows: HashMap<_, Vec<_>> = nodes.iter().map(|&node| (node, Vec::new())).collect();
        let owner = scope.subgraph().graph();
        let graph = owner.borrow();
        for &edge in scope.subgraph().ordered_edges().iter() {
            check(ctx)?;
            if !graph.has_edge(edge) {
                bail!("Path edge {edge} is inactive");
            }
            let (source, target) = graph.edge_endpoints(edge)?;
            if !rows.contains_key(&source) || !rows.contains_key(&target) {
                bail!("Path edge {edge} endpoint is outside selection");
            }
            let weight = if let Some(attr) = attr {
                match scope.edge_attribute(ctx, edge, attr)? {
                    None => 1.0,
                    Some(value) => attribute_number(&value)
                        .ok_or_else(|| anyhow!("Path weights require finite non-negative numbers"))?
                        .as_f64(),
                }
            } else {
                1.0
            };
            if !weight.is_finite() || weight < 0.0 {
                bail!("Path weights require finite non-negative numbers");
            }
            rows.get_mut(&source)
                .ok_or_else(|| anyhow!("Missing source"))?
                .push(ArcStep {
                    target,
                    edge,
                    weight,
                });
            if graph.is_undirected() && source != target {
                rows.get_mut(&target)
                    .ok_or_else(|| anyhow!("Missing target"))?
                    .push(ArcStep {
                        target: source,
                        edge,
                        weight,
                    });
            }
        }
        for row in rows.values_mut() {
            row.sort_by_key(|arc| (arc.target, arc.edge));
        }
        Ok(Self { rows })
    }
    fn require(&self, node: NodeId) -> Result<()> {
        if !self.rows.contains_key(&node) {
            bail!("Path node {node} is outside active selection");
        }
        Ok(())
    }
    fn arc_cost(&self, source: NodeId, target: NodeId) -> Result<f64> {
        self.rows[&source]
            .iter()
            .filter(|arc| arc.target == target)
            .map(|arc| arc.weight)
            .min_by(f64::total_cmp)
            .ok_or_else(|| anyhow!("Missing path arc"))
    }
    fn search(
        &self,
        ctx: &Context,
        source: NodeId,
        target: Option<NodeId>,
        excluded_nodes: &HashSet<NodeId>,
        excluded_edges: &HashSet<(NodeId, NodeId)>,
    ) -> Result<(HashMap<NodeId, f64>, HashMap<NodeId, NodeId>)> {
        check(ctx)?;
        let mut distances = HashMap::from([(source, 0.0)]);
        let mut previous = HashMap::new();
        let mut heap = BinaryHeap::from([SearchState {
            cost: 0.0,
            node: source,
        }]);
        while let Some(SearchState { cost, node }) = heap.pop() {
            check(ctx)?;
            if cost > distances[&node] {
                continue;
            }
            if Some(node) == target {
                break;
            }
            for arc in &self.rows[&node] {
                check(ctx)?;
                if excluded_nodes.contains(&arc.target)
                    || excluded_edges.contains(&(node, arc.target))
                {
                    continue;
                }
                let next = add_cost(cost, arc.weight)?;
                if next < distances.get(&arc.target).copied().unwrap_or(f64::INFINITY) {
                    distances.insert(arc.target, next);
                    previous.insert(arc.target, node);
                    heap.push(SearchState {
                        cost: next,
                        node: arc.target,
                    });
                }
            }
        }
        Ok((distances, previous))
    }
    fn path(
        &self,
        ctx: &Context,
        source: NodeId,
        target: NodeId,
        excluded_nodes: &HashSet<NodeId>,
        excluded_edges: &HashSet<(NodeId, NodeId)>,
    ) -> Result<Option<(Vec<NodeId>, f64)>> {
        let (distances, previous) =
            self.search(ctx, source, Some(target), excluded_nodes, excluded_edges)?;
        let Some(&cost) = distances.get(&target) else {
            return Ok(None);
        };
        let mut path = vec![target];
        let mut node = target;
        while node != source {
            check(ctx)?;
            node = *previous
                .get(&node)
                .ok_or_else(|| anyhow!("Missing path predecessor"))?;
            path.push(node);
        }
        path.reverse();
        Ok(Some((path, cost)))
    }
}
fn add_cost(left: f64, right: f64) -> Result<f64> {
    let value = left + right;
    if !value.is_finite() {
        bail!("Path cost overflow");
    }
    Ok(value)
}
#[derive(Clone, Copy, Debug)]
struct SearchState {
    cost: f64,
    node: NodeId,
}
impl PartialEq for SearchState {
    fn eq(&self, other: &Self) -> bool {
        self.cost.total_cmp(&other.cost).is_eq() && self.node == other.node
    }
}
impl Eq for SearchState {}
impl Ord for SearchState {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .total_cmp(&self.cost)
            .then_with(|| other.node.cmp(&self.node))
    }
}
impl PartialOrd for SearchState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Reachable distances: checked integer BFS or non-negative weighted Dijkstra.
pub struct ShortestPathMapStep {
    source: String,
    weight_attr: Option<AttrName>,
    output: String,
}
impl ShortestPathMapStep {
    pub fn new(source: String, weight_attr: Option<AttrName>, output: String) -> Self {
        Self {
            source,
            weight_attr,
            output,
        }
    }
}
impl Step for ShortestPathMapStep {
    fn supports_atomic_properties(&self) -> bool {
        true
    }
    fn id(&self) -> &'static str {
        "core.shortest_path_map"
    }
    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().into(),
            description: "Checked selected BFS/Dijkstra distances".into(),
            cost_hint: CostHint::Linearithmic,
        }
    }
    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        check(ctx)?;
        let source = resolve(scope, &self.source)?;
        let topology = PathTopology::prepare(ctx, scope, self.weight_attr.as_ref())?;
        topology.require(source)?;
        let result: HashMap<NodeId, AlgorithmParamValue> = if self.weight_attr.is_some() {
            topology
                .search(ctx, source, None, &HashSet::new(), &HashSet::new())?
                .0
                .into_iter()
                .map(|(node, distance)| (node, AlgorithmParamValue::Float(distance)))
                .collect()
        } else {
            let mut distances = HashMap::from([(source, 0i64)]);
            let mut queue = VecDeque::from([source]);
            while let Some(node) = queue.pop_front() {
                check(ctx)?;
                let distance = distances[&node]
                    .checked_add(1)
                    .ok_or_else(|| anyhow!("BFS distance overflow"))?;
                for arc in &topology.rows[&node] {
                    check(ctx)?;
                    if let std::collections::hash_map::Entry::Vacant(entry) =
                        distances.entry(arc.target)
                    {
                        entry.insert(distance);
                        queue.push_back(arc.target);
                    }
                }
            }
            distances
                .into_iter()
                .map(|(node, value)| (node, AlgorithmParamValue::Int(value)))
                .collect()
        };
        check(ctx)?;
        ctx.emit_iteration(0, result.len());
        scope
            .variables_mut()
            .set_node_map(self.output.clone(), result);
        Ok(())
    }
}

/// K distinct loopless node sequences, with minimum parallel-arc costs.
pub struct KShortestPathsStep {
    source: String,
    target: String,
    k: usize,
    weight_attr: Option<AttrName>,
    output: String,
}
impl KShortestPathsStep {
    pub fn new(
        source: String,
        target: String,
        k: usize,
        weight_attr: Option<AttrName>,
        output: String,
    ) -> Self {
        Self {
            source,
            target,
            k,
            weight_attr,
            output,
        }
    }
}
#[derive(Clone)]
struct Candidate {
    path: Vec<NodeId>,
    cost: f64,
}
impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cost.total_cmp(&other.cost).is_eq() && self.path == other.path
    }
}
impl Eq for Candidate {}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .total_cmp(&self.cost)
            .then_with(|| other.path.cmp(&self.path))
    }
}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl KShortestPathsStep {
    fn paths(
        &self,
        ctx: &Context,
        topology: &PathTopology,
        source: NodeId,
        target: NodeId,
    ) -> Result<Vec<(Vec<NodeId>, f64)>> {
        if self.k == 0 {
            return Ok(vec![]);
        }
        let Some(first) = topology.path(ctx, source, target, &HashSet::new(), &HashSet::new())?
        else {
            return Ok(vec![]);
        };
        let mut seen = HashSet::from([first.0.clone()]);
        let mut accepted = vec![first];
        let mut candidates = BinaryHeap::new();
        while accepted.len() < self.k {
            check(ctx)?;
            let previous = &accepted[accepted.len() - 1].0;
            let mut prefix_cost = 0.0;
            for index in 0..previous.len().saturating_sub(1) {
                check(ctx)?;
                let root = &previous[..=index];
                let excluded_nodes: HashSet<_> = root[..index].iter().copied().collect();
                let excluded_edges: HashSet<_> = accepted
                    .iter()
                    .filter_map(|(path, _)| {
                        if path.len() > index + 1 && path[..=index] == *root {
                            Some((path[index], path[index + 1]))
                        } else {
                            None
                        }
                    })
                    .collect();
                if let Some((spur, cost)) = topology.path(
                    ctx,
                    previous[index],
                    target,
                    &excluded_nodes,
                    &excluded_edges,
                )? {
                    let mut path = root[..index].to_vec();
                    path.extend(spur);
                    let cost = add_cost(prefix_cost, cost)?;
                    if seen.insert(path.clone()) {
                        candidates.push(Candidate { path, cost });
                    }
                }
                prefix_cost = add_cost(
                    prefix_cost,
                    topology.arc_cost(previous[index], previous[index + 1])?,
                )?;
            }
            let Some(next) = candidates.pop() else { break };
            accepted.push((next.path, next.cost));
        }
        accepted.sort_by(|(a, ac), (b, bc)| ac.total_cmp(bc).then_with(|| a.cmp(b)));
        Ok(accepted)
    }
}
impl Step for KShortestPathsStep {
    fn supports_atomic_properties(&self) -> bool {
        true
    }
    fn id(&self) -> &'static str {
        "core.k_shortest_paths"
    }
    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().into(),
            description: "Checked loopless node paths using Yen".into(),
            cost_hint: CostHint::Quadratic,
        }
    }
    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        check(ctx)?;
        let source = resolve(scope, &self.source)?;
        let target = resolve(scope, &self.target)?;
        let topology = PathTopology::prepare(ctx, scope, self.weight_attr.as_ref())?;
        topology.require(source)?;
        topology.require(target)?;
        let paths = self.paths(ctx, &topology, source, target)?;
        let output = serde_json::to_string(&paths)?;
        check(ctx)?;
        ctx.emit_iteration(0, paths.len());
        scope
            .variables_mut()
            .set_scalar(self.output.clone(), AlgorithmParamValue::Text(output));
        Ok(())
    }
}

/// Ordered seeded walks with per-edge transition multiplicity and optional restart.
pub struct RandomWalkStep {
    start_nodes: String,
    length: usize,
    restart_prob: f64,
    weight_attr: Option<AttrName>,
    seed: Option<u64>,
    output: String,
}
impl RandomWalkStep {
    pub fn new(
        start_nodes: String,
        length: usize,
        restart_prob: f64,
        weight_attr: Option<AttrName>,
        seed: Option<u64>,
        output: String,
    ) -> Self {
        Self {
            start_nodes,
            length,
            restart_prob,
            weight_attr,
            seed,
            output,
        }
    }
}
impl Step for RandomWalkStep {
    fn supports_atomic_properties(&self) -> bool {
        true
    }
    fn id(&self) -> &'static str {
        "core.random_walk"
    }
    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().into(),
            description: "Checked selected seeded random walks".into(),
            cost_hint: CostHint::Linear,
        }
    }
    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        check(ctx)?;
        if !self.restart_prob.is_finite() || !(0.0..=1.0).contains(&self.restart_prob) {
            bail!("restart_prob must be finite and in [0,1]");
        }
        let mut starts = if let Ok(map) = scope.variables().node_map(&self.start_nodes) {
            map.keys().copied().collect::<Vec<_>>()
        } else {
            vec![resolve(scope, &self.start_nodes)?]
        };
        starts.sort_unstable();
        let topology = PathTopology::prepare(ctx, scope, self.weight_attr.as_ref())?;
        for &node in &starts {
            topology.require(node)?;
        }
        let mut transitions = HashMap::new();
        if self.weight_attr.is_some() {
            for (&node, row) in &topology.rows {
                check(ctx)?;
                let maximum = row.iter().map(|arc| arc.weight).fold(0.0_f64, f64::max);
                let mut cumulative = 0.0;
                let mut choices = Vec::new();
                if maximum > 0.0 {
                    for arc in row {
                        let next = cumulative + arc.weight / maximum;
                        if next > cumulative {
                            choices.push((arc.target, next));
                        }
                        cumulative = next;
                    }
                }
                transitions.insert(node, (cumulative, choices));
            }
        }
        let mut rng = self.seed.map(fastrand::Rng::with_seed).unwrap_or_default();
        let mut walks = Vec::with_capacity(starts.len());
        for start in starts {
            check(ctx)?;
            let mut current = start;
            let mut walk = vec![start];
            for _ in 0..self.length {
                check(ctx)?;
                let next = if rng.f64() < self.restart_prob {
                    start
                } else {
                    let row = &topology.rows[&current];
                    if row.is_empty() {
                        break;
                    }
                    if let Some((total, choices)) =
                        transitions.get(&current).filter(|(total, _)| *total > 0.0)
                    {
                        let pick = rng.f64() * total;
                        choices
                            .iter()
                            .find(|(_, limit)| pick < *limit)
                            .or_else(|| choices.last())
                            .ok_or_else(|| anyhow!("Missing weighted transition"))?
                            .0
                    } else {
                        row[rng.usize(0..row.len())].target
                    }
                };
                walk.try_reserve(1)?;
                walk.push(next);
                current = next;
            }
            walks.push(walk);
        }
        let output = serde_json::to_string(&walks)?;
        check(ctx)?;
        ctx.emit_iteration(0, walks.len());
        scope
            .variables_mut()
            .set_scalar(self.output.clone(), AlgorithmParamValue::Text(output));
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::graph::Graph;
    use crate::subgraphs::Subgraph;
    use std::cell::RefCell;
    use std::collections::HashSet;
    use std::rc::Rc;

    #[test]
    fn test_shortest_path_map_unweighted() {
        let mut graph = Graph::new();
        let a = graph.add_node();
        let b = graph.add_node();
        let c = graph.add_node();
        graph.add_edge(a, b).unwrap();
        graph.add_edge(b, c).unwrap();

        let nodes: HashSet<NodeId> = [a, b, c].into_iter().collect();
        let subgraph =
            Subgraph::from_nodes(Rc::new(RefCell::new(graph)), nodes, "test".into()).unwrap();

        let mut ctx = Context::new();
        let mut vars = super::super::core::StepVariables::default();
        vars.set_scalar("source".to_string(), AlgorithmParamValue::Int(a as i64));

        let mut scope = StepScope::new(&subgraph, &mut vars);

        let step = ShortestPathMapStep::new("source".to_string(), None, "distances".to_string());
        step.apply(&mut ctx, &mut scope).unwrap();

        let distances = scope.variables().node_map("distances").unwrap();
        assert_eq!(distances.get(&a), Some(&AlgorithmParamValue::Int(0)));
        assert_eq!(distances.get(&b), Some(&AlgorithmParamValue::Int(1)));
        assert_eq!(distances.get(&c), Some(&AlgorithmParamValue::Int(2)));
    }

    #[test]
    fn test_random_walk_basic() {
        let mut graph = Graph::new();
        let a = graph.add_node();
        let b = graph.add_node();
        let c = graph.add_node();
        graph.add_edge(a, b).unwrap();
        graph.add_edge(b, c).unwrap();
        graph.add_edge(c, a).unwrap();

        let nodes: HashSet<NodeId> = [a, b, c].into_iter().collect();
        let subgraph =
            Subgraph::from_nodes(Rc::new(RefCell::new(graph)), nodes, "test".into()).unwrap();

        let mut ctx = Context::new();
        let mut vars = super::super::core::StepVariables::default();
        vars.set_scalar("source".to_string(), AlgorithmParamValue::Int(a as i64));

        let mut scope = StepScope::new(&subgraph, &mut vars);

        let step = RandomWalkStep::new(
            "source".to_string(),
            10,
            0.0,
            None,
            Some(42),
            "walks".to_string(),
        );
        step.apply(&mut ctx, &mut scope).unwrap();

        let walks_json = scope.variables().scalar("walks").unwrap();
        if let AlgorithmParamValue::Text(json) = walks_json {
            let walks: Vec<Vec<NodeId>> = serde_json::from_str(json).unwrap();
            assert_eq!(walks.len(), 1);
            assert!(walks[0].len() > 1); // Should have walked
        } else {
            panic!("Expected text output");
        }
    }
}

#[cfg(test)]
mod checked_tests {
    use super::*;
    use crate::{algorithms::steps::StepVariables, subgraphs::Subgraph, AttrValue, Graph};
    use std::{
        cell::RefCell,
        rc::Rc,
        sync::{atomic::AtomicBool, Arc},
    };

    fn enumerate(
        topology: &PathTopology,
        node: NodeId,
        target: NodeId,
        path: &mut Vec<NodeId>,
        cost: f64,
        result: &mut HashMap<Vec<NodeId>, f64>,
    ) {
        if node == target {
            result
                .entry(path.clone())
                .and_modify(|old| *old = old.min(cost))
                .or_insert(cost);
            return;
        }
        for arc in &topology.rows[&node] {
            if !path.contains(&arc.target) {
                path.push(arc.target);
                enumerate(
                    topology,
                    arc.target,
                    target,
                    path,
                    cost + arc.weight,
                    result,
                );
                path.pop();
            }
        }
    }
    #[test]
    fn yen_matches_exhaustive_simple_path_enumeration_on_all_four_node_graphs() {
        for directed in [false, true] {
            let pairs: Vec<_> = (0..4)
                .flat_map(|u| {
                    (0..4)
                        .filter(move |&v| if directed { u != v } else { u < v })
                        .map(move |v| (u, v))
                })
                .collect();
            for mask in 0..1usize << pairs.len() {
                let mut rows: HashMap<_, Vec<_>> = (0..4).map(|node| (node, vec![])).collect();
                for (edge, &(u, v)) in pairs.iter().enumerate() {
                    if mask & (1 << edge) != 0 {
                        let weight = (edge % 3) as f64;
                        rows.get_mut(&u).expect("row").push(ArcStep {
                            target: v,
                            edge,
                            weight,
                        });
                        if !directed {
                            rows.get_mut(&v).expect("row").push(ArcStep {
                                target: u,
                                edge,
                                weight,
                            });
                        }
                    }
                }
                for row in rows.values_mut() {
                    row.sort_by_key(|arc| (arc.target, arc.edge));
                }
                let topology = PathTopology { rows };
                let mut expected = HashMap::new();
                enumerate(&topology, 0, 3, &mut vec![0], 0.0, &mut expected);
                let step = KShortestPathsStep::new(
                    "source".into(),
                    "target".into(),
                    usize::MAX,
                    None,
                    "paths".into(),
                );
                let actual = step.paths(&Context::new(), &topology, 0, 3).expect("paths");
                assert_eq!(
                    actual.len(),
                    expected.len(),
                    "directed={directed}, mask={mask}"
                );
                for (path, cost) in &actual {
                    assert_eq!(expected.get(path), Some(cost));
                    assert_eq!(path.iter().collect::<HashSet<_>>().len(), path.len());
                }
                assert!(actual.windows(2).all(|pair| pair[0].1 <= pair[1].1));
                if !actual.is_empty() {
                    let first = KShortestPathsStep::new(
                        "source".into(),
                        "target".into(),
                        1,
                        None,
                        "paths".into(),
                    )
                    .paths(&Context::new(), &topology, 0, 3)
                    .expect("first");
                    assert_eq!(first[0].1, actual[0].1);
                }
            }
        }
    }
    fn selection() -> Subgraph {
        let mut graph = Graph::new_undirected();
        graph.add_nodes(4);
        for (u, v, w) in [
            (0, 1, 10.0),
            (0, 1, 2.0),
            (1, 2, 1.0),
            (0, 2, 8.0),
            (2, 3, 1.0),
        ] {
            let edge = graph.add_edge(u, v).expect("edge");
            graph
                .set_edge_attr(edge, "weight".into(), AttrValue::Float64(w))
                .expect("weight");
        }
        Subgraph::new(
            Rc::new(RefCell::new(graph)),
            HashSet::from([0, 1, 2]),
            HashSet::from([0, 1, 2, 3]),
            "selected".into(),
        )
    }
    #[test]
    fn weighted_distances_and_paths_keep_parallel_minima_undirected_reverse_and_selection() {
        let selected = selection();
        let mut vars = StepVariables::default();
        vars.set_scalar("source", AlgorithmParamValue::Int(2));
        vars.set_scalar("target", AlgorithmParamValue::Int(0));
        let mut scope = StepScope::new(&selected, &mut vars);
        ShortestPathMapStep::new("source".into(), Some("weight".into()), "distance".into())
            .apply(&mut Context::new(), &mut scope)
            .expect("distances");
        assert_eq!(
            scope.variables().node_map("distance").expect("map")[&0],
            AlgorithmParamValue::Float(3.0)
        );
        assert!(!scope
            .variables()
            .node_map("distance")
            .expect("map")
            .contains_key(&3));
        KShortestPathsStep::new(
            "source".into(),
            "target".into(),
            10,
            Some("weight".into()),
            "paths".into(),
        )
        .apply(&mut Context::new(), &mut scope)
        .expect("paths");
        let AlgorithmParamValue::Text(value) = scope.variables().scalar("paths").expect("paths")
        else {
            panic!("text")
        };
        let paths: Vec<(Vec<usize>, f64)> = serde_json::from_str(value).expect("json");
        assert_eq!(paths, [(vec![2, 1, 0], 3.0), (vec![2, 0], 8.0)]);
    }
    #[test]
    fn negative_ids_bad_weights_and_cancellation_preserve_previous_results() {
        let selected = selection();
        let mut vars = StepVariables::default();
        vars.set_scalar("source", AlgorithmParamValue::Int(-1));
        vars.set_node_map("result", HashMap::from([(0, AlgorithmParamValue::Int(99))]));
        let mut scope = StepScope::new(&selected, &mut vars);
        let step =
            ShortestPathMapStep::new("source".into(), Some("weight".into()), "result".into());
        assert!(step.apply(&mut Context::new(), &mut scope).is_err());
        scope
            .variables_mut()
            .set_scalar("source", AlgorithmParamValue::Int(0));
        let mut ctx = Context::with_cancel_token(Arc::new(AtomicBool::new(true)));
        assert!(step.apply(&mut ctx, &mut scope).is_err());
        selected
            .graph()
            .borrow_mut()
            .set_edge_attr(0, "weight".into(), AttrValue::Float64(-1.0))
            .expect("bad weight");
        assert!(step.apply(&mut Context::new(), &mut scope).is_err());
        assert_eq!(
            scope.variables().node_map("result").expect("unchanged")[&0],
            AlgorithmParamValue::Int(99)
        );
    }
    #[test]
    fn seeded_walks_are_isolated_and_huge_finite_weights_normalize_without_overflow() {
        let mut graph = Graph::new_directed();
        graph.add_nodes(3);
        for (u, v, w) in [(0, 1, 0.0), (0, 2, 1e308), (0, 2, 1e308), (2, 2, 1.0)] {
            let edge = graph.add_edge(u, v).expect("edge");
            graph
                .set_edge_attr(edge, "weight".into(), AttrValue::Float64(w))
                .expect("weight");
        }
        let selected = Subgraph::from_nodes(
            Rc::new(RefCell::new(graph)),
            (0..3).collect(),
            "walk".into(),
        )
        .expect("selection");
        let mut vars = StepVariables::default();
        vars.set_scalar("start", AlgorithmParamValue::Int(0));
        let mut scope = StepScope::new(&selected, &mut vars);
        let step = RandomWalkStep::new(
            "start".into(),
            5,
            0.0,
            Some("weight".into()),
            Some(19),
            "walk".into(),
        );
        fastrand::seed(71);
        let expected = fastrand::u64(..);
        fastrand::seed(71);
        step.apply(&mut Context::new(), &mut scope).expect("walk");
        assert_eq!(fastrand::u64(..), expected);
        let AlgorithmParamValue::Text(value) = scope.variables().scalar("walk").expect("walk")
        else {
            panic!("text")
        };
        let walks: Vec<Vec<usize>> = serde_json::from_str(value).expect("json");
        assert_eq!(walks, vec![vec![0, 2, 2, 2, 2, 2]]);
    }
}
