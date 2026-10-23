use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Result};

use crate::algorithms::pathfinding::utils::checked_dijkstra_weights;
use crate::algorithms::registry::Registry;
use crate::algorithms::{
    Algorithm, AlgorithmMetadata, AlgorithmOutput, AlgorithmParamValue, Context, CostHint,
    ParameterMetadata, ParameterType,
};
use crate::state::topology::{build_csr_from_edges_with_scratch, Csr, CsrOptions};
use crate::subgraphs::Subgraph;
use crate::traits::SubgraphOperations;
use crate::types::{AttrName, AttrValue, NodeId};

#[derive(Clone, Debug)]
pub struct AStarPathfinding {
    start_attr: AttrName,
    goal_attr: AttrName,
    weight_attr: Option<AttrName>,
    heuristic_attr: Option<AttrName>,
    output_attr: AttrName,
}

impl AStarPathfinding {
    pub fn new(
        start_attr: AttrName,
        goal_attr: AttrName,
        weight_attr: Option<AttrName>,
        heuristic_attr: Option<AttrName>,
        output_attr: AttrName,
    ) -> Self {
        Self {
            start_attr,
            goal_attr,
            weight_attr,
            heuristic_attr,
            output_attr,
        }
    }

    fn metadata_template() -> AlgorithmMetadata {
        AlgorithmMetadata {
            id: "pathfinding.astar".to_string(),
            name: "A* Pathfinding".to_string(),
            description: "A* search with optional edge weights and heuristics.".to_string(),
            version: "0.2.0".to_string(),
            cost_hint: CostHint::Linearithmic,
            supports_cancellation: true,
            parameters: vec![
                ParameterMetadata {
                    name: "start_attr".to_string(),
                    description: "Node attribute marking the start node.".to_string(),
                    value_type: ParameterType::Text,
                    required: true,
                    default_value: None,
                },
                ParameterMetadata {
                    name: "goal_attr".to_string(),
                    description: "Node attribute marking the goal node.".to_string(),
                    value_type: ParameterType::Text,
                    required: true,
                    default_value: None,
                },
                ParameterMetadata {
                    name: "weight_attr".to_string(),
                    description: "Optional edge weight attribute.".to_string(),
                    value_type: ParameterType::Text,
                    required: false,
                    default_value: None,
                },
                ParameterMetadata {
                    name: "heuristic_attr".to_string(),
                    description:
                        "Optional finite nonnegative consistent heuristic; goal must be zero."
                            .to_string(),
                    value_type: ParameterType::Text,
                    required: false,
                    default_value: None,
                },
                ParameterMetadata {
                    name: "output_attr".to_string(),
                    description: "Attribute name to store path order.".to_string(),
                    value_type: ParameterType::Text,
                    required: false,
                    default_value: Some(AlgorithmParamValue::Text("path_index".to_string())),
                },
            ],
        }
    }

    fn resolve_node(&self, subgraph: &Subgraph, attr: &AttrName) -> Result<NodeId> {
        let mut found = None;
        for &node in subgraph.ordered_nodes().iter() {
            if subgraph.get_node_attribute(node, attr)?.is_some() {
                if found.is_some() {
                    return Err(anyhow!(
                        "multiple nodes carry attribute '{attr}'; expected exactly one"
                    ));
                }
                found = Some(node);
            }
        }
        found.ok_or_else(|| anyhow!("no node in subgraph carries attribute '{attr}'"))
    }

    fn heuristics_map(&self, subgraph: &Subgraph) -> Result<HashMap<NodeId, f64>> {
        let Some(attr) = &self.heuristic_attr else {
            return Ok(HashMap::new());
        };
        let reference = subgraph.graph();
        let graph = reference.borrow();
        let pool = graph.pool();
        Ok(crate::query::weights::checked_heuristics(
            &subgraph.ordered_nodes(),
            |node| {
                graph
                    .space()
                    .get_node_attr_index(node, attr)
                    .map(|index| {
                        pool.get_attr_by_index(attr, index, true).ok_or_else(|| {
                            crate::GraphError::InvalidInput(format!(
                                "Invalid heuristic reference at node {node}"
                            ))
                        })
                    })
                    .transpose()
            },
        )?)
    }
    fn run(
        &self,
        ctx: &mut Context,
        subgraph: &Subgraph,
        weights: &HashMap<(NodeId, NodeId), f64>,
    ) -> Result<Vec<(NodeId, i64)>> {
        let start = self.resolve_node(subgraph, &self.start_attr)?;
        let goal = self.resolve_node(subgraph, &self.goal_attr)?;
        let nodes = subgraph.ordered_nodes();
        let source = nodes
            .binary_search(&start)
            .map_err(|_| anyhow!("A* start outside selection"))?;
        let target = nodes
            .binary_search(&goal)
            .map_err(|_| anyhow!("A* goal outside selection"))?;
        let heuristics = self.heuristics_map(subgraph)?;
        let add_reverse = subgraph.graph().borrow().is_undirected();
        let csr = subgraph
            .csr_cache_get(add_reverse)
            .ok_or_else(|| anyhow!("Missing prepared A* topology"))?;
        let result = search(
            || ctx.is_cancelled(),
            SearchInput {
                nodes: &nodes,
                csr: &csr,
                weights,
                heuristics: &heuristics,
            },
            source,
            target,
        )?;
        ctx.record_stat("astar.path_cost", result.cost);
        Ok(result.rows)
    }
}

/// Prepared read-only topology, weights and heuristic values for the shared A* kernel.
pub(crate) struct SearchInput<'a> {
    pub nodes: &'a [NodeId],
    pub csr: &'a Csr,
    pub weights: &'a HashMap<(NodeId, NodeId), f64>,
    pub heuristics: &'a HashMap<NodeId, f64>,
}
pub(crate) struct SearchResult {
    pub rows: Vec<(NodeId, i64)>,
    pub cost: f64,
}
pub(crate) fn search(
    is_cancelled: impl Fn() -> bool,
    input: SearchInput<'_>,
    source: usize,
    goal: usize,
) -> Result<SearchResult> {
    if is_cancelled() {
        return Err(anyhow!("astar cancelled"));
    }
    let SearchInput {
        nodes,
        csr,
        weights,
        heuristics,
    } = input;
    let heuristic = |row: usize| heuristics.get(&nodes[row]).copied().unwrap_or(0.0);
    if heuristic(goal) != 0.0 {
        return Err(anyhow!("heuristic at goal must be zero"));
    }
    for (&(u, v), &weight) in weights {
        if is_cancelled() {
            return Err(anyhow!("astar cancelled"));
        }
        if heuristics.get(&u).copied().unwrap_or(0.0)
            > weight + heuristics.get(&v).copied().unwrap_or(0.0)
        {
            return Err(anyhow!(
                "inconsistent heuristic on selected edge {u} -> {v}"
            ));
        }
    }
    #[derive(Copy, Clone, Debug)]
    struct State {
        estimated: f64,
        cost: f64,
        row: usize,
    }
    impl Eq for State {}
    impl PartialEq for State {
        fn eq(&self, other: &Self) -> bool {
            self.row == other.row
                && self.estimated.total_cmp(&other.estimated).is_eq()
                && self.cost.total_cmp(&other.cost).is_eq()
        }
    }
    impl Ord for State {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            other
                .estimated
                .total_cmp(&self.estimated)
                .then_with(|| other.cost.total_cmp(&self.cost))
                .then_with(|| self.row.cmp(&other.row))
        }
    }
    impl PartialOrd for State {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    let mut open = BinaryHeap::new();
    let mut distances = vec![f64::INFINITY; nodes.len()];
    let mut previous = vec![None; nodes.len()];
    distances[source] = 0.0;
    open.push(State {
        estimated: heuristic(source),
        cost: 0.0,
        row: source,
    });
    while let Some(State { cost, row, .. }) = open.pop() {
        if is_cancelled() {
            return Err(anyhow!("astar cancelled"));
        }
        if cost > distances[row] {
            continue;
        }
        if row == goal {
            break;
        }
        for &neighbor in csr.neighbors(row) {
            if is_cancelled() {
                return Err(anyhow!("astar cancelled"));
            }
            let weight = weights
                .get(&(nodes[row], nodes[neighbor]))
                .copied()
                .ok_or_else(|| anyhow!("missing checked A* weight"))?;
            let candidate = cost + weight;
            if !candidate.is_finite() {
                return Err(anyhow!("nonfinite A* distance"));
            }
            if candidate < distances[neighbor] {
                let estimated = candidate + heuristic(neighbor);
                if !estimated.is_finite() {
                    return Err(anyhow!("nonfinite A* priority"));
                }
                distances[neighbor] = candidate;
                previous[neighbor] = Some(row);
                open.push(State {
                    estimated,
                    cost: candidate,
                    row: neighbor,
                });
            }
        }
    }
    if !distances[goal].is_finite() {
        return Err(anyhow!("goal unreachable from start"));
    }
    let mut path = vec![goal];
    let mut current = goal;
    while current != source {
        if is_cancelled() {
            return Err(anyhow!("astar cancelled"));
        }
        current = previous[current].ok_or_else(|| anyhow!("failed to reconstruct path"))?;
        path.push(current);
        if path.len() > nodes.len() {
            return Err(anyhow!("cycle in A* predecessor path"));
        }
    }
    path.reverse();
    let rows = path
        .into_iter()
        .enumerate()
        .map(|(order, row)| Ok((nodes[row], i64::try_from(order)?)))
        .collect::<Result<_>>()?;
    if is_cancelled() {
        return Err(anyhow!("astar cancelled"));
    }
    Ok(SearchResult {
        rows,
        cost: distances[goal],
    })
}

impl Algorithm for AStarPathfinding {
    fn id(&self) -> &'static str {
        "pathfinding.astar"
    }

    fn metadata(&self) -> AlgorithmMetadata {
        Self::metadata_template()
    }

    fn execute(&self, ctx: &mut Context, subgraph: Subgraph) -> Result<Subgraph> {
        let t0 = Instant::now();
        if ctx.is_cancelled() {
            return Err(anyhow!("astar cancelled"));
        }
        let weights = checked_dijkstra_weights(&subgraph, self.weight_attr.as_ref())?;

        // Phase 1: Collect nodes
        let nodes_start = Instant::now();
        let nodes = subgraph.ordered_nodes();
        ctx.record_call("astar.collect_nodes", nodes_start.elapsed());
        ctx.record_stat("astar.count.input_nodes", nodes.len() as f64);

        // Phase 2: Build indexer
        let idx_start = Instant::now();
        let mut node_to_index = rustc_hash::FxHashMap::default();
        node_to_index.reserve(nodes.len());
        for (i, &node) in nodes.iter().enumerate() {
            node_to_index.insert(node, i);
        }
        ctx.record_call("astar.build_indexer", idx_start.elapsed());

        // Phase 3: Build or retrieve CSR
        let add_reverse = subgraph.graph().borrow().is_undirected();
        if subgraph.csr_cache_get(add_reverse).is_some() {
            ctx.record_call("astar.csr_cache_hit", std::time::Duration::from_nanos(0));
        } else {
            let csr_start = Instant::now();

            let edges = subgraph.ordered_edges();
            let graph_ref = subgraph.graph();
            let graph_borrow = graph_ref.borrow();

            let mut csr = Csr::default();
            let csr_time = build_csr_from_edges_with_scratch(
                &mut csr,
                nodes.len(),
                edges.iter().copied(),
                |nid| node_to_index.get(&nid).copied(),
                |eid| graph_borrow.edge_endpoints(eid).ok(),
                CsrOptions {
                    add_reverse_edges: add_reverse,
                    sort_neighbors: false,
                },
            );
            ctx.record_call("astar.csr_cache_miss", csr_start.elapsed());
            ctx.record_call("astar.build_csr", csr_time);
            subgraph.csr_cache_store(add_reverse, Arc::new(csr));
        }

        // Phase 3: Execute A* (will use cached CSR automatically)
        let path = self.run(ctx, &subgraph, &weights)?;
        ctx.record_stat("astar.count.path_nodes", path.len() as f64);
        if ctx.is_cancelled() {
            return Err(anyhow!("astar cancelled"));
        }

        // Phase 4: Write results
        if ctx.persist_results() {
            let attr_values: Vec<(NodeId, AttrValue)> = path
                .iter()
                .map(|&(node, index)| (node, AttrValue::Int(index)))
                .collect();

            ctx.with_scoped_timer("astar.write_attributes", || {
                subgraph.set_node_attr_column(self.output_attr.clone(), attr_values)
            })
            .map_err(|err| anyhow!("failed to persist astar path: {err}"))?;
        } else {
            ctx.add_output(
                "pathfinding.astar.path",
                AlgorithmOutput::NodeIntegers(path),
            );
        }

        ctx.record_duration("astar.total_execution", t0.elapsed());
        Ok(subgraph)
    }
}

pub fn register(registry: &Registry) -> Result<()> {
    let metadata = AStarPathfinding::metadata_template();
    let id = metadata.id.clone();
    registry.register_with_metadata(id.as_str(), metadata, |spec| {
        let start_attr = spec.params.expect_text("start_attr")?.to_string();
        let goal_attr = spec.params.expect_text("goal_attr")?.to_string();
        let weight_attr = spec
            .params
            .get_text("weight_attr")
            .map(|s| AttrName::from(s.to_string()));
        let heuristic_attr = spec
            .params
            .get_text("heuristic_attr")
            .map(|s| AttrName::from(s.to_string()));
        let output_attr = spec
            .params
            .get_text("output_attr")
            .unwrap_or("path_index")
            .to_string();
        Ok(Box::new(AStarPathfinding::new(
            start_attr.into(),
            goal_attr.into(),
            weight_attr,
            heuristic_attr,
            output_attr.into(),
        )) as Box<dyn Algorithm>)
    })
}
