use std::time::Instant;

use super::map_equation::Walk;
use anyhow::{anyhow, Result};
use std::collections::HashMap;

use crate::algorithms::registry::Registry;
use crate::algorithms::{
    Algorithm, AlgorithmMetadata, Context, CostHint, ParameterMetadata, ParameterType,
};
use crate::subgraphs::Subgraph;
use crate::traits::SubgraphOperations;
use crate::types::{AttrName, AttrValue, NodeId};

/// Infomap community detection algorithm.
///
/// Uses the map equation to minimize the description length of random walks on the graph.
/// Bounded two-level greedy trials with recorded uniform-node teleportation.
#[derive(Clone, Debug)]
pub struct Infomap {
    teleportation: f64,
    num_trials: usize,
    max_iter: usize,
    flow_max_iter: usize,
    seed: Option<u64>,
    output_attr: AttrName,
}

impl Infomap {
    pub fn new(
        teleportation: f64,
        num_trials: usize,
        max_iter: usize,
        seed: Option<u64>,
        output_attr: AttrName,
    ) -> Result<Self> {
        if !(0.0..=1.0).contains(&teleportation) {
            return Err(anyhow!("teleportation must be between 0.0 and 1.0"));
        }
        if num_trials == 0 {
            return Err(anyhow!("num_trials must be greater than zero"));
        }
        if max_iter == 0 {
            return Err(anyhow!("max_iter must be greater than zero"));
        }
        Ok(Self {
            teleportation,
            num_trials,
            max_iter,
            flow_max_iter: 10_000,
            seed,
            output_attr,
        })
    }

    /// Bound stationary-flow iterations independently of optimization sweeps.
    pub fn with_flow_budget(mut self, iterations: usize) -> Result<Self> {
        if iterations == 0 {
            return Err(anyhow!("flow_max_iter must be positive"));
        }
        self.flow_max_iter = iterations;
        Ok(self)
    }

    fn metadata_template() -> AlgorithmMetadata {
        AlgorithmMetadata {
            id: "community.infomap".to_string(),
            name: "Infomap".to_string(),
            description: "Random-walk based community detection using information theory."
                .to_string(),
            version: "0.1.0".to_string(),
            cost_hint: CostHint::Quadratic,
            supports_cancellation: true,
            parameters: vec![
                ParameterMetadata {
                    name: "teleportation".to_string(),
                    description: "Recorded uniform-node teleportation probability; dangling rows jump uniformly."
                        .to_string(),
                    value_type: ParameterType::Float,
                    required: false,
                    default_value: None,
                },
                ParameterMetadata {
                    name: "num_trials".to_string(),
                    description: "Number of random trials for optimization.".to_string(),
                    value_type: ParameterType::Int,
                    required: false,
                    default_value: None,
                },
                ParameterMetadata {
                    name: "max_iter".to_string(),
                    description: "Maximum iterations per trial.".to_string(),
                    value_type: ParameterType::Int,
                    required: false,
                    default_value: None,
                },
                ParameterMetadata {
                    name: "flow_max_iter".to_string(),
                    description: "Stationary-flow iteration budget; nonconvergence rejects before writing labels.".to_string(),
                    value_type: ParameterType::Int,
                    required: false,
                    default_value: None,
                },
                ParameterMetadata {
                    name: "seed".to_string(),
                    description: "Random seed for reproducibility.".to_string(),
                    value_type: ParameterType::Int,
                    required: false,
                    default_value: None,
                },
                ParameterMetadata {
                    name: "output_attr".to_string(),
                    description: "Node attribute name for community assignment.".to_string(),
                    value_type: ParameterType::Text,
                    required: false,
                    default_value: None,
                },
            ],
        }
    }
}

impl Algorithm for Infomap {
    fn id(&self) -> &'static str {
        "community.infomap"
    }

    fn metadata(&self) -> AlgorithmMetadata {
        Self::metadata_template()
    }

    fn execute(&self, ctx: &mut Context, subgraph: Subgraph) -> Result<Subgraph> {
        let t0 = Instant::now();

        if ctx.is_cancelled() {
            return Err(anyhow!("Infomap cancelled"));
        }
        crate::algorithms::pathfinding::utils::checked_dijkstra_weights(&subgraph, None)?;
        let nodes = subgraph.ordered_nodes();
        let index: HashMap<_, _> = nodes.iter().enumerate().map(|(i, &n)| (n, i)).collect();
        let mut edges = Vec::new();
        let directed = subgraph.graph().borrow().is_directed();
        {
            let graph = subgraph.graph();
            let graph = graph.borrow();
            for &edge in subgraph.ordered_edges().iter() {
                let (u, v) = graph.edge_endpoints(edge)?;
                edges.push((index[&u], index[&v]));
            }
        }
        let walk = Walk::new(nodes.len(), edges, directed)?;
        let flow = walk.solve(self.teleportation, self.flow_max_iter, ctx)?;
        ctx.record_stat("infomap.flow.iterations", flow.iterations as f64);
        ctx.record_stat("infomap.flow.residual", flow.residual);
        let mut rng = fastrand::Rng::with_seed(self.seed.unwrap_or(0));
        let partition = flow.optimize(self.num_trials, self.max_iter, &mut rng, ctx)?;
        ctx.record_stat(
            "infomap.count.communities",
            partition.iter().copied().max().map_or(0, |n| n + 1) as f64,
        );
        if ctx.is_cancelled() {
            return Err(anyhow!("Infomap cancelled"));
        }

        // Phase 8: Write results
        if ctx.persist_results() {
            let write_start = Instant::now();
            let attr_values: Vec<(NodeId, AttrValue)> = nodes
                .iter()
                .enumerate()
                .map(|(idx, &node)| (node, AttrValue::Int(partition[idx] as i64)))
                .collect();

            subgraph
                .set_node_attr_column(self.output_attr.clone(), attr_values)
                .map_err(|err| anyhow!("failed to persist communities: {err}"))?;

            ctx.record_call("infomap.write_attributes", write_start.elapsed());
        }

        ctx.record_call("infomap.total_execution", t0.elapsed());
        Ok(subgraph)
    }
}

pub fn register(registry: &Registry) -> Result<()> {
    let metadata = Infomap::metadata_template();
    let id = metadata.id.clone();
    registry.register_with_metadata(id.as_str(), metadata, |spec| {
        let teleportation = spec.params.get_float("teleportation").unwrap_or(0.15);
        let num_trials = usize::try_from(spec.params.get_int("num_trials").unwrap_or(10))
            .map_err(|_| anyhow!("num_trials must be positive and fit usize"))?;
        let max_iter = usize::try_from(spec.params.get_int("max_iter").unwrap_or(100))
            .map_err(|_| anyhow!("max_iter must be positive and fit usize"))?;
        let seed = spec
            .params
            .get_int("seed")
            .map(|s| u64::try_from(s).map_err(|_| anyhow!("seed must be nonnegative")))
            .transpose()?;
        let output_attr = spec
            .params
            .get_text("output_attr")
            .unwrap_or("community")
            .to_string();

        let flow_max_iter = usize::try_from(spec.params.get_int("flow_max_iter").unwrap_or(10_000))
            .map_err(|_| anyhow!("flow_max_iter must be positive and fit usize"))?;
        Infomap::new(teleportation, num_trials, max_iter, seed, output_attr)
            .and_then(|algo| algo.with_flow_budget(flow_max_iter))
            .map(|algo| Box::new(algo) as Box<dyn Algorithm>)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::NodeId;
    use crate::Graph;
    use std::cell::RefCell;
    use std::collections::HashSet;
    use std::rc::Rc;

    #[test]
    fn test_infomap_basic() {
        let mut graph = Graph::new();
        let n0 = graph.add_node();
        let n1 = graph.add_node();
        let n2 = graph.add_node();
        graph.add_edge(n0, n1).unwrap();
        graph.add_edge(n1, n2).unwrap();
        graph.add_edge(n2, n0).unwrap();

        let n3 = graph.add_node();
        let n4 = graph.add_node();
        let n5 = graph.add_node();
        graph.add_edge(n3, n4).unwrap();
        graph.add_edge(n4, n5).unwrap();
        graph.add_edge(n5, n3).unwrap();

        graph.add_edge(n2, n3).unwrap();

        let nodes: HashSet<NodeId> = [n0, n1, n2, n3, n4, n5].into_iter().collect();
        let subgraph =
            Subgraph::from_nodes(Rc::new(RefCell::new(graph)), nodes, "test".into()).unwrap();
        let algo = Infomap::new(0.15, 5, 50, Some(42), "community".to_string()).unwrap();
        let mut ctx = Context::default();

        let result = algo.execute(&mut ctx, subgraph).unwrap();
        assert_eq!(result.node_count(), 6);
    }

    #[test]
    fn test_infomap_empty() {
        let graph = Graph::new();
        let subgraph =
            Subgraph::from_nodes(Rc::new(RefCell::new(graph)), HashSet::new(), "empty".into())
                .unwrap();
        let algo = Infomap::new(0.15, 5, 50, Some(42), "community".to_string()).unwrap();
        let mut ctx = Context::default();

        let result = algo.execute(&mut ctx, subgraph).unwrap();
        assert_eq!(result.node_count(), 0);
    }

    #[test]
    fn test_infomap_parameter_validation() {
        assert!(Infomap::new(-0.1, 10, 50, None, "community".to_string()).is_err());
        assert!(Infomap::new(1.5, 10, 50, None, "community".to_string()).is_err());
        assert!(Infomap::new(0.15, 0, 50, None, "community".to_string()).is_err());
        assert!(Infomap::new(0.15, 10, 0, None, "community".to_string()).is_err());
        assert!(Infomap::new(0.15, 10, 50, None, "community".to_string()).is_ok());
    }
}
