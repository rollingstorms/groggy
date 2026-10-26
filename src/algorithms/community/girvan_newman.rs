use std::collections::HashMap;
use std::time::Instant;

use anyhow::{anyhow, Result};

use crate::algorithms::community::girvan_newman_paths::{Edge, Topology};
use crate::algorithms::community::modularity::ModularityData;
use crate::algorithms::pathfinding::utils::checked_dijkstra_weights;
use crate::algorithms::registry::Registry;
use crate::algorithms::{
    Algorithm, AlgorithmMetadata, Context, CostHint, ParameterMetadata, ParameterType,
};
use crate::subgraphs::Subgraph;
use crate::types::{AttrName, AttrValue, NodeId};

/// Girvan-Newman hierarchical community detection via edge betweenness.
///
/// This algorithm iteratively removes edges with highest betweenness centrality,
/// returning the best partition scored against the original undirected graph.
/// Edge lengths affect shortest paths; modularity uses unweighted edge instances.
#[derive(Clone, Debug)]
pub struct GirvanNewman {
    num_levels: Option<usize>,
    modularity_threshold: Option<f64>,
    weight_attr: Option<AttrName>,
    output_attr: AttrName,
}

impl GirvanNewman {
    pub fn new(
        num_levels: Option<usize>,
        modularity_threshold: Option<f64>,
        weight_attr: Option<AttrName>,
        output_attr: AttrName,
    ) -> Self {
        Self {
            num_levels,
            modularity_threshold,
            weight_attr,
            output_attr,
        }
    }

    fn metadata_template() -> AlgorithmMetadata {
        AlgorithmMetadata {
            id: "community.girvan_newman".to_string(),
            name: "Girvan-Newman".to_string(),
            description: "Undirected community detection via edge-instance betweenness removal and original-graph modularity. Intended for small graphs.".to_string(),
            version: "0.1.0".to_string(),
            cost_hint: CostHint::Cubic,
            supports_cancellation: true,
            parameters: vec![
                ParameterMetadata {
                    name: "num_levels".to_string(),
                    description: "Maximum individual edge removals; zero returns initial components. Default: all non-loop edges, subject to modularity stopping.".to_string(),
                    value_type: ParameterType::Int,
                    required: false,
                    default_value: None,
                },
                ParameterMetadata {
                    name: "modularity_threshold".to_string(),
                    description: "Stop when original-graph modularity falls below the best by more than this nonnegative threshold (default 0.0001).".to_string(),
                    value_type: ParameterType::Float,
                    required: false,
                    default_value: None,
                },
                ParameterMetadata {
                    name: "weight_attr".to_string(),
                    description: "Optional strictly positive finite edge lengths (missing one); modularity remains unweighted.".to_string(),
                    value_type: ParameterType::Text,
                    required: false,
                    default_value: None,
                },
                ParameterMetadata {
                    name: "output_attr".to_string(),
                    description: "Node attribute to store community assignments.".to_string(),
                    value_type: ParameterType::Text,
                    required: false,
                    default_value: None,
                },
            ],
        }
    }

    fn compute(&self, ctx: &mut Context, subgraph: &Subgraph) -> Result<HashMap<NodeId, usize>> {
        let t0 = Instant::now();
        if ctx.is_cancelled() {
            return Err(anyhow!("Girvan-Newman cancelled"));
        }
        let threshold = self.modularity_threshold.unwrap_or(0.0001);
        if !threshold.is_finite() || threshold < 0.0 {
            return Err(anyhow!(
                "modularity_threshold must be finite and nonnegative"
            ));
        }
        if subgraph.graph().borrow().is_directed() {
            return Err(anyhow!(
                "Girvan-Newman currently requires an undirected graph"
            ));
        }
        let checked = checked_dijkstra_weights(ctx, subgraph, self.weight_attr.as_ref())?;
        if self.weight_attr.is_some() && checked.values().any(|&weight| weight <= 0.0) {
            return Err(anyhow!(
                "Girvan-Newman requires strictly positive edge weights"
            ));
        }
        let nodes = subgraph.ordered_nodes();
        let index: HashMap<_, _> = nodes
            .iter()
            .enumerate()
            .map(|(i, &node)| (node, i))
            .collect();
        let mut edges = Vec::new();
        let mut original_edges = Vec::new();
        {
            let graph = subgraph.graph();
            let graph = graph.borrow();
            for &id in subgraph.ordered_edges().iter() {
                let (u, v) = graph.edge_endpoints(id)?;
                let value = self
                    .weight_attr
                    .as_ref()
                    .map(|attr| ctx.edge_attribute(subgraph, id, attr))
                    .transpose()?
                    .flatten();
                let length = crate::query::weights::dijkstra_weight(value.as_ref(), id)?;
                edges.push(Edge {
                    id,
                    u: index[&u],
                    v: index[&v],
                    length,
                });
                original_edges.push((u, v));
            }
        }
        let topology = Topology::new(nodes.len(), edges);
        let mut active: Vec<_> = topology.edges.iter().map(|e| e.u != e.v).collect();
        let initial_edge_count = active.iter().filter(|&&v| v).count();
        ctx.record_stat("girvan_newman.count.input_nodes", nodes.len() as f64);
        ctx.record_stat(
            "girvan_newman.count.initial_edges",
            initial_edge_count as f64,
        );
        let data = ModularityData::new(&original_edges);
        let score = |labels: &[usize]| {
            let partition: HashMap<_, _> =
                nodes.iter().zip(labels).map(|(&n, &c)| (n, c)).collect();
            crate::algorithms::community::modularity::modularity(&partition, &original_edges, &data)
        };
        let mut best = topology.components(&active);
        let mut best_modularity = score(&best);
        let limit = self
            .num_levels
            .unwrap_or(initial_edge_count)
            .min(initial_edge_count);
        let mut removals = 0;
        for iteration in 0..limit {
            if ctx.is_cancelled() {
                return Err(anyhow!("Girvan-Newman cancelled"));
            }
            let start = Instant::now();
            let scores = topology.betweenness(&active, self.weight_attr.is_some(), ctx)?;
            ctx.record_call(
                format!("girvan_newman.iteration_{iteration}.compute_betweenness"),
                start.elapsed(),
            );
            // Sorted edge IDs plus strict improvement give a deterministic lowest-ID tie break.
            let mut chosen = None;
            let mut highest = -1.0;
            for (edge, &value) in scores.iter().enumerate() {
                if active[edge] && value > highest {
                    chosen = Some(edge);
                    highest = value;
                }
            }
            let Some(edge) = chosen else {
                break;
            };
            active[edge] = false;
            removals += 1;
            let labels = topology.components(&active);
            let modularity = score(&labels);
            ctx.record_stat(
                format!("girvan_newman.iteration_{iteration}.modularity"),
                modularity,
            );
            if modularity > best_modularity {
                best_modularity = modularity;
                best = labels;
            }
            if modularity < best_modularity - threshold {
                break;
            }
        }
        ctx.record_stat("girvan_newman.count.iterations", removals as f64);
        ctx.record_stat("girvan_newman.best_modularity", best_modularity);
        ctx.record_call("girvan_newman.compute", t0.elapsed());
        Ok(nodes
            .iter()
            .zip(best)
            .map(|(&node, community)| (node, community))
            .collect())
    }
}

impl Algorithm for GirvanNewman {
    fn supports_atomic_properties(&self) -> bool {
        true
    }

    fn id(&self) -> &'static str {
        "community.girvan_newman"
    }

    fn metadata(&self) -> AlgorithmMetadata {
        Self::metadata_template()
    }

    fn execute(&self, ctx: &mut Context, subgraph: Subgraph) -> Result<Subgraph> {
        let t0 = Instant::now();

        let communities = self.compute(ctx, &subgraph)?;

        if ctx.persist_results() {
            let attr_values: Vec<(NodeId, AttrValue)> = communities
                .iter()
                .map(|(&node, &comm)| (node, AttrValue::Int(comm as i64)))
                .collect();

            {
                let started = Instant::now();
                let result =
                    ctx.attach_node_column(&subgraph, self.output_attr.clone(), attr_values);
                ctx.record_call("girvan_newman.write_attributes", started.elapsed());
                result
            }
            .map_err(|err| anyhow!("failed to persist community assignments: {err}"))?;
        }

        ctx.record_duration("girvan_newman.total_execution", t0.elapsed());
        Ok(subgraph)
    }
}

pub fn register(registry: &Registry) -> Result<()> {
    let metadata = GirvanNewman::metadata_template();
    let id = metadata.id.clone();

    registry.register_with_metadata(id.as_str(), metadata, |spec| {
        let num_levels = spec
            .params
            .get_int("num_levels")
            .map(|i| {
                usize::try_from(i)
                    .map_err(|_| anyhow!("num_levels must be nonnegative and fit usize"))
            })
            .transpose()?;

        let modularity_threshold = spec.params.get_float("modularity_threshold");

        let weight_attr = spec
            .params
            .get_text("weight_attr")
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string().into());

        let output_attr = spec
            .params
            .get_text("output_attr")
            .unwrap_or("community")
            .to_string()
            .into();

        Ok(Box::new(GirvanNewman::new(
            num_levels,
            modularity_threshold,
            weight_attr,
            output_attr,
        )) as Box<dyn Algorithm>)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::Context;
    use crate::api::graph::Graph;
    use crate::subgraphs::Subgraph;
    use crate::traits::SubgraphOperations;
    use std::cell::RefCell;
    use std::collections::HashSet;
    use std::rc::Rc;

    #[test]
    fn test_girvan_newman_small_graph() {
        let mut graph = Graph::new_manual();

        // Two triangles connected by a bridge
        let n0 = graph.add_node().unwrap();
        let n1 = graph.add_node().unwrap();
        let n2 = graph.add_node().unwrap();
        let n3 = graph.add_node().unwrap();
        let n4 = graph.add_node().unwrap();
        let n5 = graph.add_node().unwrap();

        // Triangle 1
        graph.add_edge(n0, n1).unwrap();
        graph.add_edge(n1, n2).unwrap();
        graph.add_edge(n2, n0).unwrap();

        // Bridge
        graph.add_edge(n2, n3).unwrap();

        // Triangle 2
        graph.add_edge(n3, n4).unwrap();
        graph.add_edge(n4, n5).unwrap();
        graph.add_edge(n5, n3).unwrap();

        let nodes: HashSet<NodeId> = vec![n0, n1, n2, n3, n4, n5].into_iter().collect();
        let subgraph =
            Subgraph::from_nodes(Rc::new(RefCell::new(graph)), nodes, "test".into()).unwrap();

        let algo = GirvanNewman::new(Some(5), None, None, "community".into());

        let mut ctx = Context::new();
        let result = algo.execute(&mut ctx, subgraph).unwrap();

        let attr_name: AttrName = "community".to_string();

        // Get community assignments
        let comm_0 = result.get_node_attribute(n0, &attr_name).unwrap().unwrap();
        let comm_1 = result.get_node_attribute(n1, &attr_name).unwrap().unwrap();
        let comm_2 = result.get_node_attribute(n2, &attr_name).unwrap().unwrap();
        let comm_3 = result.get_node_attribute(n3, &attr_name).unwrap().unwrap();
        let comm_4 = result.get_node_attribute(n4, &attr_name).unwrap().unwrap();
        let comm_5 = result.get_node_attribute(n5, &attr_name).unwrap().unwrap();

        // Nodes 0,1,2 should be in one community
        assert_eq!(comm_0, comm_1);
        assert_eq!(comm_1, comm_2);

        // Nodes 3,4,5 should be in another community
        assert_eq!(comm_3, comm_4);
        assert_eq!(comm_4, comm_5);

        // The two communities should be different
        assert_ne!(comm_0, comm_3);
    }

    #[test]
    fn test_girvan_newman_disconnected() {
        let mut graph = Graph::new_manual();

        // Two disconnected triangles
        let n0 = graph.add_node().unwrap();
        let n1 = graph.add_node().unwrap();
        let n2 = graph.add_node().unwrap();
        let n3 = graph.add_node().unwrap();
        let n4 = graph.add_node().unwrap();
        let n5 = graph.add_node().unwrap();

        // Triangle 1
        graph.add_edge(n0, n1).unwrap();
        graph.add_edge(n1, n2).unwrap();
        graph.add_edge(n2, n0).unwrap();

        // Triangle 2 (disconnected)
        graph.add_edge(n3, n4).unwrap();
        graph.add_edge(n4, n5).unwrap();
        graph.add_edge(n5, n3).unwrap();

        let nodes: HashSet<NodeId> = vec![n0, n1, n2, n3, n4, n5].into_iter().collect();
        let subgraph =
            Subgraph::from_nodes(Rc::new(RefCell::new(graph)), nodes, "test".into()).unwrap();

        let algo = GirvanNewman::new(Some(5), None, None, "community".into());

        let mut ctx = Context::new();
        let result = algo.execute(&mut ctx, subgraph).unwrap();

        let attr_name: AttrName = "community".to_string();

        // Get community assignments
        let comm_0 = result.get_node_attribute(n0, &attr_name).unwrap().unwrap();
        let comm_1 = result.get_node_attribute(n1, &attr_name).unwrap().unwrap();
        let comm_2 = result.get_node_attribute(n2, &attr_name).unwrap().unwrap();
        let comm_3 = result.get_node_attribute(n3, &attr_name).unwrap().unwrap();

        // Nodes within triangles should be in same community
        assert_eq!(comm_0, comm_1);
        assert_eq!(comm_1, comm_2);

        // Disconnected triangles should be in different communities
        assert_ne!(comm_0, comm_3);
    }

    #[test]
    fn test_girvan_newman_weighted() {
        let mut graph = Graph::new_manual();

        // Two triangles connected by TWO bridges with different weights
        let n0 = graph.add_node().unwrap();
        let n1 = graph.add_node().unwrap();
        let n2 = graph.add_node().unwrap();
        let n3 = graph.add_node().unwrap();
        let n4 = graph.add_node().unwrap();
        let n5 = graph.add_node().unwrap();

        // Triangle 1
        graph.add_edge(n0, n1).unwrap();
        graph.add_edge(n1, n2).unwrap();
        graph.add_edge(n2, n0).unwrap();

        // Two bridges with different weights
        let bridge1 = graph.add_edge(n2, n3).unwrap(); // Low weight bridge
        let bridge2 = graph.add_edge(n1, n4).unwrap(); // High weight bridge

        // Triangle 2
        graph.add_edge(n3, n4).unwrap();
        graph.add_edge(n4, n5).unwrap();
        graph.add_edge(n5, n3).unwrap();

        // Set edge weights
        graph
            .set_edge_attr(bridge1, "weight".into(), AttrValue::Float(1.0))
            .unwrap();
        graph
            .set_edge_attr(bridge2, "weight".into(), AttrValue::Float(100.0))
            .unwrap();

        let nodes: HashSet<NodeId> = vec![n0, n1, n2, n3, n4, n5].into_iter().collect();
        let subgraph =
            Subgraph::from_nodes(Rc::new(RefCell::new(graph)), nodes, "test".into()).unwrap();

        // Run weighted version
        let algo_weighted =
            GirvanNewman::new(Some(1), None, Some("weight".into()), "community".into());
        let mut ctx_weighted = Context::new();
        let result_weighted = algo_weighted
            .execute(&mut ctx_weighted, subgraph.clone())
            .unwrap();

        // Run unweighted version
        let algo_unweighted = GirvanNewman::new(Some(1), None, None, "community_unweighted".into());
        let mut ctx_unweighted = Context::new();
        let result_unweighted = algo_unweighted
            .execute(&mut ctx_unweighted, subgraph)
            .unwrap();

        let attr_weighted: AttrName = "community".to_string();
        let attr_unweighted: AttrName = "community_unweighted".to_string();

        // Both versions should successfully complete
        result_weighted
            .get_node_attribute(n0, &attr_weighted)
            .unwrap()
            .unwrap();
        result_unweighted
            .get_node_attribute(n0, &attr_unweighted)
            .unwrap()
            .unwrap();
    }
}
