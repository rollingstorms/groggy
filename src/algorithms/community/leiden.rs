use std::collections::HashMap;

use anyhow::{anyhow, Result};

use crate::algorithms::community::leiden_levels::{fast_moves, inherited_partition, refine};
use crate::algorithms::community::modularity_levels::Level;
use crate::algorithms::registry::Registry;
use crate::algorithms::{
    Algorithm, AlgorithmMetadata, AlgorithmParamValue, Context, CostHint, ParameterMetadata,
    ParameterType,
};
use crate::subgraphs::Subgraph;
use crate::traits::SubgraphOperations;
use crate::types::{AttrName, AttrValue};

/// Leiden community detection algorithm.
///
/// Bounded modularity optimization with constrained refinement and aggregation.
#[derive(Clone, Debug)]
pub struct Leiden {
    max_iter: usize,
    max_phases: usize,
    resolution: f64,
    seed: Option<u64>,
    theta: f64,
    output_attr: AttrName,
}

impl Leiden {
    pub fn new(
        max_iter: usize,
        max_phases: usize,
        resolution: f64,
        seed: Option<u64>,
        output_attr: AttrName,
    ) -> Result<Self> {
        if max_iter == 0 {
            return Err(anyhow!("max_iter must be greater than zero"));
        }
        if max_phases == 0 {
            return Err(anyhow!("max_phases must be greater than zero"));
        }
        if !resolution.is_finite() || resolution <= 0.0 {
            return Err(anyhow!("resolution must be finite and positive"));
        }
        Ok(Self {
            max_iter,
            max_phases,
            resolution,
            seed,
            theta: 0.01,
            output_attr,
        })
    }

    /// Set the positive finite refinement temperature in normalized modularity units.
    pub fn with_theta(mut self, theta: f64) -> Result<Self> {
        if !theta.is_finite() || theta <= 0.0 {
            return Err(anyhow!("theta must be finite and positive"));
        }
        self.theta = theta;
        Ok(self)
    }

    fn metadata_template() -> AlgorithmMetadata {
        AlgorithmMetadata {
            id: "community.leiden".to_string(),
            name: "Leiden".to_string(),
            description: "Improved modularity optimization with guaranteed connectivity."
                .to_string(),
            version: "0.1.0".to_string(),
            cost_hint: CostHint::Linearithmic,
            supports_cancellation: true,
            parameters: vec![
                ParameterMetadata {
                    name: "max_iter".to_string(),
                    description:
                        "Maximum node visits per level as a multiple of that level's node count."
                            .to_string(),
                    value_type: ParameterType::Int,
                    required: false,
                    default_value: Some(AlgorithmParamValue::Int(20)),
                },
                ParameterMetadata {
                    name: "max_phases".to_string(),
                    description: "Maximum fast-move/refinement/aggregation levels.".to_string(),
                    value_type: ParameterType::Int,
                    required: false,
                    default_value: Some(AlgorithmParamValue::Int(10)),
                },
                ParameterMetadata {
                    name: "resolution".to_string(),
                    description: "Resolution parameter for modularity.".to_string(),
                    value_type: ParameterType::Float,
                    required: false,
                    default_value: Some(AlgorithmParamValue::Float(1.0)),
                },
                ParameterMetadata {
                    name: "seed".to_string(),
                    description: "Random seed for reproducibility.".to_string(),
                    value_type: ParameterType::Int,
                    required: false,
                    default_value: None,
                },
                ParameterMetadata {
                    name: "theta".to_string(),
                    description:
                        "Positive finite refinement temperature in normalized modularity units."
                            .to_string(),
                    value_type: ParameterType::Float,
                    required: false,
                    default_value: Some(AlgorithmParamValue::Float(0.01)),
                },
                ParameterMetadata {
                    name: "output_attr".to_string(),
                    description: "Attribute name to store the resulting communities.".to_string(),
                    value_type: ParameterType::Text,
                    required: false,
                    default_value: Some(AlgorithmParamValue::Text("community".to_string())),
                },
            ],
        }
    }
}

impl Algorithm for Leiden {
    fn id(&self) -> &'static str {
        "community.leiden"
    }

    fn metadata(&self) -> AlgorithmMetadata {
        Self::metadata_template()
    }

    fn execute(&self, ctx: &mut Context, subgraph: Subgraph) -> Result<Subgraph> {
        if ctx.is_cancelled() {
            return Err(anyhow!("community detection cancelled"));
        }
        if subgraph.graph().borrow().is_directed() {
            return Err(anyhow!("leiden currently requires an undirected graph"));
        }
        crate::algorithms::pathfinding::utils::checked_dijkstra_weights(&subgraph, None)?;

        let nodes = subgraph.ordered_nodes();
        let index: HashMap<_, _> = nodes
            .iter()
            .enumerate()
            .map(|(i, &node)| (node, i))
            .collect();
        let mut edges = Vec::new();
        {
            let graph = subgraph.graph();
            let graph = graph.borrow();
            for &edge in subgraph.ordered_edges().iter() {
                let (u, v) = graph.edge_endpoints(edge)?;
                edges.push((index[&u], index[&v], 1));
            }
        }
        let mut level = Level::new(nodes.len(), edges)?;
        let mut membership: Vec<_> = (0..nodes.len()).collect();
        let mut inherited = level.singleton_labels();
        let mut final_labels = inherited.clone();
        let mut rng = fastrand::Rng::with_seed(self.seed.unwrap_or(0));
        let mut levels = 0;
        for phase in 0..self.max_phases {
            if ctx.is_cancelled() {
                return Err(anyhow!("Leiden cancelled"));
            }
            ctx.record_stat(format!("leiden.phase_{phase}.nodes"), level.n as f64);
            let before = level.modularity(&inherited, self.resolution)?;
            let parent = fast_moves(
                &level,
                &inherited,
                self.resolution,
                self.max_iter,
                &mut rng,
                ctx,
            )?;
            let after = level.modularity(&parent.labels, self.resolution)?;
            let tolerance = 64. * f64::EPSILON * (1. + before.abs().max(after.abs()));
            if after < before - tolerance {
                return Err(anyhow!("Leiden phase decreased modularity"));
            }
            final_labels = membership.iter().map(|&u| parent.labels[u]).collect();
            levels += 1;
            ctx.record_stat(format!("leiden.phase_{phase}.modularity"), after);
            ctx.record_stat(
                format!("leiden.phase_{phase}.communities"),
                parent.communities as f64,
            );
            ctx.record_stat(
                format!("leiden.phase_{phase}.visits"),
                parent.iterations as f64,
            );
            if parent.communities == level.n {
                break;
            }
            let refined = refine(&level, &parent, self.resolution, self.theta, &mut rng, ctx)?;
            let next_inherited = inherited_partition(&parent, &refined)?;
            let coarse = level.aggregate(&refined)?;
            let projected = coarse.modularity(&next_inherited, self.resolution)?;
            if (projected - after).abs() > tolerance {
                return Err(anyhow!("Leiden aggregation changed modularity"));
            }
            ctx.record_stat(
                format!("leiden.phase_{phase}.refined_groups"),
                refined.communities as f64,
            );
            for u in &mut membership {
                *u = refined.labels[*u];
            }
            inherited = next_inherited;
            level = coarse;
        }
        ctx.record_stat("leiden.total_phases", levels as f64);
        if ctx.is_cancelled() {
            return Err(anyhow!("Leiden cancelled"));
        }

        if ctx.persist_results() {
            let attr_values: Vec<_> = nodes
                .iter()
                .zip(final_labels)
                .map(|(&node, c)| (node, AttrValue::Int(c as i64)))
                .collect();

            ctx.with_scoped_timer("community.leiden.write_attrs", || {
                subgraph.set_node_attr_column(self.output_attr.clone(), attr_values)
            })?;
        }

        Ok(subgraph)
    }
}

pub fn register(registry: &Registry) -> Result<()> {
    let metadata = Leiden::metadata_template();
    let id = metadata.id.clone();
    registry.register_with_metadata(id.as_str(), metadata, |spec| {
        let max_iter = usize::try_from(spec.params.get_int("max_iter").unwrap_or(20))
            .map_err(|_| anyhow!("max_iter must be positive and fit usize"))?;
        let max_phases = usize::try_from(spec.params.get_int("max_phases").unwrap_or(10))
            .map_err(|_| anyhow!("max_phases must be positive and fit usize"))?;
        let resolution = spec.params.get_float("resolution").unwrap_or(1.0);
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
        let theta = spec.params.get_float("theta").unwrap_or(0.01);
        Leiden::new(max_iter, max_phases, resolution, seed, output_attr)
            .and_then(|algo| algo.with_theta(theta))
            .map(|algo| Box::new(algo) as Box<dyn Algorithm>)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::community::utils::find_connected_components;
    use crate::api::graph::Graph;
    use crate::types::NodeId;
    use std::cell::RefCell;
    use std::collections::HashSet;
    use std::rc::Rc;

    fn make_two_cliques() -> (Graph, Vec<NodeId>) {
        let mut graph = Graph::new();
        let nodes: Vec<NodeId> = (0..8).map(|_| graph.add_node()).collect();

        // First clique: 0-1-2-3
        for i in 0..4 {
            for j in i + 1..4 {
                graph.add_edge(nodes[i], nodes[j]).unwrap();
                graph.add_edge(nodes[j], nodes[i]).unwrap();
            }
        }

        // Second clique: 4-5-6-7
        for i in 4..8 {
            for j in i + 1..8 {
                graph.add_edge(nodes[i], nodes[j]).unwrap();
                graph.add_edge(nodes[j], nodes[i]).unwrap();
            }
        }

        // Weak link between cliques
        graph.add_edge(nodes[3], nodes[4]).unwrap();
        graph.add_edge(nodes[4], nodes[3]).unwrap();

        (graph, nodes)
    }

    #[test]
    fn test_leiden_two_cliques() {
        let (graph, nodes) = make_two_cliques();
        let node_set: HashSet<NodeId> = nodes.iter().copied().collect();
        let subgraph =
            Subgraph::from_nodes(Rc::new(RefCell::new(graph)), node_set, "test".into()).unwrap();

        let leiden = Leiden::new(20, 10, 1.0, Some(42), "community".into()).unwrap();
        let mut ctx = Context::new();
        let result = leiden.execute(&mut ctx, subgraph).unwrap();

        // Check that nodes are assigned communities
        let attr_name: AttrName = "community".into();
        let communities: HashMap<NodeId, i64> = nodes
            .iter()
            .map(|&n| {
                let val = result.get_node_attribute(n, &attr_name).unwrap().unwrap();
                (n, val.as_int().unwrap())
            })
            .collect();

        // First clique should be in same community
        let comm0 = communities[&nodes[0]];
        assert_eq!(communities[&nodes[1]], comm0);
        assert_eq!(communities[&nodes[2]], comm0);

        // Second clique should be in same community
        let comm4 = communities[&nodes[4]];
        assert_eq!(communities[&nodes[5]], comm4);
        assert_eq!(communities[&nodes[6]], comm4);

        // The two cliques should ideally be in different communities
        // (though algorithm may merge due to bridge edge)
    }

    #[test]
    fn test_leiden_empty_graph() {
        let graph = Graph::new();
        let subgraph =
            Subgraph::from_nodes(Rc::new(RefCell::new(graph)), HashSet::new(), "empty".into())
                .unwrap();

        let leiden = Leiden::new(20, 10, 1.0, None, "community".into()).unwrap();
        let mut ctx = Context::new();
        let result = leiden.execute(&mut ctx, subgraph).unwrap();

        assert_eq!(result.node_count(), 0);
    }

    #[test]
    fn test_connected_components_detection() {
        let mut adjacency: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        adjacency.insert(0, vec![1, 2]);
        adjacency.insert(1, vec![0, 2]);
        adjacency.insert(2, vec![0, 1]);
        adjacency.insert(3, vec![4]);
        adjacency.insert(4, vec![3]);

        let nodes = vec![0, 1, 2, 3, 4];
        let components = find_connected_components(&nodes, &adjacency);

        assert_eq!(components.len(), 2);
        assert!(components.iter().any(|c| c.contains(&0) && c.contains(&1)));
        assert!(components.iter().any(|c| c.contains(&3) && c.contains(&4)));
    }
}
