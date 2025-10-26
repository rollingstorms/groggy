use std::collections::{HashMap, HashSet, VecDeque};

use anyhow::{anyhow, Result};

use crate::algorithms::community::modularity::ModularityData;
use crate::algorithms::pathfinding::utils::collect_edge_weights;
use crate::algorithms::registry::Registry;
use crate::algorithms::{
    Algorithm, AlgorithmMetadata, Context, CostHint, ParameterMetadata,
    ParameterType,
};
use crate::subgraphs::Subgraph;
use crate::traits::SubgraphOperations;
use crate::types::{AttrName, AttrValue, NodeId};

/// Girvan-Newman hierarchical community detection via edge betweenness.
///
/// This algorithm iteratively removes edges with highest betweenness centrality,
/// producing a dendrogram of community structure. It's computationally expensive
/// (O(m²n)) but produces high-quality hierarchical clusterings for small graphs.
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
            description: "Hierarchical community detection via iterative edge removal based on betweenness centrality. O(m²n) complexity - best for small graphs (<10K edges).".to_string(),
            version: "0.1.0".to_string(),
            cost_hint: CostHint::Cubic,
            supports_cancellation: true,
            parameters: vec![
                ParameterMetadata {
                    name: "num_levels".to_string(),
                    description: "Maximum number of edge removal iterations. If None, stops when modularity stops improving.".to_string(),
                    value_type: ParameterType::Int,
                    required: false,
                    default_value: None,
                },
                ParameterMetadata {
                    name: "modularity_threshold".to_string(),
                    description: "Stop when modularity improvement falls below this threshold.".to_string(),
                    value_type: ParameterType::Float,
                    required: false,
                    default_value: None,
                },
                ParameterMetadata {
                    name: "weight_attr".to_string(),
                    description: "Optional edge weight attribute for weighted betweenness.".to_string(),
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

    /// Compute edge betweenness using Brandes-like algorithm
    fn compute_edge_betweenness(
        &self,
        subgraph: &Subgraph,
        weight_map: Option<&HashMap<(NodeId, NodeId), f64>>,
    ) -> Result<HashMap<(NodeId, NodeId), f64>> {
        let nodes: Vec<NodeId> = subgraph.nodes().iter().copied().collect();
        let mut edge_betweenness: HashMap<(NodeId, NodeId), f64> = HashMap::new();

        // Initialize all edges to 0
        for &node in &nodes {
            for neighbor in subgraph.neighbors(node)? {
                edge_betweenness.insert((node, neighbor), 0.0);
            }
        }

        // For each source node, compute shortest paths and accumulate edge betweenness
        for &source in &nodes {
            let (order, predecessors, sigma) =
                self.shortest_paths(subgraph, source, weight_map)?;

            // Accumulate edge betweenness from bottom up
            let mut delta: HashMap<NodeId, f64> = nodes.iter().map(|&v| (v, 0.0)).collect();
            let mut edge_flow: HashMap<(NodeId, NodeId), f64> = HashMap::new();

            for &w in order.iter().rev() {
                if let Some(preds) = predecessors.get(&w) {
                    let coeff = (1.0 + delta[&w]) / sigma[&w];
                    for &v in preds {
                        let contribution = sigma[&v] * coeff;
                        delta.entry(v).and_modify(|d| *d += contribution);

                        // Track flow on this edge
                        *edge_flow.entry((v, w)).or_insert(0.0) += contribution;
                    }
                }
            }

            // Add edge flows to betweenness (both directions for undirected)
            for ((u, v), flow) in edge_flow {
                *edge_betweenness.entry((u, v)).or_insert(0.0) += flow;
                *edge_betweenness.entry((v, u)).or_insert(0.0) += flow;
            }
        }

        // Normalize (each edge counted from both endpoints)
        for value in edge_betweenness.values_mut() {
            *value /= 2.0;
        }

        Ok(edge_betweenness)
    }

    /// Shortest paths computation (BFS for unweighted, Dijkstra for weighted)
    fn shortest_paths(
        &self,
        subgraph: &Subgraph,
        source: NodeId,
        weight_map: Option<&HashMap<(NodeId, NodeId), f64>>,
    ) -> Result<(
        Vec<NodeId>,
        HashMap<NodeId, Vec<NodeId>>,
        HashMap<NodeId, f64>,
    )> {
        let nodes: Vec<NodeId> = subgraph.nodes().iter().copied().collect();

        if weight_map.is_none() {
            // Unweighted BFS
            let mut stack = Vec::new();
            let mut predecessors: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
            let mut sigma: HashMap<NodeId, f64> = nodes.iter().map(|&v| (v, 0.0)).collect();
            sigma.insert(source, 1.0);

            let mut distance: HashMap<NodeId, i64> = nodes.iter().map(|&v| (v, -1)).collect();
            distance.insert(source, 0);

            let mut queue = VecDeque::new();
            queue.push_back(source);

            while let Some(v) = queue.pop_front() {
                stack.push(v);
                let neighbors = subgraph.neighbors(v)?;
                for w in neighbors {
                    if distance[&w] < 0 {
                        distance.insert(w, distance[&v] + 1);
                        queue.push_back(w);
                    }
                    if distance[&w] == distance[&v] + 1 {
                        let sigma_v = sigma[&v];
                        if let Some(val) = sigma.get_mut(&w) {
                            *val += sigma_v;
                        }
                        predecessors.entry(w).or_default().push(v);
                    }
                }
            }

            Ok((stack, predecessors, sigma))
        } else {
            // Weighted Dijkstra variant
            let weights = weight_map.unwrap();
            let mut predecessors: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
            let mut sigma: HashMap<NodeId, f64> = nodes.iter().map(|&v| (v, 0.0)).collect();
            sigma.insert(source, 1.0);

            let mut dist: HashMap<NodeId, f64> = nodes
                .iter()
                .map(|&v| (v, f64::INFINITY))
                .collect();
            dist.insert(source, 0.0);

            #[derive(Copy, Clone, Debug)]
            struct State {
                cost: f64,
                node: NodeId,
            }
            impl Eq for State {}
            impl PartialEq for State {
                fn eq(&self, other: &Self) -> bool {
                    self.node == other.node && (self.cost - other.cost).abs() <= 1e-9
                }
            }
            impl Ord for State {
                fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                    other
                        .cost
                        .partial_cmp(&self.cost)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| self.node.cmp(&other.node))
                }
            }
            impl PartialOrd for State {
                fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                    Some(self.cmp(other))
                }
            }

            let mut heap = std::collections::BinaryHeap::new();
            heap.push(State {
                cost: 0.0,
                node: source,
            });

            let mut order = Vec::new();

            while let Some(State { cost, node }) = heap.pop() {
                if cost > dist[&node] + 1e-9 {
                    continue;
                }
                order.push(node);
                let neighbors = subgraph.neighbors(node)?;
                for neighbor in neighbors {
                    let weight = weights.get(&(node, neighbor)).copied().unwrap_or(1.0);
                    let next = cost + weight;
                    let current = dist.get(&neighbor).copied().unwrap_or(f64::INFINITY);
                    if next + 1e-9 < current {
                        dist.insert(neighbor, next);
                        sigma.insert(neighbor, sigma[&node]);
                        predecessors.insert(neighbor, vec![node]);
                        heap.push(State {
                            cost: next,
                            node: neighbor,
                        });
                    } else if (next - current).abs() <= 1e-9 {
                        let sigma_node = sigma[&node];
                        if let Some(val) = sigma.get_mut(&neighbor) {
                            *val += sigma_node;
                        }
                        predecessors.entry(neighbor).or_default().push(node);
                    }
                }
            }

            Ok((order, predecessors, sigma))
        }
    }

    /// Remove edge from active edge set (simulating removal without modifying subgraph)
    fn remove_edge(
        active_edges: &mut HashSet<(NodeId, NodeId)>,
        u: NodeId,
        v: NodeId,
    ) {
        active_edges.remove(&(u, v));
        active_edges.remove(&(v, u));
    }

    /// Get neighbors considering only active edges
    fn active_neighbors(
        &self,
        subgraph: &Subgraph,
        node: NodeId,
        active_edges: &HashSet<(NodeId, NodeId)>,
    ) -> Result<Vec<NodeId>> {
        let all_neighbors = subgraph.neighbors(node)?;
        Ok(all_neighbors
            .into_iter()
            .filter(|&n| active_edges.contains(&(node, n)))
            .collect())
    }

    /// Compute communities from active edges using connected components
    fn compute_communities(
        &self,
        subgraph: &Subgraph,
        active_edges: &HashSet<(NodeId, NodeId)>,
    ) -> Result<HashMap<NodeId, usize>> {
        let nodes: Vec<NodeId> = subgraph.nodes().iter().copied().collect();
        let mut parent: HashMap<NodeId, NodeId> = nodes.iter().map(|&n| (n, n)).collect();
        let mut rank: HashMap<NodeId, usize> = nodes.iter().map(|&n| (n, 0)).collect();

        // Union-Find
        fn find(parent: &mut HashMap<NodeId, NodeId>, node: NodeId) -> NodeId {
            if parent[&node] != node {
                let root = find(parent, parent[&node]);
                parent.insert(node, root);
            }
            parent[&node]
        }

        fn union(
            parent: &mut HashMap<NodeId, NodeId>,
            rank: &mut HashMap<NodeId, usize>,
            u: NodeId,
            v: NodeId,
        ) {
            let root_u = find(parent, u);
            let root_v = find(parent, v);
            if root_u != root_v {
                let rank_u = rank[&root_u];
                let rank_v = rank[&root_v];
                if rank_u < rank_v {
                    parent.insert(root_u, root_v);
                } else if rank_u > rank_v {
                    parent.insert(root_v, root_u);
                } else {
                    parent.insert(root_v, root_u);
                    rank.insert(root_u, rank_u + 1);
                }
            }
        }

        // Union all active edges
        for &(u, v) in active_edges {
            union(&mut parent, &mut rank, u, v);
        }

        // Assign community IDs
        let mut root_to_comm: HashMap<NodeId, usize> = HashMap::new();
        let mut next_comm = 0;
        let mut communities: HashMap<NodeId, usize> = HashMap::new();

        for &node in &nodes {
            let root = find(&mut parent, node);
            let comm = *root_to_comm.entry(root).or_insert_with(|| {
                let c = next_comm;
                next_comm += 1;
                c
            });
            communities.insert(node, comm);
        }

        Ok(communities)
    }

    fn compute(&self, ctx: &mut Context, subgraph: &Subgraph) -> Result<HashMap<NodeId, usize>> {
        let nodes: Vec<NodeId> = subgraph.nodes().iter().copied().collect();
        let num_nodes = nodes.len();

        if num_nodes < 2 {
            return Ok(nodes.iter().map(|&n| (n, 0)).collect());
        }

        // Build initial edge list
        let mut all_edges: Vec<(NodeId, NodeId)> = Vec::new();
        for &node in &nodes {
            for neighbor in subgraph.neighbors(node)? {
                if node < neighbor {
                    all_edges.push((node, neighbor));
                }
            }
        }

        // Initialize active edges
        let mut active_edges: HashSet<(NodeId, NodeId)> = HashSet::new();
        for &(u, v) in &all_edges {
            active_edges.insert((u, v));
            active_edges.insert((v, u));
        }

        let initial_edge_count = all_edges.len();

        let weight_map = self
            .weight_attr
            .as_ref()
            .map(|attr| collect_edge_weights(subgraph, attr));

        // Create modularity data from all edges
        let mod_data = ModularityData::new(&all_edges);

        let mut best_communities: HashMap<NodeId, usize> = nodes.iter().map(|&n| (n, 0)).collect();
        let mut best_modularity = -1.0;

        let max_iterations = self.num_levels.unwrap_or(initial_edge_count);
        let mod_threshold = self.modularity_threshold.unwrap_or(0.0001);

        for iteration in 0..max_iterations {
            if ctx.is_cancelled() {
                return Err(anyhow!("Girvan-Newman cancelled"));
            }

            // Compute current communities
            let communities = self.compute_communities(subgraph, &active_edges)?;

            // Build edge list for current active edges
            let current_edges: Vec<(NodeId, NodeId)> = active_edges
                .iter()
                .filter_map(|&(u, v)| if u < v { Some((u, v)) } else { None })
                .collect();

            // Compute modularity on current partition
            let modularity = crate::algorithms::community::modularity::modularity(
                &communities,
                &current_edges,
                &mod_data,
            );

            // Track best partition
            if modularity > best_modularity {
                best_modularity = modularity;
                best_communities = communities.clone();
            }

            // Check stopping criterion
            if iteration > 0 && modularity < best_modularity - mod_threshold {
                break;
            }

            // If no edges left, stop
            if active_edges.is_empty() {
                break;
            }

            // Compute edge betweenness on remaining graph
            let edge_betweenness = self.compute_edge_betweenness(subgraph, weight_map.as_ref())?;

            // Find edge with max betweenness
            let mut max_betweenness = -1.0;
            let mut edge_to_remove: Option<(NodeId, NodeId)> = None;

            for &(u, v) in &active_edges {
                if u < v {
                    // Only consider each edge once
                    if let Some(&betweenness) = edge_betweenness.get(&(u, v)) {
                        if active_edges.contains(&(u, v)) {
                            if betweenness > max_betweenness {
                                max_betweenness = betweenness;
                                edge_to_remove = Some((u, v));
                            }
                        }
                    }
                }
            }

            // Remove the edge with highest betweenness
            if let Some((u, v)) = edge_to_remove {
                Self::remove_edge(&mut active_edges, u, v);
            } else {
                break; // No more edges to consider
            }

            ctx.emit_iteration(iteration, 0);
        }

        Ok(best_communities)
    }
}

impl Algorithm for GirvanNewman {
    fn id(&self) -> &'static str {
        "community.girvan_newman"
    }

    fn metadata(&self) -> AlgorithmMetadata {
        Self::metadata_template()
    }

    fn execute(&self, ctx: &mut Context, subgraph: Subgraph) -> Result<Subgraph> {
        let communities = self.compute(ctx, &subgraph)?;

        // Prepare bulk updates
        let attr_values: Vec<(NodeId, AttrValue)> = communities
            .into_iter()
            .map(|(node, comm)| (node, AttrValue::Int(comm as i64)))
            .collect();

        let mut updates = HashMap::new();
        updates.insert(self.output_attr.clone(), attr_values);

        subgraph.set_node_attrs(updates)?;

        Ok(subgraph)
    }
}

pub fn register(registry: &Registry) -> Result<()> {
    let metadata = GirvanNewman::metadata_template();
    let id = metadata.id.clone();
    registry.register_with_metadata(id.as_str(), metadata, |spec| {
        let num_levels = spec.params.get_int("num_levels").map(|i| i as usize);

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
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn test_girvan_newman_small_graph() {
        let mut graph = Graph::new();

        // Two triangles connected by a bridge
        let n0 = graph.add_node();
        let n1 = graph.add_node();
        let n2 = graph.add_node();
        let n3 = graph.add_node();
        let n4 = graph.add_node();
        let n5 = graph.add_node();

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
        let subgraph = Subgraph::from_nodes(
            Rc::new(RefCell::new(graph)),
            nodes,
            "test".into(),
        )
        .unwrap();

        let algo = GirvanNewman::new(Some(5), None, None, "community".into());

        let mut ctx = Context::new();
        let result = algo.execute(&mut ctx, subgraph).unwrap();

        let attr_name: AttrName = "community".to_string();

        // Get community assignments
        let comm_0 = result
            .get_node_attribute(n0, &attr_name)
            .unwrap()
            .unwrap();
        let comm_1 = result
            .get_node_attribute(n1, &attr_name)
            .unwrap()
            .unwrap();
        let comm_2 = result
            .get_node_attribute(n2, &attr_name)
            .unwrap()
            .unwrap();
        let comm_3 = result
            .get_node_attribute(n3, &attr_name)
            .unwrap()
            .unwrap();
        let comm_4 = result
            .get_node_attribute(n4, &attr_name)
            .unwrap()
            .unwrap();
        let comm_5 = result
            .get_node_attribute(n5, &attr_name)
            .unwrap()
            .unwrap();

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
        let mut graph = Graph::new();

        // Two disconnected triangles
        let n0 = graph.add_node();
        let n1 = graph.add_node();
        let n2 = graph.add_node();
        let n3 = graph.add_node();
        let n4 = graph.add_node();
        let n5 = graph.add_node();

        // Triangle 1
        graph.add_edge(n0, n1).unwrap();
        graph.add_edge(n1, n2).unwrap();
        graph.add_edge(n2, n0).unwrap();

        // Triangle 2 (disconnected)
        graph.add_edge(n3, n4).unwrap();
        graph.add_edge(n4, n5).unwrap();
        graph.add_edge(n5, n3).unwrap();

        let nodes: HashSet<NodeId> = vec![n0, n1, n2, n3, n4, n5].into_iter().collect();
        let subgraph = Subgraph::from_nodes(
            Rc::new(RefCell::new(graph)),
            nodes,
            "test".into(),
        )
        .unwrap();

        let algo = GirvanNewman::new(Some(5), None, None, "community".into());

        let mut ctx = Context::new();
        let result = algo.execute(&mut ctx, subgraph).unwrap();

        let attr_name: AttrName = "community".to_string();

        // Get community assignments
        let comm_0 = result
            .get_node_attribute(n0, &attr_name)
            .unwrap()
            .unwrap();
        let comm_1 = result
            .get_node_attribute(n1, &attr_name)
            .unwrap()
            .unwrap();
        let comm_2 = result
            .get_node_attribute(n2, &attr_name)
            .unwrap()
            .unwrap();
        let comm_3 = result
            .get_node_attribute(n3, &attr_name)
            .unwrap()
            .unwrap();

        // Nodes within triangles should be in same community
        assert_eq!(comm_0, comm_1);
        assert_eq!(comm_1, comm_2);

        // Disconnected triangles should be in different communities
        assert_ne!(comm_0, comm_3);
    }
}
