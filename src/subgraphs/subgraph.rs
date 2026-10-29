//! Subgraph - A view into a Graph with full Graph API inheritance
//!
//! *** ARCHITECTURE OVERVIEW ***
//! A Subgraph represents a subset of nodes and edges from a parent Graph.
//! Unlike traditional graph libraries, Subgraphs in Groggy have FULL Graph API.
//!
//! DESIGN PHILOSOPHY:
//! - Subgraph IS-A Graph (through trait or delegation)
//! - All Graph operations work on Subgraphs: filter_nodes, bfs, dfs, etc.
//! - Infinite composability: subgraph.filter_nodes().bfs().filter_edges()
//! - Column access: subgraph[attr_name] -> Vec<AttrValue>
//! - True inheritance enables unprecedented power
//!
//! KEY BENEFITS:
//! - Recursive filtering and analysis
//! - Batch operations at any subgraph level
//! - Consistent API regardless of graph or subgraph context
//! - Performance: operations stay in Rust core

use crate::api::graph::Graph;
use crate::errors::{GraphError, GraphResult};
use crate::state::topology::{build_csr_from_edges_with_scratch, Csr, CsrOptions};
use crate::traits::{GraphEntity, SubgraphOperations};
use crate::types::{AttrName, AttrValue, EdgeId, EntityId, NodeId, SubgraphId};
use crate::viz::VizModule;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::{Arc, OnceLock};

/// Similarity metrics for comparing subgraphs
#[derive(Debug, Clone, PartialEq)]
pub enum SimilarityMetric {
    /// Jaccard similarity: |A ∩ B| / |A ∪ B|
    Jaccard,
    /// Dice coefficient: 2 * |A ∩ B| / (|A| + |B|)
    Dice,
    /// Cosine similarity: A·B / (||A|| * ||B||)
    Cosine,
    /// Overlap coefficient: |A ∩ B| / min(|A|, |B|)
    Overlap,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ComponentCacheMode {
    Undirected,
    Weak,
    Strong,
}

#[derive(Debug, Clone)]
pub(crate) struct ComponentCacheComponent {
    pub(crate) nodes: Arc<Vec<NodeId>>,
    pub(crate) edges: Arc<Vec<EdgeId>>,
}

#[derive(Debug, Clone)]
pub(crate) struct ComponentCacheEntry {
    pub(crate) version: crate::state::space::SpaceVersion,
    pub(crate) assignments: Arc<Vec<(NodeId, i64)>>,
    pub(crate) components: Arc<Vec<ComponentCacheComponent>>,
}

#[derive(Debug, Default)]
struct ComponentCacheStore {
    entries: HashMap<ComponentCacheMode, ComponentCacheEntry>,
}

/// A Subgraph represents a subset of nodes and edges from a parent Graph
/// with full Graph API capabilities through delegation.
///
/// CORE CONCEPT: Subgraph IS-A Graph
/// - All Graph operations work: filter_nodes, bfs, dfs, algorithms
/// - Column access: subgraph[attr_name] -> Vec<AttrValue>
/// - Infinite composability: subgraph.filter().filter().bfs().set()
/// - Consistent API at every level
#[derive(Debug, Clone)]
pub struct Subgraph {
    /// Reference to the parent graph (shared across all subgraphs)
    /// This enables all Graph operations to work on Subgraphs
    /// Uses RefCell for interior mutability to allow batch operations
    graph: Rc<RefCell<Graph>>,

    /// Set of node IDs that are included in this subgraph
    /// Operations are filtered to only these nodes
    nodes: Arc<HashSet<NodeId>>,

    /// Set of edge IDs that are included in this subgraph
    /// Usually induced edges (edges between subgraph nodes)
    edges: Arc<HashSet<EdgeId>>,

    /// Metadata about how this subgraph was created
    /// Examples: "filter_nodes", "bfs_traversal", "batch_selection"
    subgraph_type: String,

    /// Unique identifier for this subgraph (for GraphEntity trait)
    /// TODO: Generate proper IDs through GraphPool storage
    subgraph_id: SubgraphId,

    /// Cached component computations keyed by traversal mode
    component_cache: Rc<RefCell<ComponentCacheStore>>,

    /// Cached ascending node order for deterministic algorithms
    ordered_nodes_cache: Arc<OnceLock<Arc<[NodeId]>>>,

    /// Cached ascending edge order for deterministic algorithms
    ordered_edges_cache: Arc<OnceLock<Arc<[EdgeId]>>>,

    /// Cached CSR adjacency keyed by traversal mode (reverse flag)
    topology_cache: RefCell<HashMap<CsrCacheKey, CsrCacheEntry>>,
    validated_scope: RefCell<Option<crate::state::space::SpaceVersion>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CsrCacheKey {
    add_reverse: bool,
}

#[derive(Debug, Clone)]
struct CsrCacheEntry {
    version: crate::state::space::SpaceVersion,
    csr: Arc<Csr>,
}

impl Subgraph {
    /// Create a new Subgraph from a Graph with specific nodes and edges
    ///
    /// PERFORMANCE: Automatically chooses between content-based hashing and simple IDs
    /// based on subgraph size. For large subgraphs (>10k nodes or >30k edges), uses
    /// simple IDs to avoid expensive sorting+hashing overhead. For small subgraphs,
    /// uses content-based IDs for deduplication benefits.
    ///
    /// Use `new_with_content_id()` if you specifically need content-based IDs for
    /// pool storage or deduplication regardless of size.
    pub fn new(
        graph: Rc<RefCell<Graph>>,
        nodes: HashSet<NodeId>,
        edges: HashSet<EdgeId>, // FIXED: Must be EdgeId, not NodeId
        subgraph_type: String,
    ) -> Self {
        // SMART STRATEGY: Choose ID generation based on size
        // Hashing overhead becomes significant above these thresholds
        const HASH_THRESHOLD_NODES: usize = 10_000;
        const HASH_THRESHOLD_EDGES: usize = 30_000;

        let use_simple_id =
            nodes.len() > HASH_THRESHOLD_NODES || edges.len() > HASH_THRESHOLD_EDGES;

        if use_simple_id {
            // Large subgraph - use simple ID for performance
            Self::new_with_simple_id(
                graph,
                nodes,
                edges,
                subgraph_type,
                Self::generate_simple_id(),
            )
        } else {
            // Small subgraph - use content-based ID for deduplication
            Self::new_with_content_id(graph, nodes, edges, subgraph_type)
        }
    }

    /// Create a new Subgraph with content-based ID (always uses hashing)
    ///
    /// Use this when you need deterministic, content-addressable IDs for:
    /// - Pool storage via `store_subgraph()`
    /// - Deduplication of identical subgraphs
    /// - Subgraphs that will be looked up by content
    pub fn new_with_content_id(
        graph: Rc<RefCell<Graph>>,
        nodes: HashSet<NodeId>,
        edges: HashSet<EdgeId>,
        subgraph_type: String,
    ) -> Self {
        // Generate subgraph ID using hash-based approach
        let subgraph_id = {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            let mut sorted_nodes: Vec<_> = nodes.iter().collect();
            sorted_nodes.sort();
            for node in sorted_nodes {
                node.hash(&mut hasher);
            }
            let mut sorted_edges: Vec<_> = edges.iter().collect();
            sorted_edges.sort();
            for edge in sorted_edges {
                edge.hash(&mut hasher);
            }
            subgraph_type.hash(&mut hasher);
            hasher.finish() as SubgraphId
        };

        let nodes = Arc::new(nodes);
        let edges = Arc::new(edges);

        Self {
            graph,
            nodes,
            edges,
            subgraph_type,
            subgraph_id,
            component_cache: Rc::new(RefCell::new(ComponentCacheStore::default())),
            ordered_nodes_cache: Arc::new(OnceLock::new()),
            ordered_edges_cache: Arc::new(OnceLock::new()),
            topology_cache: RefCell::new(HashMap::new()),
            validated_scope: RefCell::new(None),
        }
    }

    /// Generate a unique simple ID using atomic counter
    fn generate_simple_id() -> SubgraphId {
        use std::sync::atomic::{AtomicUsize, Ordering};
        // Start at 100 to avoid reserved IDs (0, 1 used for special views)
        static COUNTER: AtomicUsize = AtomicUsize::new(100);
        COUNTER.fetch_add(1, Ordering::Relaxed)
    }

    /// Create a new Subgraph with just nodes, calculating induced edges
    pub fn from_nodes(
        graph: Rc<RefCell<Graph>>,
        nodes: HashSet<NodeId>,
        subgraph_type: String,
    ) -> GraphResult<Self> {
        let edges = Self::calculate_induced_edges(&graph, &nodes)?;
        Ok(Self::new(graph, nodes, edges, subgraph_type))
    }

    /// Create a new Subgraph with a simple numeric ID (skips expensive hashing)
    ///
    /// Use this directly when you have a specific ID you want to use (e.g.,
    /// reserved IDs like 0 for delegation views, 1 for cached views).
    ///
    /// Most users should call `new()` which automatically chooses between
    /// simple and content-based IDs based on size.
    pub fn new_with_simple_id(
        graph: Rc<RefCell<Graph>>,
        nodes: HashSet<NodeId>,
        edges: HashSet<EdgeId>,
        subgraph_type: String,
        simple_id: SubgraphId,
    ) -> Self {
        let nodes = Arc::new(nodes);
        let edges = Arc::new(edges);

        Self {
            graph,
            nodes,
            edges,
            subgraph_type,
            subgraph_id: simple_id,
            component_cache: Rc::new(RefCell::new(ComponentCacheStore::default())),
            ordered_nodes_cache: Arc::new(OnceLock::new()),
            ordered_edges_cache: Arc::new(OnceLock::new()),
            topology_cache: RefCell::new(HashMap::new()),
            validated_scope: RefCell::new(None),
        }
    }

    pub(crate) fn component_cache_get(
        &self,
        mode: ComponentCacheMode,
    ) -> Option<ComponentCacheEntry> {
        let version = {
            let graph = self.graph.borrow();
            graph.space().cache_version()
        };

        let cache = self.component_cache.borrow();
        cache
            .entries
            .get(&mode)
            .filter(|entry| entry.version == version)
            .cloned()
    }

    pub(crate) fn component_cache_store(
        &self,
        mode: ComponentCacheMode,
        assignments: Arc<Vec<(NodeId, i64)>>,
        components: Arc<Vec<ComponentCacheComponent>>,
    ) {
        let version = {
            let graph = self.graph.borrow();
            graph.space().cache_version()
        };

        let mut cache = self.component_cache.borrow_mut();
        cache.entries.insert(
            mode,
            ComponentCacheEntry {
                version,
                assignments,
                components,
            },
        );
    }

    /// Calculate induced edges (edges where both endpoints are in the node set) - O(k) OPTIMIZED
    ///
    /// Uses columnar topology vectors for O(k) performance where k = number of active edges,
    /// which is much better than O(E) over all edges in the graph.
    pub fn calculate_induced_edges(
        graph: &Rc<RefCell<Graph>>,
        nodes: &HashSet<NodeId>,
    ) -> GraphResult<HashSet<EdgeId>> {
        let mut induced_edges = HashSet::new();
        let graph_borrow = graph.borrow();

        // Get columnar topology vectors (edge_ids, sources, targets) - O(1) if cached
        let (edge_ids, sources, targets) = graph_borrow.get_columnar_topology();

        // Iterate through parallel vectors - O(k) where k = active edges
        for i in 0..edge_ids.len() {
            let edge_id = edge_ids[i];
            let source = sources[i];
            let target = targets[i];

            // O(1) HashSet lookups instead of O(n) Vec::contains
            if nodes.contains(&source) && nodes.contains(&target) {
                induced_edges.insert(edge_id);
            }
        }

        Ok(induced_edges)
    }

    /// Explicitly freeze this selection's current values/topology, without changing its write owner.
    /// Fails if selected IDs are no longer active; does not reconstruct a historical state.
    pub fn pin_view(&self) -> GraphResult<crate::subgraphs::GraphView> {
        crate::subgraphs::GraphView::from_membership(
            self.graph.borrow().pin_current_state()?,
            self.nodes.as_ref().clone(),
            self.edges.as_ref().clone(),
        )
    }

    /// Get reference to the parent graph (for read operations)
    pub fn graph(&self) -> Rc<RefCell<Graph>> {
        self.graph.clone()
    }

    /// Get the nodes in this subgraph
    pub fn nodes(&self) -> &HashSet<NodeId> {
        &self.nodes
    }

    /// Get the edges in this subgraph  
    pub fn edges(&self) -> &HashSet<EdgeId> {
        &self.edges
    }

    /// Get the subgraph type metadata
    pub fn subgraph_type(&self) -> &str {
        &self.subgraph_type
    }

    /// Get node count
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get edge count
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Check if a node is in this subgraph
    pub fn has_node(&self, node_id: NodeId) -> bool {
        self.nodes.contains(&node_id)
    }

    /// Check if an edge is in this subgraph
    pub fn has_edge(&self, edge_id: EdgeId) -> bool {
        self.edges.contains(&edge_id)
    }

    /// Get all node IDs as a Vec (for compatibility with existing APIs)
    pub fn node_ids(&self) -> Vec<NodeId> {
        self.nodes.iter().copied().collect()
    }

    /// Get all edge IDs as a Vec (for compatibility with existing APIs)
    pub fn edge_ids(&self) -> Vec<EdgeId> {
        self.edges.iter().copied().collect()
    }

    /// Deterministic ordered node list (ascending).
    pub(crate) fn ordered_nodes(&self) -> Arc<[NodeId]> {
        let cache = self.ordered_nodes_cache.get_or_init(|| {
            let mut nodes_vec: Vec<NodeId> = self.nodes.iter().copied().collect();
            nodes_vec.sort_unstable();
            Arc::<[NodeId]>::from(nodes_vec.into_boxed_slice())
        });
        Arc::clone(cache)
    }

    /// Deterministic ordered edge list (ascending).
    pub(crate) fn ordered_edges(&self) -> Arc<[EdgeId]> {
        let cache = self.ordered_edges_cache.get_or_init(|| {
            let mut edges_vec: Vec<EdgeId> = self.edges.iter().copied().collect();
            edges_vec.sort_unstable();
            Arc::<[EdgeId]>::from(edges_vec.into_boxed_slice())
        });
        Arc::clone(cache)
    }

    /// Compute an owned numerical snapshot of this view's selected active edges.
    /// The view itself continues to read its live parent graph.
    pub fn adjacency_matrix(
        &self,
        sparse: bool,
        weight_attr: Option<&str>,
    ) -> GraphResult<crate::storage::matrix::GraphMatrix<f64>> {
        let graph = self.graph.borrow();
        let nodes: Vec<_> = self
            .ordered_nodes()
            .iter()
            .copied()
            .filter(|&id| graph.space().contains_node(id))
            .collect();
        let edges: Vec<_> = self
            .ordered_edges()
            .iter()
            .copied()
            .filter(|&id| graph.space().contains_edge(id))
            .collect();
        graph.adjacency_from_entities(&nodes, &edges, sparse, weight_attr)
    }

    /// Retrieve a cached CSR adjacency if the current version matches.
    pub(crate) fn csr_cache_get(&self, add_reverse: bool) -> Option<Arc<Csr>> {
        let version = {
            let graph = self.graph.borrow();
            graph.space().cache_version()
        };

        let key = CsrCacheKey { add_reverse };
        self.topology_cache
            .borrow()
            .get(&key)
            .filter(|entry| entry.version == version)
            .map(|entry| entry.csr.clone())
    }

    /// Store a CSR adjacency in the cache.
    pub(crate) fn csr_cache_store(&self, add_reverse: bool, csr: Arc<Csr>) {
        let version = {
            let graph = self.graph.borrow();
            graph.space().cache_version()
        };

        let key = CsrCacheKey { add_reverse };
        self.topology_cache
            .borrow_mut()
            .insert(key, CsrCacheEntry { version, csr });
    }

    /// Get or build CSR adjacency following STYLE_ALGO pattern.
    ///
    /// This is the internal method used by neighbors() and degree() overrides
    /// to provide CSR-optimized access patterns.
    ///
    /// # Arguments
    /// * `add_reverse` - If true, add reverse edges for undirected traversal
    ///
    /// # Returns
    /// Arc to cached or newly built CSR
    fn get_or_build_csr_internal(&self, add_reverse: bool) -> GraphResult<Arc<Csr>> {
        // Check cache first
        if let Some(cached) = self.csr_cache_get(add_reverse) {
            return Ok(cached);
        }

        // Cache miss - build CSR following STYLE_ALGO pattern
        let nodes = self.ordered_nodes();
        let edges = self.ordered_edges();

        // Build simple indexer: node_id -> CSR index mapping
        // Use HashMap for simplicity (could optimize to dense array later)
        let mut node_to_idx: HashMap<NodeId, usize> = HashMap::new();
        for (idx, &node) in nodes.iter().enumerate() {
            node_to_idx.insert(node, idx);
        }

        // Build CSR using scratch buffer (reuses allocations)
        let mut csr = Csr::default();
        {
            // Scope the borrow to ensure it's released before caching
            let graph = self.graph.borrow();
            let pool = graph.pool();

            let _build_time = build_csr_from_edges_with_scratch(
                &mut csr,
                nodes.len(),
                edges.iter().copied(),
                |nid| node_to_idx.get(&nid).copied(),
                |eid| pool.get_edge_endpoints(eid),
                CsrOptions {
                    add_reverse_edges: add_reverse,
                    sort_neighbors: false,
                },
            );
        } // graph and pool dropped here

        let csr_arc = Arc::new(csr);

        // Store in cache for future use
        self.csr_cache_store(add_reverse, csr_arc.clone());

        Ok(csr_arc)
    }

    /// Capture the current selection and export source-bound node columns.
    pub fn nodes_table(&self) -> GraphResult<crate::storage::table::NodesTable> {
        self.pin_view()?.nodes_table()
    }

    /// Capture the current selection and export source-bound edge columns.
    pub fn edges_table(&self) -> GraphResult<crate::storage::table::EdgesTable> {
        self.pin_view()?.edges_table()
    }

    /// Export both components from one immutable capture of the current selection.
    /// The returned table is pinned; this legacy Subgraph itself remains connected.
    pub fn table(&self) -> GraphResult<crate::storage::table::GraphTable> {
        self.pin_view()?.table()
    }
}

/// Core Graph operations implemented for Subgraph through delegation
/// This makes Subgraph truly behave like a Graph
impl Subgraph {
    fn validate_query_scope(&self, graph: &Graph) -> GraphResult<()> {
        let version = graph.space().cache_version();
        if self.validated_scope.borrow().as_ref() == Some(&version) {
            return Ok(());
        }
        for &node in self.nodes.iter() {
            if !graph.space().contains_node(node) {
                return Err(GraphError::InvalidInput(format!(
                    "Inactive query node {node}"
                )));
            }
        }
        for &edge in self.edges.iter() {
            let (source, target) = graph.edge_endpoints(edge)?;
            if !self.nodes.contains(&source) || !self.nodes.contains(&target) {
                return Err(GraphError::InvalidInput(
                    "Query edge endpoints must belong to the selection".into(),
                ));
            }
        }
        *self.validated_scope.borrow_mut() = Some(version);
        Ok(())
    }
    /// Filter the current selection without reintroducing excluded nodes or edges.
    pub fn filter_nodes(&self, filter: &crate::query::NodeFilter) -> GraphResult<Self> {
        self.filter_nodes_named(filter, "filtered_nodes")
    }
    fn filter_nodes_named(
        &self,
        filter: &crate::query::NodeFilter,
        suffix: &str,
    ) -> GraphResult<Self> {
        let graph = self.graph.borrow();
        self.validate_query_scope(&graph)?;
        let pool = graph.pool();
        let source = crate::query::filtering::Live::new(&pool, graph.space(), Some(&self.edges));
        let nodes: HashSet<_> =
            crate::query::filtering::nodes(&source, &self.ordered_nodes(), filter)?
                .into_iter()
                .collect();
        let mut edges = HashSet::new();
        for &edge in self.edges.iter() {
            let (s, t) = graph.edge_endpoints(edge)?;
            if nodes.contains(&s) && nodes.contains(&t) {
                edges.insert(edge);
            }
        }
        Ok(Self::new(
            self.graph.clone(),
            nodes,
            edges,
            format!("{}_{suffix}", self.subgraph_type),
        ))
    }
    /// Filter current selected edge instances, keeping only their selected endpoints.
    pub fn filter_edges(&self, filter: &crate::query::EdgeFilter) -> GraphResult<Self> {
        let graph = self.graph.borrow();
        self.validate_query_scope(&graph)?;
        let pool = graph.pool();
        let source = crate::query::filtering::Live::new(&pool, graph.space(), Some(&self.edges));
        let edges: HashSet<_> =
            crate::query::filtering::edges(&source, &self.ordered_edges(), filter)?
                .into_iter()
                .collect();
        let mut nodes = HashSet::new();
        for &edge in &edges {
            let (s, t) = graph.edge_endpoints(edge)?;
            nodes.extend([s, t]);
        }
        Ok(Self::new(
            self.graph.clone(),
            nodes,
            edges,
            format!("{}_filtered_edges", self.subgraph_type),
        ))
    }
    /// Conjoin attribute predicates within the current selection; an empty map keeps it.
    pub fn filter_nodes_by_attributes(
        &self,
        filters: &HashMap<AttrName, crate::query::AttributeFilter>,
    ) -> GraphResult<Self> {
        let filter = if filters.is_empty() {
            crate::query::NodeFilter::NodeSet(self.nodes.as_ref().clone())
        } else {
            crate::query::NodeFilter::And(
                filters
                    .iter()
                    .map(|(name, filter)| crate::query::NodeFilter::AttributeFilter {
                        name: name.clone(),
                        filter: filter.clone(),
                    })
                    .collect(),
            )
        };
        self.filter_nodes_named(&filter, "attr_filtered")
    }
    /// Filter by a property value without expanding the parent edge scope.
    pub fn filter_nodes_by_attribute(
        &self,
        name: &AttrName,
        value: &AttrValue,
    ) -> GraphResult<Self> {
        self.filter_nodes_named(
            &crate::query::NodeFilter::AttributeEquals {
                name: name.clone(),
                value: value.clone(),
            },
            "value_filtered",
        )
    }

    /// Check if the subgraph is connected (has exactly one connected component)
    pub fn is_connected(&self) -> GraphResult<bool> {
        // Use the trait method instead of the old method
        use crate::traits::SubgraphOperations;
        let components = SubgraphOperations::connected_components(self)?;
        Ok(components.len() == 1 && !self.nodes.is_empty())
    }

    /// Convert this subgraph to NetworkX format
    ///
    /// This creates a NetworkXGraph representation that contains only
    /// the nodes and edges from this subgraph.
    ///
    /// # Returns
    /// * `NetworkXGraph` - A representation compatible with NetworkX
    ///
    /// # Examples
    /// ```ignore
    /// # use groggy::core::subgraph::Subgraph;
    /// # use groggy::errors::GraphResult;
    /// # fn example() -> GraphResult<()> {
    /// # let subgraph: Subgraph = todo!(); // Placeholder for actual subgraph
    /// let nx_graph = subgraph.to_networkx()?;
    /// # Ok(())
    /// # }
    /// ```ignore
    pub fn to_networkx(&self) -> GraphResult<crate::utils::convert::NetworkXGraph> {
        crate::utils::convert::subgraph_to_networkx(self)
    }

    /// Check if there is a path between two nodes within this subgraph
    ///
    /// This is more efficient than `shortest_path_subgraph` when you only need
    /// to know if a path exists, not the actual path.
    ///
    /// # Arguments
    /// * `node1_id` - The starting node ID
    /// * `node2_id` - The destination node ID
    ///
    /// # Returns
    /// * `Ok(true)` if a path exists between the nodes within this subgraph
    /// * `Ok(false)` if no path exists or either node is not in this subgraph
    /// * `Err(GraphError)` if there's an error during traversal
    ///
    /// # Examples
    /// ```ignore
    /// # use groggy::core::subgraph::Subgraph;
    /// # use groggy::errors::GraphResult;
    /// # fn example() -> GraphResult<()> {
    /// # let subgraph: Subgraph = todo!(); // Placeholder for actual subgraph
    /// // Check if there's a path between node 1 and node 5 in the subgraph
    /// let path_exists = subgraph.has_path(1, 5)?;
    /// # Ok(())
    /// # }
    /// ```ignore
    pub fn has_path(&self, node1_id: NodeId, node2_id: NodeId) -> GraphResult<bool> {
        let graph = self.graph.borrow();
        self.validate_query_scope(&graph)?;
        crate::query::scoped_traversal::has_path(
            &crate::query::scoped_traversal::Live::new(&graph, &self.nodes, &self.edges),
            node1_id,
            node2_id,
        )
    }

    /// Create visualization module for this subgraph
    /// Extracts subgraph data into thread-safe structures and returns VizModule
    pub fn viz(&self) -> VizModule {
        let data_source = crate::subgraphs::visualization::SubgraphDataSource::from_subgraph(self);

        use std::sync::Arc;
        let data_source: Arc<dyn crate::viz::streaming::data_source::DataSource> =
            Arc::new(data_source);
        VizModule::new(data_source)
    }
}

/// Column access operations for bulk attribute extraction
impl Subgraph {
    /// Get all values for a node attribute within this subgraph
    /// This is the key column access operation: subgraph[attr_name] -> Vec<AttrValue>
    pub fn get_node_attribute_column(&self, attr_name: &AttrName) -> GraphResult<Vec<AttrValue>> {
        let mut values = Vec::new();
        let graph_borrow = self.graph.borrow();

        for &node_id in self.nodes.iter() {
            if let Some(attr_value) = graph_borrow.get_node_attr(node_id, attr_name)? {
                values.push(attr_value);
            } else {
                // Handle missing attributes - could use None or skip
                // For now, we'll skip missing attributes
            }
        }

        Ok(values)
    }

    /// Get all values for an edge attribute within this subgraph
    pub fn get_edge_attribute_column(&self, attr_name: &AttrName) -> GraphResult<Vec<AttrValue>> {
        let mut values = Vec::new();
        let graph_borrow = self.graph.borrow();

        for &edge_id in self.edges.iter() {
            if let Some(attr_value) = graph_borrow.get_edge_attr(edge_id, attr_name)? {
                values.push(attr_value);
            } else {
                // Handle missing attributes - skip for now
            }
        }

        Ok(values)
    }

    /// Get node attribute values for specific nodes within this subgraph
    /// This enables: g.nodes[[1,2,3]][attr_name] -> [val1, val2, val3]
    pub fn get_node_attributes_for_nodes(
        &self,
        node_ids: &[NodeId],
        attr_name: &AttrName,
    ) -> GraphResult<Vec<AttrValue>> {
        let mut values = Vec::new();
        let graph_borrow = self.graph.borrow();

        for &node_id in node_ids {
            // Ensure the node is in this subgraph
            if !self.has_node(node_id) {
                return Err(GraphError::node_not_found(
                    node_id,
                    "subgraph attribute access",
                ));
            }

            if let Some(attr_value) = graph_borrow.get_node_attr(node_id, attr_name)? {
                values.push(attr_value);
            }
        }

        Ok(values)
    }
}

/// Batch operations for setting attributes on multiple nodes/edges in subgraph
impl Subgraph {
    /// Set an attribute on all nodes in this subgraph
    /// This enables: subgraph.set_node_attribute_bulk("department", "Engineering")
    /// OPTIMIZED: Uses bulk API instead of O(n) individual calls
    pub fn set_node_attribute_bulk(
        &self,
        attr_name: &AttrName,
        attr_value: AttrValue,
    ) -> GraphResult<()> {
        let graph_borrow = self.graph.borrow_mut();

        // Use the optimized bulk API instead of individual calls
        let mut attrs_values = std::collections::HashMap::new();
        let node_value_pairs: Vec<(NodeId, AttrValue)> = self
            .nodes
            .iter()
            .map(|&node_id| (node_id, attr_value.clone()))
            .collect();

        attrs_values.insert(attr_name.clone(), node_value_pairs);

        // Single bulk operation instead of O(n) individual calls - now using trait system
        drop(graph_borrow); // Release the borrow before calling trait method
        self.set_node_attrs(attrs_values)?;

        Ok(())
    }

    /// Set attributes on specific nodes within this subgraph
    /// OPTIMIZED: Uses bulk API instead of O(n) individual calls
    pub fn set_node_attributes_for_nodes(
        &self,
        node_ids: &[NodeId],
        attr_name: &AttrName,
        attr_value: AttrValue,
    ) -> GraphResult<()> {
        // Validate all nodes exist in this subgraph upfront
        for &node_id in node_ids {
            if !self.has_node(node_id) {
                return Err(GraphError::node_not_found(
                    node_id,
                    "subgraph batch operation",
                ));
            }
        }

        // Use optimized bulk API
        let graph_borrow = self.graph.borrow_mut();
        let mut attrs_values = std::collections::HashMap::new();
        let node_value_pairs: Vec<(NodeId, AttrValue)> = node_ids
            .iter()
            .map(|&node_id| (node_id, attr_value.clone()))
            .collect();

        attrs_values.insert(attr_name.clone(), node_value_pairs);
        drop(graph_borrow); // Release the borrow before calling trait method
        self.set_node_attrs(attrs_values)?;

        Ok(())
    }

    /// Set multiple attributes on all nodes in this subgraph
    /// This enables: subgraph.set_bulk({attr1: val1, attr2: val2})
    /// OPTIMIZED: Uses bulk API instead of O(n*m) individual calls
    pub fn set_node_attributes_bulk(
        &self,
        attributes: std::collections::HashMap<AttrName, AttrValue>,
    ) -> GraphResult<()> {
        let graph_borrow = self.graph.borrow_mut();

        // Transform to bulk API format: HashMap<AttrName, Vec<(NodeId, AttrValue)>>
        let mut attrs_values = std::collections::HashMap::new();

        for (attr_name, attr_value) in attributes {
            let node_value_pairs: Vec<(NodeId, AttrValue)> = self
                .nodes
                .iter()
                .map(|&node_id| (node_id, attr_value.clone()))
                .collect();
            attrs_values.insert(attr_name, node_value_pairs);
        }

        // Single bulk operation instead of O(n*m) individual calls - now using trait system
        drop(graph_borrow); // Release the borrow before calling trait method
        self.set_node_attrs(attrs_values)?;

        Ok(())
    }

    /// Set an attribute on all edges in this subgraph
    pub fn set_edge_attribute_bulk(
        &self,
        attr_name: &AttrName,
        attr_value: AttrValue,
    ) -> GraphResult<()> {
        let values = self
            .edges
            .iter()
            .map(|&edge| (edge, attr_value.clone()))
            .collect();
        self.set_edge_attrs(HashMap::from([(attr_name.clone(), values)]))
    }

    // === STRUCTURAL METRICS ===
    // Remember: All operations are on this subgraph (active nodes/edges)

    /// Calculate clustering coefficient for a node or average for all nodes
    /// Formula: 2 * triangles / (degree * (degree - 1)) for directed graphs
    pub fn clustering_coefficient(&self, node_id: Option<NodeId>) -> GraphResult<f64> {
        match node_id {
            Some(nid) => {
                if !self.has_node(nid) {
                    return Err(GraphError::InvalidInput(format!(
                        "Node {} not in this subgraph",
                        nid
                    )));
                }
                self.calculate_node_clustering_coefficient(nid)
            }
            None => {
                // Average clustering coefficient for all nodes in subgraph
                let mut total = 0.0;
                let mut count = 0;

                for &node_id in self.nodes.iter() {
                    let coefficient = self.calculate_node_clustering_coefficient(node_id)?;
                    total += coefficient;
                    count += 1;
                }

                if count == 0 {
                    Ok(0.0)
                } else {
                    Ok(total / count as f64)
                }
            }
        }
    }

    /// Helper method to calculate clustering coefficient for a single node
    fn calculate_node_clustering_coefficient(&self, node_id: NodeId) -> GraphResult<f64> {
        let graph = self.graph.borrow();

        // Get neighbors of this node that are also in the subgraph
        let neighbors = graph.neighbors_filtered(node_id, &self.nodes)?;
        let degree = neighbors.len();

        if degree < 2 {
            return Ok(0.0); // Clustering coefficient is 0 for nodes with degree < 2
        }

        // Count triangles: edges between neighbors
        let mut triangles = 0;
        for i in 0..neighbors.len() {
            for j in (i + 1)..neighbors.len() {
                if graph.has_edge_between_filtered(neighbors[i], neighbors[j], &self.edges)? {
                    triangles += 1;
                }
            }
        }

        // Calculate clustering coefficient
        let possible_edges = degree * (degree - 1) / 2; // For undirected graphs
        Ok(triangles as f64 / possible_edges as f64)
    }

    /// Calculate global clustering coefficient (transitivity)
    /// Formula: 3 * triangles / triads
    pub fn transitivity(&self) -> GraphResult<f64> {
        let mut total_triangles = 0;
        let mut total_triads = 0;

        let graph = self.graph.borrow();

        // Count each triangle only once by using node ordering
        let mut nodes_vec: Vec<_> = self.nodes.iter().copied().collect();
        nodes_vec.sort();

        for (i, &node_a) in nodes_vec.iter().enumerate() {
            let neighbors_a = graph.neighbors_filtered(node_a, &self.nodes)?;

            for (j, &node_b) in nodes_vec.iter().enumerate().skip(i + 1) {
                if !neighbors_a.contains(&node_b) {
                    continue; // No edge between a and b
                }

                for &node_c in nodes_vec.iter().skip(j + 1) {
                    if neighbors_a.contains(&node_c) {
                        let neighbors_b = graph.neighbors_filtered(node_b, &self.nodes)?;
                        if neighbors_b.contains(&node_c) {
                            // Triangle found: a-b-c
                            total_triangles += 1;
                        }
                    }
                }
            }
        }

        // Count triads (connected triples)
        for &node_id in self.nodes.iter() {
            let neighbors = graph.neighbors_filtered(node_id, &self.nodes)?;
            let degree = neighbors.len();

            if degree >= 2 {
                total_triads += degree * (degree - 1) / 2;
            }
        }

        if total_triads == 0 {
            Ok(0.0)
        } else {
            Ok(3.0 * total_triangles as f64 / total_triads as f64)
        }
    }

    /// Calculate subgraph density
    /// Formula: 2 * edges / (nodes * (nodes - 1)) for undirected graphs
    pub fn density(&self) -> f64 {
        let node_count = self.nodes.len();

        if node_count < 2 {
            return 0.0;
        }

        let edge_count = self.edges.len();
        let max_possible_edges = node_count * (node_count - 1) / 2; // Undirected graph

        edge_count as f64 / max_possible_edges as f64
    }

    // === SUBGRAPH SET OPERATIONS ===

    fn common_views(&self, other: &Subgraph) -> GraphResult<(super::GraphView, super::GraphView)> {
        if !Rc::ptr_eq(&self.graph, &other.graph) {
            return Err(GraphError::InvalidInput(
                "Subgraph algebra requires the same graph owner".into(),
            ));
        }
        let left = self.pin_view()?;
        let right = super::GraphView::from_membership(
            left.read_state().clone(),
            other.nodes.as_ref().clone(),
            other.edges.as_ref().clone(),
        )?;
        Ok((left, right))
    }

    fn connected_named_result(&self, view: super::GraphView, name: String) -> Self {
        let (nodes, edges) = view.into_membership();
        Self::new(self.graph.clone(), nodes, edges, name)
    }

    /// Exact membership union; reject unrelated graph owners.
    pub fn merge_with(&self, other: &Subgraph) -> GraphResult<Subgraph> {
        let (left, right) = self.common_views(other)?;
        Ok(self.connected_named_result(
            left.merge_with(&right)?,
            format!("merge_{}_{}", self.subgraph_type, other.subgraph_type),
        ))
    }

    /// Exact membership intersection; reject unrelated graph owners.
    pub fn intersect_with(&self, other: &Subgraph) -> GraphResult<Subgraph> {
        let (left, right) = self.common_views(other)?;
        Ok(self.connected_named_result(
            left.intersect_with(&right)?,
            format!(
                "intersection_{}_{}",
                self.subgraph_type, other.subgraph_type
            ),
        ))
    }

    /// Remove nodes/edges and prune edges incident to removed nodes.
    pub fn subtract_from(&self, other: &Subgraph) -> GraphResult<Subgraph> {
        let (left, right) = self.common_views(other)?;
        Ok(self.connected_named_result(
            left.subtract_from(&right)?,
            format!("difference_{}_{}", self.subgraph_type, other.subgraph_type),
        ))
    }

    /// Compare node membership only, within the same graph owner.
    pub fn calculate_similarity(
        &self,
        other: &Subgraph,
        metric: SimilarityMetric,
    ) -> GraphResult<f64> {
        if !Rc::ptr_eq(&self.graph, &other.graph) {
            return Err(GraphError::InvalidInput(
                "Subgraph similarity requires the same graph owner".into(),
            ));
        }
        Ok(super::view_algebra::membership_similarity(
            self.nodes(),
            other.nodes(),
            metric,
        ))
    }

    /// Nonempty intersections in caller order.
    pub fn find_overlaps(&self, others: Vec<&Subgraph>) -> GraphResult<Vec<Subgraph>> {
        others
            .into_iter()
            .map(|other| self.intersect_with(other))
            .filter_map(|result| match result {
                Ok(view) if view.node_count() == 0 => None,
                other => Some(other),
            })
            .collect()
    }

    /// Group present node values with the canonical immutable grouping kernel.
    pub fn group_by_nodes(&self, attr: &AttrName) -> GraphResult<Vec<Subgraph>> {
        Ok(self
            .pin_view()?
            .grouped_nodes(attr)?
            .into_iter()
            .map(|(value, view)| {
                self.connected_named_result(view, format!("{attr}_group_{value:?}"))
            })
            .collect())
    }

    /// Group present edge values, preserving exact selected instances.
    pub fn group_by_edges(&self, attr: &AttrName) -> GraphResult<Vec<Subgraph>> {
        Ok(self
            .pin_view()?
            .grouped_edges(attr)?
            .into_iter()
            .map(|(value, view)| {
                self.connected_named_result(view, format!("{attr}_edge_group_{value:?}"))
            })
            .collect())
    }
}

// Note: Index syntax (subgraph[attr_name]) is not implemented because we can't return
// references to computed Vec<AttrValue>. Use get_node_attribute_column() instead.

impl std::fmt::Display for Subgraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Subgraph(nodes={}, edges={}, type={})",
            self.node_count(),
            self.edge_count(),
            self.subgraph_type
        )
    }
}

/// GraphEntity trait implementation for Subgraph
/// This integrates Subgraph into the universal entity system
impl GraphEntity for Subgraph {
    fn entity_id(&self) -> EntityId {
        EntityId::Subgraph(self.subgraph_id)
    }

    fn entity_type(&self) -> &'static str {
        "subgraph"
    }

    fn graph_ref(&self) -> Rc<RefCell<Graph>> {
        self.graph.clone()
    }

    fn related_entities(&self) -> GraphResult<Vec<Box<dyn GraphEntity>>> {
        // Return node entities contained in this subgraph
        // TODO: Create EntityNode wrappers for the nodes in this subgraph
        Ok(Vec::new()) // Placeholder for now
    }

    fn summary(&self) -> String {
        format!(
            "Subgraph(type={}, id={}, nodes={}, edges={})",
            self.subgraph_type,
            self.subgraph_id,
            self.nodes.len(),
            self.edges.len()
        )
    }
}

/// SubgraphOperations trait implementation for Subgraph
/// This provides the standard subgraph interface using existing efficient storage
impl SubgraphOperations for Subgraph {
    fn node_set(&self) -> &HashSet<NodeId> {
        self.nodes.as_ref()
    }

    fn edge_set(&self) -> &HashSet<EdgeId> {
        self.edges.as_ref()
    }

    fn induced_subgraph(&self, nodes: &[NodeId]) -> GraphResult<Box<dyn SubgraphOperations>> {
        // Preserve the legacy request intersection, but never expand the parent's edge scope.
        let nodes: Vec<_> = nodes
            .iter()
            .copied()
            .filter(|node| self.nodes.contains(node))
            .collect();
        let view = self.pin_view()?.select_nodes(&nodes)?;
        Ok(Box::new(self.connected_named_result(
            view,
            format!("{}_induced", self.subgraph_type),
        )))
    }

    fn subgraph_from_edges(&self, edges: &[EdgeId]) -> GraphResult<Box<dyn SubgraphOperations>> {
        let edges: Vec<_> = edges
            .iter()
            .copied()
            .filter(|edge| self.edges.contains(edge))
            .collect();
        let view = self.pin_view()?.select_edges(&edges)?;
        Ok(Box::new(self.connected_named_result(
            view,
            format!("{}_from_edges", self.subgraph_type),
        )))
    }

    fn connected_components(&self) -> GraphResult<Vec<Box<dyn SubgraphOperations>>> {
        if let Some(cached) = self.component_cache_get(ComponentCacheMode::Undirected) {
            let mut component_subgraphs = Vec::with_capacity(cached.components.len());
            for (i, component) in cached.components.iter().enumerate() {
                // Convert Vec to HashSet on demand - cheaper than storing HashSet
                let nodes_set: HashSet<NodeId> = component.nodes.iter().copied().collect();
                let edges_set: HashSet<EdgeId> = component.edges.iter().copied().collect();
                let component_subgraph = Subgraph::new(
                    self.graph.clone(),
                    nodes_set,
                    edges_set,
                    format!("{}_component_{}", self.subgraph_type, i),
                );
                component_subgraphs
                    .push(Box::new(component_subgraph) as Box<dyn SubgraphOperations>);
            }
            return Ok(component_subgraphs);
        }

        {
            let graph = self.graph.borrow();
            self.validate_query_scope(&graph)?;
        }
        let nodes = self.ordered_nodes();
        if nodes.len() >= u32::MAX as usize {
            return Err(GraphError::InvalidInput(
                "Components require fewer than u32::MAX nodes".into(),
            ));
        }
        let csr = self.get_or_build_csr_internal(true)?;
        let mut labels = vec![0; nodes.len()];
        let count = crate::algorithms::community::components::bfs_components(
            &csr,
            &mut labels,
            &mut crate::algorithms::Context::new(),
            None,
        )
        .map_err(|error| GraphError::InvalidInput(error.to_string()))? as usize;
        let mut grouped_nodes = vec![Vec::new(); count];
        let mut grouped_edges = vec![Vec::new(); count];
        let positions: HashMap<_, _> = nodes
            .iter()
            .enumerate()
            .map(|(index, &node)| (node, index))
            .collect();
        for (index, &node) in nodes.iter().enumerate() {
            grouped_nodes[labels[index] as usize].push(node);
        }
        {
            let graph = self.graph.borrow();
            for &edge in self.ordered_edges().iter() {
                let (source, target) = graph.edge_endpoints(edge)?;
                let group = labels[positions[&source]] as usize;
                if group == labels[positions[&target]] as usize {
                    grouped_edges[group].push(edge);
                }
            }
        }
        let mut component_subgraphs = Vec::with_capacity(count);
        let mut cache_components = Vec::with_capacity(count);
        let assignments = nodes
            .iter()
            .zip(&labels)
            .map(|(&node, &label)| (node, i64::from(label)))
            .collect();
        for (index, (nodes, edges)) in grouped_nodes.into_iter().zip(grouped_edges).enumerate() {
            component_subgraphs.push(Box::new(Self::new(
                self.graph.clone(),
                nodes.iter().copied().collect(),
                edges.iter().copied().collect(),
                format!("{}_component_{index}", self.subgraph_type),
            )) as Box<dyn SubgraphOperations>);
            cache_components.push(ComponentCacheComponent {
                nodes: Arc::new(nodes),
                edges: Arc::new(edges),
            });
        }
        if !cache_components.is_empty() {
            self.component_cache_store(
                ComponentCacheMode::Undirected,
                Arc::new(assignments),
                Arc::new(cache_components),
            );
        }

        Ok(component_subgraphs)
    }

    fn bfs(
        &self,
        start: NodeId,
        max_depth: Option<usize>,
    ) -> GraphResult<Box<dyn SubgraphOperations>> {
        let graph = self.graph.borrow();
        self.validate_query_scope(&graph)?;
        let (nodes, edges) = crate::query::scoped_traversal::bfs(
            &crate::query::scoped_traversal::Live::new(&graph, &self.nodes, &self.edges),
            start,
            max_depth,
        )?;
        Ok(Box::new(Self::new(
            self.graph.clone(),
            nodes,
            edges,
            format!("{}_bfs_from_{start}", self.subgraph_type),
        )))
    }
    fn dfs(
        &self,
        start: NodeId,
        max_depth: Option<usize>,
    ) -> GraphResult<Box<dyn SubgraphOperations>> {
        let graph = self.graph.borrow();
        self.validate_query_scope(&graph)?;
        let (nodes, edges) = crate::query::scoped_traversal::dfs(
            &crate::query::scoped_traversal::Live::new(&graph, &self.nodes, &self.edges),
            start,
            max_depth,
        )?;
        Ok(Box::new(Self::new(
            self.graph.clone(),
            nodes,
            edges,
            format!("{}_dfs_from_{start}", self.subgraph_type),
        )))
    }
    fn shortest_path_subgraph(
        &self,
        source: NodeId,
        target: NodeId,
    ) -> GraphResult<Option<Box<dyn SubgraphOperations>>> {
        let graph = self.graph.borrow();
        self.validate_query_scope(&graph)?;
        Ok(crate::query::scoped_traversal::shortest_path(
            &crate::query::scoped_traversal::Live::new(&graph, &self.nodes, &self.edges),
            source,
            target,
        )?
        .map(|(nodes, edges)| {
            Box::new(Self::new(
                self.graph.clone(),
                nodes,
                edges,
                format!("{}_path_{source}_{target}", self.subgraph_type),
            )) as Box<dyn SubgraphOperations>
        }))
    }

    /// Create a VizModule for this subgraph to enable visualization
    /// Uses the thread-safe SubgraphDataSource wrapper to bridge subgraphs to the viz system
    fn viz(&self) -> crate::viz::VizModule {
        use crate::subgraphs::visualization::SubgraphDataSource;
        use std::sync::Arc;

        // Create thread-safe data source wrapper
        let data_source = SubgraphDataSource::from_subgraph_operations(self);
        let data_source: Arc<dyn crate::viz::streaming::data_source::DataSource> =
            Arc::new(data_source);

        // Create and return VizModule
        crate::viz::VizModule::new(data_source)
    }

    // ============================================================================
    // CSR-OPTIMIZED NEIGHBOR ACCESS (PHASE 1 OF STEP PRIMITIVES OPTIMIZATION)
    // ============================================================================
    // Override default implementations to use cached CSR for 10-50x speedup
    // Following STYLE_ALGO pattern used by PageRank, LPA, and other algorithms

    fn neighbors(&self, node_id: NodeId) -> GraphResult<Vec<NodeId>> {
        // Build CSR with reverse edges when the backing graph is undirected so that
        // single stored edges appear for both endpoints.
        let add_reverse = {
            let graph = self.graph.borrow();
            graph.is_undirected()
        };

        // Get or build CSR (cached across calls)
        let csr = self.get_or_build_csr_internal(add_reverse)?;

        // Get ordered nodes for indexing
        let ordered_nodes = self.ordered_nodes();

        // Find node index using binary search (ordered_nodes is sorted)
        let node_idx =
            ordered_nodes
                .binary_search(&node_id)
                .map_err(|_| GraphError::NodeNotFound {
                    node_id,
                    operation: "neighbors".to_string(),
                    suggestion: "Node not found in subgraph".to_string(),
                })?;

        // Get neighbor indices from CSR (O(1) slice access!)
        let neighbor_indices = csr.neighbors(node_idx);

        // Map CSR indices back to NodeIds
        Ok(neighbor_indices
            .iter()
            .map(|&idx| ordered_nodes[idx])
            .collect())
    }

    fn degree(&self, node_id: NodeId) -> GraphResult<usize> {
        // Match neighbors() behavior: include reverse edges for undirected graphs.
        let add_reverse = {
            let graph = self.graph.borrow();
            graph.is_undirected()
        };

        // Get or build CSR (cached across calls)
        let csr = self.get_or_build_csr_internal(add_reverse)?;

        // Get ordered nodes for indexing
        let ordered_nodes = self.ordered_nodes();

        // Find node index using binary search
        let node_idx =
            ordered_nodes
                .binary_search(&node_id)
                .map_err(|_| GraphError::NodeNotFound {
                    node_id,
                    operation: "degree".to_string(),
                    suggestion: "Node not found in subgraph".to_string(),
                })?;

        // Get degree from CSR (just neighbor count)
        Ok(csr.neighbors(node_idx).len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::graph::Graph;
    use crate::types::AttrValue;

    fn mutation_fixture(publishing: bool) -> (Rc<RefCell<Graph>>, Subgraph) {
        let mut graph = if publishing {
            Graph::new_publishing()
        } else {
            Graph::new_manual()
        };
        graph.insert_nodes(3).expect("nodes");
        graph.add_edges(&[(0, 1), (1, 2)]).expect("edges");
        if !publishing {
            graph
                .commit("fixture".into(), "test".into())
                .expect("initial state");
        }
        let shared = Rc::new(RefCell::new(graph));
        let selection = Subgraph::from_nodes(
            shared.clone(),
            [0, 1, 2].into_iter().collect(),
            "mutation".into(),
        )
        .expect("selection");
        (shared, selection)
    }

    #[test]
    fn node_bulk_and_column_reject_inactive_rows_without_partial_changes() {
        for publishing in [false, true] {
            let (shared, selection) = mutation_fixture(publishing);
            shared.borrow_mut().remove_node(2).expect("deactivate");
            if !publishing {
                shared
                    .borrow_mut()
                    .commit("remove".into(), "test".into())
                    .expect("publish removal");
            }
            let state = shared.borrow().current_state_id();
            for column in [false, true] {
                let values = vec![
                    (0, AttrValue::Int(10)),
                    (2, AttrValue::Int(20)),
                    (0, AttrValue::Int(30)),
                    (99, AttrValue::Int(40)),
                ];
                let error = if column {
                    selection.set_node_attr_column("score".into(), values)
                } else {
                    selection.set_node_attrs(HashMap::from([
                        ("score".into(), values),
                        ("valid".into(), vec![(1, AttrValue::Int(50))]),
                    ]))
                }
                .expect_err("whole operation must reject")
                .to_string();
                assert!(error.contains("score[1]: node 2"), "{error}");
                assert!(error.contains("score[2]: duplicate node"), "{error}");
                assert!(error.contains("score[3]: node 99"), "{error}");
                let graph = shared.borrow();
                assert_eq!(graph.current_state_id(), state);
                assert!(!graph.has_uncommitted_changes());
                assert_eq!(
                    graph.get_node_attr(0, &"score".into()).expect("score"),
                    None
                );
                assert_eq!(
                    graph.get_node_attr(1, &"valid".into()).expect("valid"),
                    None
                );
            }
            assert!(selection
                .set_node_attr_column("score".into(), vec![(2, AttrValue::Int(1))])
                .is_err());
        }
    }

    #[test]
    fn convenience_node_and_edge_writes_reject_stale_membership_as_a_whole() {
        for publishing in [false, true] {
            let (shared, selection) = mutation_fixture(publishing);
            shared.borrow_mut().remove_node(2).expect("deactivate");
            if !publishing {
                shared
                    .borrow_mut()
                    .commit("remove".into(), "test".into())
                    .expect("publish removal");
            }
            let state = shared.borrow().current_state_id();
            assert!(selection
                .set_node_attribute_bulk(&"x".into(), AttrValue::Int(1))
                .is_err());
            assert!(selection
                .set_node_attributes_for_nodes(&[0, 2], &"x".into(), AttrValue::Int(1))
                .is_err());
            assert!(selection
                .set_node_attributes_bulk(HashMap::from([
                    ("x".into(), AttrValue::Int(1)),
                    ("y".into(), AttrValue::Int(2))
                ]))
                .is_err());
            assert!(selection
                .set_edge_attribute_bulk(&"w".into(), AttrValue::Int(1))
                .is_err());
            let graph = shared.borrow();
            assert_eq!(graph.get_node_attr(0, &"x".into()).expect("x"), None);
            assert_eq!(graph.get_edge_attr(0, &"w".into()).expect("w"), None);
            assert_eq!(graph.current_state_id(), state);
            assert!(!graph.has_uncommitted_changes());
        }
    }

    #[test]
    fn valid_convenience_bulk_writes_accept_once_and_empty_writes_are_noops() {
        let (shared, selection) = mutation_fixture(true);
        let before = shared.borrow().current_state_id();
        selection
            .set_edge_attribute_bulk(&"w".into(), AttrValue::Int(7))
            .expect("bulk edges");
        assert_eq!(shared.borrow().current_state_id(), before + 1);
        selection
            .set_node_attributes_bulk(HashMap::from([
                ("x".into(), AttrValue::Int(1)),
                ("y".into(), AttrValue::Int(2)),
            ]))
            .expect("bulk columns");
        assert_eq!(shared.borrow().current_state_id(), before + 2);
        let graph = shared.borrow();
        assert!(!graph.has_uncommitted_changes());
        for edge in [0, 1] {
            assert_eq!(
                graph.get_edge_attr(edge, &"w".into()).expect("w"),
                Some(AttrValue::Int(7))
            );
        }
        drop(graph);
        selection
            .set_node_attrs(HashMap::new())
            .expect("empty attributes");
        selection
            .set_node_attr_column("x".into(), Vec::new())
            .expect("empty column");
        let empty = Subgraph::new(
            shared.clone(),
            HashSet::new(),
            HashSet::new(),
            "empty".into(),
        );
        empty
            .set_edge_attribute_bulk(&"w".into(), AttrValue::Int(8))
            .expect("empty edges");
        assert_eq!(shared.borrow().current_state_id(), before + 2);
    }

    #[test]
    fn test_subgraph_creation() {
        // Test basic subgraph creation and functionality
        let mut graph = Graph::new_manual();

        // Add test nodes
        let node1 = graph.add_node().unwrap();
        let node2 = graph.add_node().unwrap();

        // Set attributes
        graph
            .set_node_attr(
                node1,
                "name".to_string(),
                AttrValue::Text("Alice".to_string()),
            )
            .unwrap();
        graph
            .set_node_attr(
                node2,
                "name".to_string(),
                AttrValue::Text("Bob".to_string()),
            )
            .unwrap();

        // Create subgraph
        let graph_rc = Rc::new(RefCell::new(graph));
        let node_subset = HashSet::from([node1, node2]);
        let subgraph = Subgraph::from_nodes(graph_rc, node_subset, "test".to_string()).unwrap();

        // Test basic properties
        assert_eq!(subgraph.node_count(), 2);
        assert!(subgraph.has_node(node1));
        assert!(subgraph.has_node(node2));

        // Test column access
        let names = subgraph
            .get_node_attribute_column(&"name".to_string())
            .unwrap();
        assert_eq!(names.len(), 2);
    }

    #[test]
    fn test_subgraph_algorithms() {
        // Test the core SubgraphOperations algorithms
        let mut graph = Graph::new_manual();

        // Create a small connected graph for testing
        let node1 = graph.add_node().unwrap();
        let node2 = graph.add_node().unwrap();
        let node3 = graph.add_node().unwrap();
        let node4 = graph.add_node().unwrap(); // Isolated node to test components

        let edge1 = graph.add_edge(node1, node2).unwrap();
        let edge2 = graph.add_edge(node2, node3).unwrap();
        let _edge3 = graph.add_edge(node3, node1).unwrap(); // Triangle

        let graph_rc = Rc::new(RefCell::new(graph));
        let node_subset = HashSet::from([node1, node2, node3, node4]);
        let subgraph = Subgraph::from_nodes(graph_rc, node_subset, "test".to_string()).unwrap();

        // Test BFS subgraph
        let bfs_result = subgraph.bfs(node1, Some(2)).unwrap();
        assert!(bfs_result.contains_node(node1));
        // BFS test completed

        // Test DFS subgraph
        let dfs_result = subgraph.dfs(node1, Some(2)).unwrap();
        assert!(dfs_result.contains_node(node1));
        // DFS test completed

        // Test connected components
        use crate::traits::SubgraphOperations;
        let components = SubgraphOperations::connected_components(&subgraph).unwrap();
        // Connected components test completed
        assert!(!components.is_empty()); // At least the triangle should form one component

        // Test induced subgraph
        let induced = subgraph.induced_subgraph(&[node1, node2]).unwrap();
        assert_eq!(induced.node_count(), 2);
        assert!(induced.contains_node(node1));
        assert!(induced.contains_node(node2));

        // Test subgraph from edges
        let edge_subset = [edge1, edge2];
        let edge_subgraph = subgraph.subgraph_from_edges(&edge_subset).unwrap();
        assert!(edge_subgraph.contains_edge(edge1));
        assert!(edge_subgraph.contains_edge(edge2));

        // Test shortest path (if path exists)
        if let Some(path) = subgraph.shortest_path_subgraph(node1, node3).unwrap() {
            assert!(path.contains_node(node1));
            assert!(path.contains_node(node3));
            // Shortest path found
        }
    }

    #[test]
    fn test_structural_metrics() {
        // Create a triangle graph for testing clustering coefficient
        let mut graph = Graph::new_manual();
        let node1 = graph.add_node().unwrap();
        let node2 = graph.add_node().unwrap();
        let node3 = graph.add_node().unwrap();

        // Create edges to form a triangle
        graph.add_edge(node1, node2).unwrap();
        graph.add_edge(node2, node3).unwrap();
        graph.add_edge(node3, node1).unwrap();

        // Create subgraph with all nodes
        let graph_rc = Rc::new(RefCell::new(graph));
        let all_nodes = HashSet::from([node1, node2, node3]);
        let subgraph = Subgraph::from_nodes(graph_rc, all_nodes, "triangle".to_string()).unwrap();

        // Test density - triangle graph should have density = 1.0
        let density = subgraph.density();
        assert!((density - 1.0).abs() < f64::EPSILON);

        // Test connectivity
        assert!(subgraph.is_connected().unwrap());

        // Test clustering coefficient - each node should have coefficient = 1.0 (complete triangle)
        let avg_clustering = subgraph.clustering_coefficient(None).unwrap();
        assert!((avg_clustering - 1.0).abs() < f64::EPSILON);

        // Test transitivity
        let transitivity = subgraph.transitivity().unwrap();
        assert!((transitivity - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_subgraph_set_operations() {
        let mut graph = Graph::new_manual();
        let node1 = graph.add_node().unwrap();
        let node2 = graph.add_node().unwrap();
        let node3 = graph.add_node().unwrap();
        let node4 = graph.add_node().unwrap();

        graph.add_edge(node1, node2).unwrap();
        graph.add_edge(node2, node3).unwrap();
        graph.add_edge(node3, node4).unwrap();

        let graph_rc = Rc::new(RefCell::new(graph));

        // Create two overlapping subgraphs
        let subgraph1_nodes = HashSet::from([node1, node2, node3]);
        let subgraph1 =
            Subgraph::from_nodes(graph_rc.clone(), subgraph1_nodes, "sub1".to_string()).unwrap();

        let subgraph2_nodes = HashSet::from([node2, node3, node4]);
        let subgraph2 =
            Subgraph::from_nodes(graph_rc.clone(), subgraph2_nodes, "sub2".to_string()).unwrap();

        // Test merge (union)
        let merged = subgraph1.merge_with(&subgraph2).unwrap();
        assert_eq!(merged.node_count(), 4); // All 4 nodes
        assert!(merged.has_node(node1));
        assert!(merged.has_node(node4));

        // Test intersection
        let intersection = subgraph1.intersect_with(&subgraph2).unwrap();
        assert_eq!(intersection.node_count(), 2); // node2, node3
        assert!(intersection.has_node(node2));
        assert!(intersection.has_node(node3));
        assert!(!intersection.has_node(node1));
        assert!(!intersection.has_node(node4));

        // Test subtraction
        let difference = subgraph1.subtract_from(&subgraph2).unwrap();
        assert_eq!(difference.node_count(), 1); // only node1
        assert!(difference.has_node(node1));
        assert!(!difference.has_node(node2));

        // Test similarity metrics
        let jaccard = subgraph1
            .calculate_similarity(&subgraph2, SimilarityMetric::Jaccard)
            .unwrap();
        assert!((jaccard - 0.5).abs() < f64::EPSILON); // |intersection| / |union| = 2/4 = 0.5

        let dice = subgraph1
            .calculate_similarity(&subgraph2, SimilarityMetric::Dice)
            .unwrap();
        assert!((dice - 2.0 / 3.0).abs() < f64::EPSILON); // 2 * 2 / (3 + 3) = 4/6 = 2/3
    }
}
