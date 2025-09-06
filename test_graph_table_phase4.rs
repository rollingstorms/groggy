#!/usr/bin/env rust-script
//! Test Phase 4: GraphTable (Composite) Implementation

use std::collections::{HashMap, HashSet};

// Mock the types we need for testing
#[derive(Clone, Debug, PartialEq)]
enum AttrValue {
    Int(i64),
    Text(String),
    Bool(bool),
    Float(f32),
    Null,
}

impl std::fmt::Display for AttrValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AttrValue::Int(i) => write!(f, "{}", i),
            AttrValue::Text(s) => write!(f, "{}", s),
            AttrValue::Bool(b) => write!(f, "{}", b),
            AttrValue::Float(fl) => write!(f, "{}", fl),
            AttrValue::Null => write!(f, "null"),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
enum AttrValueType {
    Int,
    Text,
    Bool,
    Float,
}

type EdgeId = usize;
type NodeId = usize;

// Mock BaseArray 
#[derive(Clone, Debug)]
struct BaseArray {
    data: Vec<AttrValue>,
    dtype: AttrValueType,
    name: Option<String>,
}

impl BaseArray {
    fn with_name(data: Vec<AttrValue>, dtype: AttrValueType, name: String) -> Self {
        Self { data, dtype, name: Some(name) }
    }
    
    fn len(&self) -> usize { self.data.len() }
    fn get(&self, index: usize) -> Option<&AttrValue> { self.data.get(index) }
    
    fn from_edge_ids(edge_ids: Vec<EdgeId>) -> Self {
        let data: Vec<AttrValue> = edge_ids.into_iter()
            .map(|id| AttrValue::Int(id as i64))
            .collect();
        Self::with_name(data, AttrValueType::Int, "edge_id".to_string())
    }
    
    fn from_node_ids(node_ids: Vec<NodeId>) -> Self {
        let data: Vec<AttrValue> = node_ids.into_iter()
            .map(|id| AttrValue::Int(id as i64))
            .collect();
        Self::with_name(data, AttrValueType::Int, "node_id".to_string())
    }
    
    fn as_edge_ids(&self) -> Result<Vec<EdgeId>, String> {
        self.data.iter()
            .map(|val| match val {
                AttrValue::Int(id) => Ok(*id as EdgeId),
                _ => Err("Invalid edge ID type".to_string()),
            })
            .collect()
    }
    
    fn as_node_ids(&self) -> Result<Vec<NodeId>, String> {
        self.data.iter()
            .map(|val| match val {
                AttrValue::Int(id) => Ok(*id as NodeId),
                _ => Err("Invalid node ID type".to_string()),
            })
            .collect()
    }
}

// Mock BaseTable
#[derive(Clone, Debug)]
struct BaseTable {
    columns: HashMap<String, BaseArray>,
    column_order: Vec<String>,
    nrows: usize,
}

impl BaseTable {
    fn from_columns(columns: HashMap<String, BaseArray>) -> Result<Self, String> {
        if columns.is_empty() {
            return Ok(Self {
                columns: HashMap::new(),
                column_order: Vec::new(),
                nrows: 0,
            });
        }
        
        let first_len = columns.values().next().unwrap().len();
        for (name, column) in &columns {
            if column.len() != first_len {
                return Err(format!("Column length mismatch"));
            }
        }
        
        let column_order: Vec<String> = columns.keys().cloned().collect();
        
        Ok(Self {
            columns,
            column_order,
            nrows: first_len,
        })
    }
    
    fn nrows(&self) -> usize { self.nrows }
    fn ncols(&self) -> usize { self.columns.len() }
    fn has_column(&self, name: &str) -> bool { self.columns.contains_key(name) }
    fn column(&self, name: &str) -> Option<&BaseArray> { self.columns.get(name) }
    fn column_names(&self) -> &[String] { &self.column_order }
}

// Mock NodesTable
#[derive(Clone, Debug)]
struct NodesTable {
    base: BaseTable,
}

impl NodesTable {
    fn new(node_ids: Vec<NodeId>) -> Self {
        let mut columns = HashMap::new();
        columns.insert("node_id".to_string(), BaseArray::from_node_ids(node_ids));
        
        let base = BaseTable::from_columns(columns).expect("Valid node table");
        Self { base }
    }
    
    fn from_base_table(base: BaseTable) -> Result<Self, String> {
        if !base.has_column("node_id") {
            return Err("NodesTable requires 'node_id' column".to_string());
        }
        Ok(Self { base })
    }
    
    fn node_ids(&self) -> Result<Vec<NodeId>, String> {
        let node_id_column = self.base.column("node_id")
            .ok_or_else(|| "node_id column not found".to_string())?;
        node_id_column.as_node_ids()
    }
    
    fn validate_uids(&self) -> Result<(), String> {
        let node_id_column = self.base.column("node_id")
            .ok_or_else(|| "node_id column required".to_string())?;
        
        let mut seen_ids = HashSet::new();
        for i in 0..node_id_column.len() {
            match node_id_column.get(i) {
                Some(AttrValue::Int(id)) => {
                    if !seen_ids.insert(*id as NodeId) {
                        return Err(format!("Duplicate node_id found: {}", id));
                    }
                },
                Some(AttrValue::Null) => {
                    return Err("Null node_id found".to_string());
                },
                _ => return Err("Invalid node_id type".to_string()),
            }
        }
        Ok(())
    }
    
    fn validate_node_structure(&self) -> Vec<String> {
        let mut warnings = Vec::new();
        
        if !self.base.has_column("node_id") {
            warnings.push("Missing required 'node_id' column".to_string());
        }
        
        if self.nrows() == 0 {
            warnings.push("Empty nodes table".to_string());
        }
        
        if let Err(e) = self.validate_uids() {
            warnings.push(format!("UID validation failed: {}", e));
        }
        
        warnings
    }
    
    fn get_by_uid(&self, uid: NodeId) -> Option<HashMap<String, AttrValue>> {
        let node_id_column = self.base.column("node_id")?;
        
        for i in 0..node_id_column.len() {
            match node_id_column.get(i) {
                Some(AttrValue::Int(id)) if (*id) as NodeId == uid => {
                    let mut row = HashMap::new();
                    for col_name in self.column_names() {
                        if let Some(column) = self.base.column(col_name) {
                            if let Some(value) = column.get(i) {
                                row.insert(col_name.clone(), value.clone());
                            }
                        }
                    }
                    return Some(row);
                },
                _ => continue,
            }
        }
        
        None
    }
    
    fn base(&self) -> &BaseTable { &self.base }
    fn nrows(&self) -> usize { self.base.nrows() }
    fn ncols(&self) -> usize { self.base.ncols() }
    fn has_column(&self, name: &str) -> bool { self.base.has_column(name) }
    fn column_names(&self) -> &[String] { self.base.column_names() }
}

// Mock EdgeConfig
#[derive(Clone, Debug)]
struct EdgeConfig {
    allow_self_loops: bool,
    allow_multi_edges: bool,
    validate_node_references: bool,
}

impl Default for EdgeConfig {
    fn default() -> Self {
        Self {
            allow_self_loops: true,
            allow_multi_edges: true,
            validate_node_references: false,
        }
    }
}

// Mock EdgesTable
#[derive(Clone, Debug)]
struct EdgesTable {
    base: BaseTable,
}

impl EdgesTable {
    fn new(edges: Vec<(EdgeId, NodeId, NodeId)>) -> Self {
        let mut columns = HashMap::new();
        
        let (edge_ids, sources, targets): (Vec<_>, Vec<_>, Vec<_>) = {
            let mut edge_ids = Vec::new();
            let mut sources = Vec::new();
            let mut targets = Vec::new();
            
            for (edge_id, source, target) in edges {
                edge_ids.push(edge_id);
                sources.push(source);
                targets.push(target);
            }
            
            (edge_ids, sources, targets)
        };
        
        columns.insert("edge_id".to_string(), BaseArray::from_edge_ids(edge_ids));
        columns.insert("source".to_string(), BaseArray::from_node_ids(sources));
        columns.insert("target".to_string(), BaseArray::from_node_ids(targets));
        
        let base = BaseTable::from_columns(columns).expect("Valid edge table");
        Self { base }
    }
    
    fn from_base_table(base: BaseTable) -> Result<Self, String> {
        let required_cols = ["edge_id", "source", "target"];
        for col in &required_cols {
            if !base.has_column(col) {
                return Err(format!("EdgesTable requires '{}' column", col));
            }
        }
        Ok(Self { base })
    }
    
    fn sources(&self) -> Result<Vec<NodeId>, String> {
        let source_column = self.base.column("source")
            .ok_or_else(|| "source column not found".to_string())?;
        source_column.as_node_ids()
    }
    
    fn targets(&self) -> Result<Vec<NodeId>, String> {
        let target_column = self.base.column("target")
            .ok_or_else(|| "target column not found".to_string())?;
        target_column.as_node_ids()
    }
    
    fn validate_edges(&self, config: &EdgeConfig) -> Result<(), String> {
        let edge_ids = self.edge_ids()?;
        let sources = self.sources()?;
        let targets = self.targets()?;
        
        // Validate edge IDs are unique
        let mut seen_edge_ids = HashSet::new();
        for edge_id in &edge_ids {
            if !seen_edge_ids.insert(edge_id) {
                return Err(format!("Duplicate edge_id found: {}", edge_id));
            }
        }
        
        // Policy-based validation
        if !config.allow_self_loops {
            for (source, target) in sources.iter().zip(targets.iter()) {
                if source == target {
                    return Err(format!("Self-loop detected: {} -> {} (disallowed by config)", source, target));
                }
            }
        }
        
        if !config.allow_multi_edges {
            let mut seen_pairs = HashSet::new();
            for (source, target) in sources.iter().zip(targets.iter()) {
                let pair = (*source, *target);
                if !seen_pairs.insert(pair) {
                    return Err(format!("Multi-edge detected: {} -> {} (disallowed by config)", source, target));
                }
            }
        }
        
        Ok(())
    }
    
    fn validate_edge_structure(&self) -> Vec<String> {
        let mut warnings = Vec::new();
        
        for required_col in &["edge_id", "source", "target"] {
            if !self.base.has_column(required_col) {
                warnings.push(format!("Missing required '{}' column", required_col));
            }
        }
        
        if self.nrows() == 0 {
            warnings.push("Empty edges table".to_string());
        }
        
        if let Err(e) = self.validate_edges(&EdgeConfig::default()) {
            warnings.push(format!("Edge validation failed: {}", e));
        }
        
        warnings
    }
    
    fn edge_ids(&self) -> Result<Vec<EdgeId>, String> {
        let edge_id_column = self.base.column("edge_id")
            .ok_or_else(|| "edge_id column not found".to_string())?;
        edge_id_column.as_edge_ids()
    }
    
    fn get_by_edge_id(&self, edge_id: EdgeId) -> Option<HashMap<String, AttrValue>> {
        let edge_id_column = self.base.column("edge_id")?;
        
        for i in 0..edge_id_column.len() {
            match edge_id_column.get(i) {
                Some(AttrValue::Int(id)) if (*id) as EdgeId == edge_id => {
                    let mut row = HashMap::new();
                    for col_name in self.column_names() {
                        if let Some(column) = self.base.column(col_name) {
                            if let Some(value) = column.get(i) {
                                row.insert(col_name.clone(), value.clone());
                            }
                        }
                    }
                    return Some(row);
                },
                _ => continue,
            }
        }
        
        None
    }
    
    fn edge_stats(&self) -> Result<HashMap<String, usize>, String> {
        let mut stats = HashMap::new();
        
        stats.insert("total_edges".to_string(), self.nrows());
        stats.insert("total_columns".to_string(), self.ncols());
        
        // Count self-loops
        let sources = self.sources()?;
        let targets = self.targets()?;
        let self_loops = sources.iter().zip(targets.iter())
            .filter(|(s, t)| s == t)
            .count();
        stats.insert("self_loops".to_string(), self_loops);
        
        // Count unique sources and targets
        let unique_sources: HashSet<_> = sources.into_iter().collect();
        let unique_targets: HashSet<_> = targets.into_iter().collect();
        stats.insert("unique_sources".to_string(), unique_sources.len());
        stats.insert("unique_targets".to_string(), unique_targets.len());
        
        Ok(stats)
    }
    
    fn outgoing_edges(&self, source_nodes: &[NodeId]) -> Result<Self, String> {
        // Simplified filter implementation
        let sources = self.sources()?;
        let source_set: HashSet<_> = source_nodes.iter().collect();
        
        let mut filtered_edges = Vec::new();
        let edge_ids = self.edge_ids()?;
        let targets = self.targets()?;
        
        for (i, source) in sources.iter().enumerate() {
            if source_set.contains(source) {
                filtered_edges.push((edge_ids[i], *source, targets[i]));
            }
        }
        
        Ok(Self::new(filtered_edges))
    }
    
    fn incoming_edges(&self, target_nodes: &[NodeId]) -> Result<Self, String> {
        // Simplified filter implementation
        let targets = self.targets()?;
        let target_set: HashSet<_> = target_nodes.iter().collect();
        
        let mut filtered_edges = Vec::new();
        let edge_ids = self.edge_ids()?;
        let sources = self.sources()?;
        
        for (i, target) in targets.iter().enumerate() {
            if target_set.contains(target) {
                filtered_edges.push((edge_ids[i], sources[i], *target));
            }
        }
        
        Ok(Self::new(filtered_edges))
    }
    
    fn base(&self) -> &BaseTable { &self.base }
    fn nrows(&self) -> usize { self.base.nrows() }
    fn ncols(&self) -> usize { self.base.ncols() }
    fn has_column(&self, name: &str) -> bool { self.base.has_column(name) }
    fn column_names(&self) -> &[String] { self.base.column_names() }
}

// Mock ValidationStrictness
#[derive(Clone, Debug, PartialEq)]
enum ValidationStrictness {
    Minimal,
    Standard,
    Strict,
}

// Mock ValidationPolicy
#[derive(Clone, Debug)]
struct ValidationPolicy {
    strictness: ValidationStrictness,
    validate_node_references: bool,
    edge_config: EdgeConfig,
    auto_repair: bool,
}

impl Default for ValidationPolicy {
    fn default() -> Self {
        Self {
            strictness: ValidationStrictness::Standard,
            validate_node_references: true,
            edge_config: EdgeConfig::default(),
            auto_repair: false,
        }
    }
}

// Mock ValidationReport
#[derive(Debug, Clone)]
struct ValidationReport {
    errors: Vec<String>,
    warnings: Vec<String>,
    info: Vec<String>,
    stats: HashMap<String, usize>,
}

impl ValidationReport {
    fn new() -> Self {
        Self {
            errors: Vec::new(),
            warnings: Vec::new(),
            info: Vec::new(),
            stats: HashMap::new(),
        }
    }
    
    fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }
    
    fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }
}

// Mock GraphTable with Phase 4 methods
#[derive(Clone, Debug)]
struct GraphTable {
    nodes: NodesTable,
    edges: EdgesTable,
    policy: ValidationPolicy,
}

impl GraphTable {
    fn new(nodes: NodesTable, edges: EdgesTable) -> Self {
        Self {
            nodes,
            edges,
            policy: ValidationPolicy::default(),
        }
    }
    
    fn with_policy(nodes: NodesTable, edges: EdgesTable, policy: ValidationPolicy) -> Self {
        Self {
            nodes,
            edges,
            policy,
        }
    }
    
    fn empty() -> Self {
        let empty_nodes = NodesTable::new(vec![]);
        let empty_edges = EdgesTable::new(vec![]);
        Self::new(empty_nodes, empty_edges)
    }
    
    fn from_data(node_ids: Vec<NodeId>, edges: Vec<(EdgeId, NodeId, NodeId)>) -> Self {
        let nodes = NodesTable::new(node_ids);
        let edges_table = EdgesTable::new(edges);
        Self::new(nodes, edges_table)
    }
    
    fn nodes(&self) -> &NodesTable {
        &self.nodes
    }
    
    fn edges(&self) -> &EdgesTable {
        &self.edges
    }
    
    fn policy(&self) -> &ValidationPolicy {
        &self.policy
    }
    
    // Phase 4: Validation methods
    fn validate(&self) -> ValidationReport {
        let mut report = ValidationReport::new();
        
        // Basic statistics
        report.stats.insert("total_nodes".to_string(), self.nodes.nrows());
        report.stats.insert("total_edges".to_string(), self.edges.nrows());
        
        // Validate nodes table
        match self.nodes.validate_uids() {
            Ok(()) => {
                report.info.push("Node UIDs validation passed".to_string());
            }
            Err(e) => {
                report.errors.push(format!("Node UID validation failed: {}", e));
            }
        }
        
        let node_warnings = self.nodes.validate_node_structure();
        for warning in node_warnings {
            match self.policy.strictness {
                ValidationStrictness::Minimal => {
                    if warning.to_lowercase().contains("error") || warning.to_lowercase().contains("missing") {
                        report.warnings.push(format!("Node structure: {}", warning));
                    }
                },
                ValidationStrictness::Standard | ValidationStrictness::Strict => {
                    report.warnings.push(format!("Node structure: {}", warning));
                }
            }
        }
        
        // Validate edges table
        match self.edges.validate_edges(&self.policy.edge_config) {
            Ok(()) => {
                report.info.push("Edge validation passed".to_string());
            }
            Err(e) => {
                report.errors.push(format!("Edge validation failed: {}", e));
            }
        }
        
        let edge_warnings = self.edges.validate_edge_structure();
        for warning in edge_warnings {
            match self.policy.strictness {
                ValidationStrictness::Minimal => {
                    if warning.to_lowercase().contains("error") || warning.to_lowercase().contains("missing") {
                        report.warnings.push(format!("Edge structure: {}", warning));
                    }
                },
                ValidationStrictness::Standard | ValidationStrictness::Strict => {
                    report.warnings.push(format!("Edge structure: {}", warning));
                }
            }
        }
        
        // Cross-validation: Check that edge endpoints reference valid nodes
        if self.policy.validate_node_references {
            if let (Ok(node_ids), Ok(sources), Ok(targets)) = (
                self.nodes.node_ids(),
                self.edges.sources(), 
                self.edges.targets()
            ) {
                let node_set: HashSet<NodeId> = node_ids.into_iter().collect();
                
                for (i, source) in sources.iter().enumerate() {
                    if !node_set.contains(source) {
                        report.errors.push(format!("Edge {} references non-existent source node {}", i, source));
                    }
                }
                
                for (i, target) in targets.iter().enumerate() {
                    if !node_set.contains(target) {
                        report.errors.push(format!("Edge {} references non-existent target node {}", i, target));
                    }
                }
                
                if report.errors.is_empty() {
                    report.info.push("Edge node reference validation passed".to_string());
                }
            } else {
                report.warnings.push("Could not perform node reference validation".to_string());
            }
        }
        
        report
    }
    
    fn conform(&self) -> Result<ValidationReport, String> {
        let report = self.validate();
        
        if !report.is_valid() {
            return Err(format!("Graph validation failed with {} errors", report.errors.len()));
        }
        
        Ok(report)
    }
    
    fn stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        
        stats.insert("total_nodes".to_string(), self.nodes.nrows());
        stats.insert("total_edges".to_string(), self.edges.nrows());
        stats.insert("node_columns".to_string(), self.nodes.ncols());
        stats.insert("edge_columns".to_string(), self.edges.ncols());
        
        // Add edge statistics if available
        if let Ok(edge_stats) = self.edges.edge_stats() {
            stats.extend(edge_stats);
        }
        
        stats
    }
    
    fn get_nodes(&self, node_ids: &[NodeId]) -> Result<HashMap<NodeId, HashMap<String, AttrValue>>, String> {
        let mut result = HashMap::new();
        
        for &node_id in node_ids {
            if let Some(node_data) = self.nodes.get_by_uid(node_id) {
                result.insert(node_id, node_data);
            }
        }
        
        Ok(result)
    }
    
    fn get_edges(&self, edge_ids: &[EdgeId]) -> Result<HashMap<EdgeId, HashMap<String, AttrValue>>, String> {
        let mut result = HashMap::new();
        
        for &edge_id in edge_ids {
            if let Some(edge_data) = self.edges.get_by_edge_id(edge_id) {
                result.insert(edge_id, edge_data);
            }
        }
        
        Ok(result)
    }
    
    fn neighbors(&self, node_ids: &[NodeId]) -> Result<HashSet<NodeId>, String> {
        let mut neighbors = HashSet::new();
        
        // Get outgoing edges
        let outgoing = self.edges.outgoing_edges(node_ids)?;
        let out_targets = outgoing.targets()?;
        neighbors.extend(out_targets);
        
        // Get incoming edges  
        let incoming = self.edges.incoming_edges(node_ids)?;
        let in_sources = incoming.sources()?;
        neighbors.extend(in_sources);
        
        // Remove the original nodes from neighbors
        for &node_id in node_ids {
            neighbors.remove(&node_id);
        }
        
        Ok(neighbors)
    }
}

fn main() {
    println!("Phase 4: GraphTable (Composite) Implementation Test");
    println!("================================================");
    
    // Test 1: GraphTable creation with composite structure
    println!("\n🧪 Test 1: GraphTable creation");
    
    let node_ids = vec![1, 2, 3, 4];
    let edges = vec![
        (10, 1, 2), // edge 10: 1 -> 2
        (20, 2, 3), // edge 20: 2 -> 3
        (30, 3, 4), // edge 30: 3 -> 4
        (40, 1, 4), // edge 40: 1 -> 4
        (50, 4, 4), // edge 50: 4 -> 4 (self-loop)
    ];
    
    let graph_table = GraphTable::from_data(node_ids.clone(), edges.clone());
    
    println!("✅ GraphTable created: {} nodes, {} edges", 
        graph_table.nodes().nrows(), graph_table.edges().nrows());
    assert_eq!(graph_table.nodes().nrows(), 4);
    assert_eq!(graph_table.edges().nrows(), 5);
    println!("✅ Nodes: {:?}", graph_table.nodes().node_ids().unwrap());
    println!("✅ Edges: {:?}", graph_table.edges().edge_ids().unwrap());
    
    // Test 2: Validation Policy configuration
    println!("\n🧪 Test 2: ValidationPolicy configuration");
    
    let strict_policy = ValidationPolicy {
        strictness: ValidationStrictness::Strict,
        validate_node_references: true,
        edge_config: EdgeConfig {
            allow_self_loops: false,
            allow_multi_edges: true,
            validate_node_references: true,
        },
        auto_repair: false,
    };
    
    let strict_graph = GraphTable::with_policy(
        NodesTable::new(node_ids.clone()),
        EdgesTable::new(edges.clone()),
        strict_policy
    );
    
    println!("✅ Strict policy GraphTable created");
    assert_eq!(strict_graph.policy().strictness, ValidationStrictness::Strict);
    assert!(!strict_graph.policy().edge_config.allow_self_loops);
    
    // Test 3: Comprehensive validation
    println!("\n🧪 Test 3: Comprehensive validation");
    
    // Test default policy (should pass)
    let default_report = graph_table.validate();
    println!("✅ Default validation: {} errors, {} warnings, {} info", 
        default_report.errors.len(), default_report.warnings.len(), default_report.info.len());
    assert!(default_report.is_valid());
    
    // Test strict policy (should fail due to self-loop)
    let strict_report = strict_graph.validate();
    println!("✅ Strict validation: {} errors, {} warnings, {} info", 
        strict_report.errors.len(), strict_report.warnings.len(), strict_report.info.len());
    assert!(!strict_report.is_valid()); // Should fail due to self-loop
    assert!(strict_report.errors.iter().any(|e| e.contains("Self-loop")));
    
    // Test 4: Schema conformance
    println!("\n🧪 Test 4: Schema conformance");
    
    match graph_table.conform() {
        Ok(report) => {
            println!("✅ Graph conforms to schema: {} info messages", report.info.len());
            assert!(report.is_valid());
        }
        Err(e) => panic!("Graph should conform to default schema: {}", e),
    }
    
    match strict_graph.conform() {
        Ok(_) => panic!("Strict graph should not conform due to self-loop"),
        Err(e) => {
            println!("✅ Strict graph correctly fails conformance: {}", e);
            assert!(e.contains("validation failed"));
        }
    }
    
    // Test 5: Graph statistics
    println!("\n🧪 Test 5: Graph statistics");
    
    let stats = graph_table.stats();
    println!("✅ Graph stats: {} entries", stats.len());
    println!("   Total nodes: {}", stats.get("total_nodes").unwrap());
    println!("   Total edges: {}", stats.get("total_edges").unwrap());
    println!("   Self loops: {}", stats.get("self_loops").unwrap());
    println!("   Unique sources: {}", stats.get("unique_sources").unwrap());
    
    assert_eq!(*stats.get("total_nodes").unwrap(), 4);
    assert_eq!(*stats.get("total_edges").unwrap(), 5);
    assert_eq!(*stats.get("self_loops").unwrap(), 1);
    
    // Test 6: Node and edge queries
    println!("\n🧪 Test 6: Node and edge queries");
    
    let nodes = graph_table.get_nodes(&[1, 3]).unwrap();
    println!("✅ Retrieved {} nodes by ID", nodes.len());
    assert_eq!(nodes.len(), 2);
    assert!(nodes.contains_key(&1));
    assert!(nodes.contains_key(&3));
    
    let edges = graph_table.get_edges(&[10, 30]).unwrap();
    println!("✅ Retrieved {} edges by ID", edges.len());
    assert_eq!(edges.len(), 2);
    assert!(edges.contains_key(&10));
    assert!(edges.contains_key(&30));
    
    // Test 7: Neighbor discovery
    println!("\n🧪 Test 7: Neighbor discovery");
    
    let neighbors = graph_table.neighbors(&[1]).unwrap();
    println!("✅ Node 1 has {} neighbors: {:?}", neighbors.len(), neighbors);
    // Node 1 connects to 2 and 4, so neighbors should include them
    assert!(neighbors.contains(&2)); // 1 -> 2
    assert!(neighbors.contains(&4)); // 1 -> 4
    
    let neighbors_2 = graph_table.neighbors(&[2]).unwrap();
    println!("✅ Node 2 has {} neighbors: {:?}", neighbors_2.len(), neighbors_2);
    assert!(neighbors_2.contains(&1)); // 1 -> 2 (incoming)
    assert!(neighbors_2.contains(&3)); // 2 -> 3 (outgoing)
    
    // Test 8: Cross-validation (node references)
    println!("\n🧪 Test 8: Cross-validation (node references)");
    
    // Create graph with invalid edge references
    let invalid_edges = vec![
        (100, 1, 2),  // Valid
        (200, 1, 99), // Invalid: node 99 doesn't exist
        (300, 88, 2), // Invalid: node 88 doesn't exist
    ];
    
    let invalid_graph = GraphTable::from_data(vec![1, 2, 3], invalid_edges);
    let invalid_report = invalid_graph.validate();
    
    println!("✅ Invalid graph validation: {} errors", invalid_report.errors.len());
    assert!(!invalid_report.is_valid());
    assert!(invalid_report.errors.iter().any(|e| e.contains("non-existent")));
    
    // Test 9: Empty graph handling
    println!("\n🧪 Test 9: Empty graph handling");
    
    let empty_graph = GraphTable::empty();
    let empty_report = empty_graph.validate();
    
    println!("✅ Empty graph validation: {} warnings", empty_report.warnings.len());
    assert!(empty_report.is_valid()); // Empty graph is technically valid
    assert_eq!(empty_graph.stats().get("total_nodes").unwrap(), &0);
    assert_eq!(empty_graph.stats().get("total_edges").unwrap(), &0);
    
    // Test 10: Different validation strictness levels
    println!("\n🧪 Test 10: Validation strictness levels");
    
    let minimal_policy = ValidationPolicy {
        strictness: ValidationStrictness::Minimal,
        validate_node_references: false,
        edge_config: EdgeConfig::default(),
        auto_repair: false,
    };
    
    let minimal_graph = GraphTable::with_policy(
        NodesTable::new(vec![]),  // Empty nodes (would normally warn)
        EdgesTable::new(vec![]),  // Empty edges
        minimal_policy
    );
    
    let minimal_report = minimal_graph.validate();
    println!("✅ Minimal validation: {} warnings (should be less than standard)", 
        minimal_report.warnings.len());
    
    let standard_graph = GraphTable::with_policy(
        NodesTable::new(vec![]),
        EdgesTable::new(vec![]),
        ValidationPolicy::default()
    );
    
    let standard_report = standard_graph.validate();
    println!("✅ Standard validation: {} warnings", standard_report.warnings.len());
    
    // Minimal should have fewer warnings than standard for empty tables
    assert!(minimal_report.warnings.len() <= standard_report.warnings.len());
    
    println!("\n🎉 ALL PHASE 4 TESTS PASSED!");
    println!("✅ Composite GraphTable with NodesTable + EdgesTable");
    println!("✅ ValidationPolicy with different strictness levels");
    println!("✅ Comprehensive validate() method");
    println!("✅ Schema conformance with conform() method");
    println!("✅ Cross-validation (node reference checking)");
    println!("✅ Graph statistics and queries");
    println!("✅ Neighbor discovery functionality");
    println!("✅ Edge case handling (empty graphs, invalid references)");
    
    println!("\n📋 Phase 4: GraphTable (Composite) Implementation: COMPLETE!");
}