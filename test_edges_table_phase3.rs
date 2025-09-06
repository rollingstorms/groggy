#!/usr/bin/env rust-script
//! Test Phase 3: EdgesTable Implementation

use std::collections::HashMap;

// Mock the types we need for testing
#[derive(Clone, Debug, PartialEq)]
enum AttrValue {
    Int(i64),
    Text(String),
    Bool(bool),
    Float(f32),
    Null,
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
#[derive(Clone)]
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
    fn dtype(&self) -> &AttrValueType { &self.dtype }
    fn name(&self) -> Option<&String> { self.name.as_ref() }
    
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
#[derive(Clone)]
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
    
    // Phase 3: Conversion method
    fn to_edges(self) -> Result<EdgesTable, String> {
        let required_cols = ["edge_id", "source", "target"];
        for col in &required_cols {
            if !self.has_column(col) {
                return Err(format!("EdgesTable requires '{}' column", col));
            }
        }
        
        EdgesTable::from_base_table(self)
    }
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

// Mock EdgesTable with Phase 3 methods
#[derive(Clone)]
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
    
    fn edge_ids(&self) -> Result<Vec<EdgeId>, String> {
        let edge_id_column = self.base.column("edge_id")
            .ok_or_else(|| "edge_id column not found".to_string())?;
        edge_id_column.as_edge_ids()
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
    
    // Phase 3: Validation methods
    fn validate_edges(&self, config: &EdgeConfig) -> Result<(), String> {
        let edge_ids = self.edge_ids()?;
        let sources = self.sources()?;
        let targets = self.targets()?;
        
        // Validate edge IDs are unique
        let mut seen_edge_ids = std::collections::HashSet::new();
        for edge_id in &edge_ids {
            if !seen_edge_ids.insert(edge_id) {
                return Err(format!("Duplicate edge_id found: {}", edge_id));
            }
        }
        
        // Check for null values in required columns
        let edge_id_column = self.base.column("edge_id").unwrap();
        let source_column = self.base.column("source").unwrap();
        let target_column = self.base.column("target").unwrap();
        
        for i in 0..edge_id_column.len() {
            if let Some(AttrValue::Null) = edge_id_column.get(i) {
                return Err("Null edge_id found".to_string());
            }
            if let Some(AttrValue::Null) = source_column.get(i) {
                return Err("Null source found".to_string());
            }
            if let Some(AttrValue::Null) = target_column.get(i) {
                return Err("Null target found".to_string());
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
            let mut seen_pairs = std::collections::HashSet::new();
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
        
        // Check required columns exist
        for required_col in &["edge_id", "source", "target"] {
            if !self.base.has_column(required_col) {
                warnings.push(format!("Missing required '{}' column", required_col));
            }
        }
        
        // Check for reasonable edge count
        if self.nrows() == 0 {
            warnings.push("Empty edges table".to_string());
        } else if self.nrows() > 10_000_000 {
            warnings.push(format!("Large edges table ({} rows) - consider partitioning", self.nrows()));
        }
        
        // Check for suspicious column patterns
        for col_name in self.column_names() {
            if col_name.contains(' ') {
                warnings.push(format!("Column '{}' contains spaces - may cause query issues", col_name));
            }
        }
        
        // Validate edges with default config
        if let Err(e) = self.validate_edges(&EdgeConfig::default()) {
            warnings.push(format!("Edge validation failed: {}", e));
        }
        
        warnings
    }
    
    fn get_by_edge_id(&self, edge_id: EdgeId) -> Option<HashMap<String, AttrValue>> {
        let edge_id_column = self.base.column("edge_id")?;
        
        // Find the row with matching edge_id
        for i in 0..edge_id_column.len() {
            match edge_id_column.get(i) {
                Some(AttrValue::Int(id)) if *id as EdgeId == edge_id => {
                    // Found the row, collect all column values
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
        let unique_sources: std::collections::HashSet<_> = sources.into_iter().collect();
        let unique_targets: std::collections::HashSet<_> = targets.into_iter().collect();
        stats.insert("unique_sources".to_string(), unique_sources.len());
        stats.insert("unique_targets".to_string(), unique_targets.len());
        
        Ok(stats)
    }
    
    // Phase 3: Base table access  
    fn base(&self) -> &BaseTable { &self.base }
    fn into_base(self) -> BaseTable { self.base }
    
    // Delegate methods
    fn nrows(&self) -> usize { self.base.nrows() }
    fn ncols(&self) -> usize { self.base.ncols() }
    fn has_column(&self, name: &str) -> bool { self.base.has_column(name) }
    fn column_names(&self) -> &[String] { self.base.column_names() }
}

fn main() {
    println!("Phase 3: EdgesTable Implementation Test");
    println!("======================================");
    
    // Test 1: EdgesTable creation and basic functionality
    println!("\n🧪 Test 1: EdgesTable creation");
    
    let edges = vec![
        (1, 10, 20), // edge_id: 1, source: 10, target: 20
        (2, 20, 30), // edge_id: 2, source: 20, target: 30
        (3, 10, 30), // edge_id: 3, source: 10, target: 30
        (4, 30, 30), // edge_id: 4, source: 30, target: 30 (self-loop)
    ];
    let edges_table = EdgesTable::new(edges.clone());
    
    println!("✅ EdgesTable created: {} rows, {} columns", edges_table.nrows(), edges_table.ncols());
    assert_eq!(edges_table.nrows(), 4);
    assert_eq!(edges_table.ncols(), 3);
    assert!(edges_table.has_column("edge_id"));
    assert!(edges_table.has_column("source"));
    assert!(edges_table.has_column("target"));
    
    let retrieved_edge_ids = edges_table.edge_ids().expect("Should get edge IDs");
    assert_eq!(retrieved_edge_ids, vec![1, 2, 3, 4]);
    println!("✅ Edge IDs retrieved correctly: {:?}", retrieved_edge_ids);
    
    // Test 2: EdgeConfig and validation
    println!("\n🧪 Test 2: EdgeConfig and validation");
    
    // Test default config (allows everything)
    let default_config = EdgeConfig::default();
    edges_table.validate_edges(&default_config).expect("Should validate with default config");
    println!("✅ Default config validation passed");
    
    // Test strict config (no self-loops)
    let strict_config = EdgeConfig {
        allow_self_loops: false,
        allow_multi_edges: true,
        validate_node_references: false,
    };
    
    match edges_table.validate_edges(&strict_config) {
        Err(e) => {
            println!("✅ Strict config correctly detected self-loop: {}", e);
            assert!(e.contains("Self-loop detected"));
        },
        Ok(_) => panic!("Should have detected self-loop"),
    }
    
    // Test 3: Structure validation
    println!("\n🧪 Test 3: Structure validation");
    
    let warnings = edges_table.validate_edge_structure();
    println!("✅ Structure validation: {} warnings", warnings.len());
    for warning in &warnings {
        println!("   Warning: {}", warning);
    }
    
    // Test 4: BaseTable conversion
    println!("\n🧪 Test 4: BaseTable conversion");
    
    // Create BaseTable with required edge columns  
    let mut columns = HashMap::new();
    columns.insert("edge_id".to_string(), BaseArray::with_name(
        vec![AttrValue::Int(100), AttrValue::Int(200), AttrValue::Int(300)],
        AttrValueType::Int,
        "edge_id".to_string()
    ));
    columns.insert("source".to_string(), BaseArray::with_name(
        vec![AttrValue::Int(1), AttrValue::Int(2), AttrValue::Int(3)],
        AttrValueType::Int,
        "source".to_string()
    ));
    columns.insert("target".to_string(), BaseArray::with_name(
        vec![AttrValue::Int(4), AttrValue::Int(5), AttrValue::Int(6)],
        AttrValueType::Int,
        "target".to_string()
    ));
    columns.insert("weight".to_string(), BaseArray::with_name(
        vec![AttrValue::Float(1.0), AttrValue::Float(2.5), AttrValue::Float(3.7)],
        AttrValueType::Float,
        "weight".to_string()
    ));
    
    let base_table = BaseTable::from_columns(columns).expect("Base table creation");
    let edges_from_base = base_table.to_edges().expect("Conversion should work");
    
    println!("✅ BaseTable → EdgesTable conversion: {} rows", edges_from_base.nrows());
    assert_eq!(edges_from_base.nrows(), 3);
    assert_eq!(edges_from_base.ncols(), 4);
    
    // Test 5: Access methods (get_by_edge_id)
    println!("\n🧪 Test 5: Access methods");
    
    let edge_100_row = edges_from_base.get_by_edge_id(100).expect("Should find edge 100");
    println!("✅ Found edge by ID 100: {} attributes", edge_100_row.len());
    
    match edge_100_row.get("weight") {
        Some(AttrValue::Float(weight)) => {
            println!("✅ Edge 100's weight: {}", weight);
            assert!((weight - 1.0).abs() < 0.001);
        },
        _ => panic!("Expected weight for edge 100"),
    }
    
    // Test 6: Edge statistics
    println!("\n🧪 Test 6: Edge statistics");
    
    let stats = edges_table.edge_stats().expect("Should get stats");
    println!("✅ Edge stats: {} entries", stats.len());
    
    assert_eq!(*stats.get("total_edges").unwrap(), 4);
    assert_eq!(*stats.get("self_loops").unwrap(), 1); // One self-loop: 30->30
    assert_eq!(*stats.get("unique_sources").unwrap(), 3); // 10, 20, 30 are all unique sources
    println!("   Total edges: {}", stats.get("total_edges").unwrap());
    println!("   Self loops: {}", stats.get("self_loops").unwrap());
    println!("   Unique sources: {}", stats.get("unique_sources").unwrap());
    println!("   Unique targets: {}", stats.get("unique_targets").unwrap());
    
    // Test 7: Base table access
    println!("\n🧪 Test 7: Base table access");
    
    let base_ref = edges_from_base.base();
    println!("✅ Base table access: {} x {}", base_ref.nrows(), base_ref.ncols());
    
    let base_owned = edges_from_base.into_base();
    println!("✅ Into base table: {} x {}", base_owned.nrows(), base_owned.ncols());
    
    // Test 8: Multi-edge detection
    println!("\n🧪 Test 8: Multi-edge validation");
    
    let multi_edges = vec![
        (1, 10, 20), 
        (2, 10, 20), // Duplicate connection!
        (3, 20, 30),
    ];
    let multi_edges_table = EdgesTable::new(multi_edges);
    
    let no_multi_config = EdgeConfig {
        allow_self_loops: true,
        allow_multi_edges: false,
        validate_node_references: false,
    };
    
    match multi_edges_table.validate_edges(&no_multi_config) {
        Err(e) => {
            println!("✅ Multi-edge detection worked: {}", e);
            assert!(e.contains("Multi-edge detected"));
        },
        Ok(_) => panic!("Should have detected multi-edge"),
    }
    
    println!("\n🎉 ALL PHASE 3 TESTS PASSED!");
    println!("✅ EdgesTable struct with BaseTable composition");
    println!("✅ EdgeConfig for validation policies");
    println!("✅ validate_edges() - Check duplicates, nulls, self-loops, multi-edges");
    println!("✅ validate_edge_structure() - Edge-specific lint");
    println!("✅ get_by_edge_id() - Find edge by ID");
    println!("✅ edge_stats() - Compute edge statistics");
    println!("✅ BaseTable::to_edges() conversion method");
    println!("✅ base() and into_base() access methods");
    
    println!("\n📋 Phase 3 EdgesTable Implementation: COMPLETE!");
}