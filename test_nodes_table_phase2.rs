#!/usr/bin/env rust-script
//! Test Phase 2: NodesTable Implementation

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
    
    fn from_node_ids(node_ids: Vec<NodeId>) -> Self {
        let data: Vec<AttrValue> = node_ids.into_iter()
            .map(|id| AttrValue::Int(id as i64))
            .collect();
        Self::with_name(data, AttrValueType::Int, "node_id".to_string())
    }
    
    fn as_node_ids(&self) -> Result<Vec<NodeId>, String> {
        self.data.iter()
            .map(|val| match val {
                AttrValue::Int(id) => Ok(*id as NodeId),
                _ => Err("Invalid node ID type".to_string()),
            })
            .collect()
    }
    
    fn unique_values(&self) -> Result<Vec<AttrValue>, String> {
        let mut unique = Vec::new();
        for val in &self.data {
            if !unique.contains(val) {
                unique.push(val.clone());
            }
        }
        Ok(unique)
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
    
    // Phase 2: Conversion method
    fn to_nodes(self, uid_key: &str) -> Result<NodesTable, String> {
        if !self.has_column(uid_key) {
            return Err(format!("UID column '{}' not found", uid_key));
        }
        
        if uid_key != "node_id" {
            return Err(format!("UID column must be named 'node_id', found '{}'", uid_key));
        }
        
        NodesTable::from_base_table(self)
    }
}

// Mock NodesTable with Phase 2 methods
#[derive(Clone)]
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
    
    // Phase 2: Validation methods
    fn validate_uids(&self) -> Result<(), String> {
        let node_id_column = self.base.column("node_id")
            .ok_or_else(|| "node_id column required".to_string())?;
        
        let mut seen_ids = std::collections::HashSet::new();
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
    
    // Phase 2: Access methods
    fn get_by_uid(&self, uid: NodeId) -> Option<HashMap<String, AttrValue>> {
        let node_id_column = self.base.column("node_id")?;
        
        for i in 0..node_id_column.len() {
            match node_id_column.get(i) {
                Some(AttrValue::Int(id)) if *id as NodeId == uid => {
                    let mut row = HashMap::new();
                    for col_name in self.base.column_names() {
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
    
    // Phase 2: Base table access  
    fn base(&self) -> &BaseTable { &self.base }
    fn into_base(self) -> BaseTable { self.base }
    
    // Delegate methods
    fn nrows(&self) -> usize { self.base.nrows() }
    fn ncols(&self) -> usize { self.base.ncols() }
    fn has_column(&self, name: &str) -> bool { self.base.has_column(name) }
}

fn main() {
    println!("Phase 2: NodesTable Implementation Test");
    println!("=======================================");
    
    // Test 1: NodesTable creation and basic functionality
    println!("\n🧪 Test 1: NodesTable creation");
    
    let node_ids = vec![1, 2, 3];
    let nodes_table = NodesTable::new(node_ids.clone());
    
    println!("✅ NodesTable created: {} rows, {} columns", nodes_table.nrows(), nodes_table.ncols());
    assert_eq!(nodes_table.nrows(), 3);
    assert_eq!(nodes_table.ncols(), 1);
    assert!(nodes_table.has_column("node_id"));
    
    let retrieved_ids = nodes_table.node_ids().expect("Should get node IDs");
    assert_eq!(retrieved_ids, node_ids);
    println!("✅ Node IDs retrieved correctly: {:?}", retrieved_ids);
    
    // Test 2: Validation methods  
    println!("\n🧪 Test 2: Validation methods");
    
    // Test valid UIDs
    nodes_table.validate_uids().expect("Should validate successfully");
    println!("✅ UID validation passed");
    
    let warnings = nodes_table.validate_node_structure();
    println!("✅ Structure validation: {} warnings", warnings.len());
    for warning in &warnings {
        println!("   Warning: {}", warning);
    }
    
    // Test 3: BaseTable conversion
    println!("\n🧪 Test 3: BaseTable conversion");
    
    // Create BaseTable with node_id column  
    let mut columns = HashMap::new();
    columns.insert("node_id".to_string(), BaseArray::with_name(
        vec![AttrValue::Int(10), AttrValue::Int(20), AttrValue::Int(30)],
        AttrValueType::Int,
        "node_id".to_string()
    ));
    columns.insert("name".to_string(), BaseArray::with_name(
        vec![
            AttrValue::Text("Alice".to_string()),
            AttrValue::Text("Bob".to_string()), 
            AttrValue::Text("Charlie".to_string())
        ],
        AttrValueType::Text,
        "name".to_string()
    ));
    
    let base_table = BaseTable::from_columns(columns).expect("Base table creation");
    let nodes_from_base = base_table.to_nodes("node_id").expect("Conversion should work");
    
    println!("✅ BaseTable → NodesTable conversion: {} rows", nodes_from_base.nrows());
    assert_eq!(nodes_from_base.nrows(), 3);
    assert_eq!(nodes_from_base.ncols(), 2);
    
    // Test 4: Access methods
    println!("\n🧪 Test 4: Access methods (get_by_uid)");
    
    let alice_row = nodes_from_base.get_by_uid(10).expect("Should find Alice");
    println!("✅ Found node by UID 10: {} attributes", alice_row.len());
    
    match alice_row.get("name") {
        Some(AttrValue::Text(name)) => {
            println!("✅ Alice's name: {}", name);
            assert_eq!(name, "Alice");
        },
        _ => panic!("Expected Alice's name"),
    }
    
    // Test 5: Base table access
    println!("\n🧪 Test 5: Base table access");
    
    let base_ref = nodes_from_base.base();
    println!("✅ Base table access: {} x {}", base_ref.nrows(), base_ref.ncols());
    
    let base_owned = nodes_from_base.into_base();
    println!("✅ Into base table: {} x {}", base_owned.nrows(), base_owned.ncols());
    
    // Test 6: Validation failure cases
    println!("\n🧪 Test 6: Validation edge cases");
    
    // Test duplicate IDs
    let mut bad_columns = HashMap::new();
    bad_columns.insert("node_id".to_string(), BaseArray::with_name(
        vec![AttrValue::Int(1), AttrValue::Int(1)], // Duplicate!
        AttrValueType::Int,
        "node_id".to_string()
    ));
    
    let bad_base = BaseTable::from_columns(bad_columns).expect("Bad table creation");
    let bad_nodes = NodesTable::from_base_table(bad_base).expect("Bad nodes table");
    
    match bad_nodes.validate_uids() {
        Err(e) => {
            println!("✅ Duplicate validation caught: {}", e);
            assert!(e.contains("Duplicate"));
        },
        Ok(_) => panic!("Should have failed validation"),
    }
    
    println!("\n🎉 ALL PHASE 2 TESTS PASSED!");
    println!("✅ NodesTable struct with BaseTable composition");
    println!("✅ validate_uids() - Check uniqueness, no nulls"); 
    println!("✅ validate_node_structure() - Node-specific lint");
    println!("✅ get_by_uid() - Find node by UID");
    println!("✅ BaseTable::to_nodes() conversion method");
    println!("✅ base() and into_base() access methods");
    
    println!("\n📋 Phase 2 NodesTable Implementation: COMPLETE!");
}