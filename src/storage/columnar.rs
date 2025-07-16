//! ColumnarStore: Columnar storage for attributes with efficient bulk operations
//! Implements fast batch and vectorized access for Groggy graphs.

use pyo3::prelude::*;
use serde_json::Value;
use std::collections::{HashMap, HashSet};

use bitvec::prelude::*;
use dashmap::DashMap;
use serde_json::Value as JsonValue;

/// Supported attribute types for columnar storage
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum AttributeType {
    Int,
    Float,
    Bool,
    Str,
    Json, // fallback for mixed/complex types
}

/// Unique identifier for attributes across the entire graph
#[derive(
    Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct AttrUID(pub u64);

/// Enum for typed columnar data
#[derive(Debug, Clone)]
pub enum ColumnData {
    Int(Vec<Option<i64>>),
    Float(Vec<Option<f64>>),
    Bool(Vec<Option<bool>>),
    Str(Vec<Option<String>>),
    Json(std::collections::HashMap<usize, serde_json::Value>),
}

/// Columnar storage for attributes with efficient bulk operations, UID mapping, schema, and bitmap indices
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColumnKind {
    Node,
    Edge,
}

pub struct ColumnarStore {
    /// Maps attribute names to their unique IDs
    pub attr_name_to_uid: DashMap<String, AttrUID>,
    /// Maps attribute UIDs back to names
    pub attr_uid_to_name: DashMap<AttrUID, String>,
    /// Next available attribute UID
    pub next_attr_uid: std::sync::atomic::AtomicU64,

    /// Attribute type/schema: attr_uid -> AttributeType
    pub attr_schema: DashMap<AttrUID, AttributeType>,

    /// Unified columnar storage: (ColumnKind, attr_uid) -> ColumnData
    pub columns: DashMap<(ColumnKind, AttrUID), ColumnData>,
    /// Sparse fallback storage (legacy): attr_uid -> HashMap<entity_index, attr_value>
    pub node_attributes: DashMap<AttrUID, HashMap<usize, JsonValue>>,
    pub edge_attributes: DashMap<AttrUID, HashMap<usize, JsonValue>>,

    /// Bitmap indices for fast filtering (built on-demand)
    /// Maps (attr_uid, attr_value) -> BitVec indicating which nodes/edges have this value
    pub node_value_bitmaps: DashMap<(AttrUID, JsonValue), BitVec>,
    pub edge_value_bitmaps: DashMap<(AttrUID, JsonValue), BitVec>,

    /// Track if bitmaps are up-to-date
    pub bitmaps_dirty: std::sync::atomic::AtomicBool,
    
    /// Track maximum entity indices for bitmap sizing
    pub max_node_index: std::sync::atomic::AtomicUsize,
    pub max_edge_index: std::sync::atomic::AtomicUsize,
}

impl Clone for ColumnarStore {
    fn clone(&self) -> Self {
        Self {
            attr_name_to_uid: self.attr_name_to_uid.clone(),
            attr_uid_to_name: self.attr_uid_to_name.clone(),
            next_attr_uid: std::sync::atomic::AtomicU64::new(self.next_attr_uid.load(std::sync::atomic::Ordering::Relaxed)),
            attr_schema: self.attr_schema.clone(),
            columns: self.columns.clone(),
            node_attributes: self.node_attributes.clone(),
            edge_attributes: self.edge_attributes.clone(),
            node_value_bitmaps: self.node_value_bitmaps.clone(),
            edge_value_bitmaps: self.edge_value_bitmaps.clone(),
            bitmaps_dirty: std::sync::atomic::AtomicBool::new(self.bitmaps_dirty.load(std::sync::atomic::Ordering::Relaxed)),
            max_node_index: std::sync::atomic::AtomicUsize::new(self.max_node_index.load(std::sync::atomic::Ordering::Relaxed)),
            max_edge_index: std::sync::atomic::AtomicUsize::new(self.max_edge_index.load(std::sync::atomic::Ordering::Relaxed)),
        }
    }
}

impl ColumnarStore {
    /// Returns all node columns as Vec<(AttrUID, ColumnData)> (cloned). For inspection/bulk ops only.
    pub fn node_columns(&self) -> Vec<(AttrUID, ColumnData)> {
        self.columns.iter()
            .filter_map(|entry| {
                let ((kind, uid), col) = entry.pair();
                if *kind == ColumnKind::Node {
                    Some((uid.clone(), col.clone()))
                } else {
                    None
                }
            })
            .collect()
    }
    /// Returns all edge columns as Vec<(AttrUID, ColumnData)> (cloned). For inspection/bulk ops only.
    pub fn edge_columns(&self) -> Vec<(AttrUID, ColumnData)> {
        self.columns.iter()
            .filter_map(|entry| {
                let ((kind, uid), col) = entry.pair();
                if *kind == ColumnKind::Edge {
                    Some((uid.clone(), col.clone()))
                } else {
                    None
                }
            })
            .collect()
    }
}

impl ColumnarStore {
    pub fn new() -> Self {
        Self {
            attr_name_to_uid: DashMap::new(),
            attr_uid_to_name: DashMap::new(),
            next_attr_uid: std::sync::atomic::AtomicU64::new(1),
            attr_schema: DashMap::new(),
        columns: DashMap::new(),
            node_attributes: DashMap::new(),
            edge_attributes: DashMap::new(),
            node_value_bitmaps: DashMap::new(),
            edge_value_bitmaps: DashMap::new(),
            bitmaps_dirty: std::sync::atomic::AtomicBool::new(true),
            max_node_index: std::sync::atomic::AtomicUsize::new(0),
            max_edge_index: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Registers an attribute name and returns its UID. If already present, returns existing UID.
    pub fn register_attr(&self, attr_name: String) -> u64 {
        if let Some(uid) = self.attr_name_to_uid.get(&attr_name) {
            return uid.0;
        }
        let uid = self.next_attr_uid.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let attr_uid = AttrUID(uid);
        self.attr_name_to_uid.insert(attr_name.clone(), attr_uid.clone());
        self.attr_uid_to_name.insert(attr_uid.clone(), attr_name);
        // Default to Json type for backward compatibility
        self.attr_schema.insert(attr_uid.clone(), AttributeType::Json);
        uid
    }

    /// Registers an attribute name and type, returning its UID. If already present, enforces type match.
    pub fn register_attr_with_type(&self, attr_name: String, attr_type: AttributeType, is_node: bool) -> Result<u64, String> {
        if let Some(uid) = self.attr_name_to_uid.get(&attr_name) {
            // If already present, check schema
            let schema = self.attr_schema.get(&uid);
            if let Some(existing_type) = schema {
                if *existing_type != attr_type {
                    return Err(format!("Attribute '{}' already registered with type {:?}, got {:?}", attr_name, existing_type, attr_type));
                }
            }
            return Ok(uid.0);
        }
        let uid = self.next_attr_uid.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let attr_uid = AttrUID(uid);
        self.attr_name_to_uid.insert(attr_name.clone(), attr_uid.clone());
        self.attr_uid_to_name.insert(attr_uid.clone(), attr_name);
        self.attr_schema.insert(attr_uid.clone(), attr_type);
        // Initialize empty column of correct type
        match attr_type {
            AttributeType::Int => {
                let kind = if is_node { ColumnKind::Node } else { ColumnKind::Edge };
                self.columns.insert((kind, attr_uid.clone()), ColumnData::Int(Vec::new()));
            }
            AttributeType::Float => {
                let kind = if is_node { ColumnKind::Node } else { ColumnKind::Edge };
                self.columns.insert((kind, attr_uid.clone()), ColumnData::Float(Vec::new()));
            }
            AttributeType::Bool => {
                let kind = if is_node { ColumnKind::Node } else { ColumnKind::Edge };
                self.columns.insert((kind, attr_uid.clone()), ColumnData::Bool(Vec::new()));
            }
            AttributeType::Str => {
                let kind = if is_node { ColumnKind::Node } else { ColumnKind::Edge };
                self.columns.insert((kind, attr_uid.clone()), ColumnData::Str(Vec::new()));
            }
            AttributeType::Json => {
                let kind = if is_node { ColumnKind::Node } else { ColumnKind::Edge };
                self.columns.insert((kind, attr_uid.clone()), ColumnData::Json(HashMap::new()));
            }
        }
        Ok(uid)
    }

    // --- Typed node attribute set/get ---

    /// Set a single int value for a node attribute and entity index
    pub fn set_node_int(&self, attr_name: String, idx: usize, value: i64) -> Result<(), String> {
        let uid = self.attr_name_to_uid.get(&attr_name).ok_or("Attribute not found")?.clone();
        let schema = self.attr_schema.get(&uid).ok_or("Schema not found")?;
        if *schema != AttributeType::Int {
            return Err("Type mismatch: attribute is not Int".to_string());
        }
        let mut col = self.columns.get_mut(&(ColumnKind::Node, uid.clone())).ok_or("Column not found")?;
        match &mut *col {
            ColumnData::Int(vec) => {
                if idx >= vec.len() {
                    vec.resize(idx + 1, None);
                }
                vec[idx] = Some(value);
                Ok(())
            }
            _ => Err("Column type mismatch".to_string()),
        }
    }

    /// Get a single int value for a node attribute and entity index
    pub fn get_node_int(&self, attr_name: String, idx: usize) -> Option<i64> {
        let uid = self.attr_name_to_uid.get(&attr_name)?;
        let schema = self.attr_schema.get(&uid)?;
        if *schema != AttributeType::Int {
            return None;
        }
        let col = self.columns.get(&(ColumnKind::Node, uid.clone()))?;
        match *col {
            ColumnData::Int(ref vec) => vec.get(idx).and_then(|v| *v),
            _ => None,
        }
    }

    /// Batch set int values for a node attribute
    pub fn set_node_int_batch(&self, attr_name: String, indices: &[usize], values: &[i64]) -> Result<(), String> {
        if indices.len() != values.len() {
            return Err("Indices and values must have same length".to_string());
        }
        let uid = self.attr_name_to_uid.get(&attr_name).ok_or("Attribute not found")?.clone();
        let schema = self.attr_schema.get(&uid).ok_or("Schema not found")?;
        if *schema != AttributeType::Int {
            return Err("Type mismatch: attribute is not Int".to_string());
        }
        let mut col = self.columns.get_mut(&(ColumnKind::Node, uid.clone())).ok_or("Column not found")?;
        match &mut *col {
            ColumnData::Int(vec) => {
                for (&idx, &value) in indices.iter().zip(values.iter()) {
                    if idx >= vec.len() {
                        vec.resize(idx + 1, None);
                    }
                    vec[idx] = Some(value);
                }
                Ok(())
            }
            _ => Err("Column type mismatch".to_string()),
        }
    }

    /// Batch get int values for a node attribute
    pub fn get_node_int_batch(&self, attr_name: String, indices: &[usize]) -> Option<Vec<Option<i64>>> {
        let uid = self.attr_name_to_uid.get(&attr_name)?;
        let schema = self.attr_schema.get(&uid)?;
        if *schema != AttributeType::Int {
            return None;
        }
        let binding = self.node_columns();
        let col = binding.iter().find(|(k,_)| *k == *uid).map(|(_,c)| c)?;
        match col {
            ColumnData::Int(vec) => Some(indices.iter().map(|&idx| vec.get(idx).copied().flatten()).collect()),
            _ => None,
        }
    }

    // --- Repeat for Float, Bool, Str, Json (Json uses fallback map) ---

    /// Set a single float value for a node attribute and entity index
    pub fn set_node_float(&self, attr_name: String, idx: usize, value: f64) -> Result<(), String> {
        let uid = self.attr_name_to_uid.get(&attr_name).ok_or("Attribute not found")?.clone();
        let schema = self.attr_schema.get(&uid).ok_or("Schema not found")?;
        if *schema != AttributeType::Float {
            return Err("Type mismatch: attribute is not Float".to_string());
        }
        let mut col = self.columns.get_mut(&(ColumnKind::Node, uid.clone())).ok_or("Column not found")?;
        match &mut *col {
            ColumnData::Float(vec) => {
                if idx >= vec.len() {
                    vec.resize(idx + 1, None);
                }
                vec[idx] = Some(value);
                Ok(())
            }
            _ => Err("Column type mismatch".to_string()),
        }
    }

    /// Get a single float value for a node attribute and entity index
    pub fn get_node_float(&self, attr_name: String, idx: usize) -> Option<f64> {
        let uid = self.attr_name_to_uid.get(&attr_name)?;
        let schema = self.attr_schema.get(&uid)?;
        if *schema != AttributeType::Float {
            return None;
        }
        let col = self.columns.get(&(ColumnKind::Node, uid.clone()))?;
        match *col {
            ColumnData::Float(ref vec) => vec.get(idx).and_then(|v| *v),
            _ => None,
        }
    }

    /// Set a single bool value for a node attribute and entity index
    pub fn set_node_bool(&self, attr_name: String, idx: usize, value: bool) -> Result<(), String> {
        let uid = self.attr_name_to_uid.get(&attr_name).ok_or("Attribute not found")?.clone();
        let schema = self.attr_schema.get(&uid).ok_or("Schema not found")?;
        if *schema != AttributeType::Bool {
            return Err("Type mismatch: attribute is not Bool".to_string());
        }
        let mut col = self.columns.get_mut(&(ColumnKind::Node, uid.clone())).ok_or("Column not found")?;
        match &mut *col {
            ColumnData::Bool(vec) => {
                if idx >= vec.len() {
                    vec.resize(idx + 1, None);
                }
                vec[idx] = Some(value);
                Ok(())
            }
            _ => Err("Column type mismatch".to_string()),
        }
    }

    /// Get a single bool value for a node attribute and entity index
    pub fn get_node_bool(&self, attr_name: String, idx: usize) -> Option<bool> {
        let uid = self.attr_name_to_uid.get(&attr_name)?;
        let schema = self.attr_schema.get(&uid)?;
        if *schema != AttributeType::Bool {
            return None;
        }
        let col = self.columns.get(&(ColumnKind::Node, uid.clone()))?;
        match *col {
            ColumnData::Bool(ref vec) => vec.get(idx).and_then(|v| *v),
            _ => None,
        }
    }

    /// Set a single string value for a node attribute and entity index
    pub fn set_node_str(&self, attr_name: String, idx: usize, value: String) -> Result<(), String> {
        let uid = self.attr_name_to_uid.get(&attr_name).ok_or("Attribute not found")?.clone();
        let schema = self.attr_schema.get(&uid).ok_or("Schema not found")?;
        if *schema != AttributeType::Str {
            return Err("Type mismatch: attribute is not Str".to_string());
        }
        let mut col = self.columns.get_mut(&(ColumnKind::Node, uid.clone())).ok_or("Column not found")?;
        match &mut *col {
            ColumnData::Str(vec) => {
                if idx >= vec.len() {
                    vec.resize(idx + 1, None);
                }
                vec[idx] = Some(value);
                Ok(())
            }
            _ => Err("Column type mismatch".to_string()),
        }
    }

    /// Get a single string value for a node attribute and entity index
    pub fn get_node_str(&self, attr_name: String, idx: usize) -> Option<String> {
        let uid = self.attr_name_to_uid.get(&attr_name)?;
        let schema = self.attr_schema.get(&uid)?;
        if *schema != AttributeType::Str {
            return None;
        }
        let col = self.columns.get(&(ColumnKind::Node, uid.clone()))?;
        match *col {
            ColumnData::Str(ref vec) => vec.get(idx).and_then(|v| v.clone()),
            _ => None,
        }
    }

    /// Set a single value for a node attribute and entity index (Json fallback)
    pub fn set_node_json(&self, attr_name: String, idx: usize, value: JsonValue) -> Result<(), String> {
        let uid = self.attr_name_to_uid.get(&attr_name).ok_or("Attribute not found")?.clone();
        let schema = self.attr_schema.get(&uid).ok_or("Schema not found")?;
        if *schema != AttributeType::Json {
            return Err("Type mismatch: attribute is not Json".to_string());
        }
        let mut col = self.columns.get_mut(&(ColumnKind::Node, uid.clone())).ok_or("Column not found")?;
        match &mut *col {
            ColumnData::Json(map) => {
                map.insert(idx, value);
                Ok(())
            }
            _ => Err("Column type mismatch".to_string()),
        }
    }

    /// Get a single value for a node attribute and entity index (Json fallback)
    pub fn get_node_json(&self, attr_name: String, idx: usize) -> Option<JsonValue> {
        let uid = self.attr_name_to_uid.get(&attr_name)?;
        let schema = self.attr_schema.get(&uid)?;
        if *schema != AttributeType::Json {
            return None;
        }
        let col = self.columns.get(&(ColumnKind::Node, uid.clone()))?;
        match *col {
            ColumnData::Json(ref map) => map.get(&idx).cloned(),
            _ => None,
        }
    }

    // --- Repeat all for edges ---

    /// Set a single int value for an edge attribute and entity index
    pub fn set_edge_int(&self, attr_name: String, idx: usize, value: i64) -> Result<(), String> {
        let uid = self.attr_name_to_uid.get(&attr_name).ok_or("Attribute not found")?.clone();
        let schema = self.attr_schema.get(&uid).ok_or("Schema not found")?;
        if *schema != AttributeType::Int {
            return Err("Type mismatch: attribute is not Int".to_string());
        }
        let mut binding = self.edge_columns();
        let col = binding.iter_mut().find(|(k,_)| *k == uid).map(|(_,c)| c).ok_or("Column not found")?;
        match col {
            ColumnData::Int(vec) => {
                if idx >= vec.len() {
                    vec.resize(idx + 1, None);
                }
                vec[idx] = Some(value);
                Ok(())
            }
            _ => Err("Column type mismatch".to_string()),
        }
    }

    /// Get a single int value for an edge attribute and entity index
    pub fn get_edge_int(&self, attr_name: String, idx: usize) -> Option<i64> {
        let uid = self.attr_name_to_uid.get(&attr_name)?;
        let schema = self.attr_schema.get(&uid)?;
        if *schema != AttributeType::Int {
            return None;
        }
        let binding = self.edge_columns();
        let col = binding.iter().find(|(k,_)| *k == *uid).map(|(_,c)| c)?;
        match col {
            ColumnData::Int(vec) => vec.get(idx).and_then(|v| *v),
            _ => None,
        }
    }

    /// Set a single float value for an edge attribute and entity index
    pub fn set_edge_float(&self, attr_name: String, idx: usize, value: f64) -> Result<(), String> {
        let uid = self.attr_name_to_uid.get(&attr_name).ok_or("Attribute not found")?.clone();
        let schema = self.attr_schema.get(&uid).ok_or("Schema not found")?;
        if *schema != AttributeType::Float {
            return Err("Type mismatch: attribute is not Float".to_string());
        }
        let mut col = self.columns.get_mut(&(ColumnKind::Edge, uid.clone())).ok_or("Column not found")?;
        match &mut *col {
            ColumnData::Float(vec) => {
                if idx >= vec.len() {
                    vec.resize(idx + 1, None);
                }
                vec[idx] = Some(value);
                Ok(())
            }
            _ => Err("Column type mismatch".to_string()),
        }
    }

    /// Get a single float value for an edge attribute and entity index
    pub fn get_edge_float(&self, attr_name: String, idx: usize) -> Option<f64> {
        let uid = self.attr_name_to_uid.get(&attr_name)?;
        let schema = self.attr_schema.get(&uid)?;
        if *schema != AttributeType::Float {
            return None;
        }
        let col = self.columns.get(&(ColumnKind::Edge, uid.clone()))?;
        match *col {
            ColumnData::Float(ref vec) => vec.get(idx).and_then(|v| *v),
            _ => None,
        }
    }

    /// Set a single bool value for an edge attribute and entity index
    pub fn set_edge_bool(&self, attr_name: String, idx: usize, value: bool) -> Result<(), String> {
        let uid = self.attr_name_to_uid.get(&attr_name).ok_or("Attribute not found")?.clone();
        let schema = self.attr_schema.get(&uid).ok_or("Schema not found")?;
        if *schema != AttributeType::Bool {
            return Err("Type mismatch: attribute is not Bool".to_string());
        }
        let mut col = self.columns.get_mut(&(ColumnKind::Edge, uid.clone())).ok_or("Column not found")?;
        match &mut *col {
            ColumnData::Bool(vec) => {
                if idx >= vec.len() {
                    vec.resize(idx + 1, None);
                }
                vec[idx] = Some(value);
                Ok(())
            }
            _ => Err("Column type mismatch".to_string()),
        }
    }

    /// Get a single bool value for an edge attribute and entity index
    pub fn get_edge_bool(&self, attr_name: String, idx: usize) -> Option<bool> {
        let uid = self.attr_name_to_uid.get(&attr_name)?;
        let schema = self.attr_schema.get(&uid)?;
        if *schema != AttributeType::Bool {
            return None;
        }
        let col = self.columns.get(&(ColumnKind::Edge, uid.clone()))?;
        match *col {
            ColumnData::Bool(ref vec) => vec.get(idx).and_then(|v| *v),
            _ => None,
        }
    }

    /// Set a single string value for an edge attribute and entity index
    pub fn set_edge_str(&self, attr_name: String, idx: usize, value: String) -> Result<(), String> {
        let uid = self.attr_name_to_uid.get(&attr_name).ok_or("Attribute not found")?.clone();
        let schema = self.attr_schema.get(&uid).ok_or("Schema not found")?;
        if *schema != AttributeType::Str {
            return Err("Type mismatch: attribute is not Str".to_string());
        }
        let mut col = self.columns.get_mut(&(ColumnKind::Edge, uid.clone())).ok_or("Column not found")?;
        match &mut *col {
            ColumnData::Str(vec) => {
                if idx >= vec.len() {
                    vec.resize(idx + 1, None);
                }
                vec[idx] = Some(value);
                Ok(())
            }
            _ => Err("Column type mismatch".to_string()),
        }
    }

    /// Get a single string value for an edge attribute and entity index
    pub fn get_edge_str(&self, attr_name: String, idx: usize) -> Option<String> {
        let uid = self.attr_name_to_uid.get(&attr_name)?;
        let schema = self.attr_schema.get(&uid)?;
        if *schema != AttributeType::Str {
            return None;
        }
        let col = self.columns.get(&(ColumnKind::Edge, uid.clone()))?;
        match *col {
            ColumnData::Str(ref vec) => vec.get(idx).and_then(|v| v.clone()),
            _ => None,
        }
    }

    /// Set a single value for an edge attribute and entity index (Json fallback)
    pub fn set_edge_json(&self, attr_name: String, idx: usize, value: JsonValue) -> Result<(), String> {
        let uid = self.attr_name_to_uid.get(&attr_name).ok_or("Attribute not found")?.clone();
        let schema = self.attr_schema.get(&uid).ok_or("Schema not found")?;
        if *schema != AttributeType::Json {
            return Err("Type mismatch: attribute is not Json".to_string());
        }
        let mut col = self.columns.get_mut(&(ColumnKind::Edge, uid.clone())).ok_or("Column not found")?;
        match &mut *col {
            ColumnData::Json(map) => {
                map.insert(idx, value);
                Ok(())
            }
            _ => Err("Column type mismatch".to_string()),
        }
    }

    /// Get a single value for an edge attribute and entity index (Json fallback)
    pub fn get_edge_json(&self, attr_name: String, idx: usize) -> Option<JsonValue> {
        let uid = self.attr_name_to_uid.get(&attr_name)?;
        let schema = self.attr_schema.get(&uid)?;
        if *schema != AttributeType::Json {
            return None;
        }
        let col = self.columns.get(&(ColumnKind::Edge, uid.clone()))?;
        match *col {
            ColumnData::Json(ref map) => map.get(&idx).cloned(),
            _ => None,
        }
    }

    // --- Type-specific bitmap filtering for Int, Bool, Str columns (nodes) ---

    /// Filter node indices by int value for a column (scalar fallback)
    pub fn filter_nodes_by_int(&self, attr_name: String, value: i64) -> Vec<usize> {
        let uid = match self.attr_name_to_uid.get(&attr_name) {
            Some(u) => u.clone(),
            None => return Vec::new(),
        };
        let schema = match self.attr_schema.get(&uid) {
            Some(s) => *s,
            None => return Vec::new(),
        };
        if schema != AttributeType::Int {
            return Vec::new();
        }
        let binding = self.node_columns();
let col = match binding.iter().find(|(k,_)| *k == uid).map(|(_,c)| c) {
            Some(c) => c,
            None => return Vec::new(),
        };
        match col {
            ColumnData::Int(vec) => vec.iter().enumerate().filter_map(|(i, v)| v.filter(|&x| x == value).map(|_| i)).collect(),
            _ => Vec::new(),
        }
    }

    /// Internal helper: SIMD/vectorized filter for int columns (uses std::simd if available)
    #[cfg(feature = "simd")] // Enable this with --features=simd
    fn filter_int_simd_helper(col: &ColumnData, value: i64) -> Vec<usize> {
        use std::simd::{Simd, SimdPartialEq};
        const LANES: usize = 8;
        let mut result = Vec::new();
        if let ColumnData::Int(vec) = col {
            let mut i = 0;
            while i + LANES <= vec.len() {
                let simd_vals = Simd::<i64, LANES>::from_array([
                    vec[i].unwrap_or(i64::MIN),
                    vec[i+1].unwrap_or(i64::MIN),
                    vec[i+2].unwrap_or(i64::MIN),
                    vec[i+3].unwrap_or(i64::MIN),
                    vec[i+4].unwrap_or(i64::MIN),
                    vec[i+5].unwrap_or(i64::MIN),
                    vec[i+6].unwrap_or(i64::MIN),
                    vec[i+7].unwrap_or(i64::MIN),
                ]);
                let mask = simd_vals.simd_eq(Simd::splat(value));
                for lane in 0..LANES {
                    if mask.test(lane) && vec[i+lane].is_some() {
                        result.push(i+lane);
                    }
                }
                i += LANES;
            }
            for j in i..vec.len() {
                if vec[j] == Some(value) {
                    result.push(j);
                }
            }
        }
        result
    }
    #[cfg(not(feature = "simd"))]
    fn filter_int_simd_helper(col: &ColumnData, value: i64) -> Vec<usize> {
        if let ColumnData::Int(vec) = col {
            vec.iter().enumerate().filter_map(|(i, v)| v.filter(|&x| x == value).map(|_| i)).collect()
        } else {
            Vec::new()
        }
    }
    /// SIMD/vectorized filter for node int columns (delegates to helper)
    pub fn filter_nodes_int_simd(&self, attr_name: String, value: i64) -> Vec<usize> {
        let uid = match self.attr_name_to_uid.get(&attr_name) {
            Some(u) => u.clone(),
            None => return Vec::new(),
        };
        let schema = match self.attr_schema.get(&uid) {
            Some(s) => *s,
            None => return Vec::new(),
        };
        if schema != AttributeType::Int {
            return Vec::new();
        }
        let binding = self.node_columns();
        let col = match binding.iter().find(|(k,_)| *k == uid).map(|(_,c)| c) {
            Some(c) => c,
            None => return Vec::new(),
        };
        Self::filter_int_simd_helper(&*col, value)
    }
    /// SIMD/vectorized filter for edge int columns (delegates to helper)
    pub fn filter_edges_int_simd(&self, attr_name: String, value: i64) -> Vec<usize> {
        let uid = match self.attr_name_to_uid.get(&attr_name) {
            Some(u) => u.clone(),
            None => return Vec::new(),
        };
        let schema = match self.attr_schema.get(&uid) {
            Some(s) => *s,
            None => return Vec::new(),
        };
        if schema != AttributeType::Int {
            return Vec::new();
        }
        let binding = self.edge_columns();
let col = match binding.iter().find(|(k,_)| *k == uid).map(|(_,c)| c) {
            Some(c) => c,
            None => return Vec::new(),
        };
        Self::filter_int_simd_helper(&*col, value)
    }

    /// Batch set/get for node float columns
    pub fn set_node_float_batch(&self, attr_name: String, indices: &[usize], values: &[f64]) -> Result<(), String> {
        if indices.len() != values.len() {
            return Err("Indices and values must have same length".to_string());
        }
        let uid = self.attr_name_to_uid.get(&attr_name).ok_or("Attribute not found")?.clone();
        let schema = self.attr_schema.get(&uid).ok_or("Schema not found")?;
        if *schema != AttributeType::Float {
            return Err("Type mismatch: attribute is not Float".to_string());
        }
        let mut binding = self.node_columns();
        let col = binding.iter_mut().find(|(k,_)| *k == uid).map(|(_,c)| c).ok_or("Column not found")?;
        match col {
            ColumnData::Float(vec) => {
                for (&idx, &val) in indices.iter().zip(values.iter()) {
                    if idx >= vec.len() {
                        vec.resize(idx + 1, None);
                    }
                    vec[idx] = Some(val);
                }
                Ok(())
            }
            _ => Err("Column type mismatch".to_string()),
        }
    }
    pub fn get_node_float_batch(&self, attr_name: String, indices: &[usize]) -> Option<Vec<Option<f64>>> {
        let uid = self.attr_name_to_uid.get(&attr_name)?;
        let schema = self.attr_schema.get(&uid)?;
        if *schema != AttributeType::Float {
            return None;
        }
        let binding = self.node_columns();
        let col = binding.iter().find(|(k,_)| *k == *uid).map(|(_,c)| c)?;
        match col {
            ColumnData::Float(vec) => Some(indices.iter().map(|&idx| vec.get(idx).copied().flatten()).collect()),
            _ => None,
        }
    }
    /// Batch set/get for node bool columns
    pub fn set_node_bool_batch(&self, attr_name: String, indices: &[usize], values: &[bool]) -> Result<(), String> {
        if indices.len() != values.len() {
            return Err("Indices and values must have same length".to_string());
        }
        let uid = self.attr_name_to_uid.get(&attr_name).ok_or("Attribute not found")?.clone();
        let schema = self.attr_schema.get(&uid).ok_or("Schema not found")?;
        if *schema != AttributeType::Bool {
            return Err("Type mismatch: attribute is not Bool".to_string());
        }
        let mut binding = self.node_columns();
        let col = binding.iter_mut().find(|(k,_)| *k == uid).map(|(_,c)| c).ok_or("Column not found")?;
        match col {
            ColumnData::Bool(vec) => {
                for (&idx, &val) in indices.iter().zip(values.iter()) {
                    if idx >= vec.len() {
                        vec.resize(idx + 1, None);
                    }
                    vec[idx] = Some(val);
                }
                Ok(())
            }
            _ => Err("Column type mismatch".to_string()),
        }
    }
    pub fn get_node_bool_batch(&self, attr_name: String, indices: &[usize]) -> Option<Vec<Option<bool>>> {
        let uid = self.attr_name_to_uid.get(&attr_name)?;
        let schema = self.attr_schema.get(&uid)?;
        if *schema != AttributeType::Bool {
            return None;
        }
        let binding = self.node_columns();
        let col = binding.iter().find(|(k,_)| *k == *uid).map(|(_,c)| c)?;
        match col {
            ColumnData::Bool(vec) => Some(indices.iter().map(|&idx| vec.get(idx).copied().flatten()).collect()),
            _ => None,
        }
    }
    /// Batch set/get for node str columns
    pub fn set_node_str_batch(&self, attr_name: String, indices: &[usize], values: &[String]) -> Result<(), String> {
        if indices.len() != values.len() {
            return Err("Indices and values must have same length".to_string());
        }
        let uid = self.attr_name_to_uid.get(&attr_name).ok_or("Attribute not found")?.clone();
        let schema = self.attr_schema.get(&uid).ok_or("Schema not found")?;
        if *schema != AttributeType::Str {
            return Err("Type mismatch: attribute is not Str".to_string());
        }
        let mut binding = self.node_columns();
        let col = binding.iter_mut().find(|(k,_)| *k == uid).map(|(_,c)| c).ok_or("Column not found")?;
        match col {
            ColumnData::Str(vec) => {
                for (idx, val) in indices.iter().zip(values.iter()) {
                    if *idx >= vec.len() {
                        vec.resize(*idx + 1, None);
                    }
                    vec[*idx] = Some(val.clone());
                }
                Ok(())
            }
            _ => Err("Column type mismatch".to_string()),
        }
    }
    pub fn get_node_str_batch(&self, attr_name: String, indices: &[usize]) -> Option<Vec<Option<String>>> {
        let uid = self.attr_name_to_uid.get(&attr_name)?;
        let schema = self.attr_schema.get(&uid)?;
        if *schema != AttributeType::Str {
            return None;
        }
        let binding = self.node_columns();
        let col = binding.iter().find(|(k,_)| *k == *uid).map(|(_,c)| c)?;
        match col {
            ColumnData::Str(vec) => Some(indices.iter().map(|&idx| vec.get(idx).and_then(|v| v.clone())).collect()),
            _ => None,
        }
    }
    // --- Repeat batch set/get for edges (float, bool, str) ---
    pub fn set_edge_float_batch(&self, attr_name: String, indices: &[usize], values: &[f64]) -> Result<(), String> {
        if indices.len() != values.len() {
            return Err("Indices and values must have same length".to_string());
        }
        let uid = self.attr_name_to_uid.get(&attr_name).ok_or("Attribute not found")?.clone();
        let schema = self.attr_schema.get(&uid).ok_or("Schema not found")?;
        if *schema != AttributeType::Float {
            return Err("Type mismatch: attribute is not Float".to_string());
        }
        let mut binding = self.edge_columns();
        let col = binding.iter_mut().find(|(k,_)| *k == uid).map(|(_,c)| c).ok_or("Column not found")?;
        match col {
            ColumnData::Float(vec) => {
                for (&idx, &val) in indices.iter().zip(values.iter()) {
                    if idx >= vec.len() {
                        vec.resize(idx + 1, None);
                    }
                    vec[idx] = Some(val);
                }
                Ok(())
            }
            _ => Err("Column type mismatch".to_string()),
        }
    }
    pub fn get_edge_float_batch(&self, attr_name: String, indices: &[usize]) -> Option<Vec<Option<f64>>> {
        let uid = self.attr_name_to_uid.get(&attr_name)?;
        let schema = self.attr_schema.get(&uid)?;
        if *schema != AttributeType::Float {
            return None;
        }
        let binding = self.edge_columns();
        let col = binding.iter().find(|(k,_)| *k == *uid).map(|(_,c)| c)?;
        match col {
            ColumnData::Float(vec) => Some(indices.iter().map(|&idx| vec.get(idx).copied().flatten()).collect()),
            _ => None,
        }
    }
    pub fn set_edge_bool_batch(&self, attr_name: String, indices: &[usize], values: &[bool]) -> Result<(), String> {
        if indices.len() != values.len() {
            return Err("Indices and values must have same length".to_string());
        }
        let uid = self.attr_name_to_uid.get(&attr_name).ok_or("Attribute not found")?.clone();
        let schema = self.attr_schema.get(&uid).ok_or("Schema not found")?;
        if *schema != AttributeType::Bool {
            return Err("Type mismatch: attribute is not Bool".to_string());
        }
        let mut binding = self.edge_columns();
        let col = binding.iter_mut().find(|(k,_)| *k == uid).map(|(_,c)| c).ok_or("Column not found")?;
        match col {
            ColumnData::Bool(vec) => {
                for (&idx, &val) in indices.iter().zip(values.iter()) {
                    if idx >= vec.len() {
                        vec.resize(idx + 1, None);
                    }
                    vec[idx] = Some(val);
                }
                Ok(())
            }
            _ => Err("Column type mismatch".to_string()),
        }
    }
    pub fn get_edge_bool_batch(&self, attr_name: String, indices: &[usize]) -> Option<Vec<Option<bool>>> {
        let uid = self.attr_name_to_uid.get(&attr_name)?;
        let schema = self.attr_schema.get(&uid)?;
        if *schema != AttributeType::Bool {
            return None;
        }
        let binding = self.edge_columns();
        let col = binding.iter().find(|(k,_)| *k == *uid).map(|(_,c)| c)?;
        match col {
            ColumnData::Bool(vec) => Some(indices.iter().map(|&idx| vec.get(idx).copied().flatten()).collect()),
            _ => None,
        }
    }
    pub fn set_edge_str_batch(&self, attr_name: String, indices: &[usize], values: &[String]) -> Result<(), String> {
        if indices.len() != values.len() {
            return Err("Indices and values must have same length".to_string());
        }
        let uid = self.attr_name_to_uid.get(&attr_name).ok_or("Attribute not found")?.clone();
        let schema = self.attr_schema.get(&uid).ok_or("Schema not found")?;
        if *schema != AttributeType::Str {
            return Err("Type mismatch: attribute is not Str".to_string());
        }
        let mut binding = self.edge_columns();
        let col = binding.iter_mut().find(|(k,_)| *k == uid).map(|(_,c)| c).ok_or("Column not found")?;
        match col {
            ColumnData::Str(vec) => {
                for (idx, val) in indices.iter().zip(values.iter()) {
                    if *idx >= vec.len() {
                        vec.resize(*idx + 1, None);
                    }
                    vec[*idx] = Some(val.clone());
                }
                Ok(())
            }
            _ => Err("Column type mismatch".to_string()),
        }
    }
    pub fn get_edge_str_batch(&self, attr_name: String, indices: &[usize]) -> Option<Vec<Option<String>>> {
        let uid = self.attr_name_to_uid.get(&attr_name)?;
        let schema = self.attr_schema.get(&uid)?;
        if *schema != AttributeType::Str {
            return None;
        }
        let binding = self.edge_columns();
        let col = binding.iter().find(|(k,_)| *k == *uid).map(|(_,c)| c)?;
        match col {
            ColumnData::Str(vec) => Some(indices.iter().map(|&idx| vec.get(idx).and_then(|v| v.clone())).collect()),
            _ => None,
        }
    }


    /// Filter node indices by bool value for a column
    pub fn filter_nodes_by_bool(&self, attr_name: String, value: bool) -> Vec<usize> {
        let uid = match self.attr_name_to_uid.get(&attr_name) {
            Some(u) => u.clone(),
            None => return Vec::new(),
        };
        let schema = match self.attr_schema.get(&uid) {
            Some(s) => *s,
            None => return Vec::new(),
        };
        if schema != AttributeType::Bool {
            return Vec::new();
        }
        let binding = self.node_columns();
        let col = match binding.iter().find(|(k,_)| *k == uid).map(|(_,c)| c) {
            Some(c) => c,
            None => return Vec::new(),
        };
        match col {
            ColumnData::Bool(vec) => vec.iter().enumerate().filter_map(|(i, v)| v.filter(|&x| x == value).map(|_| i)).collect(),
            _ => Vec::new(),
        }
    }

    /// Filter node indices by string value for a column
    pub fn filter_nodes_by_str(&self, attr_name: String, value: String) -> Vec<usize> {
        let uid = match self.attr_name_to_uid.get(&attr_name) {
            Some(u) => u.clone(),
            None => return Vec::new(),
        };
        let schema = match self.attr_schema.get(&uid) {
            Some(s) => *s,
            None => return Vec::new(),
        };
        if schema != AttributeType::Str {
            return Vec::new();
        }
        let binding = self.node_columns();
        let col = match binding.iter().find(|(k,_)| *k == uid).map(|(_,c)| c) {
            Some(c) => c,
            None => return Vec::new(),
        };
        match col {
            ColumnData::Str(vec) => vec.iter().enumerate().filter_map(|(i, v)| v.as_ref().filter(|x| **x == value).map(|_| i)).collect(),
            _ => Vec::new(),
        }
    }


    /// Get all node attribute names
    pub fn node_attr_names(&self) -> Vec<String> {
        self.attr_name_to_uid.iter().map(|e| e.key().clone()).collect()
    }

    /// Get column stats for node attributes (Python API)
    pub fn get_column_stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        for entry in self.node_attributes.iter() {
            if let Some(name) = self.attr_uid_to_name.get(entry.key()) {
                stats.insert(name.clone(), entry.value().len());
            }
        }
        stats
    }

    /// Rebuild all node bitmaps (Python API)
    pub fn rebuild_node_bitmaps(&self) {
        self.node_value_bitmaps.clear();
        for entry in self.node_attributes.iter() {
            let attr_uid = entry.key().clone();
            for (idx, value) in entry.value().iter() {
                let key = (attr_uid.clone(), value.clone());
                let mut bv = self.node_value_bitmaps.entry(key).or_insert_with(|| BitVec::repeat(false, self.max_node_index.load(std::sync::atomic::Ordering::SeqCst) + 1));
                if *idx >= bv.len() {
                    bv.resize(*idx + 1, false);
                }
                bv.set(*idx, true);
            }
        }
        self.bitmaps_dirty.store(false, std::sync::atomic::Ordering::SeqCst);
    }

    /// Filter nodes by attribute value (returns indices)
    pub fn filter_nodes_by_value(&self, attr_name: String, value: JsonValue) -> Vec<usize> {
        let uid = match self.attr_name_to_uid.get(&attr_name) {
            Some(u) => u.clone(),
            None => return Vec::new(),
        };
        let key = (uid, value);
        if self.bitmaps_dirty.load(std::sync::atomic::Ordering::SeqCst) {
            self.rebuild_node_bitmaps();
        }
        if let Some(bv) = self.node_value_bitmaps.get(&key) {
            bv.iter_ones().collect()
        } else {
            Vec::new()
        }
    }

    // --- Edge attribute support ---

    /// Get all values for an edge attribute by name (Python API)
    pub fn get_edge_attr(&self, attr_name: String) -> Option<HashMap<usize, JsonValue>> {
        let uid = self.attr_name_to_uid.get(&attr_name)?.clone();
        self.edge_attributes.get(&uid).map(|m| m.clone())
    }

    /// Set all values for an edge attribute by name (Python API)
    pub fn set_edge_attr(&self, attr_name: String, data: HashMap<usize, JsonValue>) {
        let uid = self.register_attr(attr_name);
        self.edge_attributes.insert(AttrUID(uid), data);
        self.bitmaps_dirty.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// Get a single value for an edge attribute and entity index
    pub fn get_edge_value(&self, attr_name: String, idx: usize) -> Option<JsonValue> {
        let uid = self.attr_name_to_uid.get(&attr_name)?.clone();
        self.edge_attributes.get(&uid)?.get(&idx).cloned()
    }

    /// Set a single value for an edge attribute and entity index
    pub fn set_edge_value(&self, attr_name: String, idx: usize, value: JsonValue) {
        let uid = self.register_attr(attr_name);
        self.edge_attributes.entry(AttrUID(uid)).or_default().insert(idx, value);
        self.bitmaps_dirty.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// Get all edge attribute names
    pub fn edge_attr_names(&self) -> Vec<String> {
        // Only return names for which there is edge data
        let mut names = Vec::new();
        for entry in self.edge_attributes.iter() {
            if let Some(name) = self.attr_uid_to_name.get(entry.key()) {
                names.push(name.clone());
            }
        }
        names
    }

    /// Get column stats for edge attributes (Python API)
    pub fn get_edge_column_stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        for entry in self.edge_attributes.iter() {
            if let Some(name) = self.attr_uid_to_name.get(entry.key()) {
                stats.insert(name.clone(), entry.value().len());
            }
        }
        stats
    }

    /// Rebuild all edge bitmaps (Python API)
    pub fn rebuild_edge_bitmaps(&self) {
        self.edge_value_bitmaps.clear();
        for entry in self.edge_attributes.iter() {
            let attr_uid = entry.key().clone();
            for (idx, value) in entry.value().iter() {
                let key = (attr_uid.clone(), value.clone());
                let mut bv = self.edge_value_bitmaps.entry(key).or_insert_with(|| BitVec::repeat(false, self.max_edge_index.load(std::sync::atomic::Ordering::SeqCst) + 1));
                if *idx >= bv.len() {
                    bv.resize(*idx + 1, false);
                }
                bv.set(*idx, true);
            }
        }
        // No separate dirty flag for edges; reuse bitmaps_dirty if needed
    }

    /// Filter edges by attribute value (returns indices)
    pub fn filter_edges_by_value(&self, attr_name: String, value: JsonValue) -> Vec<usize> {
        let uid = match self.attr_name_to_uid.get(&attr_name) {
            Some(u) => u.clone(),
            None => return Vec::new(),
        };
        let key = (uid, value);
        // No separate dirty flag for edges; always rebuild for now
        self.rebuild_edge_bitmaps();
        if let Some(bv) = self.edge_value_bitmaps.get(&key) {
            bv.iter_ones().collect()
        } else {
            Vec::new()
        }
    }
}


impl ColumnarStore {
    /// Fast column access for internal use
    /// Fast column access for internal use
    pub fn get_column_internal(&self, kind: ColumnKind, attr: &str) -> Option<ColumnData> {
        if let Some(uid) = self.attr_name_to_uid.get(attr) {
            self.columns.get(&(kind, uid.clone())).map(|col| col.clone())
        } else {
            None
        }
    }

    /// Fast column mutation for internal use
    pub fn set_column_internal(&mut self, attr: &str, values: Vec<Value>) {
        // attr is &str (attribute name); need to resolve to AttrUID and type
        if let Some(uid) = self.attr_name_to_uid.get(attr) {
            if let Some(schema) = self.attr_schema.get(&uid) {
                let kind = ColumnKind::Node; // or infer from context if needed
                let key = (kind, uid.clone());
                let col_data = match *schema {
                    AttributeType::Int => ColumnData::Int(values.into_iter().map(|v| v.as_i64()).map(|x| x.map(Some).unwrap_or(None)).collect()),
                    AttributeType::Float => ColumnData::Float(values.into_iter().map(|v| v.as_f64()).map(|x| x.map(Some).unwrap_or(None)).collect()),
                    AttributeType::Bool => ColumnData::Bool(values.into_iter().map(|v| v.as_bool()).map(|x| x.map(Some).unwrap_or(None)).collect()),
                    AttributeType::Str => ColumnData::Str(values.into_iter().map(|v| v.as_str().map(|s| s.to_string())).collect()),
                    AttributeType::Json => ColumnData::Json(values.into_iter().enumerate().map(|(i, v)| (i, v)).collect()),
                };
                self.columns.insert(key, col_data);
            }
        }
    }

    /// Bulk attribute retrieval (vectorized)
    /// Bulk attribute retrieval (vectorized)
    pub fn bulk_get_internal(&self, keys: &[(ColumnKind, String)]) -> HashMap<(ColumnKind, AttrUID), ColumnData> {
        let mut result = HashMap::new();
        for (kind, attr_name) in keys.iter() {
            if let Some(uid) = self.attr_name_to_uid.get(attr_name) {
                let key = (*kind, uid.clone());
                if let Some(col) = self.columns.get(&key) {
                    result.insert(key, col.clone());
                }
            }
        }
        result
    }

    /// Bulk attribute setting (vectorized)
    pub fn bulk_set_internal(&mut self, data: HashMap<String, Vec<Value>>) {
        for (attr, values) in data {
            // attr is String (attribute name), values is Vec<Value>
            if let Some(uid) = self.attr_name_to_uid.get(&attr) {
                if let Some(schema) = self.attr_schema.get(&uid) {
                    let kind = ColumnKind::Node; // or infer from context if needed
                    let key = (kind, uid.clone());
                    let col_data = match *schema {
                        AttributeType::Int => ColumnData::Int(values.into_iter().map(|v| v.as_i64()).map(|x| x.map(Some).unwrap_or(None)).collect()),
                        AttributeType::Float => ColumnData::Float(values.into_iter().map(|v| v.as_f64()).map(|x| x.map(Some).unwrap_or(None)).collect()),
                        AttributeType::Bool => ColumnData::Bool(values.into_iter().map(|v| v.as_bool()).map(|x| x.map(Some).unwrap_or(None)).collect()),
                        AttributeType::Str => ColumnData::Str(values.into_iter().map(|v| v.as_str().map(|s| s.to_string())).collect()),
                        AttributeType::Json => ColumnData::Json(values.into_iter().enumerate().map(|(i, v)| (i, v)).collect()),
                    };
                    self.columns.insert(key, col_data);
                }
            }
        }
    }

    /// Fast column filtering with SIMD (placeholder)
    pub fn filter_column_internal(&self, _attr: &str, _predicate: &dyn Fn(&Value) -> bool) -> Vec<usize> {
        // Return indices matching predicate (not implemented)
        Vec::new()
    }

    /// Type-safe column access (placeholder)
    pub fn get_typed_column<T: 'static>(&self, _attr: &str) -> Option<&Vec<T>> {
        // Not implemented: would require typed storage
        None
    }

    /// Pre-allocate column space (placeholder)
    pub fn ensure_column_capacity(&mut self, attr: &str, capacity: usize) {
        // You must choose the correct variant based on the attribute type. Here, default to Int for demonstration.
        // attr is &str (attribute name); need to resolve to AttrUID and type
        if let Some(uid) = self.attr_name_to_uid.get(attr) {
            if let Some(schema) = self.attr_schema.get(&uid) {
                let kind = ColumnKind::Node; // or infer from context if needed
                let key = (kind, uid.clone());
                self.columns.entry(key).or_insert_with(|| match *schema {
                    AttributeType::Int => ColumnData::Int(Vec::with_capacity(capacity)),
                    AttributeType::Float => ColumnData::Float(Vec::with_capacity(capacity)),
                    AttributeType::Bool => ColumnData::Bool(Vec::with_capacity(capacity)),
                    AttributeType::Str => ColumnData::Str(Vec::with_capacity(capacity)),
                    AttributeType::Json => ColumnData::Json(HashMap::with_capacity(capacity)),
                });
            }
        }
    }


    /// Get all values for a node attribute by name (Python API)
    pub fn get_node_attr(&self, attr_name: String) -> Option<HashMap<usize, JsonValue>> {
        let uid = self.attr_name_to_uid.get(&attr_name)?.clone();
        self.node_attributes.get(&uid).map(|m| m.clone())
    }

    /// Set all values for a node attribute by name (Python API)
    pub fn set_node_attr(&self, attr_name: String, data: HashMap<usize, JsonValue>) {
        let uid = self.register_attr(attr_name);
        self.node_attributes.insert(AttrUID(uid), data);
        self.bitmaps_dirty.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// Get a single value for a node attribute and entity index
    pub fn get_node_value(&self, attr_name: String, idx: usize) -> Option<JsonValue> {
        let uid = self.attr_name_to_uid.get(&attr_name)?.clone();
        self.node_attributes.get(&uid)?.get(&idx).cloned()
    }

    /// Set a single value for a node attribute and entity index
    pub fn set_node_value(&self, attr_name: String, idx: usize, value: JsonValue) {
        let uid = self.register_attr(attr_name);
        self.node_attributes.entry(AttrUID(uid)).or_default().insert(idx, value);
        self.bitmaps_dirty.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// Filter edges by bool value for a column
    pub fn filter_edges_by_bool(&self, attr_name: String, value: bool) -> Vec<usize> {
        let uid = match self.attr_name_to_uid.get(&attr_name) {
            Some(u) => u.clone(),
            None => return Vec::new(),
        };
        let schema = match self.attr_schema.get(&uid) {
            Some(s) => *s,
            None => return Vec::new(),
        };
        if schema != AttributeType::Bool {
            return Vec::new();
        }
        let binding = self.edge_columns();
        let col = match binding.iter().find(|(k,_)| *k == uid).map(|(_,c)| c) {
            Some(c) => c,
            None => return Vec::new(),
        };
        match col {
            ColumnData::Bool(vec) => vec.iter().enumerate().filter_map(|(i, v)| v.filter(|&x| x == value).map(|_| i)).collect(),
            _ => Vec::new(),
        }
    }

    /// Get edge endpoints (source, target) by edge ID - placeholder implementation
    pub fn edge_endpoints(&self, _edge_id: &crate::graph::types::EdgeId) -> Option<(crate::graph::types::NodeId, crate::graph::types::NodeId)> {
        // TODO: Implement proper edge endpoint storage and retrieval
        // For now, return a placeholder to prevent compilation errors
        None
    }
}
