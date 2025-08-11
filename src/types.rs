//! Core type definitions for the graph library.
//! 
//! DESIGN PRINCIPLE: Keep types simple and focused. The Graph should be the main
//! manager that coordinates between components, not layers calling layers.

use std::hash::{Hash, Hasher};

/*
=== FUNDAMENTAL TYPES ===
These are the core primitives that everything else builds on.
Keep them simple, efficient, and well-documented.
*/

/// Node identifier - should be opaque, incrementing, and reusable after deletion
/// DESIGN: Use usize for cache-friendly indexing into arrays/vectors
pub type NodeId = usize;

/// Edge identifier - same principles as NodeId
pub type EdgeId = usize;

/// Attribute name - human-readable string key for node/edge properties
pub type AttrName = String;

/// State identifier for version control - should be globally unique and ordered
/// DESIGN: Use u64 for plenty of space and natural ordering
pub type StateId = u64;

/// Branch name for git-like workflow
pub type BranchName = String;

/*
=== ATTRIBUTE VALUE SYSTEM ===
This is the heart of the flexible attribute system. Should support:
- Common data types (numbers, strings, bools)
- Vector embeddings for ML workloads  
- Efficient hashing for deduplication
- JSON-like flexibility without the performance cost
*/

/// Efficient storage for attribute values supporting multiple data types
/// DESIGN: Enum dispatch is fast, Hash implementation handles f32 properly
#[derive(Debug, Clone, PartialEq)]
pub enum AttrValue {
    /// 32-bit float (embeddings, coordinates, ML features)
    Float(f32),
    /// 64-bit signed integer (counts, IDs, timestamps)
    Int(i64),
    /// UTF-8 string (names, descriptions, categories)
    Text(String),
    /// Vector of floats (embeddings, coordinates, feature vectors)
    /// PERFORMANCE: Vec<f32> is more cache-friendly than Vec<AttrValue>
    FloatVec(Vec<f32>),
    /// Boolean flag (active, enabled, etc.)
    Bool(bool),
}

// IMPLEMENTATION NOTES:
// - Hash trait needed for deduplication in change tracking
// - Use f32.to_bits() for consistent hashing of floats
// - Sort hash inputs to ensure deterministic ordering
impl Hash for AttrValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            AttrValue::Float(f) => {
                0u8.hash(state);  // Discriminant for Float variant
                f.to_bits().hash(state);  // Use to_bits() for consistent float hashing
            }
            AttrValue::Int(i) => {
                1u8.hash(state);  // Discriminant for Int variant
                i.hash(state);
            }
            AttrValue::Text(s) => {
                2u8.hash(state);  // Discriminant for Text variant
                s.hash(state);
            }
            AttrValue::FloatVec(v) => {
                3u8.hash(state);  // Discriminant for FloatVec variant
                v.len().hash(state);  // Hash length first
                for f in v {
                    f.to_bits().hash(state);  // Hash each float using to_bits()
                }
            }
            AttrValue::Bool(b) => {
                4u8.hash(state);  // Discriminant for Bool variant
                b.hash(state);
            }
        }
    }
}

impl AttrValue {
    /// Get runtime type information as string
    pub fn type_name(&self) -> &'static str {
        match self {
            AttrValue::Float(_) => "Float",
            AttrValue::Int(_) => "Int", 
            AttrValue::Text(_) => "Text",
            AttrValue::FloatVec(_) => "FloatVec",
            AttrValue::Bool(_) => "Bool",
        }
    }
    
    /// Try to convert to specific type with error handling
    pub fn as_float(&self) -> Option<f32> {
        match self {
            AttrValue::Float(f) => Some(*f),
            _ => None,
        }
    }
    
    pub fn as_int(&self) -> Option<i64> {
        match self {
            AttrValue::Int(i) => Some(*i),
            _ => None,
        }
    }
    
    pub fn as_text(&self) -> Option<&str> {
        match self {
            AttrValue::Text(s) => Some(s),
            _ => None,
        }
    }
    
    pub fn as_float_vec(&self) -> Option<&[f32]> {
        match self {
            AttrValue::FloatVec(v) => Some(v),
            _ => None,
        }
    }
    
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            AttrValue::Bool(b) => Some(*b),
            _ => None,
        }
    }
}
