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

/// Subgraph identifier for entity system
pub type SubgraphId = usize;

/// Universal entity identifier for the GraphEntity trait system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EntityId {
    /// Node entity
    Node(NodeId),
    /// Edge entity
    Edge(EdgeId),
    /// Subgraph entity (base subgraph)
    Subgraph(SubgraphId),
    /// Neighborhood subgraph entity
    Neighborhood(SubgraphId),
    /// Component subgraph entity
    Component(SubgraphId),
    /// Path subgraph entity
    Path(SubgraphId),
    /// Filter subgraph entity
    Filter(SubgraphId),
    /// Meta-node entity (node containing a subgraph)
    MetaNode(NodeId),
}

impl EntityId {
    /// Check if this EntityId represents a valid identifier
    pub fn is_valid(&self) -> bool {
        match self {
            EntityId::Node(id) | EntityId::MetaNode(id) => *id != usize::MAX,
            EntityId::Edge(id) => *id != usize::MAX,
            EntityId::Subgraph(id)
            | EntityId::Neighborhood(id)
            | EntityId::Component(id)
            | EntityId::Path(id)
            | EntityId::Filter(id) => *id != usize::MAX,
        }
    }

    /// Get the underlying numeric ID regardless of entity type
    pub fn numeric_id(&self) -> usize {
        match self {
            EntityId::Node(id) | EntityId::MetaNode(id) => *id,
            EntityId::Edge(id) => *id,
            EntityId::Subgraph(id)
            | EntityId::Neighborhood(id)
            | EntityId::Component(id)
            | EntityId::Path(id)
            | EntityId::Filter(id) => *id,
        }
    }
}

impl Default for EntityId {
    fn default() -> Self {
        EntityId::Node(0)
    }
}

/*
=== GRAPH STRUCTURE TYPES ===
Fundamental graph properties that affect behavior across the entire system.
*/

/// Graph directionality - determines edge interpretation throughout the system
/// DESIGN: This affects adjacency caching, traversal algorithms, and matrix operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphType {
    /// Directed graph - edges have direction (A→B ≠ B→A)
    /// Compatible with NetworkX DiGraph
    Directed,
    /// Undirected graph - edges are bidirectional (A↔B = B↔A)
    /// Compatible with NetworkX Graph
    Undirected,
}

impl Default for GraphType {
    /// Default to undirected for backward compatibility
    fn default() -> Self {
        Self::Undirected
    }
}

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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
    /// Vector of integers (IDs, counts, indices)
    IntVec(Vec<i64>),
    /// Vector of strings (tags, labels, categories)
    TextVec(Vec<String>),
    /// Vector of booleans (flags, masks, selections)
    BoolVec(Vec<bool>),
    /// Boolean flag (active, enabled, etc.)
    Bool(bool),
    /// Memory-optimized compact string for short text values (Memory Optimization 1)
    /// Uses inline storage for strings <= 15 bytes, avoiding heap allocation
    CompactText(CompactString),
    /// Memory-optimized small integer for values that fit in smaller types
    SmallInt(i32),
    /// Byte array for binary data (more efficient than Vec<u8> as AttrValue)
    Bytes(Vec<u8>),
    /// Compressed large text data (Memory Optimization 3)
    CompressedText(CompressedData),
    /// Compressed large float vector (Memory Optimization 3)
    CompressedFloatVec(CompressedData),
    /// Null/missing value - preserves the fact that an attribute is missing
    Null,
    /// Reference to a subgraph stored in GraphPool (for hierarchical meta-nodes)
    SubgraphRef(SubgraphId),
    /// Array of node IDs (for storing node collections as attributes)
    NodeArray(Vec<NodeId>),
    /// Array of edge IDs (for storing edge collections as attributes)
    EdgeArray(Vec<EdgeId>),
    /// JSON-serialized complex data (lists, dicts, nested structures)
    Json(String),
}

/// Custom PartialEq implementation that compares logical content across storage variants
/// This allows Text("hello") to equal CompactText("hello") and CompressedText("hello")
impl PartialEq for AttrValue {
    fn eq(&self, other: &Self) -> bool {
        use AttrValue::*;
        match (self, other) {
            // Exact type matches
            (Float(a), Float(b)) => a == b,
            (Int(a), Int(b)) => a == b,
            (Bool(a), Bool(b)) => a == b,
            (SmallInt(a), SmallInt(b)) => a == b,
            (FloatVec(a), FloatVec(b)) => a == b,
            (IntVec(a), IntVec(b)) => a == b,
            (TextVec(a), TextVec(b)) => a == b,
            (BoolVec(a), BoolVec(b)) => a == b,
            (Bytes(a), Bytes(b)) => a == b,
            (Json(a), Json(b)) => a == b,

            // Cross-type integer comparisons
            (Int(a), SmallInt(b)) => *a == *b as i64,
            (SmallInt(a), Int(b)) => *a as i64 == *b,

            // Cross-type string comparisons - compare content, not storage format
            (Text(a), Text(b)) => a == b,
            (Text(a), CompactText(b)) => a.as_str() == b.as_str(),
            (Text(a), CompressedText(b)) => {
                if let Ok(decompressed) = b.decompress_text() {
                    a.as_str() == decompressed.as_str()
                } else {
                    false
                }
            }
            (CompactText(a), Text(b)) => a.as_str() == b.as_str(),
            (CompactText(a), CompactText(b)) => a.as_str() == b.as_str(),
            (CompactText(a), CompressedText(b)) => {
                if let Ok(decompressed) = b.decompress_text() {
                    a.as_str() == decompressed.as_str()
                } else {
                    false
                }
            }
            (CompressedText(a), Text(b)) => {
                if let Ok(decompressed) = a.decompress_text() {
                    decompressed.as_str() == b.as_str()
                } else {
                    false
                }
            }
            (CompressedText(a), CompactText(b)) => {
                if let Ok(decompressed) = a.decompress_text() {
                    decompressed.as_str() == b.as_str()
                } else {
                    false
                }
            }
            (CompressedText(a), CompressedText(b)) => {
                // For compressed text, try to decompress both and compare
                match (a.decompress_text(), b.decompress_text()) {
                    (Ok(a_text), Ok(b_text)) => a_text == b_text,
                    _ => false,
                }
            }

            // Cross-type float vector comparisons
            (FloatVec(a), CompressedFloatVec(b)) => match b.decompress_float_vec() {
                Ok(decompressed_b) => a == &decompressed_b,
                Err(_) => false,
            },
            (CompressedFloatVec(a), FloatVec(b)) => match a.decompress_float_vec() {
                Ok(decompressed_a) => &decompressed_a == b,
                Err(_) => false,
            },
            (CompressedFloatVec(a), CompressedFloatVec(b)) => {
                match (a.decompress_float_vec(), b.decompress_float_vec()) {
                    (Ok(decompressed_a), Ok(decompressed_b)) => decompressed_a == decompressed_b,
                    _ => false,
                }
            }

            // Null comparisons - Null only equals Null
            (Null, Null) => true,

            // All other combinations are not equal
            _ => false,
        }
    }
}

/// Memory-efficient string storage that avoids heap allocation for short strings
/// MEMORY OPTIMIZATION: Stores strings <= 15 bytes inline, larger ones on heap
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum CompactString {
    /// Inline storage for strings up to 15 bytes (fits in 16 bytes total)
    Inline { data: [u8; 15], len: u8 },
    /// Heap storage for longer strings
    Heap(String),
}

impl CompactString {
    /// Create a new compact string from a regular string
    pub fn new(s: &str) -> Self {
        let bytes = s.as_bytes();
        if bytes.len() <= 15 {
            let mut data = [0u8; 15];
            data[..bytes.len()].copy_from_slice(bytes);
            CompactString::Inline {
                data,
                len: bytes.len() as u8,
            }
        } else {
            CompactString::Heap(s.to_string())
        }
    }

    /// Get the string content as a &str
    pub fn as_str(&self) -> &str {
        match self {
            CompactString::Inline { data, len } => {
                std::str::from_utf8(&data[..*len as usize]).unwrap()
            }
            CompactString::Heap(s) => s.as_str(),
        }
    }

    /// Get the memory usage in bytes
    pub fn memory_size(&self) -> usize {
        match self {
            CompactString::Inline { .. } => 16, // 15 bytes data + 1 byte length
            CompactString::Heap(s) => std::mem::size_of::<String>() + s.capacity(),
        }
    }
}

// IMPLEMENTATION NOTES:
// - Hash trait needed for deduplication in change tracking
// - Use f32.to_bits() for consistent hashing of floats
// - Sort hash inputs to ensure deterministic ordering
impl Hash for AttrValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            AttrValue::Float(f) => {
                0u8.hash(state); // Discriminant for Float variant
                f.to_bits().hash(state); // Use to_bits() for consistent float hashing
            }
            AttrValue::Int(i) => {
                1u8.hash(state); // Discriminant for Int variant
                i.hash(state);
            }
            AttrValue::Text(s) => {
                2u8.hash(state); // Discriminant for Text variant
                s.hash(state);
            }
            AttrValue::FloatVec(v) => {
                3u8.hash(state); // Discriminant for FloatVec variant
                v.len().hash(state); // Hash length first
                for f in v {
                    f.to_bits().hash(state); // Hash each float using to_bits()
                }
            }
            AttrValue::IntVec(v) => {
                4u8.hash(state); // Discriminant for IntVec variant
                v.hash(state);
            }
            AttrValue::TextVec(v) => {
                5u8.hash(state); // Discriminant for TextVec variant
                v.hash(state);
            }
            AttrValue::BoolVec(v) => {
                6u8.hash(state); // Discriminant for BoolVec variant
                v.hash(state);
            }
            AttrValue::Bool(b) => {
                7u8.hash(state); // Discriminant for Bool variant
                b.hash(state);
            }
            AttrValue::CompactText(cs) => {
                8u8.hash(state); // Discriminant for CompactText variant
                cs.as_str().hash(state);
            }
            AttrValue::SmallInt(i) => {
                9u8.hash(state); // Discriminant for SmallInt variant
                i.hash(state);
            }
            AttrValue::Bytes(bytes) => {
                10u8.hash(state); // Discriminant for Bytes variant
                bytes.hash(state);
            }
            AttrValue::CompressedText(cd) => {
                11u8.hash(state); // Discriminant for CompressedText variant
                cd.data.hash(state);
                cd.original_size.hash(state);
            }
            AttrValue::CompressedFloatVec(cd) => {
                12u8.hash(state); // Discriminant for CompressedFloatVec variant
                cd.data.hash(state);
                cd.original_size.hash(state);
            }
            AttrValue::Null => {
                13u8.hash(state); // Discriminant for Null variant
                                  // No additional data to hash for Null
            }
            AttrValue::SubgraphRef(id) => {
                14u8.hash(state); // Discriminant for SubgraphRef variant
                id.hash(state);
            }
            AttrValue::NodeArray(nodes) => {
                15u8.hash(state); // Discriminant for NodeArray variant
                nodes.hash(state);
            }
            AttrValue::EdgeArray(edges) => {
                16u8.hash(state); // Discriminant for EdgeArray variant
                edges.hash(state);
            }
            AttrValue::Json(json_str) => {
                17u8.hash(state); // Discriminant for Json variant
                json_str.hash(state);
            }
        }
    }
}

// Manual Eq implementation to handle f32 comparison
impl Eq for AttrValue {}

impl Default for AttrValue {
    fn default() -> Self {
        AttrValue::Null
    }
}

impl PartialOrd for AttrValue {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for AttrValue {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        use AttrValue::*;

        match (self, other) {
            // Same variant comparisons
            (Int(a), Int(b)) => a.cmp(b),
            (SmallInt(a), SmallInt(b)) => a.cmp(b),
            (Float(a), Float(b)) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
            (Bool(a), Bool(b)) => a.cmp(b),
            (Text(a), Text(b)) => AttrValue::cmp_text_content(a, b),
            (CompactText(a), CompactText(b)) => AttrValue::cmp_text_content(a.as_str(), b.as_str()),
            (FloatVec(a), FloatVec(b)) => a.len().cmp(&b.len()).then_with(|| {
                // Compare vectors lexicographically
                for (va, vb) in a.iter().zip(b.iter()) {
                    match va.partial_cmp(vb).unwrap_or(Ordering::Equal) {
                        Ordering::Equal => continue,
                        other => return other,
                    }
                }
                Ordering::Equal
            }),
            (Bytes(a), Bytes(b)) => a.cmp(b),
            (CompressedText(a), CompressedText(b)) => {
                match (a.decompress_text(), b.decompress_text()) {
                    (Ok(lhs), Ok(rhs)) => AttrValue::cmp_text_content(lhs.as_str(), rhs.as_str()),
                    _ => a.data.cmp(&b.data),
                }
            }
            (CompressedFloatVec(a), CompressedFloatVec(b)) => a.data.cmp(&b.data),

            // Cross-type comparisons - order by type discriminant first
            (Int(a), SmallInt(b)) => a.cmp(&(*b as i64)),
            (SmallInt(a), Int(b)) => (*a as i64).cmp(b),

            // Text comparisons across storage types
            (Text(a), CompactText(b)) => AttrValue::cmp_text_content(a.as_str(), b.as_str()),
            (CompactText(a), Text(b)) => AttrValue::cmp_text_content(a.as_str(), b.as_str()),
            (Text(a), CompressedText(b)) => match b.decompress_text() {
                Ok(text) => AttrValue::cmp_text_content(a.as_str(), text.as_str()),
                Err(_) => std::cmp::Ordering::Equal,
            },
            (CompressedText(a), Text(b)) => match a.decompress_text() {
                Ok(text) => AttrValue::cmp_text_content(text.as_str(), b.as_str()),
                Err(_) => std::cmp::Ordering::Equal,
            },
            (CompactText(a), CompressedText(b)) => match b.decompress_text() {
                Ok(text) => AttrValue::cmp_text_content(a.as_str(), text.as_str()),
                Err(_) => std::cmp::Ordering::Equal,
            },
            (CompressedText(a), CompactText(b)) => match a.decompress_text() {
                Ok(text) => AttrValue::cmp_text_content(text.as_str(), b.as_str()),
                Err(_) => std::cmp::Ordering::Equal,
            },

            // Different types - order by discriminant (type priority)
            _ => self.type_discriminant().cmp(&other.type_discriminant()),
        }
    }
}

impl AttrValue {
    /// Get type discriminant for ordering different types
    fn type_discriminant(&self) -> u8 {
        match self {
            AttrValue::Null => 0, // Null sorts first
            AttrValue::Int(_) | AttrValue::SmallInt(_) => 1,
            AttrValue::Float(_) => 2,
            AttrValue::Bool(_) => 3,
            AttrValue::Text(_) | AttrValue::CompactText(_) | AttrValue::CompressedText(_) => 4,
            AttrValue::FloatVec(_) | AttrValue::CompressedFloatVec(_) => 5,
            AttrValue::IntVec(_) => 6,
            AttrValue::TextVec(_) => 7,
            AttrValue::BoolVec(_) => 8,
            AttrValue::Bytes(_) => 9,
            AttrValue::SubgraphRef(_) => 10,
            AttrValue::NodeArray(_) => 11,
            AttrValue::EdgeArray(_) => 12,
            AttrValue::Json(_) => 13,
        }
    }
}

impl AttrValue {
    fn cmp_text_content(lhs: &str, rhs: &str) -> std::cmp::Ordering {
        use std::cmp::Ordering;

        fn parse_numeric(input: &str) -> Option<f64> {
            let trimmed = input.trim();
            if trimmed.is_empty() {
                None
            } else {
                trimmed.parse::<f64>().ok()
            }
        }

        match (parse_numeric(lhs), parse_numeric(rhs)) {
            (Some(a), Some(b)) => a.partial_cmp(&b).unwrap_or(Ordering::Equal),
            _ => lhs.cmp(rhs),
        }
    }
}

/*
=== NUMERIC TYPE SYSTEM ===
Type infrastructure for the advanced NumArray type system.
This supports seamless conversions between different numeric representations
and enables type-aware operations across BaseArray, NumArray, and Matrix.
*/

/// Numeric type classification for the advanced type system
/// This enum represents the fundamental numeric types that can be stored
/// in NumArrays and supports automatic type inference and conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NumericType {
    /// 32-bit signed integer
    Int32,
    /// 64-bit signed integer  
    Int64,
    /// 32-bit floating point
    Float32,
    /// 64-bit floating point
    Float64,
    /// Boolean values (can be treated as 0/1)
    Bool,
}

impl NumericType {
    /// Check if this type can be losslessly converted to another type
    pub fn can_convert_to(&self, target: NumericType) -> bool {
        use NumericType::*;
        match (self, target) {
            // Same types can always convert
            (a, b) if *a == b => true,

            // Integer promotions (lossless)
            (Int32, Int64) => true,
            (Int32, Float64) => true, // i32 can fit in f64 exactly
            (Bool, Int32) | (Bool, Int64) => true,
            (Bool, Float32) | (Bool, Float64) => true,

            // Float promotions (lossless)
            (Float32, Float64) => true,

            // All other conversions are potentially lossy
            _ => false,
        }
    }

    /// Get the "highest" type that can represent both types without loss
    pub fn promote_with(&self, other: NumericType) -> NumericType {
        use NumericType::*;
        match (*self, other) {
            // Same types
            (a, b) if a == b => a,

            // Bool promotes to anything
            (Bool, other) | (other, Bool) => other,

            // Float64 is the ultimate promotion target
            (Float64, _) | (_, Float64) => Float64,
            (Float32, _) | (_, Float32) => Float32,

            // Integer promotions
            (Int64, _) | (_, Int64) => Int64,
            // Note: (Int32, _) | (_, Int32) => Int32 is unreachable due to above patterns
            // If we reach here, both types must be Int32 (caught by same types) or lower precision
            _ => Int32, // Default fallback for any remaining cases
        }
    }

    /// Get the size in bytes of this numeric type
    pub fn size_bytes(&self) -> usize {
        match self {
            NumericType::Bool => 1,
            NumericType::Int32 => 4,
            NumericType::Float32 => 4,
            NumericType::Int64 => 8,
            NumericType::Float64 => 8,
        }
    }

    /// Check if this type is an integer type
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            NumericType::Int32 | NumericType::Int64 | NumericType::Bool
        )
    }

    /// Check if this type is a floating point type
    pub fn is_float(&self) -> bool {
        matches!(self, NumericType::Float32 | NumericType::Float64)
    }
}

impl AttrValue {
    /// Get the numeric type of this AttrValue, if it represents a numeric value
    pub fn numeric_type(&self) -> Option<NumericType> {
        match self {
            AttrValue::Bool(_) => Some(NumericType::Bool),
            AttrValue::SmallInt(_) => Some(NumericType::Int32),
            AttrValue::Int(_) => Some(NumericType::Int64),
            AttrValue::Float(_) => Some(NumericType::Float32),
            _ => None,
        }
    }

    /// Check if this AttrValue represents a numeric value
    pub fn is_numeric(&self) -> bool {
        self.numeric_type().is_some()
    }

    /// Convert this AttrValue to a specific numeric type, if possible
    #[allow(dead_code)]
    pub fn to_numeric_type(&self, target: NumericType) -> Option<AttrValue> {
        match (self, target) {
            // Bool conversions
            (AttrValue::Bool(b), NumericType::Bool) => Some(AttrValue::Bool(*b)),
            (AttrValue::Bool(b), NumericType::Int32) => {
                Some(AttrValue::SmallInt(if *b { 1 } else { 0 }))
            }
            (AttrValue::Bool(b), NumericType::Int64) => {
                Some(AttrValue::Int(if *b { 1 } else { 0 }))
            }
            (AttrValue::Bool(b), NumericType::Float32) => {
                Some(AttrValue::Float(if *b { 1.0 } else { 0.0 }))
            }
            (AttrValue::Bool(b), NumericType::Float64) => {
                // Note: We store f32 in AttrValue, so convert to f32
                Some(AttrValue::Float(if *b { 1.0 } else { 0.0 }))
            }

            // SmallInt (i32) conversions
            (AttrValue::SmallInt(i), NumericType::Bool) => Some(AttrValue::Bool(*i != 0)),
            (AttrValue::SmallInt(i), NumericType::Int32) => Some(AttrValue::SmallInt(*i)),
            (AttrValue::SmallInt(i), NumericType::Int64) => Some(AttrValue::Int(*i as i64)),
            (AttrValue::SmallInt(i), NumericType::Float32) => Some(AttrValue::Float(*i as f32)),
            (AttrValue::SmallInt(i), NumericType::Float64) => Some(AttrValue::Float(*i as f32)), // Still stored as f32

            // Int (i64) conversions
            (AttrValue::Int(i), NumericType::Bool) => Some(AttrValue::Bool(*i != 0)),
            (AttrValue::Int(i), NumericType::Int32) => {
                if *i >= i32::MIN as i64 && *i <= i32::MAX as i64 {
                    Some(AttrValue::SmallInt(*i as i32))
                } else {
                    None // Overflow
                }
            }
            (AttrValue::Int(i), NumericType::Int64) => Some(AttrValue::Int(*i)),
            (AttrValue::Int(i), NumericType::Float32) => Some(AttrValue::Float(*i as f32)), // May lose precision
            (AttrValue::Int(i), NumericType::Float64) => Some(AttrValue::Float(*i as f32)), // Still stored as f32

            // Float (f32) conversions
            (AttrValue::Float(f), NumericType::Bool) => Some(AttrValue::Bool(*f != 0.0)),
            (AttrValue::Float(f), NumericType::Int32) => {
                let rounded = f.round();
                if rounded >= i32::MIN as f32 && rounded <= i32::MAX as f32 {
                    Some(AttrValue::SmallInt(rounded as i32))
                } else {
                    None // Overflow
                }
            }
            (AttrValue::Float(f), NumericType::Int64) => {
                let rounded = f.round();
                if rounded.is_finite() {
                    Some(AttrValue::Int(rounded as i64))
                } else {
                    None // Infinity or NaN
                }
            }
            (AttrValue::Float(f), NumericType::Float32) => Some(AttrValue::Float(*f)),
            (AttrValue::Float(f), NumericType::Float64) => Some(AttrValue::Float(*f)), // Still stored as f32

            _ => None, // Non-numeric input
        }
    }
}

/// Compressed data storage for large values (Memory Optimization 3)
/// Uses simple run-length encoding and basic compression
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CompressedData {
    /// Compressed bytes
    pub data: Vec<u8>,
    /// Original size before compression
    original_size: usize,
    /// Compression algorithm used
    algorithm: CompressionAlgorithm,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum CompressionAlgorithm {
    /// No compression (passthrough)
    None,
    /// Simple run-length encoding for repetitive data
    RunLength,
    /// Basic LZ77-style compression for text
    Basic,
}

impl CompressedData {
    /// Compress text data
    pub fn compress_text(text: &str) -> Self {
        let bytes = text.as_bytes();

        // For small text, don't compress
        if bytes.len() < 100 {
            return Self {
                data: bytes.to_vec(),
                original_size: bytes.len(),
                algorithm: CompressionAlgorithm::None,
            };
        }

        // Simple run-length encoding for repetitive text
        let compressed = Self::run_length_encode(bytes);

        // Only use compression if it actually saves space
        if compressed.len() < bytes.len() {
            Self {
                data: compressed,
                original_size: bytes.len(),
                algorithm: CompressionAlgorithm::RunLength,
            }
        } else {
            Self {
                data: bytes.to_vec(),
                original_size: bytes.len(),
                algorithm: CompressionAlgorithm::None,
            }
        }
    }

    /// Compress float vector data
    pub fn compress_float_vec(vec: &[f32]) -> Self {
        let bytes = unsafe {
            std::slice::from_raw_parts(vec.as_ptr() as *const u8, std::mem::size_of_val(vec))
        };

        // For small vectors, don't compress
        if vec.len() < 25 {
            return Self {
                data: bytes.to_vec(),
                original_size: bytes.len(),
                algorithm: CompressionAlgorithm::None,
            };
        }

        // Run-length encoding can work well for sparse vectors
        let compressed = Self::run_length_encode(bytes);

        if compressed.len() < bytes.len() {
            Self {
                data: compressed,
                original_size: bytes.len(),
                algorithm: CompressionAlgorithm::RunLength,
            }
        } else {
            Self {
                data: bytes.to_vec(),
                original_size: bytes.len(),
                algorithm: CompressionAlgorithm::None,
            }
        }
    }

    /// Decompress to text
    pub fn decompress_text(&self) -> Result<String, &'static str> {
        let bytes = match self.algorithm {
            CompressionAlgorithm::None => self.data.clone(),
            CompressionAlgorithm::RunLength => Self::run_length_decode(&self.data)?,
            CompressionAlgorithm::Basic => return Err("Basic compression not implemented"),
        };

        String::from_utf8(bytes).map_err(|_| "Invalid UTF-8")
    }

    /// Decompress to float vector
    pub fn decompress_float_vec(&self) -> Result<Vec<f32>, &'static str> {
        let bytes = match self.algorithm {
            CompressionAlgorithm::None => self.data.clone(),
            CompressionAlgorithm::RunLength => Self::run_length_decode(&self.data)?,
            CompressionAlgorithm::Basic => return Err("Basic compression not implemented"),
        };

        if bytes.len() % std::mem::size_of::<f32>() != 0 {
            return Err("Invalid float vector data");
        }

        let float_count = bytes.len() / std::mem::size_of::<f32>();
        let mut result = Vec::with_capacity(float_count);

        unsafe {
            let float_ptr = bytes.as_ptr() as *const f32;
            for i in 0..float_count {
                result.push(*float_ptr.add(i));
            }
        }

        Ok(result)
    }

    /// Simple run-length encoding
    fn run_length_encode(input: &[u8]) -> Vec<u8> {
        if input.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::new();
        let mut current_byte = input[0];
        let mut count = 1u8;

        for &byte in &input[1..] {
            if byte == current_byte && count < 255 {
                count += 1;
            } else {
                result.push(count);
                result.push(current_byte);
                current_byte = byte;
                count = 1;
            }
        }

        // Add final run
        result.push(count);
        result.push(current_byte);

        result
    }

    /// Decode run-length encoded data
    fn run_length_decode(input: &[u8]) -> Result<Vec<u8>, &'static str> {
        if !input.len().is_multiple_of(2) {
            return Err("Invalid run-length encoded data");
        }

        let mut result = Vec::new();

        for chunk in input.chunks_exact(2) {
            let count = chunk[0];
            let byte = chunk[1];

            for _ in 0..count {
                result.push(byte);
            }
        }

        Ok(result)
    }

    /// Get compression ratio (compressed_size / original_size)
    pub fn compression_ratio(&self) -> f32 {
        if self.original_size == 0 {
            return 1.0;
        }
        self.data.len() as f32 / self.original_size as f32
    }

    /// Get memory usage in bytes
    pub fn memory_size(&self) -> usize {
        std::mem::size_of::<Self>() + self.data.capacity()
    }
}

impl Hash for CompactString {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_str().hash(state);
    }
}

/// Type enumeration for AttrValue variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttrValueType {
    Float,
    Int,
    Text,
    FloatVec,
    IntVec,
    TextVec,
    BoolVec,
    Bool,
    CompactText,
    SmallInt,
    Bytes,
    CompressedText,
    CompressedFloatVec,
    Null,
    SubgraphRef,
    NodeArray,
    EdgeArray,
    Json,
}

impl AttrValueType {
    /// Check if this type is numeric
    pub fn is_numeric(&self) -> bool {
        matches!(
            self,
            AttrValueType::Float | AttrValueType::Int | AttrValueType::SmallInt
        )
    }
}

impl std::fmt::Display for AttrValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AttrValue::Float(val) => {
                if val.is_nan() {
                    write!(f, "nan")
                } else {
                    write!(f, "{}", val)
                }
            }
            AttrValue::Int(val) => write!(f, "{}", val),
            AttrValue::Text(val) => write!(f, "{}", val),
            AttrValue::Bool(val) => write!(f, "{}", val),
            AttrValue::SmallInt(val) => write!(f, "{}", val),
            AttrValue::CompactText(val) => write!(f, "{}", val.as_str()),
            AttrValue::FloatVec(val) => write!(f, "{:?}", val),
            AttrValue::IntVec(val) => write!(f, "{:?}", val),
            AttrValue::TextVec(val) => write!(f, "{:?}", val),
            AttrValue::BoolVec(val) => write!(f, "{:?}", val),
            AttrValue::Bytes(val) => write!(f, "{:?}", val),
            AttrValue::CompressedText(val) => match val.decompress_text() {
                Ok(text) => write!(f, "{}", text),
                Err(_) => write!(f, "[compressed text]"),
            },
            AttrValue::CompressedFloatVec(val) => match val.decompress_float_vec() {
                Ok(vec) => write!(f, "{:?}", vec),
                Err(_) => write!(f, "[compressed float vec]"),
            },
            AttrValue::Null => write!(f, "nan"),
            AttrValue::SubgraphRef(id) => write!(f, "SubgraphRef({})", id),
            AttrValue::NodeArray(nodes) => write!(f, "NodeArray({:?})", nodes),
            AttrValue::EdgeArray(edges) => write!(f, "EdgeArray({:?})", edges),
            AttrValue::Json(json) => write!(f, "Json({})", json),
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
            AttrValue::IntVec(_) => "IntVec",
            AttrValue::TextVec(_) => "TextVec",
            AttrValue::BoolVec(_) => "BoolVec",
            AttrValue::Bool(_) => "Bool",
            AttrValue::CompactText(_) => "CompactText",
            AttrValue::SmallInt(_) => "SmallInt",
            AttrValue::Bytes(_) => "Bytes",
            AttrValue::CompressedText(_) => "CompressedText",
            AttrValue::CompressedFloatVec(_) => "CompressedFloatVec",
            AttrValue::Null => "Null",
            AttrValue::SubgraphRef(_) => "SubgraphRef",
            AttrValue::NodeArray(_) => "NodeArray",
            AttrValue::EdgeArray(_) => "EdgeArray",
            AttrValue::Json(_) => "Json",
        }
    }

    /// Get the type enum for this value
    pub fn dtype(&self) -> AttrValueType {
        match self {
            AttrValue::Float(_) => AttrValueType::Float,
            AttrValue::Int(_) => AttrValueType::Int,
            AttrValue::Text(_) => AttrValueType::Text,
            AttrValue::FloatVec(_) => AttrValueType::FloatVec,
            AttrValue::IntVec(_) => AttrValueType::IntVec,
            AttrValue::TextVec(_) => AttrValueType::TextVec,
            AttrValue::BoolVec(_) => AttrValueType::BoolVec,
            AttrValue::Bool(_) => AttrValueType::Bool,
            AttrValue::CompactText(_) => AttrValueType::CompactText,
            AttrValue::SmallInt(_) => AttrValueType::SmallInt,
            AttrValue::Bytes(_) => AttrValueType::Bytes,
            AttrValue::CompressedText(_) => AttrValueType::CompressedText,
            AttrValue::CompressedFloatVec(_) => AttrValueType::CompressedFloatVec,
            AttrValue::Null => AttrValueType::Null,
            AttrValue::SubgraphRef(_) => AttrValueType::SubgraphRef,
            AttrValue::NodeArray(_) => AttrValueType::NodeArray,
            AttrValue::EdgeArray(_) => AttrValueType::EdgeArray,
            AttrValue::Json(_) => AttrValueType::Json,
        }
    }

    /// Calculate memory usage in bytes (Memory Optimization 1)
    pub fn memory_size(&self) -> usize {
        match self {
            AttrValue::Float(_) => std::mem::size_of::<f32>(),
            AttrValue::Int(_) => std::mem::size_of::<i64>(),
            AttrValue::Text(s) => std::mem::size_of::<String>() + s.capacity(),
            AttrValue::FloatVec(v) => {
                std::mem::size_of::<Vec<f32>>() + v.capacity() * std::mem::size_of::<f32>()
            }
            AttrValue::IntVec(v) => {
                std::mem::size_of::<Vec<i64>>() + v.capacity() * std::mem::size_of::<i64>()
            }
            AttrValue::TextVec(v) => {
                std::mem::size_of::<Vec<String>>()
                    + v.iter()
                        .map(|s| std::mem::size_of::<String>() + s.capacity())
                        .sum::<usize>()
            }
            AttrValue::BoolVec(v) => {
                std::mem::size_of::<Vec<bool>>() + v.capacity() * std::mem::size_of::<bool>()
            }
            AttrValue::Bool(_) => std::mem::size_of::<bool>(),
            AttrValue::CompactText(cs) => cs.memory_size(),
            AttrValue::SmallInt(_) => std::mem::size_of::<i32>(),
            AttrValue::Bytes(b) => std::mem::size_of::<Vec<u8>>() + b.capacity(),
            AttrValue::CompressedText(cd) => cd.memory_size(),
            AttrValue::CompressedFloatVec(cd) => cd.memory_size(),
            AttrValue::Null => 0, // Null values take no memory
            AttrValue::SubgraphRef(_) => std::mem::size_of::<crate::types::SubgraphId>(),
            AttrValue::NodeArray(nodes) => {
                std::mem::size_of::<Vec<crate::types::NodeId>>()
                    + nodes.capacity() * std::mem::size_of::<crate::types::NodeId>()
            }
            AttrValue::EdgeArray(edges) => {
                std::mem::size_of::<Vec<crate::types::EdgeId>>()
                    + edges.capacity() * std::mem::size_of::<crate::types::EdgeId>()
            }
            AttrValue::Json(s) => std::mem::size_of::<String>() + s.capacity(),
        }
    }

    /// Check if this value can be stored more efficiently as a compact variant
    pub fn can_optimize(&self) -> bool {
        match self {
            AttrValue::Text(s) if s.len() <= 15 => true,
            AttrValue::Text(s) if s.len() > 100 => true, // Can compress large text
            AttrValue::Int(i) if *i >= i32::MIN as i64 && *i <= i32::MAX as i64 => true,
            AttrValue::FloatVec(v) if v.len() > 25 => true, // Can compress large vectors
            _ => false,
        }
    }

    /// Convert to a more memory-efficient variant if possible (Memory Optimization 1 & 3)
    pub fn optimize(self) -> Self {
        match self {
            AttrValue::Text(s) if s.len() <= 15 => AttrValue::CompactText(CompactString::new(&s)),
            AttrValue::Text(s) if s.len() > 100 => {
                AttrValue::CompressedText(CompressedData::compress_text(&s))
            }
            AttrValue::Int(i) if i >= i32::MIN as i64 && i <= i32::MAX as i64 => {
                AttrValue::SmallInt(i as i32)
            }
            AttrValue::FloatVec(v) if v.len() > 25 => {
                AttrValue::CompressedFloatVec(CompressedData::compress_float_vec(&v))
            }
            other => other,
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
            AttrValue::SmallInt(i) => Some(*i as i64),
            _ => None,
        }
    }

    pub fn as_text(&self) -> Option<&str> {
        match self {
            AttrValue::Text(s) => Some(s),
            AttrValue::CompactText(cs) => Some(cs.as_str()),
            // Note: Compressed text requires decompression, so we can't return &str
            _ => None,
        }
    }

    /// Get text content, decompressing if necessary
    pub fn get_text(&self) -> Option<String> {
        match self {
            AttrValue::Text(s) => Some(s.clone()),
            AttrValue::CompactText(cs) => Some(cs.as_str().to_string()),
            AttrValue::CompressedText(cd) => cd.decompress_text().ok(),
            _ => None,
        }
    }

    pub fn as_float_vec(&self) -> Option<&[f32]> {
        match self {
            AttrValue::FloatVec(v) => Some(v),
            // Note: Compressed vectors require decompression, so we can't return &[f32]
            _ => None,
        }
    }

    /// Get float vector content, decompressing if necessary
    pub fn get_float_vec(&self) -> Option<Vec<f32>> {
        match self {
            AttrValue::FloatVec(v) => Some(v.clone()),
            AttrValue::CompressedFloatVec(cd) => cd.decompress_float_vec().ok(),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            AttrValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            AttrValue::Bytes(b) => Some(b),
            _ => None,
        }
    }
}

/// Memory usage statistics (Memory Optimization 4)
#[derive(Debug, Clone)]
pub struct MemoryStatistics {
    /// Pool memory usage in bytes
    pub pool_memory_bytes: usize,
    /// Space memory usage in bytes
    pub space_memory_bytes: usize,
    /// History memory usage in bytes
    pub history_memory_bytes: usize,
    /// Change tracker memory usage in bytes
    pub change_tracker_memory_bytes: usize,
    /// Total memory usage in bytes
    pub total_memory_bytes: usize,
    /// Total memory usage in megabytes
    pub total_memory_mb: f64,
    /// Memory efficiency metrics
    pub memory_efficiency: MemoryEfficiency,
    /// Compression statistics
    pub compression_stats: CompressionStatistics,
}

/// Memory efficiency metrics
#[derive(Debug, Clone)]
pub struct MemoryEfficiency {
    /// Average bytes per node
    pub bytes_per_node: f64,
    /// Average bytes per edge
    pub bytes_per_edge: f64,
    /// Average bytes per entity (node or edge)
    pub bytes_per_entity: f64,
    /// Memory overhead ratio
    pub overhead_ratio: f64,
    /// Cache efficiency (0.0 to 1.0)
    pub cache_efficiency: f64,
}

/// Data compression statistics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CompressionStatistics {
    /// Number of compressed attributes
    pub compressed_attributes: usize,
    /// Total number of attributes
    pub total_attributes: usize,
    /// Average compression ratio
    pub average_compression_ratio: f32,
    /// Memory saved through compression in bytes
    pub memory_saved_bytes: usize,
    /// Memory saved as percentage
    pub memory_saved_percentage: f64,
}
