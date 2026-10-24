//! Versioned logical property encoding, independent of the in-memory layout.
use crate::{AttrValue, GraphError};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Stable property tags used by version 2 graph exports.
/// Float bit patterns retain NaNs, infinities and signed zero exactly.
#[derive(Serialize, Deserialize)]
#[serde(
    tag = "type",
    content = "value",
    rename_all = "snake_case",
    deny_unknown_fields
)]
enum StoredValue {
    Null,
    Int(i64),
    Float(u32),
    Float64(u64),
    Float64Vec(Vec<u64>),
    UtcInstant(crate::types::UtcInstant),
    Date(crate::types::CalendarDate),
    Duration(crate::types::FixedDuration),
    Bool(bool),
    Text(String),
    FloatVec(Vec<u32>),
    IntVec(Vec<i64>),
    BoolVec(Vec<bool>),
    TextVec(Vec<String>),
    Bytes(Vec<u8>),
    NodeArray(Vec<usize>),
    EdgeArray(Vec<usize>),
    Json(String),
}

fn invalid(message: impl std::fmt::Display) -> GraphError {
    GraphError::InvalidInput(format!("Invalid stored property: {message}"))
}

// Validate syntax without converting JSON numbers into bounded Rust numbers.
// Json retains its original serialized representation, including large integers.
fn validate_json(text: &str) -> Result<(), GraphError> {
    let mut parser = serde_json::Deserializer::from_str(text);
    serde::de::IgnoredAny::deserialize(&mut parser).map_err(invalid)?;
    parser.end().map_err(invalid)
}

/// Encode the logical value; storage optimizations are decoded and validated.
pub(crate) fn encode(value: &AttrValue) -> Result<Value, GraphError> {
    use AttrValue::*;
    let stored =
        match value {
            Null => StoredValue::Null,
            Int(v) => StoredValue::Int(*v),
            SmallInt(v) => StoredValue::Int(i64::from(*v)),
            Float(v) => StoredValue::Float(v.to_bits()),
            Float64(v) => StoredValue::Float64(v.to_bits()),
            Float64Vec(v) => StoredValue::Float64Vec(v.iter().map(|f| f.to_bits()).collect()),
            UtcInstant(v) => StoredValue::UtcInstant(*v),
            Date(v) => StoredValue::Date(*v),
            Duration(v) => StoredValue::Duration(*v),
            Bool(v) => StoredValue::Bool(*v),
            Text(v) => StoredValue::Text(v.clone()),
            CompactText(v) => StoredValue::Text(v.as_str().to_owned()),
            CompressedText(v) => StoredValue::Text(v.decompress_text().map_err(invalid)?),
            FloatVec(v) => StoredValue::FloatVec(v.iter().map(|f| f.to_bits()).collect()),
            CompressedFloatVec(v) => StoredValue::FloatVec(
                v.decompress_float_vec()
                    .map_err(invalid)?
                    .iter()
                    .map(|f| f.to_bits())
                    .collect(),
            ),
            IntVec(v) => StoredValue::IntVec(v.clone()),
            BoolVec(v) => StoredValue::BoolVec(v.clone()),
            TextVec(v) => StoredValue::TextVec(v.clone()),
            Bytes(v) => StoredValue::Bytes(v.clone()),
            NodeArray(v) => StoredValue::NodeArray(v.clone()),
            EdgeArray(v) => StoredValue::EdgeArray(v.clone()),
            Json(v) => {
                validate_json(v)?;
                StoredValue::Json(v.clone())
            }
            SubgraphRef(_) => return Err(invalid(
                "SubgraphRef requires its pool contents; active graph export cannot preserve it",
            )),
        };
    serde_json::to_value(stored).map_err(invalid)
}

/// Decode a strict typed property, rejecting unknown tags or malformed payloads.
pub(crate) fn decode(value: Value) -> Result<AttrValue, GraphError> {
    Ok(match serde_json::from_value(value).map_err(invalid)? {
        StoredValue::Null => AttrValue::Null,
        StoredValue::Int(v) => AttrValue::Int(v),
        StoredValue::Float(v) => AttrValue::Float(f32::from_bits(v)),
        StoredValue::Float64(v) => AttrValue::Float64(f64::from_bits(v)),
        StoredValue::Float64Vec(v) => {
            AttrValue::Float64Vec(v.into_iter().map(f64::from_bits).collect())
        }
        StoredValue::UtcInstant(v) => AttrValue::UtcInstant(v),
        StoredValue::Date(v) => AttrValue::Date(v),
        StoredValue::Duration(v) => AttrValue::Duration(v),
        StoredValue::Bool(v) => AttrValue::Bool(v),
        StoredValue::Text(v) => AttrValue::Text(v),
        StoredValue::FloatVec(v) => {
            AttrValue::FloatVec(v.into_iter().map(f32::from_bits).collect())
        }
        StoredValue::IntVec(v) => AttrValue::IntVec(v),
        StoredValue::BoolVec(v) => AttrValue::BoolVec(v),
        StoredValue::TextVec(v) => AttrValue::TextVec(v),
        StoredValue::Bytes(v) => AttrValue::Bytes(v),
        StoredValue::NodeArray(v) => AttrValue::NodeArray(v),
        StoredValue::EdgeArray(v) => AttrValue::EdgeArray(v),
        StoredValue::Json(v) => {
            validate_json(&v)?;
            AttrValue::Json(v)
        }
    })
}

/// Unversioned legacy exports used untagged JSON scalars. Accept containers too,
/// without guessing whether an array represents bytes, vectors or entity IDs.
pub(crate) fn decode_legacy(value: Value) -> Result<AttrValue, GraphError> {
    Ok(match value {
        Value::Null => AttrValue::Null,
        Value::Bool(v) => AttrValue::Bool(v),
        Value::String(v) => AttrValue::Text(v),
        Value::Number(v) => match v.as_i64() {
            Some(v) => AttrValue::Int(v),
            None if v.is_f64() => {
                AttrValue::Float(v.as_f64().ok_or_else(|| invalid("invalid number"))? as f32)
            }
            None => return Err(invalid("integer exceeds i64")),
        },
        v => AttrValue::Json(serde_json::to_string(&v).map_err(invalid)?),
    })
}
