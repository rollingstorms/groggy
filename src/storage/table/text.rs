//! Plain table interchange adapters; the tagged graph codec owns lossless storage.
use crate::{AttrValue, GraphError, GraphResult};

pub(super) fn csv_field(value: &AttrValue) -> String {
    match value {
        AttrValue::Int(v) => v.to_string(),
        AttrValue::SmallInt(v) => v.to_string(),
        AttrValue::Float(v) => v.to_string(),
        AttrValue::Float64(v) => v.to_string(),
        AttrValue::UtcInstant(v) => v.to_string(),
        AttrValue::Date(v) => v.to_string(),
        AttrValue::Duration(v) => v.to_string(),
        AttrValue::Text(v) => v.clone(),
        AttrValue::CompactText(v) => v.as_str().to_string(),
        AttrValue::Bool(v) => v.to_string(),
        AttrValue::Null => String::new(),
        AttrValue::FloatVec(v) => vector_csv(v),
        AttrValue::Float64Vec(v) => vector_csv(v),
        AttrValue::Bytes(v) => format!("bytes:{}", v.len()),
        _ => format!("{value:?}"),
    }
}
fn vector_csv(values: &[impl std::fmt::Display]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",")
    )
}
fn number(value: f64) -> GraphResult<serde_json::Value> {
    serde_json::Number::from_f64(value)
        .map(serde_json::Value::Number)
        .ok_or_else(|| GraphError::InvalidInput("Plain table JSON requires finite floating values; use tagged graph storage for nonfinite values".into()))
}
pub(super) fn json_field(value: &AttrValue) -> GraphResult<serde_json::Value> {
    use serde_json::Value as J;
    Ok(match value {
        AttrValue::Int(v) => J::from(*v),
        AttrValue::SmallInt(v) => J::from(*v),
        AttrValue::Float(v) => number(f64::from(*v))?,
        AttrValue::Float64(v) => number(*v)?,
        AttrValue::UtcInstant(v) => J::String(v.to_string()),
        AttrValue::Date(v) => J::String(v.to_string()),
        AttrValue::Duration(v) => J::String(v.to_string()),
        AttrValue::Text(v) => J::String(v.clone()),
        AttrValue::CompactText(v) => J::String(v.as_str().to_string()),
        AttrValue::Bool(v) => J::Bool(*v),
        AttrValue::Null => J::Null,
        AttrValue::FloatVec(v) => J::Array(
            v.iter()
                .map(|&v| number(f64::from(v)))
                .collect::<GraphResult<_>>()?,
        ),
        AttrValue::Float64Vec(v) => {
            J::Array(v.iter().map(|&v| number(v)).collect::<GraphResult<_>>()?)
        }
        _ => J::String(format!("{value:?}")),
    })
}
