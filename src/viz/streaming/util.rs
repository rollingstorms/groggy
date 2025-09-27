// --- Utility helpers ---

pub fn attr_value_to_display_text(attr: &crate::types::AttrValue) -> String {
    use crate::types::AttrValue;

    match attr {
        AttrValue::Int(i) => i.to_string(),
        AttrValue::Float(f) => f.to_string(),
        AttrValue::Text(s) => s.clone(),
        AttrValue::CompactText(s) => s.as_str().to_string(),
        AttrValue::SmallInt(i) => i.to_string(),
        AttrValue::Bool(b) => b.to_string(),
        AttrValue::FloatVec(v) => format!("[{} floats]", v.len()),
        AttrValue::Bytes(b) => format!("[{} bytes]", b.len()),
        AttrValue::CompressedText(_) => "[Compressed Text]".to_string(),
        AttrValue::CompressedFloatVec(_) => "[Compressed FloatVec]".to_string(),
        AttrValue::SubgraphRef(id) => format!("[Subgraph:{}]", id),
        AttrValue::NodeArray(nodes) => format!("[{} nodes]", nodes.len()),
        AttrValue::EdgeArray(edges) => format!("[{} edges]", edges.len()),
        AttrValue::Null => "null".to_string(),
        AttrValue::Json(json_str) => json_str.clone(),
        AttrValue::IntVec(v) => format!("[{} ints]", v.len()),
        AttrValue::TextVec(v) => format!("[{} strings]", v.len()),
        AttrValue::BoolVec(v) => format!("[{} bools]", v.len()),
    }
}

fn attr_value_to_json(attr: &crate::types::AttrValue) -> serde_json::Value {
    use crate::types::AttrValue;

    match attr {
        AttrValue::Int(i) => serde_json::Value::Number(serde_json::Number::from(*i)),
        AttrValue::Float(f) => serde_json::Number::from_f64(*f as f64)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        AttrValue::Text(s) => serde_json::Value::String(s.clone()),
        AttrValue::Bool(b) => serde_json::Value::Bool(*b),
        AttrValue::CompactText(s) => serde_json::Value::String(s.as_str().to_string()),
        AttrValue::SmallInt(i) => serde_json::Value::Number(serde_json::Number::from(*i)),
        AttrValue::FloatVec(v) => {
            let vec: Vec<serde_json::Value> = v
                .iter()
                .map(|&f| {
                    serde_json::Number::from_f64(f as f64)
                        .map(serde_json::Value::Number)
                        .unwrap_or(serde_json::Value::Null)
                })
                .collect();
            serde_json::Value::Array(vec)
        }
        AttrValue::Bytes(b) => serde_json::Value::String(format!("[{} bytes]", b.len())),
        AttrValue::CompressedText(_) => serde_json::Value::String("[Compressed Text]".to_string()),
        AttrValue::CompressedFloatVec(_) => {
            serde_json::Value::String("[Compressed FloatVec]".to_string())
        }
        AttrValue::SubgraphRef(id) => serde_json::Value::String(format!("[Subgraph:{}]", id)),
        AttrValue::NodeArray(nodes) => {
            serde_json::Value::String(format!("[{} nodes]", nodes.len()))
        }
        AttrValue::EdgeArray(edges) => {
            serde_json::Value::String(format!("[{} edges]", edges.len()))
        }
        AttrValue::Null => serde_json::Value::Null,
        AttrValue::Json(json_str) => {
            // Try to parse as JSON, fallback to string if invalid
            serde_json::from_str(json_str)
                .unwrap_or_else(|_| serde_json::Value::String(json_str.clone()))
        }
        AttrValue::IntVec(v) => serde_json::Value::Array(
            v.iter()
                .map(|&i| serde_json::Value::Number(i.into()))
                .collect(),
        ),
        AttrValue::TextVec(v) => serde_json::Value::Array(
            v.iter()
                .map(|s| serde_json::Value::String(s.clone()))
                .collect(),
        ),
        AttrValue::BoolVec(v) => {
            serde_json::Value::Array(v.iter().map(|&b| serde_json::Value::Bool(b)).collect())
        }
    }
}
