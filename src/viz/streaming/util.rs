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
