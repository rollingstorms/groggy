use groggy::types::{CompactString, CompressedData};
use groggy::{AttrValue, Graph};
use std::collections::{BTreeSet, HashSet};
use std::hash::{Hash, Hasher};

fn hash(value: &AttrValue) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

fn values() -> Vec<AttrValue> {
    let text = "x".repeat(1000);
    vec![
        AttrValue::Null,
        AttrValue::Int(1),
        AttrValue::SmallInt(1),
        AttrValue::Float(0.0),
        AttrValue::Float(-0.0),
        AttrValue::Float(f32::NAN),
        AttrValue::Float(f32::from_bits(0xffc0_0001)),
        AttrValue::Float(f32::INFINITY),
        AttrValue::Float(f32::NEG_INFINITY),
        AttrValue::Bool(false),
        AttrValue::Bool(true),
        AttrValue::Text("1".into()),
        AttrValue::Text("01".into()),
        AttrValue::Text("2".into()),
        AttrValue::Text("10".into()),
        AttrValue::Text("1a".into()),
        AttrValue::Text(text.clone()),
        AttrValue::CompactText(CompactString::new(&text)),
        AttrValue::CompressedText(CompressedData::compress_text(&text)),
        AttrValue::FloatVec(vec![0.0, f32::NAN]),
        AttrValue::CompressedFloatVec(CompressedData::compress_float_vec(&[-0.0, f32::NAN])),
        AttrValue::FloatVec(vec![1.0, 3.0]),
        AttrValue::FloatVec(vec![2.0]),
        AttrValue::IntVec(vec![1]),
        AttrValue::IntVec(vec![2]),
        AttrValue::TextVec(vec!["a".into()]),
        AttrValue::TextVec(vec!["b".into()]),
        AttrValue::BoolVec(vec![false]),
        AttrValue::BoolVec(vec![true]),
        AttrValue::Bytes(vec![1]),
        AttrValue::Bytes(vec![2]),
        AttrValue::NodeArray(vec![1]),
        AttrValue::NodeArray(vec![2]),
        AttrValue::EdgeArray(vec![1]),
        AttrValue::EdgeArray(vec![2]),
        AttrValue::SubgraphRef(1),
        AttrValue::SubgraphRef(2),
        AttrValue::Json("{}".into()),
        AttrValue::Json("[]".into()),
    ]
}

#[test]
fn equality_hashing_and_ordering_agree_across_types_and_storage_forms() {
    let values = values();
    for left in &values {
        assert_eq!(left, &left.clone());
        for right in &values {
            assert_eq!(left.cmp(right), right.cmp(left).reverse());
            assert_eq!(*left == *right, left.cmp(right).is_eq());
            if left == right {
                assert_eq!(hash(left), hash(right), "{left:?} vs {right:?}");
            }
            for third in &values {
                if left <= right && right <= third {
                    assert!(
                        left <= third,
                        "Nontransitive: {left:?}, {right:?}, {third:?}"
                    );
                }
            }
        }
    }
    let hashed: HashSet<_> = values.iter().cloned().collect();
    let ordered: BTreeSet<_> = values.iter().cloned().collect();
    assert_eq!(hashed.len(), ordered.len());
}

#[test]
fn compression_preserves_float_bits_without_alignment_assumptions() {
    let values = [1.0f32, -0.0, f32::from_bits(0x7fc0_0001), f32::INFINITY];
    let compressed = CompressedData::compress_float_vec(&values);
    let expected: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    assert_eq!(compressed.data, expected);
    let restored = compressed.decompress_float_vec().unwrap();
    assert_eq!(
        values.map(f32::to_bits).to_vec(),
        restored
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
}

#[test]
fn corrupt_compression_is_detected_and_still_has_reflexive_value_identity() {
    let mut compressed = CompressedData::compress_text("text");
    compressed.data.push(0xff);
    assert!(compressed.decompress_text().is_err());
    let value = AttrValue::CompressedText(compressed);
    assert_eq!(value, value.clone());
    assert_eq!(hash(&value), hash(&value.clone()));
    assert_ne!(value, AttrValue::Text("text".into()));

    let mut compressed = CompressedData::compress_float_vec(&[1.0]);
    compressed.data.push(0);
    assert!(compressed.decompress_float_vec().is_err());
}

#[test]
fn typed_values_null_and_type_changes_survive_recorded_snapshots() {
    let mut graph = Graph::new();
    let nodes = graph.add_nodes(3);
    let edges = graph
        .add_edges(&[
            (nodes[0], nodes[1]),
            (nodes[1], nodes[2]),
            (nodes[2], nodes[0]),
        ])
        .unwrap();
    let node = nodes[0];
    let edge = edges[0];
    // Referent reconstruction belongs to storage; this test covers inline values.
    let values: Vec<_> = values()
        .into_iter()
        .filter(|value| !matches!(value, AttrValue::SubgraphRef(_)))
        .collect();
    for (position, value) in values.iter().enumerate() {
        graph
            .set_node_attr(node, format!("p{position}"), value.clone())
            .unwrap();
        graph
            .set_edge_attr(edge, format!("p{position}"), value.clone())
            .unwrap();
    }
    graph
        .set_node_attr(node, "changing".into(), AttrValue::Int(1))
        .unwrap();
    let first = graph.commit("values".into(), "test".into()).unwrap();
    graph
        .set_node_attr(node, "changing".into(), AttrValue::Text("now text".into()))
        .unwrap();
    let second = graph.commit("type change".into(), "test".into()).unwrap();
    let prior = graph.snapshot_at_commit(first).unwrap();
    let current = graph.snapshot_at_commit(second).unwrap();
    for (position, value) in values.iter().enumerate() {
        assert_eq!(
            prior.node_attr(node, &format!("p{position}")),
            Some(value.clone())
        );
        assert_eq!(
            prior.edge_attr(edge, &format!("p{position}")),
            Some(value.clone())
        );
    }
    assert_eq!(prior.node_attr(node, &"p0".into()), Some(AttrValue::Null));
    assert_eq!(prior.node_attr(node, &"missing".into()), None);
    assert_eq!(
        prior.node_attr(node, &"changing".into()),
        Some(AttrValue::Int(1))
    );
    assert_eq!(
        current.node_attr(node, &"changing".into()),
        Some(AttrValue::Text("now text".into()))
    );
}
