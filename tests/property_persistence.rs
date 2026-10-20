use groggy::types::{CompactString, CompressedData};
use groggy::{AttrValue, Graph};
use serde_json::json;

fn values() -> Vec<AttrValue> {
    vec![
        AttrValue::Null,
        AttrValue::Int(i64::MIN),
        AttrValue::Int(i64::MAX),
        AttrValue::SmallInt(12),
        AttrValue::Float(0.5),
        AttrValue::Float(-0.0),
        AttrValue::Float(f32::from_bits(0x7fc0_0001)),
        AttrValue::Float(f32::NEG_INFINITY),
        AttrValue::Bool(false),
        AttrValue::Text("quoted \" unicode λ".into()),
        AttrValue::CompactText(CompactString::new("compact")),
        AttrValue::CompressedText(CompressedData::compress_text(&"large".repeat(1000))),
        AttrValue::FloatVec(vec![-0.0, f32::from_bits(0xffc0_0010), f32::INFINITY]),
        AttrValue::CompressedFloatVec(CompressedData::compress_float_vec(&[0.0; 512])),
        AttrValue::IntVec(vec![i64::MIN, i64::MAX]),
        AttrValue::BoolVec(vec![true, false]),
        AttrValue::TextVec(vec!["a".into(), "b".into()]),
        AttrValue::Bytes(vec![0, 255]),
        AttrValue::NodeArray(vec![0, 1]),
        AttrValue::EdgeArray(vec![0]),
        AttrValue::Json("{\"nested\":[null,true,{\"value\":7}]}".into()),
        AttrValue::Json(format!("{{\"large\":1{}}}", "0".repeat(400))),
        AttrValue::FloatVec(vec![]),
        AttrValue::IntVec(vec![]),
        AttrValue::BoolVec(vec![]),
        AttrValue::TextVec(vec![]),
        AttrValue::Bytes(vec![]),
    ]
}

#[test]
fn export_round_trips_every_inline_value_and_float_bits_on_nodes_and_edges() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("graph.json");
    let mut graph = Graph::new_directed();
    let nodes = graph.add_nodes(2);
    let edge = graph.add_edge(nodes[0], nodes[1]).unwrap();
    for (i, value) in values().into_iter().enumerate() {
        graph
            .set_node_attr(nodes[0], format!("p{i}"), value.clone())
            .unwrap();
        graph.set_edge_attr(edge, format!("p{i}"), value).unwrap();
    }
    graph.save_to_path(&path).unwrap();
    let first = std::fs::read(&path).unwrap();
    graph.save_to_path(&path).unwrap();
    assert_eq!(first, std::fs::read(&path).unwrap());
    let loaded = Graph::load_from_path(&path).unwrap();
    assert!(loaded.is_directed());
    assert_eq!(loaded.edge_endpoints(edge).unwrap(), (nodes[0], nodes[1]));
    for (i, value) in values().into_iter().enumerate() {
        let name = format!("p{i}");
        let node_value = loaded.get_node_attr(nodes[0], &name).unwrap().unwrap();
        let edge_value = loaded.get_edge_attr(edge, &name).unwrap().unwrap();
        assert_eq!(node_value, value);
        assert_eq!(edge_value, value);
        if let AttrValue::Float(before) = value {
            let AttrValue::Float(after) = node_value else {
                panic!("lost float type")
            };
            assert_eq!(before.to_bits(), after.to_bits());
        }
        if let AttrValue::FloatVec(before) = value {
            let after = node_value.get_float_vec().unwrap();
            assert_eq!(
                before.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                after.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    }
    assert_eq!(
        loaded.get_node_attr(nodes[0], &"missing".into()).unwrap(),
        None
    );
}

#[test]
fn legacy_scalars_sparse_ids_and_untyped_containers_remain_readable() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("legacy.json");
    std::fs::write(
        &path,
        json!({
            "config":{"graph_type":"undirected"},
            "nodes":[{"id":8,"attributes":{"n":3,"f":0.5,"null":null,"array":[1,2]}}, {"id":2}],
            "edges":[{"id":5,"source":8,"target":2,"attributes":{"b":true}}]
        })
        .to_string(),
    )
    .unwrap();
    let loaded = Graph::load_from_path(&path).unwrap();
    assert!(loaded.is_undirected());
    assert_eq!(
        loaded.get_node_attr(8, &"n".into()).unwrap(),
        Some(AttrValue::Int(3))
    );
    assert_eq!(
        loaded.get_node_attr(8, &"null".into()).unwrap(),
        Some(AttrValue::Null)
    );
    assert!(matches!(
        loaded.get_node_attr(8, &"array".into()).unwrap(),
        Some(AttrValue::Json(_))
    ));
    assert_eq!(loaded.edge_endpoints(5).unwrap(), (8, 2));
}

#[test]
fn malformed_documents_fail_instead_of_dropping_records_or_values() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("bad.json");
    let base = json!({"format":"groggy.graph","version":2,"nodes":[{"id":0}],"edges":[]});
    let mut cases = Vec::new();
    let mut v = base.clone();
    v["version"] = json!(999);
    cases.push(v);
    let mut v = base.clone();
    v["nodes"][0]["attributes"] = json!({"x":{"type":"future","value":1}});
    cases.push(v);
    let mut v = base.clone();
    v["nodes"][0]["attributes"] = json!({"x":{"type":"int","value":"bad"}});
    cases.push(v);
    let mut v = base.clone();
    v["nodes"][0]["attributes"] = json!({"x":{"type":"json","value":"{"}});
    cases.push(v);
    let mut v = base.clone();
    v["nodes"][0]["attributes"] = json!({"x":{"type":"null","value":42}});
    cases.push(v);
    let mut v = base.clone();
    v["nodes"] = json!([{"id":0},{"id":0}]);
    cases.push(v);
    let mut v = base.clone();
    v["edges"] = json!([{"id":0,"source":0,"target":9}]);
    cases.push(v);
    let mut v = base.clone();
    v["edges"] = json!([{"source":0,"target":0}]);
    cases.push(v);
    let mut v = base.clone();
    v["nodes"] = json!([{}]);
    cases.push(v);
    for case in cases {
        std::fs::write(&path, case.to_string()).unwrap();
        assert!(Graph::load_from_path(&path).is_err(), "Accepted {case}");
    }
}

#[test]
fn encoding_errors_preserve_an_existing_destination() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("existing.json");
    for value in [AttrValue::SubgraphRef(0), AttrValue::Json("{".into())] {
        let mut graph = Graph::new();
        let node = graph.add_node();
        graph.set_node_attr(node, "bad".into(), value).unwrap();
        std::fs::write(&path, b"previous saved data").unwrap();
        assert!(graph.save_to_path(&path).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"previous saved data");
    }
}
