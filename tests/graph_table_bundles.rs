use groggy::{
    storage::{
        array::BaseArray,
        table::{
            BaseTable, EdgesTable, GraphTable, NodesTable, Table, ValidationPolicy,
            ValidationStrictness,
        },
    },
    types::{CompactString, CompressedData},
    AttrValue as V, Graph,
};
use serde_json::Value;
use std::path::Path;
fn base(columns: Vec<(&str, Vec<V>)>) -> BaseTable {
    let order = columns.iter().map(|(n, _)| n.to_string()).collect();
    BaseTable::with_column_order(
        columns
            .into_iter()
            .map(|(n, v)| (n.to_string(), BaseArray::from_attr_values(v)))
            .collect(),
        order,
    )
    .unwrap()
}
fn fixture(values: Vec<V>) -> GraphTable {
    let n = values.len();
    GraphTable::new(
        NodesTable::from_base_table(base(vec![
            (
                "node_id",
                (0..n).map(|r| V::Int((r * 7 + 3) as i64)).collect(),
            ),
            ("odd,\"\nname", values.clone()),
        ]))
        .unwrap(),
        EdgesTable::from_base_table(base(vec![
            (
                "edge_id",
                (0..n).map(|r| V::Int((r * 11 + 2) as i64)).collect(),
            ),
            ("source", vec![V::Int(3); n]),
            ("target", vec![V::Int(3); n]),
            ("x", values),
        ]))
        .unwrap(),
    )
}
fn values() -> Vec<V> {
    vec![
        V::Null,
        V::Int(i64::MIN),
        V::Int(i64::MAX),
        V::SmallInt(7),
        V::Float(-0.0),
        V::Float(f32::from_bits(0xffc00001)),
        V::Float64(1.0000000000000002),
        V::Float64(-0.0),
        V::Float64(f64::from_bits(0xfff8000000000001)),
        V::Float64(f64::INFINITY),
        V::FloatVec(vec![-0.0, f32::NAN]),
        V::Float64Vec(vec![-0.0, f64::from_bits(0x7ff8000000000001)]),
        V::Text("true".into()),
        V::Text("".into()),
        V::Text("a,\"b\nλ".into()),
        V::CompactText(CompactString::new("short")),
        V::CompressedText(CompressedData::compress_text(&"large".repeat(100))),
        V::CompressedFloatVec(CompressedData::compress_float_vec(&[1.0; 16])),
        V::Bytes(vec![0, 255]),
        V::IntVec(vec![-1, i64::MAX]),
        V::BoolVec(vec![true, false]),
        V::TextVec(vec!["a".into()]),
        V::NodeArray(vec![3]),
        V::EdgeArray(vec![2]),
        V::Json("{\"large\":123456789012345678901234567890}".into()),
        V::Bool(false),
        V::UtcInstant(groggy::types::UtcInstant::new(1_600_000_000, 123456).unwrap()),
        V::Date(groggy::types::CalendarDate::new(2024, 2, 29).unwrap()),
        V::Duration(groggy::types::FixedDuration::new(-1, 1, 123).unwrap()),
    ]
}
fn bits_equal(expected: &V, actual: &V) {
    assert_eq!(expected, actual);
    match (expected, actual) {
        (V::Float(a), V::Float(b)) => assert_eq!(a.to_bits(), b.to_bits()),
        (V::Float64(a), V::Float64(b)) => assert_eq!(a.to_bits(), b.to_bits()),
        (V::FloatVec(a), V::FloatVec(b)) => assert_eq!(
            a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        ),
        (V::Float64Vec(a), V::Float64Vec(b)) => assert_eq!(
            a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        ),
        _ => {}
    }
}
#[test]
fn typed_binary_roundtrip_preserves_logical_values_bits_order_policy_and_ids() {
    let dir = tempfile::tempdir().unwrap();
    let mut table = fixture(values());
    let policy = ValidationPolicy {
        strictness: ValidationStrictness::Strict,
        auto_repair: false,
        ..ValidationPolicy::default()
    };
    table.set_policy(policy);
    table.save_bundle(dir.path()).unwrap();
    let loaded = GraphTable::load_bundle(dir.path()).unwrap();
    assert_eq!(
        loaded.nodes().node_ids().unwrap(),
        table.nodes().node_ids().unwrap()
    );
    assert_eq!(
        loaded.edges().edge_ids().unwrap(),
        table.edges().edge_ids().unwrap()
    );
    assert_eq!(loaded.nodes().column_names(), table.nodes().column_names());
    assert_eq!(loaded.policy().strictness, ValidationStrictness::Strict);
    for (before, after) in values()
        .iter()
        .zip(loaded.nodes().column("odd,\"\nname").unwrap().iter())
    {
        bits_equal(before, after);
    }
    for (before, after) in values()
        .iter()
        .zip(loaded.edges().column("x").unwrap().iter())
    {
        bits_equal(before, after);
    }
    let info = GraphTable::bundle_metadata_value(dir.path()).unwrap();
    assert_eq!(info["version"], "3.0");
    assert_eq!(info["checksum_algorithm"], "xxh3_64");
    assert_eq!(
        GraphTable::validate_bundle(dir.path()).unwrap().node_count,
        values().len()
    );
}
#[test]
fn empty_tables_preserve_declared_column_order_and_zero_rows() {
    let dir = tempfile::tempdir().unwrap();
    let table = fixture(vec![]);
    table.save_bundle(dir.path()).unwrap();
    let loaded = GraphTable::load_bundle(dir.path()).unwrap();
    assert_eq!(loaded.nodes().column_names(), table.nodes().column_names());
    assert!(loaded.is_empty());
}
#[test]
fn corruption_truncation_trailing_bytes_and_unknown_headers_reject() {
    let dir = tempfile::tempdir().unwrap();
    fixture(vec![V::Int(1)]).save_bundle(dir.path()).unwrap();
    let path = dir.path().join("bundle.cbor");
    let good = std::fs::read(&path).unwrap();
    let mut cases = vec![
        vec![],
        good[..7].to_vec(),
        good[..23].to_vec(),
        good[..good.len() - 1].to_vec(),
    ];
    let mut changed = good.clone();
    changed[0] ^= 1;
    cases.push(changed);
    let mut changed = good.clone();
    changed[24] ^= 1;
    cases.push(changed);
    let mut changed = good.clone();
    changed.push(0);
    cases.push(changed);
    for bytes in cases {
        std::fs::write(&path, bytes).unwrap();
        assert!(GraphTable::load_bundle(dir.path()).is_err());
        assert!(GraphTable::validate_bundle(dir.path()).is_err());
    }
}
fn rewrite(path: &Path, value: &Value) {
    let mut data = Vec::new();
    ciborium::into_writer(value, &mut data).unwrap();
    let mut bytes = b"GROGTB03".to_vec();
    bytes.extend_from_slice(&(data.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&xxhash_rust::xxh3::xxh3_64(&data).to_le_bytes());
    bytes.extend(data);
    std::fs::write(path, bytes).unwrap();
}
#[test]
fn checksum_valid_but_malformed_schema_tags_versions_and_rowcounts_reject() {
    let dir = tempfile::tempdir().unwrap();
    fixture(vec![V::Int(1)]).save_bundle(dir.path()).unwrap();
    let path = dir.path().join("bundle.cbor");
    let bytes = std::fs::read(&path).unwrap();
    let good: Value = ciborium::from_reader(&bytes[24..]).unwrap();
    let mut cases = Vec::new();
    let mut v = good.clone();
    v["version"] = 99.into();
    cases.push(v);
    let mut v = good.clone();
    v["nodes"]["rows"] = 2.into();
    cases.push(v);
    let mut v = good.clone();
    v["nodes"]["columns"][0]["name"] = "wrong".into();
    cases.push(v);
    let mut v = good.clone();
    let duplicate = v["nodes"]["columns"][0].clone();
    v["nodes"]["columns"]
        .as_array_mut()
        .unwrap()
        .push(duplicate);
    cases.push(v);
    let mut v = good.clone();
    v["nodes"]["columns"][1]["values"][0]["type"] = "mystery".into();
    cases.push(v);
    let mut v = good;
    v["unexpected"] = true.into();
    cases.push(v);
    for value in cases {
        rewrite(&path, &value);
        assert!(GraphTable::load_bundle(dir.path()).is_err());
    }
}
#[test]
fn failed_encoding_preserves_previous_file_and_successful_save_replaces_it() {
    let dir = tempfile::tempdir().unwrap();
    fixture(vec![V::Int(1)]).save_bundle(dir.path()).unwrap();
    let path = dir.path().join("bundle.cbor");
    let bytes = std::fs::read(&path).unwrap();
    for value in [V::SubgraphRef(0), V::Json("{".into())] {
        assert!(fixture(vec![value]).save_bundle(dir.path()).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
    }
    let mut malformed = fixture(vec![V::Int(1)]);
    *malformed.nodes_mut().base_table_mut() = BaseTable::new();
    assert!(malformed.save_bundle(dir.path()).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
    fixture(vec![V::Int(2)]).save_bundle(dir.path()).unwrap();
    assert_eq!(
        GraphTable::load_bundle(dir.path())
            .unwrap()
            .nodes()
            .column("odd,\"\nname")
            .unwrap()
            .get(0),
        Some(&V::Int(2))
    );
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
}
fn copy_legacy(path: &Path) {
    for name in [
        "metadata.json",
        "nodes.csv",
        "edges.csv",
        "MANIFEST.json",
        "validation_report.json",
    ] {
        std::fs::copy(
            Path::new("tests/data/graph_table_bundle_v2").join(name),
            path.join(name),
        )
        .unwrap();
    }
}
#[test]
fn legacy_v2_reads_verify_historical_checksums_and_migrate_by_saving() {
    let dir = tempfile::tempdir().unwrap();
    copy_legacy(dir.path());
    let loaded = GraphTable::load_bundle(dir.path()).unwrap();
    assert_eq!(loaded.nodes().node_ids().unwrap(), vec![8, 3]);
    assert_eq!(loaded.edges().sources().unwrap(), vec![8]);
    assert_eq!(
        GraphTable::validate_bundle(dir.path()).unwrap().version,
        "2.0"
    );
    loaded.save_bundle(dir.path()).unwrap();
    // New authoritative file wins over stale legacy sidecars, even if they corrupt.
    std::fs::write(dir.path().join("metadata.json"), "broken").unwrap();
    assert_eq!(
        GraphTable::validate_bundle(dir.path()).unwrap().version,
        "3.0"
    );
}
#[test]
fn changed_legacy_data_metadata_and_manifest_never_pass_verification() {
    for name in ["metadata.json", "nodes.csv", "edges.csv", "MANIFEST.json"] {
        let dir = tempfile::tempdir().unwrap();
        copy_legacy(dir.path());
        let path = dir.path().join(name);
        let mut bytes = std::fs::read(&path).unwrap();
        bytes.push(b' ');
        std::fs::write(&path, &bytes).unwrap();
        // Whitespace is allowed around JSON, but changing data bytes is always detected.
        if name.ends_with(".csv") {
            assert!(GraphTable::validate_bundle(dir.path()).is_err());
        }
        std::fs::write(&path, "broken").unwrap();
        assert!(GraphTable::validate_bundle(dir.path()).is_err());
    }
}
#[test]
fn frozen_source_captures_are_detached_after_bundle_roundtrip() {
    let mut graph = Graph::new();
    graph.add_node();
    graph.set_node_attr(0, "x".into(), V::Int(1)).unwrap();
    let table = graph.read_view().unwrap().table().unwrap();
    let dir = tempfile::tempdir().unwrap();
    table.save_bundle(dir.path()).unwrap();
    let loaded = GraphTable::load_bundle(dir.path()).unwrap();
    assert!(loaded.nodes().base_table().source_views().is_empty());
    assert_eq!(loaded.nodes().base_table().source_entity_ids(), None);
}
