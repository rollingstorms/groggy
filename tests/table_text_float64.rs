use groggy::{
    storage::{
        array::BaseArray,
        table::{BaseTable, Table},
    },
    AttrValue as V,
};
use std::collections::HashMap;
fn table(columns: Vec<(&str, Vec<V>)>) -> BaseTable {
    BaseTable::from_columns(
        columns
            .into_iter()
            .map(|(k, v)| (k.into(), BaseArray::new(v)))
            .collect(),
    )
    .unwrap()
}
#[test]
fn plain_json_preserves_finite_f64_numbers_and_vectors() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("table.json");
    let x = 1. + 2f64.powi(-40);
    let source = table(vec![
        ("x", vec![V::Float64(x), V::Float64(1e300), V::Float64(-0.)]),
        (
            "v",
            vec![
                V::Float64Vec(vec![x, 1e300]),
                V::Float64Vec(vec![]),
                V::Float64Vec(vec![-0.]),
            ],
        ),
    ]);
    source.to_json(&path).unwrap();
    let loaded = BaseTable::from_json(&path).unwrap();
    assert_eq!(
        loaded.column("x").unwrap().data(),
        source.column("x").unwrap().data()
    );
    assert_eq!(
        loaded.column("v").unwrap().data(),
        source.column("v").unwrap().data()
    );
    let V::Float64(zero) = loaded.column("x").unwrap().data()[2] else {
        panic!("expected float")
    };
    assert_eq!(zero.to_bits(), (-0.0f64).to_bits());
    for value in [
        V::Float(f32::NAN),
        V::Float64(f64::INFINITY),
        V::FloatVec(vec![f32::INFINITY]),
        V::Float64Vec(vec![f64::NAN]),
    ] {
        std::fs::write(&path, "existing").unwrap();
        assert!(table(vec![("x", vec![value])]).to_json(&path).is_err());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "existing");
    }
}
#[test]
fn csv_uses_f64_inference_and_one_layer_of_text_escaping() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("table.csv");
    let source = table(vec![
        (
            "x",
            vec![V::Float64(1. + 2f64.powi(-40)), V::Float64(1e300)],
        ),
        (
            "text",
            vec![
                V::Text("comma, quote\" newline\nλ".into()),
                V::CompactText(groggy::types::CompactString::new("compact,\"quoted")),
            ],
        ),
    ]);
    source.to_csv(&path).unwrap();
    let loaded = BaseTable::from_csv(&path).unwrap();
    assert_eq!(
        loaded.column("x").unwrap().data(),
        source.column("x").unwrap().data()
    );
    for (actual, expected) in loaded
        .column("text")
        .unwrap()
        .data()
        .iter()
        .zip(source.column("text").unwrap().data())
    {
        assert_eq!(actual.get_text(), expected.get_text());
    }
    let vectors = table(vec![(
        "vector",
        vec![V::FloatVec(vec![0.25, 1.]), V::Float64Vec(vec![0.25, 1.])],
    )]);
    vectors.to_csv(&path).unwrap();
    assert_eq!(
        BaseTable::from_csv(&path)
            .unwrap()
            .column("vector")
            .unwrap()
            .data(),
        &[V::Text("[0.25,1]".into()), V::Text("[0.25,1]".into())]
    );
}
#[test]
fn grouping_keeps_typed_keys_instead_of_reparsing_debug_strings() {
    let x = 1. + 2f64.powi(-40);
    let keys = vec![
        V::Float64(x),
        V::Float64(x),
        V::Float(1.),
        V::Int(1),
        V::Text("1".into()),
        V::Text("NULL".into()),
        V::Null,
        V::Float64Vec(vec![x]),
    ];
    let source = table(vec![("key", keys), ("value", vec![V::Int(1); 8])]);
    let grouped = source
        .group_by_agg(
            &["key".into()],
            HashMap::from([("value".into(), "sum".into())]),
        )
        .unwrap();
    let got: HashMap<_, _> = grouped
        .column("key")
        .unwrap()
        .data()
        .iter()
        .cloned()
        .zip(grouped.column("value").unwrap().data().iter().cloned())
        .collect();
    assert_eq!(got.len(), 7);
    assert_eq!(got[&V::Float64(x)], V::Int(2));
    for key in [
        V::Float(1.),
        V::Int(1),
        V::Text("1".into()),
        V::Text("NULL".into()),
        V::Null,
        V::Float64Vec(vec![x]),
    ] {
        assert_eq!(got[&key], V::Int(1));
    }
}
