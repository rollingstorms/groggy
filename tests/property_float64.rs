use groggy::{
    algorithms::AlgorithmParamValue, types::NumericType, AttrValue as V, BatchNodeRef as N,
    DurableGraph, Graph, GraphBatch, GraphConfig,
};
use std::{
    collections::{HashMap, HashSet},
    hash::{Hash, Hasher},
};
fn bits(value: &V) -> Vec<u64> {
    match value {
        V::Float64(v) => vec![v.to_bits()],
        V::Float64Vec(v) => v.iter().map(|v| v.to_bits()).collect(),
        _ => panic!("expected f64 tag"),
    }
}
#[test]
fn property_identity_order_and_explicit_conversion_keep_f64_precision() {
    let x = 1. + 2f64.powi(-40);
    let values = [
        V::Float64(x),
        V::Float64(1e300),
        V::Float64(-0.),
        V::Float64(f64::from_bits(0x7ff8_0000_0000_0042)),
        V::Float64Vec(vec![x, 1e300, f64::NEG_INFINITY, -0.]),
    ];
    assert_ne!(values[0], V::Float64(1.));
    assert_ne!(values[0], V::Float(x as f32)); // Storage widths remain explicit.
    assert_eq!(V::Float64(-0.), V::Float64(0.));
    for (a, b) in [
        (V::Float64(-0.), V::Float64(0.)),
        (
            V::Float64(f64::NAN),
            V::Float64(f64::from_bits(0xfff8_0000_0000_0010)),
        ),
    ] {
        let hash = |value: &V| {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            value.hash(&mut h);
            h.finish()
        };
        assert_eq!(a, b);
        assert_eq!(hash(&a), hash(&b));
    }
    assert_eq!(HashSet::from([V::Float64(0.), V::Float64(-0.)]).len(), 1);
    assert_eq!(
        V::Float(0.5).to_numeric_type(NumericType::Float64),
        Some(V::Float64(0.5))
    );
    assert!(V::Float64(1e300)
        .to_numeric_type(NumericType::Float32)
        .is_none());
    assert!(V::Float64(2f64.powi(63))
        .to_numeric_type(NumericType::Int64)
        .is_none());
    assert_eq!(V::Float64(x).numeric_type(), Some(NumericType::Float64));
    assert_eq!(
        bits(&AlgorithmParamValue::Float(x).as_attr_value().unwrap()),
        vec![x.to_bits()]
    );
    for value in values {
        assert_eq!(bits(&value.clone().optimize()), bits(&value));
    }
}
#[test]
fn export_history_and_journal_retain_f64_bits_and_legacy_f32_tags() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("values.json");
    let journal = dir.path().join("values.ggj");
    let values = [
        V::Float64(1. + 2f64.powi(-40)),
        V::Float64(f64::from_bits(0x7ff8_0000_0000_0042)),
        V::Float64Vec(vec![-0., f64::NEG_INFINITY, 1e300]),
        V::Float64Vec(vec![]),
    ];
    let mut graph = Graph::new_manual();
    let node = graph.add_node().unwrap();
    graph
        .set_node_attr(node, "old".into(), V::Float(0.1))
        .unwrap();
    for (i, value) in values.iter().enumerate() {
        graph
            .set_node_attr(node, format!("p{i}"), value.clone())
            .unwrap();
    }
    let state = graph.commit("typed".into(), "test".into()).unwrap();
    let pin = graph.read_view().unwrap();
    graph
        .set_node_attr(node, "p0".into(), V::Float64(2.))
        .unwrap();
    assert_eq!(
        bits(&pin.node_attr(node, &"p0".into()).unwrap().unwrap()),
        bits(&values[0])
    );
    graph.reset_hard().unwrap();
    assert_eq!(graph.current_state_id(), state);
    graph.save_to_path(&path).unwrap();
    let loaded = Graph::load_from_path(&path).unwrap();
    assert!(matches!(
        loaded.get_node_attr(node, &"old".into()).unwrap(),
        Some(V::Float(_))
    ));
    for (i, value) in values.iter().enumerate() {
        assert_eq!(
            bits(
                &loaded
                    .get_node_attr(node, &format!("p{i}"))
                    .unwrap()
                    .unwrap()
            ),
            bits(value)
        );
    }
    let mut store = DurableGraph::create(&journal, GraphConfig::new(), None).unwrap();
    let result = store
        .apply_batch(
            GraphBatch {
                add_nodes: 1,
                node_attrs: HashMap::from([
                    ("wide".into(), vec![(N::Added(0), values[2].clone())]),
                    ("scalar".into(), vec![(N::Added(0), values[1].clone())]),
                ]),
                ..Default::default()
            },
            0,
            "typed".into(),
            "test".into(),
        )
        .unwrap();
    drop(store);
    let restored = DurableGraph::open(&journal).unwrap();
    assert_eq!(
        restored.graph().unwrap().current_state_id(),
        result.state_id
    );
    assert_eq!(
        bits(
            &restored
                .graph()
                .unwrap()
                .get_node_attr(0, &"wide".into())
                .unwrap()
                .unwrap()
        ),
        bits(&values[2])
    );
    assert_eq!(
        bits(
            &restored
                .graph()
                .unwrap()
                .get_node_attr(0, &"scalar".into())
                .unwrap()
                .unwrap()
        ),
        bits(&values[1])
    );
}
