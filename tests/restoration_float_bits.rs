//! Property equality stays logical while accepted-state restoration keeps exact bits.
use groggy::{AttrValue as V, BatchEdgeRef as E, BatchNodeRef as N, Graph, GraphBatch};
use std::collections::HashMap;
fn values(negative: bool) -> HashMap<String, V> {
    let f = if negative { -0.0f32 } else { 0.0 };
    let d = if negative { -0.0f64 } else { 0.0 };
    let n = f32::from_bits(if negative { 0x7fc0007b } else { 0x7fc00001 });
    let q = f64::from_bits(if negative {
        0x7ff800000000007b
    } else {
        0x7ff8000000000001
    });
    HashMap::from([
        ("f32".into(), V::Float(f)),
        ("f64".into(), V::Float64(d)),
        ("nan32".into(), V::Float(n)),
        ("nan64".into(), V::Float64(q)),
        ("vec32".into(), V::FloatVec(vec![f, n])),
        ("vec64".into(), V::Float64Vec(vec![d, q])),
        (
            "compressed32".into(),
            V::FloatVec([vec![f; 30], vec![n]].concat()),
        ),
    ])
}
fn same_bits(a: V, b: V) {
    match (a, b) {
        (V::Float(a), V::Float(b)) => assert_eq!(a.to_bits(), b.to_bits()),
        (V::Float64(a), V::Float64(b)) => assert_eq!(a.to_bits(), b.to_bits()),
        (V::Float64Vec(a), V::Float64Vec(b)) => assert_eq!(
            a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        ),
        (a, V::FloatVec(b)) => {
            let a = match a {
                V::FloatVec(v) => v,
                V::CompressedFloatVec(v) => v.decompress_float_vec().unwrap(),
                _ => panic!("wrong value type"),
            };
            assert_eq!(
                a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
        _ => panic!("wrong type"),
    }
}
#[test]
fn switches_and_owned_archive_restore_exact_scalar_vector_and_compressed_float_bits_without_growth()
{
    let mut graph = Graph::new_publishing();
    graph
        .apply_batch(
            GraphBatch {
                add_nodes: 2,
                add_edges: vec![(N::Added(0), N::Added(1))],
                node_attrs: values(false)
                    .into_iter()
                    .map(|(name, v)| (name, vec![(N::Added(0), v)]))
                    .collect(),
                edge_attrs: values(false)
                    .into_iter()
                    .map(|(name, v)| (name, vec![(E::Added(0), v)]))
                    .collect(),
                effective_at: Some(10),
                ..Default::default()
            },
            0,
            "positive".into(),
            "test".into(),
        )
        .unwrap();
    let first = graph.current_state_id();
    graph.create_branch("positive".into()).unwrap();
    graph
        .apply_batch(
            GraphBatch {
                node_attrs: values(true)
                    .into_iter()
                    .map(|(name, v)| (name, vec![(N::Existing(0), v)]))
                    .collect(),
                edge_attrs: values(true)
                    .into_iter()
                    .map(|(name, v)| (name, vec![(E::Existing(0), v)]))
                    .collect(),
                effective_at: Some(20),
                ..Default::default()
            },
            first,
            "negative".into(),
            "test".into(),
        )
        .unwrap();
    let last = graph.current_state_id();
    let pin = graph.read_view().unwrap();
    let counts = |graph: &Graph| {
        values(false)
            .keys()
            .map(|name| {
                (
                    name.clone(),
                    (
                        graph.pool().get_node_attribute_column(name).unwrap().len(),
                        graph.pool().get_edge_attribute_column(name).unwrap().len(),
                    ),
                )
            })
            .collect::<HashMap<_, _>>()
    };
    let before = counts(&graph);
    for _ in 0..3 {
        graph.checkout_branch("positive".into()).unwrap();
        for (name, value) in values(false) {
            same_bits(
                graph.get_node_attr(0, &name).unwrap().unwrap(),
                value.clone(),
            );
            same_bits(graph.get_edge_attr(0, &name).unwrap().unwrap(), value);
        }
        graph.checkout_branch("main".into()).unwrap();
        for (name, value) in values(true) {
            same_bits(
                graph.get_node_attr(0, &name).unwrap().unwrap(),
                value.clone(),
            );
            same_bits(
                graph.get_edge_attr(0, &name).unwrap().unwrap(),
                value.clone(),
            );
            same_bits(pin.node_attr(0, &name).unwrap().unwrap(), value);
        }
    }
    assert_eq!(counts(&graph), before);
    let owned = graph
        .owned_snapshot_at_commit(last)
        .unwrap()
        .to_graph()
        .unwrap();
    assert_eq!(counts(&owned), before);
    for (name, value) in values(true) {
        same_bits(owned.get_node_attr(0, &name).unwrap().unwrap(), value);
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bits");
    graph.save_archive(&path).unwrap();
    let copy = Graph::load_archive(&path).unwrap();
    assert_eq!(counts(&copy), before);
    for (name, value) in values(true) {
        same_bits(copy.get_node_attr(0, &name).unwrap().unwrap(), value);
    }
}
