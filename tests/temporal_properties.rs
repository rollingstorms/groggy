use groggy::{
    query::AttributeFilter, storage::array::BaseArray, AttrValue as V, BatchNodeRef as N,
    CalendarDate as Date, DurableGraph, FixedDuration as Duration, Graph, GraphBatch, GraphConfig,
    GraphSchema, UtcInstant as Instant,
};
use serde_json::json;
use std::collections::{HashMap, HashSet};

fn values() -> Vec<V> {
    vec![
        V::UtcInstant(Instant::new(-1, 999_999).unwrap()),
        V::Date(Date::new(2000, 2, 29).unwrap()),
        V::Duration(Duration::new(-1, 86_399, 999_999).unwrap()),
    ]
}

#[test]
fn temporal_constructors_validate_canonical_bounds_and_leap_years() {
    for (year, month, day) in [(1, 1, 1), (9999, 12, 31), (2000, 2, 29)] {
        assert!(Date::new(year, month, day).is_ok());
    }
    for (year, month, day) in [(0, 1, 1), (10000, 1, 1), (1900, 2, 29), (2024, 4, 31)] {
        assert!(Date::new(year, month, day).is_err());
    }
    let date = Date::new(1969, 12, 31).unwrap();
    let instant = Instant::from_calendar(date, 23, 59, 59, 999_999).unwrap();
    assert_eq!(instant, Instant::new(-1, 999_999).unwrap());
    assert_eq!(instant.calendar().unwrap(), (date, 23, 59, 59));
    assert_eq!(instant.to_string(), "1969-12-31T23:59:59.999999Z");
    assert!(Instant::new(0, 1_000_000).is_err());
    assert!(Instant::new(i64::MAX, 0).is_err());
    assert!(Instant::from_calendar(date, 24, 0, 0, 0).is_err());
    assert!(Instant::from_calendar(date, 23, 59, 60, 0).is_err());
    for invalid in [
        (-1_000_000_000, 0, 0),
        (1_000_000_000, 0, 0),
        (0, 86_400, 0),
        (0, 0, 1_000_000),
    ] {
        assert!(Duration::new(invalid.0, invalid.1, invalid.2).is_err());
    }
    assert!(Duration::new(-999_999_999, 0, 0).is_ok());
    assert!(Duration::new(999_999_999, 86_399, 999_999).is_ok());
}

#[test]
fn temporal_identity_ordering_and_predicates_remain_distinct_from_numeric_values() {
    let source = values();
    assert_eq!(
        source
            .iter()
            .chain(source.iter())
            .collect::<HashSet<_>>()
            .len(),
        3
    );
    for value in source {
        assert_eq!(value.clone().optimize(), value);
        assert!(value.numeric_type().is_none());
        assert_ne!(value, V::Int(0));
        assert_ne!(value, V::Text(value.to_string()));
        let array = BaseArray::new(vec![value.clone(), V::Null]);
        assert_eq!(array.compare_scalar(&value, "==").unwrap(), [true, false]);
        assert_eq!(array.compare_scalar(&value, "<=").unwrap(), [true, true]);
        assert!(array.compare_scalar(&V::Int(0), ">").is_err());
        assert!(AttributeFilter::GreaterThanOrEqual(value.clone()).matches(&value));
        assert!(!AttributeFilter::GreaterThan(V::Int(0)).matches(&value));
    }
    for pair in [
        (
            V::UtcInstant(Instant::new(-1, 999_999).unwrap()),
            V::UtcInstant(Instant::new(0, 0).unwrap()),
        ),
        (
            V::Date(Date::new(2000, 2, 29).unwrap()),
            V::Date(Date::new(2000, 3, 1).unwrap()),
        ),
        (
            V::Duration(Duration::new(-1, 86_399, 999_999).unwrap()),
            V::Duration(Duration::new(0, 0, 0).unwrap()),
        ),
    ] {
        assert!(pair.0 < pair.1);
        assert!(AttributeFilter::LessThan(pair.1).matches(&pair.0));
    }
}

#[test]
fn temporal_history_export_schema_and_journal_round_trip_losslessly() {
    let fixture = Graph::load_from_path(std::path::Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/temporal_properties_v1.json"
    )))
    .unwrap();
    for (i, value) in values().into_iter().enumerate() {
        assert_eq!(
            fixture.get_node_attr(0, &format!("p{i}")).unwrap(),
            Some(value)
        );
    }
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("temporal.json");
    let mut graph = Graph::new();
    let node = graph.add_node();
    let edge = graph.add_edge(node, node).unwrap();
    graph
        .set_node_attr(node, "labels".into(), V::TextVec(vec!["Event".into()]))
        .unwrap();
    for (i, value) in values().into_iter().enumerate() {
        graph
            .set_node_attr(node, format!("p{i}"), value.clone())
            .unwrap();
        graph.set_edge_attr(edge, format!("p{i}"), value).unwrap();
    }
    graph.commit("temporal".into(), "test".into()).unwrap();
    let frozen = graph.read_view().unwrap();
    graph.set_node_attr(node, "p0".into(), V::Null).unwrap();
    assert_eq!(
        frozen.node_attr(node, &"p0".into()).unwrap(),
        Some(values()[0].clone())
    );
    graph.reset_hard().unwrap();
    let schema = GraphSchema::from_json(
        &json!({"node_labels":{"Event":{
            "p0":{"type":"utc_instant"}, "p1":{"type":"date"}, "p2":{"type":"duration"}
        }}})
        .to_string(),
    )
    .unwrap();
    assert!(graph.validate_schema(&schema).unwrap().is_empty());
    graph.save_to_path(&path).unwrap();
    let loaded = Graph::load_from_path(&path).unwrap();
    for (i, value) in values().into_iter().enumerate() {
        assert_eq!(
            loaded.get_node_attr(node, &format!("p{i}")).unwrap(),
            Some(value.clone())
        );
        assert_eq!(
            loaded.get_edge_attr(edge, &format!("p{i}")).unwrap(),
            Some(value)
        );
    }
    let journal = directory.path().join("temporal.ggj");
    let mut durable = DurableGraph::create(&journal, GraphConfig::new(), None).unwrap();
    let attrs = values()
        .into_iter()
        .enumerate()
        .map(|(i, v)| (format!("p{i}"), vec![(N::Added(0), v)]))
        .collect::<HashMap<_, _>>();
    durable
        .apply_batch(
            GraphBatch {
                add_nodes: 1,
                node_attrs: attrs,
                ..Default::default()
            },
            0,
            "temporal".into(),
            "test".into(),
        )
        .unwrap();
    drop(durable);
    let recovered = DurableGraph::open(&journal).unwrap();
    for (i, value) in values().into_iter().enumerate() {
        assert_eq!(
            recovered
                .graph()
                .unwrap()
                .get_node_attr(0, &format!("p{i}"))
                .unwrap(),
            Some(value)
        );
    }
}

#[test]
fn temporal_deserialization_rejects_unknown_versions_and_invalid_components() {
    for payload in [
        json!({"Date":{"version":1,"year":1900,"month":2,"day":29}}),
        json!({"Date":{"version":2,"year":2000,"month":2,"day":29}}),
        json!({"Duration":{"version":1,"days":0,"seconds":86400,"microseconds":0}}),
        json!({"UtcInstant":{"version":1,"seconds":0,"microseconds":1000000}}),
        json!({"UtcInstant":{"version":1,"seconds":0,"microseconds":0,"extra":0}}),
    ] {
        assert!(serde_json::from_value::<V>(payload).is_err());
    }
    for value in values() {
        assert_eq!(
            serde_json::from_str::<V>(&serde_json::to_string(&value).unwrap()).unwrap(),
            value
        );
    }
}
