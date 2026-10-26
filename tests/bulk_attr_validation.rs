use groggy::{AttrValue, Graph};
use std::collections::HashMap;

#[test]
fn node_bulk_reports_all_invalid_assignments_without_writing() {
    let mut graph = Graph::new_manual();
    let node = graph.add_node().unwrap();
    graph.commit("initial".into(), "test".into()).unwrap();

    let mut values = HashMap::new();
    values.insert(
        "score".into(),
        vec![
            (node, AttrValue::Int(1)),
            (100, AttrValue::Int(2)),
            (node, AttrValue::Int(3)),
            (101, AttrValue::Int(4)),
        ],
    );
    values.insert("label".into(), vec![(102, AttrValue::Text("bad".into()))]);

    let error = graph.set_node_attrs(values).unwrap_err().to_string();
    for part in [
        "label[0]: node 102",
        "score[1]: node 100",
        "score[2]: duplicate node 0 (first at score[0])",
        "score[3]: node 101",
    ] {
        assert!(error.contains(part), "missing {part}: {error}");
    }
    assert_eq!(graph.get_node_attr(node, &"score".into()).unwrap(), None);
    assert!(!graph.has_uncommitted_changes());
}

#[test]
fn edge_bulk_reports_deleted_and_unknown_edges_without_writing() {
    let mut graph = Graph::new_manual();
    let nodes = graph.add_nodes(2).unwrap();
    let live = graph.add_edge(nodes[0], nodes[1]).unwrap();
    let deleted = graph.add_edge(nodes[0], nodes[1]).unwrap();
    graph.remove_edge(deleted).unwrap();
    graph.commit("initial".into(), "test".into()).unwrap();

    let mut values = HashMap::new();
    values.insert(
        "weight".into(),
        vec![
            (live, AttrValue::Float(1.0)),
            (deleted, AttrValue::Float(2.0)),
            (500, AttrValue::Float(3.0)),
        ],
    );
    let error = graph.set_edge_attrs(values).unwrap_err().to_string();
    assert!(error.contains("weight[1]: edge 1"), "{error}");
    assert!(error.contains("weight[2]: edge 500"), "{error}");
    assert_eq!(graph.get_edge_attr(live, &"weight".into()).unwrap(), None);
    assert!(!graph.has_uncommitted_changes());
}

#[test]
fn column_setter_rejects_duplicates_and_invalid_ids_without_writing() {
    let mut graph = Graph::new_manual();
    let node = graph.add_node().unwrap();
    graph.commit("initial".into(), "test".into()).unwrap();

    let error = graph
        .set_node_attr_column(
            "score".into(),
            vec![
                (node, AttrValue::Int(1)),
                (30, AttrValue::Int(2)),
                (node, AttrValue::Int(3)),
                (31, AttrValue::Int(4)),
            ],
        )
        .unwrap_err()
        .to_string();
    for part in [
        "score[1]: node 30",
        "score[2]: duplicate node 0 (first at score[0])",
        "score[3]: node 31",
    ] {
        assert!(error.contains(part), "missing {part}: {error}");
    }
    assert_eq!(graph.get_node_attr(node, &"score".into()).unwrap(), None);
    assert!(!graph.has_uncommitted_changes());
}

#[test]
fn bulk_node_setters_cannot_change_system_entity_type() {
    let mut graph = Graph::new_manual();
    let node = graph.add_node().unwrap();
    let mut attrs = HashMap::new();
    attrs.insert(
        "entity_type".into(),
        vec![(node, AttrValue::Text("meta".into()))],
    );
    assert!(graph
        .set_node_attrs(attrs)
        .unwrap_err()
        .to_string()
        .contains("entity_type"));
    assert!(graph
        .set_node_attr_column(
            "entity_type".into(),
            vec![(node, AttrValue::Text("meta".into()))],
        )
        .unwrap_err()
        .to_string()
        .contains("entity_type"));
    assert_eq!(
        graph.get_node_attr(node, &"entity_type".into()).unwrap(),
        Some(AttrValue::Text("base".into()))
    );
}
