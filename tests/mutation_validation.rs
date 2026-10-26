use groggy::{AttrValue, Graph};
use std::collections::HashMap;

#[test]
fn node_creation_rejects_all_reserved_properties_before_allocating() {
    let mut graph = Graph::new_manual();
    let row = HashMap::from([("entity_type".into(), AttrValue::Text("meta".into()))]);
    let error = graph
        .add_nodes_with_attrs(vec![HashMap::new(), row.clone(), row])
        .unwrap_err()
        .to_string();
    assert!(error.contains("1.entity_type"), "{error}");
    assert!(error.contains("2.entity_type"), "{error}");
    assert!(graph.node_ids().is_empty());
    assert!(!graph.has_uncommitted_changes());
    assert_eq!(graph.add_node().unwrap(), 0);
}

#[test]
fn attributed_node_creation_preserves_sparse_rows_and_history() {
    let mut graph = Graph::new_manual();
    let rows = vec![
        HashMap::from([("name".into(), AttrValue::Text("Ada".into()))]),
        HashMap::new(),
        HashMap::from([("score".into(), AttrValue::Int(3))]),
    ];
    let ids = graph.add_nodes_with_attrs(rows).unwrap();
    let state = graph.commit("created".into(), "test".into()).unwrap();
    let snapshot = graph.snapshot_at_commit(state).unwrap();
    assert_eq!(
        snapshot.node_attr(ids[0], &"name".into()),
        Some(AttrValue::Text("Ada".into()))
    );
    assert_eq!(snapshot.node_attr(ids[1], &"name".into()), None);
    assert_eq!(
        snapshot.node_attr(ids[2], &"score".into()),
        Some(AttrValue::Int(3))
    );
}

#[test]
fn invalid_bulk_removals_preserve_topology_and_change_tracking() {
    let mut graph = Graph::new_manual();
    let nodes = graph.add_nodes(3).unwrap();
    let edges = graph
        .add_edges(&[(nodes[0], nodes[1]), (nodes[1], nodes[2])])
        .unwrap();
    graph.commit("initial".into(), "test".into()).unwrap();

    let error = graph
        .remove_nodes(&[nodes[0], 90, nodes[0], 91])
        .unwrap_err()
        .to_string();
    for part in ["1: node 90", "2: duplicate node", "3: node 91"] {
        assert!(error.contains(part), "missing {part}: {error}");
    }
    let error = graph
        .remove_edges(&[edges[0], 90, edges[0], 91])
        .unwrap_err()
        .to_string();
    for part in ["1: edge 90", "2: duplicate edge", "3: edge 91"] {
        assert!(error.contains(part), "missing {part}: {error}");
    }
    assert_eq!(graph.node_ids().len(), 3);
    assert_eq!(graph.edge_ids().len(), 2);
    assert!(!graph.has_uncommitted_changes());

    // Adjacent removals share an incident edge; it must be removed only once.
    graph.remove_nodes(&nodes[..2]).unwrap();
    assert_eq!(graph.node_ids(), vec![nodes[2]]);
    assert!(graph.edge_ids().is_empty());
    graph.remove_nodes(&[]).unwrap();
    graph.remove_edges(&[]).unwrap();
}

#[test]
fn preflight_reports_original_positions_without_writing() {
    let mut graph = Graph::new_manual();
    let node = graph.add_node().unwrap();
    graph.commit("initial".into(), "test".into()).unwrap();
    let attrs = HashMap::from([(
        "score".into(),
        vec![
            (node, AttrValue::Int(1)),
            (90, AttrValue::Null),
            (node, AttrValue::Int(3)),
        ],
    )]);
    let positions = HashMap::from([("score".into(), vec![1, 2, 3])]);
    let error = graph
        .validate_node_attrs(&attrs, Some(&positions))
        .unwrap_err()
        .to_string();
    assert!(error.contains("score[2]: node 90"), "{error}");
    assert!(
        error.contains("score[3]: duplicate node 0 (first at score[1])"),
        "{error}"
    );
    assert_eq!(graph.get_node_attr(node, &"score".into()).unwrap(), None);
    assert!(!graph.has_uncommitted_changes());
}
