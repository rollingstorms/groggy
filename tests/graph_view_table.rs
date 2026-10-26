use groggy::{storage::table::Table, subgraphs::Subgraph, AttrValue, Graph};
use std::{cell::RefCell, rc::Rc};

#[test]
fn sparse_properties_follow_node_and_edge_identity_in_live_and_frozen_exports() {
    let graph = Rc::new(RefCell::new(Graph::new_manual()));
    let mut owner = graph.borrow_mut();
    let nodes = owner.add_nodes(4).unwrap();
    let edges = [
        owner.add_edge(nodes[0], nodes[2]).unwrap(),
        owner.add_edge(nodes[2], nodes[3]).unwrap(),
    ];
    owner
        .set_node_attr(nodes[3], "late".into(), AttrValue::Int(77))
        .unwrap();
    owner
        .set_node_attr(nodes[0], "first".into(), AttrValue::Bool(true))
        .unwrap();
    owner
        .set_node_attr(nodes[2], "explicit_null".into(), AttrValue::Null)
        .unwrap();
    owner
        .set_edge_attr(edges[1], "late".into(), AttrValue::Int(88))
        .unwrap();
    let view = owner.read_view().unwrap();
    drop(owner);
    let legacy = Subgraph::new(
        graph.clone(),
        nodes.iter().copied().collect(),
        edges.into_iter().collect(),
        "table_regression".into(),
    );
    for table in [
        graph.borrow().table().unwrap(),
        legacy.table().unwrap(),
        view.table().unwrap(),
    ] {
        assert_eq!(table.nodes().node_ids().unwrap(), nodes);
        assert_eq!(
            table.nodes().column_names(),
            &["node_id", "entity_type", "explicit_null", "first", "late"]
        );
        assert_eq!(
            table.nodes().column("late").unwrap().as_slice(),
            &[
                AttrValue::Null,
                AttrValue::Null,
                AttrValue::Null,
                AttrValue::Int(77)
            ]
        );
        assert_eq!(
            table.nodes().column("first").unwrap().get(0),
            Some(&AttrValue::Bool(true))
        );
        assert_eq!(
            table.edges().column("late").unwrap().as_slice(),
            &[AttrValue::Null, AttrValue::Int(88)]
        );
        assert_eq!(table.edges().edge_ids().unwrap(), edges);
        assert_eq!(table.edges().sources().unwrap(), vec![nodes[0], nodes[2]]);
        assert_eq!(table.edges().targets().unwrap(), vec![nodes[2], nodes[3]]);
    }
    graph
        .borrow_mut()
        .set_node_attr(nodes[3], "late".into(), AttrValue::Int(999))
        .unwrap();
    assert_eq!(
        view.nodes_table().unwrap().column("late").unwrap().get(3),
        Some(&AttrValue::Int(77))
    );
    assert_eq!(
        legacy.nodes_table().unwrap().column("late").unwrap().get(3),
        Some(&AttrValue::Int(999))
    );
    graph.borrow_mut().remove_node(nodes[3]).unwrap();
    assert!(legacy.nodes_table().is_err());
    assert!(legacy.edges_table().is_err());
    assert_eq!(view.edges_table().unwrap().nrows(), 2);
}

#[test]
fn selected_tables_keep_values_types_empty_schema_and_ownership() {
    let mut graph = Graph::new_manual();
    let nodes = graph.add_nodes(3).unwrap();
    let first = graph.add_edge(nodes[0], nodes[1]).unwrap();
    let excluded = graph.add_edge(nodes[0], nodes[1]).unwrap();
    graph
        .set_edge_attr(excluded, "excluded".into(), AttrValue::Int(99))
        .unwrap();
    graph
        .set_node_attr(nodes[1], "vector".into(), AttrValue::FloatVec(vec![1., 2.]))
        .unwrap();
    graph
        .set_node_attr(nodes[1], "large".into(), AttrValue::Int(i64::MAX))
        .unwrap();
    let view = graph.read_view().unwrap().select_edges(&[first]).unwrap();
    drop(graph);
    let mut table = view.table().unwrap();
    assert_eq!(table.nodes().node_ids().unwrap(), &nodes[..2]);
    assert!(!table.edges().has_column("excluded"));
    assert_eq!(
        table.nodes().column("large").unwrap().get(1),
        Some(&AttrValue::Int(i64::MAX))
    );
    assert_eq!(
        table.nodes().column("vector").unwrap().get(1),
        Some(&AttrValue::FloatVec(vec![1., 2.]))
    );
    *table.nodes_mut() = table
        .nodes()
        .with_column(
            "local".into(),
            groggy::storage::array::BaseArray::from_attr_values(vec![AttrValue::Int(1); 2]),
        )
        .unwrap();
    assert!(!view.nodes_table().unwrap().has_column("local"));
    let empty = view.select_nodes(&[]).unwrap().table().unwrap();
    assert_eq!(empty.nodes().column_names(), &["node_id"]);
    assert_eq!(
        empty.edges().column_names(),
        &["edge_id", "source", "target"]
    );
    assert_eq!(empty.nodes().nrows(), 0);
    assert_eq!(empty.edges().nrows(), 0);
}

#[test]
fn structural_column_collisions_fail_without_changing_graph_properties() {
    let mut graph = Graph::new_manual();
    let nodes = graph.add_nodes(2).unwrap();
    let edge = graph.add_edge(nodes[0], nodes[1]).unwrap();
    graph
        .set_node_attr(nodes[0], "node_id".into(), AttrValue::Int(999))
        .unwrap();
    graph
        .set_edge_attr(edge, "source".into(), AttrValue::Int(999))
        .unwrap();
    let view = graph.read_view().unwrap();
    assert!(view
        .nodes_table()
        .unwrap_err()
        .to_string()
        .contains("structural"));
    assert!(view
        .edges_table()
        .unwrap_err()
        .to_string()
        .contains("structural"));
    assert_eq!(
        view.node_attr(nodes[0], &"node_id".into()).unwrap(),
        Some(AttrValue::Int(999))
    );
    assert_eq!(view.edge_endpoints(edge).unwrap(), (nodes[0], nodes[1]));
    assert!(view
        .select_nodes(&[nodes[1]])
        .unwrap()
        .nodes_table()
        .is_ok());
}
