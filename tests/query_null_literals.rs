use groggy::{
    query::{EdgeFilter, NodeFilter, QueryParser},
    AttrValue, Graph,
};
#[test]
fn null_is_a_contextual_typed_literal_for_both_entity_kinds() {
    for literal in ["null", "NULL", "nUlL"] {
        assert_eq!(
            QueryParser::new()
                .parse_node_query(&format!("x == {literal}"))
                .unwrap(),
            NodeFilter::AttributeEquals {
                name: "x".into(),
                value: AttrValue::Null
            }
        );
        assert_eq!(
            QueryParser::new()
                .parse_edge_query(&format!("x == {literal}"))
                .unwrap(),
            EdgeFilter::AttributeEquals {
                name: "x".into(),
                value: AttrValue::Null
            }
        );
    }
    assert_eq!(
        QueryParser::new().parse_node_query("null == null").unwrap(),
        NodeFilter::AttributeEquals {
            name: "null".into(),
            value: AttrValue::Null
        }
    );
    assert_eq!(
        QueryParser::new().parse_node_query("x == 'null'").unwrap(),
        NodeFilter::AttributeEquals {
            name: "x".into(),
            value: AttrValue::Text("null".into())
        }
    );
    for bad in [
        "x == null extra",
        "x == null null",
        "x == null)",
        "x == nul",
    ] {
        assert!(QueryParser::new().parse_node_query(bad).is_err());
        assert!(QueryParser::new().parse_edge_query(bad).is_err());
    }
}
#[test]
fn null_matching_preserves_explicit_null_versus_absence_and_text() {
    let mut graph = Graph::new();
    graph.insert_nodes(3).unwrap();
    graph.set_node_attr(0, "x".into(), AttrValue::Null).unwrap();
    graph
        .set_node_attr(1, "x".into(), AttrValue::Text("null".into()))
        .unwrap();
    let view = graph.read_view().unwrap();
    assert_eq!(
        view.filter_nodes(&QueryParser::new().parse_node_query("x == null").unwrap())
            .unwrap()
            .node_ids(),
        vec![0]
    );
    assert_eq!(
        view.filter_nodes(&QueryParser::new().parse_node_query("x != null").unwrap())
            .unwrap()
            .node_ids(),
        vec![1]
    );
    assert_eq!(
        view.filter_nodes(
            &QueryParser::new()
                .parse_node_query("NOT (x == null)")
                .unwrap()
        )
        .unwrap()
        .node_ids(),
        vec![1, 2]
    );
}
