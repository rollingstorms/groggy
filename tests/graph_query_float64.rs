use groggy::{
    query::{NodeFilter, QueryError, QueryParser},
    AttrValue as V, Graph,
};
use std::collections::HashSet;
fn ids(values: impl IntoIterator<Item = usize>) -> HashSet<usize> {
    values.into_iter().collect()
}
#[test]
fn floating_literals_keep_precision_width_and_scientific_notation() {
    let mut graph = Graph::new_directed_manual();
    let nodes = graph.add_nodes(5).unwrap();
    for (node, value) in nodes.iter().zip([
        V::Float64(1.),
        V::Float64(1. + 2f64.powi(-40)),
        V::Float64(1e300),
        V::Float(1.),
        V::Int(9_007_199_254_740_993),
    ]) {
        graph
            .set_node_attr(*node, "x".into(), value.clone())
            .unwrap();
        let edge = graph.add_edge(*node, *node).unwrap();
        graph.set_edge_attr(edge, "x".into(), value).unwrap();
    }
    let frozen = graph.read_view().unwrap();
    let mut parser = QueryParser::new();
    let cases = [
        ("x == 1.0", vec![nodes[0]]),
        ("x == 1f32", vec![nodes[3]]),
        ("x == 1.0000000000009095f64", vec![nodes[1]]),
        ("x == +1E+300", vec![nodes[2]]),
        ("x > 9007199254740992.0", vec![nodes[2], nodes[4]]),
        ("x >= 1e300", vec![nodes[2]]),
        ("x < .5", vec![]),
    ];
    for (query, expected) in cases {
        let filter = parser.parse_node_query(query).unwrap();
        let expected = ids(expected);
        assert_eq!(
            ids(graph.find_nodes(filter.clone()).unwrap()),
            expected,
            "live {query}"
        );
        assert_eq!(
            ids(frozen.filter_nodes(&filter).unwrap().node_ids()),
            expected,
            "frozen {query}"
        );
        let filter = parser.parse_edge_query(query).unwrap();
        assert_eq!(
            ids(graph.find_edges(filter.clone()).unwrap()),
            expected,
            "live edge {query}"
        );
        assert_eq!(
            ids(frozen.filter_edges(&filter).unwrap().edge_ids()),
            expected,
            "frozen edge {query}"
        );
    }
    graph
        .set_node_attr(nodes[1], "x".into(), V::Float64(0.))
        .unwrap();
    let filter = parser.parse_node_query("x == 1.0000000000009095").unwrap();
    assert!(graph.find_nodes(filter.clone()).unwrap().is_empty());
    assert_eq!(
        frozen.filter_nodes(&filter).unwrap().node_ids(),
        vec![nodes[1]]
    );
}
#[test]
fn invalid_numeric_tokens_and_trailing_input_reject_for_both_entity_kinds() {
    let mut parser = QueryParser::new();
    for query in [
        "x > 1e",
        "x == 1e+",
        "x == 1e400",
        "x == 1e40f32",
        "x == 1.2.3",
        "x == 1f16",
        "x == 1_000",
        "x == 1.0junk",
        "x == 9223372036854775808",
        "x == 1 2",
        "x == 1 trailing",
        "x == 1)",
        "x == 'unterminated",
        "x == --1",
        "x == .",
    ] {
        assert!(
            parser.parse_node_query(query).is_err(),
            "node accepted {query}"
        );
        assert!(
            parser.parse_edge_query(query).is_err(),
            "edge accepted {query}"
        );
    }
    assert!(matches!(
        parser.parse_node_query("label == 'λ' AND x == 1e+"),
        Err(QueryError::InvalidValue { position: 23, .. })
    ));
    for literal in [
        "-9223372036854775808",
        "9223372036854775807",
        "1e-300",
        "-.5",
        ".5f32",
        "1.0f64",
    ] {
        assert!(parser.parse_node_query(&format!("x == {literal}")).is_ok());
    }
    let NodeFilter::AttributeEquals { value, .. } = parser.parse_node_query("x == -0.0").unwrap()
    else {
        panic!("expected equality")
    };
    let V::Float64(value) = value else {
        panic!("expected f64")
    };
    assert_eq!(value.to_bits(), (-0.0f64).to_bits());
}
