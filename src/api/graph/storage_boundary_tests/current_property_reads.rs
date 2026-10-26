//! The fixed family of twelve current-property readers shares index validation.
use crate::{AttrValue as V, Graph, GraphResult};
use std::collections::HashMap;

fn readers(graph: &Graph, node: bool, id: usize, name: &str) -> Vec<GraphResult<()>> {
    let name = name.to_string();
    if node {
        vec![
            graph.get_node_attr(id, &name).map(|_| ()),
            graph.get_node_attrs(id).map(|_| ()),
            graph
                .get_node_attrs_bulk(vec![id], vec![name.clone()])
                .map(|_| ()),
            graph.get_nodes_attrs(&name, &[id]).map(|_| ()),
            graph._get_node_attribute_column(&name).map(|_| ()),
            graph
                ._get_node_attributes_for_nodes(&[id], &name)
                .map(|_| ()),
        ]
    } else {
        vec![
            graph.get_edge_attr(id, &name).map(|_| ()),
            graph.get_edge_attrs(id).map(|_| ()),
            graph
                .get_edge_attrs_bulk(vec![id], vec![name.clone()])
                .map(|_| ()),
            graph.get_edges_attrs(&name, &[id]).map(|_| ()),
            graph._get_edge_attribute_column(&name).map(|_| ()),
            graph
                ._get_edge_attributes_for_edges(&[id], &name)
                .map(|_| ()),
        ]
    }
}

#[test]
fn every_current_reader_rejects_missing_columns_and_out_of_bounds_indices() {
    for node in [false, true] {
        for column_exists in [false, true] {
            let mut graph = Graph::new_manual();
            graph.add_nodes(2).unwrap();
            graph.add_edge(0, 1).unwrap();
            if column_exists {
                if node {
                    graph.set_node_attr(0, "v".into(), V::Int(7)).unwrap();
                } else {
                    graph.set_edge_attr(0, "v".into(), V::Int(7)).unwrap();
                }
            }
            let before = graph.commit("before".into(), "test".into()).unwrap();
            let pin = graph.read_view().unwrap();
            if node {
                graph.space_mut().set_node_attr_index(0, "v".into(), 99);
            } else {
                graph.space_mut().set_edge_attr_index(0, "v".into(), 99);
            }
            for result in readers(&graph, node, 0, "v") {
                let error = result.unwrap_err().to_string();
                assert!(error.contains("Missing retained value"), "{error}");
                assert!(error.contains("attribute v, index 99"), "{error}");
            }
            assert_eq!(graph.current_state_id(), before);
            assert!(!graph.has_uncommitted_changes());
            let frozen = if node {
                pin.node_attr(0, &"v".into())
            } else {
                pin.edge_attr(0, &"v".into())
            };
            assert_eq!(
                frozen.unwrap(),
                if column_exists { Some(V::Int(7)) } else { None }
            );
            // A projection of genuinely absent/unrelated keys remains independent.
            let rows = if node {
                graph.get_node_attrs_bulk(vec![0], vec!["absent".into()])
            } else {
                graph.get_edge_attrs_bulk(vec![0], vec!["absent".into()])
            };
            assert_eq!(rows.unwrap()[&0], HashMap::new());
        }
    }
}

#[test]
fn aligned_columns_preserve_order_duplicates_null_absence_and_liveness() {
    let mut graph = Graph::new_manual();
    graph.add_nodes(3).unwrap();
    graph.add_edges(&[(0, 0), (0, 1), (1, 2)]).unwrap();
    for node in [false, true] {
        if node {
            graph.set_node_attr(0, "v".into(), V::Null).unwrap();
            graph
                .set_node_attr(1, "v".into(), V::Float64(1.0000000000000002))
                .unwrap();
        } else {
            graph.set_edge_attr(0, "v".into(), V::Null).unwrap();
            graph
                .set_edge_attr(1, "v".into(), V::Float64(1.0000000000000002))
                .unwrap();
        }
        let ids = [1, 0, 99, 2, 1];
        let expected = vec![
            Some(V::Float64(1.0000000000000002)),
            Some(V::Null),
            None,
            None,
            Some(V::Float64(1.0000000000000002)),
        ];
        let actual = if node {
            graph.get_nodes_attrs(&"v".into(), &ids)
        } else {
            graph.get_edges_attrs(&"v".into(), &ids)
        };
        assert_eq!(actual.unwrap(), expected);
        let rows = if node {
            graph.get_node_attrs_bulk(vec![0, 2], vec!["v".into(), "absent".into()])
        } else {
            graph.get_edge_attrs_bulk(vec![0, 2], vec!["v".into(), "absent".into()])
        };
        assert_eq!(
            rows.unwrap(),
            HashMap::from([
                (0, HashMap::from([("v".into(), V::Null)])),
                (2, HashMap::new())
            ])
        );
        let missing = if node {
            graph.get_node_attr(99, &"v".into()).map(|_| ())
        } else {
            graph.get_edge_attr(99, &"v".into()).map(|_| ())
        };
        assert!(missing.is_err());
        let missing_rows = if node {
            graph.get_node_attrs_bulk(vec![0, 99], vec!["v".into()])
        } else {
            graph.get_edge_attrs_bulk(vec![0, 99], vec!["v".into()])
        };
        assert!(missing_rows.is_err());
    }
    graph.remove_node(0).unwrap();
    assert_eq!(
        graph.get_nodes_attrs(&"v".into(), &[0]).unwrap(),
        vec![None]
    );
    assert_eq!(
        graph.get_edges_attrs(&"v".into(), &[0]).unwrap(),
        vec![None]
    );
    assert!(graph.get_node_attrs(0).is_err());
    assert!(graph.get_edge_attrs(0).is_err());
}

#[test]
fn bulk_requested_projection_ignores_corruption_outside_its_scope() {
    let mut graph = Graph::new_manual();
    graph.add_nodes(2).unwrap();
    graph.add_edges(&[(0, 0), (1, 1)]).unwrap();
    graph
        .set_node_attr(0, "v".into(), V::Bytes(vec![0, 255]))
        .unwrap();
    graph
        .set_edge_attr(0, "v".into(), V::TextVec(vec!["kept".into()]))
        .unwrap();
    graph.space_mut().set_node_attr_index(1, "v".into(), 99);
    graph.space_mut().set_edge_attr_index(1, "v".into(), 99);
    assert_eq!(
        graph
            .get_node_attrs_bulk(vec![0], vec!["v".into()])
            .unwrap()[&0]["v"],
        V::Bytes(vec![0, 255])
    );
    assert_eq!(
        graph
            .get_edge_attrs_bulk(vec![0], vec!["v".into()])
            .unwrap()[&0]["v"],
        V::TextVec(vec!["kept".into()])
    );
    assert!(graph
        .get_node_attrs_bulk(vec![], vec!["v".into()])
        .unwrap()
        .is_empty());
    assert!(graph
        .get_edge_attrs_bulk(vec![], vec!["v".into()])
        .unwrap()
        .is_empty());
    assert!(graph.get_nodes_attrs(&"v".into(), &[]).unwrap().is_empty());
    assert!(graph.get_edges_attrs(&"v".into(), &[]).unwrap().is_empty());
}
