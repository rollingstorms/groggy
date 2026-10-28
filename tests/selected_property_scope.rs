use groggy::{subgraphs::Subgraph, traits::SubgraphOperations, AttrValue as V, Graph};
use std::collections::HashMap;

#[test]
fn selected_property_reads_check_membership_without_changing_explicit_owner_writeback() {
    for publishing in [false, true] {
        let mut graph = if publishing {
            Graph::new()
        } else {
            Graph::new_manual()
        };
        graph.add_nodes(3).unwrap();
        graph.add_edges(&[(0, 1), (0, 1)]).unwrap();
        graph.set_node_attr(2, "value".into(), V::Int(7)).unwrap();
        graph.set_edge_attr(1, "value".into(), V::Int(8)).unwrap();
        if !publishing {
            graph.commit("fixture".into(), "test".into()).unwrap();
        }
        let owner = graph.into_shared();
        let selected = Subgraph::new(owner.clone(), [0, 1].into(), [0].into(), "selected".into());
        let head = owner.borrow().current_state_id();
        assert!(selected
            .get_node_attribute(2, &"value".into())
            .unwrap_err()
            .to_string()
            .contains("outside"));
        assert!(selected
            .get_edge_attribute(1, &"value".into())
            .unwrap_err()
            .to_string()
            .contains("outside"));
        assert_eq!(
            selected.get_node_attribute(0, &"value".into()).unwrap(),
            None
        );
        assert_eq!(
            selected.get_edge_attribute(0, &"value".into()).unwrap(),
            None
        );
        assert_eq!(owner.borrow().current_state_id(), head);
        // Explicit native writeback targets owner IDs. Historical diff maps
        // intentionally use this even for an empty invocation selection.
        selected
            .set_node_attrs(HashMap::from([("value".into(), vec![(2, V::Int(9))])]))
            .unwrap();
        selected
            .set_edge_attrs(HashMap::from([("value".into(), vec![(1, V::Int(10))])]))
            .unwrap();
        assert_eq!(
            owner.borrow().get_node_attr(2, &"value".into()).unwrap(),
            Some(V::Int(9))
        );
        assert_eq!(
            owner.borrow().get_edge_attr(1, &"value".into()).unwrap(),
            Some(V::Int(10))
        );
        assert!(selected.get_node_attribute(2, &"value".into()).is_err());
        assert!(selected.get_edge_attribute(1, &"value".into()).is_err());
        owner.borrow_mut().remove_node(0).unwrap();
        assert!(selected.get_node_attribute(0, &"value".into()).is_err());
        assert!(selected.get_edge_attribute(0, &"value".into()).is_err());
    }
}
