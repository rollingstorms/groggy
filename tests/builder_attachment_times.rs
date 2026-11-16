//! Direct node/map/column and edge attachments share canonical per-property clocks.
use groggy::algorithms::{global_registry, Context, Pipeline, PipelineBuilder};
use groggy::ownership::OwnerCell;
use groggy::{subgraphs::Subgraph, AttrValue, BatchNodeRef as N, Graph, GraphBatch};
use serde_json::{json, Value};
use std::sync::Arc;

fn pipeline(steps: Value) -> anyhow::Result<Pipeline> {
    PipelineBuilder::new()
        .with_algorithm("builder.step_pipeline", |p| {
            p.set_text("steps", json!({"atomic":true,"steps":steps}).to_string());
        })
        .build(global_registry())
}
#[test]
fn timed_direct_attachments_reconstruct_maps_columns_and_added_entities() {
    for kind in ["nodes", "column", "edges"] {
        let mut graph = Graph::new();
        graph
            .apply_batch(
                GraphBatch {
                    add_nodes: 3,
                    add_edges: vec![(N::Added(0), N::Added(1)), (N::Added(1), N::Added(2))],
                    effective_at: Some(0),
                    ..Default::default()
                },
                0,
                "fixture".into(),
                "test".into(),
            )
            .unwrap();
        let head = graph.current_state_id();
        let owner = Arc::new(OwnerCell::new(graph));
        let input = Subgraph::from_frozen_view(
            owner.clone(),
            owner.borrow().read_view().unwrap(),
            "root".into(),
        );
        let load = match kind {
            "column" => {
                json!({"id":"core.community_seed","params":{"strategy":"singleton","target":"values"}})
            }
            "edges" => json!({"id":"core.init_edges","params":{"value":9,"target":"values"}}),
            _ => json!({"id":"core.init_nodes","params":{"value":9,"target":"values"}}),
        };
        let attach = if kind == "edges" {
            "core.attach_edge_attr"
        } else {
            "core.attach_node_attr"
        };
        pipeline(json!([
            {"id":"graph.add_nodes","params":{"count":1,"output":"new","effective_at":10}},
            {"id":"graph.add_edges","params":{"edges":[[0,{"added":0}]],"output":"edge","effective_at":20}},
            load,
            {"id":attach,"params":{"source":"values","attr":"timed","effective_at":80}}
        ])).unwrap().run(&mut Context::new(), input).unwrap();
        let graph = owner.borrow();
        assert_eq!(graph.current_state_id(), head + 1);
        let before = graph.read_view_at(79, None).unwrap();
        let after = graph.read_view_at(80, None).unwrap();
        let ids = if kind == "edges" {
            after.edge_ids()
        } else {
            after.node_ids()
        };
        for id in ids {
            let prior = if kind == "edges" {
                before.edge_attr(id, &"timed".into())
            } else {
                before.node_attr(id, &"timed".into())
            }
            .unwrap();
            let value = if kind == "edges" {
                after.edge_attr(id, &"timed".into())
            } else {
                after.node_attr(id, &"timed".into())
            }
            .unwrap();
            assert_eq!(prior, None);
            assert_eq!(
                value.as_ref().and_then(AttrValue::as_int),
                Some(if kind == "column" { id as i64 } else { 9 })
            );
        }
    }
}

#[test]
fn bad_direct_attachment_parameters_are_not_silently_ignored() {
    for id in ["core.attach_node_attr", "core.attach_edge_attr"] {
        for extra in [
            json!({"effective_at":true}),
            json!({"effective_at":1.5}),
            json!({"effective_time":40}),
        ] {
            let mut params = json!({"source":"values","attr":"timed"});
            params
                .as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            assert!(pipeline(json!([{"id":id,"params":params}])).is_err());
        }
    }
}
