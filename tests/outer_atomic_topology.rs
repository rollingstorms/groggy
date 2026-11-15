//! Multiple builders share one topology plan, original selection and acceptance.
use groggy::algorithms::{global_registry, AlgorithmOutput, Context, Pipeline, PipelineBuilder};
use groggy::ownership::OwnerCell;
use groggy::{subgraphs::Subgraph, AttrValue, BatchNodeRef as N, Graph, GraphBatch};
use serde_json::{json, Value};
use std::sync::Arc;

fn fixture(directed: bool) -> Arc<OwnerCell<Graph>> {
    let mut graph = if directed {
        Graph::new_directed()
    } else {
        Graph::new_publishing()
    };
    graph
        .apply_batch(
            GraphBatch {
                add_nodes: 7,
                add_edges: [(0, 2), (2, 4), (2, 4), (4, 4), (5, 6)]
                    .into_iter()
                    .map(|(s, t)| (N::Added(s), N::Added(t)))
                    .collect(),
                effective_at: Some(0),
                ..Default::default()
            },
            0,
            "fixture".into(),
            "test".into(),
        )
        .unwrap();
    let head = graph.current_state_id();
    graph
        .apply_batch(
            GraphBatch {
                remove_nodes: vec![N::Existing(1), N::Existing(3)],
                effective_at: Some(1),
                ..Default::default()
            },
            head,
            "sparse IDs".into(),
            "test".into(),
        )
        .unwrap();
    Arc::new(OwnerCell::new(graph))
}

fn selected(owner: &Arc<OwnerCell<Graph>>) -> Subgraph {
    let view = owner
        .borrow()
        .read_view()
        .unwrap()
        .select_nodes(&[0, 2, 4])
        .unwrap();
    Subgraph::from_frozen_view(owner.clone(), view, "selected".into())
}

fn pipeline(programs: Vec<Value>) -> Pipeline {
    let mut builder = PipelineBuilder::new();
    for steps in programs {
        builder = builder.with_algorithm("builder.step_pipeline", |params| {
            params.set_text("steps", json!({"atomic":true,"steps":steps}).to_string());
        });
    }
    builder.build(global_registry()).unwrap()
}

fn create_nodes() -> Value {
    json!([{"id":"graph.add_nodes","params":{
        "count":1,"output":"created","attributes":{"score":1.0000000000000002},"effective_at":10
    }}])
}

fn connect(at: i64) -> Value {
    json!([
        {"id":"graph.add_nodes","params":{"count":1,"output":"more","effective_at":12}},
        {"id":"graph.add_edges","params":{"output":"edges","effective_at":at,
            "edges":[[0,{"added":0}],[{"added":0},{"added":1}],
                     [{"added":0},{"added":1}],[{"added":1},{"added":1}]]}},
        {"id":"graph.capture_staged","params":{"output":"pending"}},
        {"id":"graph.emit","params":{"source":"pending","name":"before_removal"}}
    ])
}

#[test]
fn shared_additions_captures_and_cascades_publish_one_exact_selection() {
    for directed in [false, true] {
        let graph = fixture(directed);
        let retained = graph.borrow().read_view().unwrap();
        let head = graph.borrow().current_state_id();
        let mut context = Context::new();
        let result = pipeline(vec![create_nodes(), connect(20), json!([
            {"id":"graph.remove_nodes","params":{"ids":[2],"effective_at":40}},
            {"id":"graph.capture_staged","params":{"output":"removed"}},
            {"id":"graph.emit","params":{"source":"removed","name":"after_removal"}}
        ]), json!([
            {"id":"graph.set_edge_endpoints","params":{"revisions":[[0,0,4]],"effective_at":30}}
        ]), json!([
            {"id":"graph.capture_staged","params":{"output":"final"}},
            {"id":"graph.algorithm_table","params":{"source":"final","output":"paths","algorithm":"bfs","start":0}},
            {"id":"graph.emit","params":{"source":"final","name":"final"}},
            {"id":"table.emit","params":{"source":"paths","name":"paths"}}
        ])]).run_atomic_properties(&mut context, selected(&graph)).unwrap();
        assert_eq!(graph.borrow().current_state_id(), head + 1);
        assert_eq!(
            result
                .nodes()
                .iter()
                .copied()
                .collect::<std::collections::HashSet<_>>(),
            [0, 4, 7, 8].into()
        );
        assert_eq!(
            result
                .edges()
                .iter()
                .copied()
                .collect::<std::collections::HashSet<_>>(),
            [0, 3, 5, 6, 7, 8].into()
        );
        assert!(graph.borrow().has_edge(4)); // Omitted outside topology stays untouched.
        let outputs = context.take_outputs();
        let AlgorithmOutput::GraphView(before) = &outputs["before_removal"] else {
            panic!("graph")
        };
        let AlgorithmOutput::GraphView(after) = &outputs["final"] else {
            panic!("graph")
        };
        let AlgorithmOutput::GraphView(removed) = &outputs["after_removal"] else {
            panic!("graph")
        };
        assert!(!removed.edge_ids().contains(&0));
        assert_eq!(before.node_ids(), vec![0, 2, 4, 7, 8]);
        assert_eq!(before.edge_endpoints(0).unwrap(), (0, 2));
        assert_eq!(after.node_ids(), vec![0, 4, 7, 8]);
        assert_eq!(after.edge_endpoints(0).unwrap(), (0, 4));
        assert_eq!(
            after.node_attr(7, &"score".into()).unwrap(),
            Some(AttrValue::Float64(1.0000000000000002))
        );
        assert_eq!(after.write_state_id(), None);
        assert_eq!(retained.edge_endpoints(0).unwrap(), (0, 2));
        assert!(!retained.contains_node(7));
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("outer.archive");
        graph.borrow().save_archive(&path).unwrap();
        let recovered = Graph::load_archive(&path).unwrap();
        assert_eq!(recovered.current_state_id(), head + 1);
        assert_eq!(recovered.edge_endpoints(0).unwrap(), (0, 4));
        assert_eq!(
            recovered
                .snapshot_at_commit(head)
                .unwrap()
                .read_state()
                .snapshot()
                .edges[&0],
            (0, 2)
        );
    }
}

#[test]
fn late_execution_and_final_timeline_errors_restore_outputs_and_ids() {
    for programs in [
        vec![
            create_nodes(),
            connect(20),
            json!([{"id":"graph.remove_edges","params":{"ids":[999]}}]),
        ],
        vec![create_nodes(), connect(5)], // Edge precedes its newly created endpoints.
    ] {
        let graph = fixture(false);
        let head = graph.borrow().current_state_id();
        let mut context = Context::new();
        context.add_output("original", AlgorithmOutput::Histogram(vec![11]));
        assert!(pipeline(programs)
            .run_atomic_properties(&mut context, selected(&graph))
            .is_err());
        assert_eq!(graph.borrow().current_state_id(), head);
        assert!(!graph.borrow().has_node(7));
        let outputs = context.take_outputs();
        assert_eq!(outputs.len(), 1);
        assert!(matches!(&outputs["original"], AlgorithmOutput::Histogram(v) if v == &[11]));
        // Reuse the same context successfully; failed staging reserves no IDs.
        let result = pipeline(vec![create_nodes()])
            .run_atomic_properties(&mut context, selected(&graph))
            .unwrap();
        assert!(result.nodes().contains(&7));
        assert_eq!(graph.borrow().current_state_id(), head + 1);
        assert_eq!(graph.borrow_mut().add_node().unwrap(), 8);
        assert_eq!(graph.borrow_mut().add_edge(0, 4).unwrap(), 5);
    }
}

#[test]
fn unadapted_algorithm_after_topology_rejects_before_all_prefixes() {
    let graph = fixture(false);
    let head = graph.borrow().current_state_id();
    let mut builder = PipelineBuilder::new().with_algorithm("builder.step_pipeline", |params| {
        params.set_text(
            "steps",
            json!({"atomic":true,"steps":create_nodes()}).to_string(),
        );
    });
    builder = builder.with_algorithm("centrality.pagerank", |_| {});
    let error = builder
        .build(global_registry())
        .unwrap()
        .run_atomic_properties(&mut Context::new(), selected(&graph))
        .unwrap_err();
    assert!(error
        .to_string()
        .contains("cannot run after staged topology"));
    assert_eq!(graph.borrow().current_state_id(), head);
    assert_eq!(graph.borrow_mut().add_node().unwrap(), 7);
}

#[test]
fn outer_projection_does_not_induce_omitted_parallel_edges_or_loops() {
    let graph = fixture(false);
    let view = graph
        .borrow()
        .read_view()
        .unwrap()
        .select_edges(&[1])
        .unwrap();
    let input = Subgraph::from_frozen_view(graph.clone(), view, "exact".into());
    let mut context = Context::new();
    let result = pipeline(vec![
        create_nodes(),
        json!([
            {"id":"graph.capture_staged","params":{"output":"pending"}},
            {"id":"graph.emit","params":{"source":"pending","name":"pending"}}
        ]),
    ])
    .run_atomic_properties(&mut context, input)
    .unwrap();
    assert_eq!(result.nodes().len(), 3);
    assert!(result.nodes().contains(&7));
    assert_eq!(result.edges().iter().copied().collect::<Vec<_>>(), vec![1]);
    let outputs = context.take_outputs();
    let AlgorithmOutput::GraphView(view) = &outputs["pending"] else {
        panic!("graph")
    };
    assert_eq!(view.node_ids(), vec![2, 4, 7]);
    assert_eq!(view.edge_ids(), vec![1]);
    assert!(graph.borrow().has_edge(2) && graph.borrow().has_edge(3));
}

#[test]
fn pending_properties_include_added_ids_and_exclude_cascaded_entities() {
    for directed in [false, true] {
        let graph = fixture(directed);
        let head = graph.borrow().current_state_id();
        let mut context = Context::new();
        let computations = json!([
            {"id":"graph.remove_nodes","params":{"ids":[2]}},
            {"id":"core.load_node_attr","params":{"target":"scores","attr":"score","default":2}},
            {"id":"core.init_scalar","params":{"target":"two","value":2}},
            {"id":"core.mul","params":{"left":"scores","right":"two","target":"scaled"}},
            {"id":"core.attach_node_attr","params":{"source":"scaled","attr":"scaled"}},
            {"id":"core.load_node_attr","params":{"target":"reload","attr":"scaled"}},
            {"id":"core.attach_node_attr","params":{"source":"reload","attr":"copied"}},
            {"id":"core.init_edges","params":{"target":"weights","value":5}},
            {"id":"core.attach_edge_attr","params":{"source":"weights","attr":"weight"}},
            {"id":"graph.capture_staged","params":{"output":"final"}},
            {"id":"graph.emit","params":{"source":"final","name":"final"}}
        ]);
        let result = pipeline(vec![create_nodes(), computations])
            .run_atomic_properties(&mut context, selected(&graph))
            .unwrap();
        assert_eq!(graph.borrow().current_state_id(), head + 1);
        assert_eq!(
            result
                .nodes()
                .iter()
                .copied()
                .collect::<std::collections::HashSet<_>>(),
            [0, 4, 7].into()
        );
        let outputs = context.take_outputs();
        let AlgorithmOutput::GraphView(view) = &outputs["final"] else {
            panic!("view")
        };
        assert_eq!(
            view.node_attr(7, &"scaled".into()).unwrap(),
            Some(AttrValue::Float64(2.0000000000000004))
        );
        assert_eq!(
            view.node_attr(7, &"copied".into()).unwrap(),
            view.node_attr(7, &"scaled".into()).unwrap()
        );
        assert_eq!(
            view.node_attr(0, &"scaled".into()).unwrap(),
            Some(AttrValue::Int(4))
        );
        assert_eq!(
            view.edge_attr(3, &"weight".into()).unwrap(),
            Some(AttrValue::Int(5))
        );
        assert!(!view.contains_node(2));
        assert!(!view.contains_edge(0));
    }
}

#[test]
fn stale_property_map_after_removal_rolls_back_and_context_can_be_reused() {
    let graph = fixture(false);
    let head = graph.borrow().current_state_id();
    let mut context = Context::new();
    let program = pipeline(vec![json!([
        {"id":"core.init_nodes","params":{"target":"old","value":7}},
        {"id":"graph.add_nodes","params":{"count":1,"output":"new"}},
        {"id":"graph.remove_nodes","params":{"ids":[2]}},
        {"id":"core.attach_node_attr","params":{"source":"old","attr":"stale"}}
    ])]);
    assert!(program
        .run_atomic_properties(&mut context, selected(&graph))
        .is_err());
    assert_eq!(graph.borrow().current_state_id(), head);
    assert!(graph.borrow().has_node(2));
    assert!(context.take_outputs().is_empty());
    pipeline(vec![create_nodes()])
        .run_atomic_properties(&mut context, selected(&graph))
        .unwrap();
    assert!(graph.borrow().has_node(7));
    assert!(!graph.borrow().has_node(8));
}

#[test]
fn structural_neighbor_and_path_primitives_read_one_pending_source() {
    for directed in [false, true] {
        let graph = fixture(directed);
        let head = graph.borrow().current_state_id();
        let mut context = Context::new();
        let program = json!([
            {"id":"graph.add_nodes","params":{"count":1,"output":"created"}},
            {"id":"graph.add_edges","params":{"edges":[[0,{"added":0}],[{"added":0},4]],"output":"connected","attributes":{"weight":2.5}}},
            {"id":"graph.set_edge_endpoints","params":{"revisions":[[0,4,0]]}},
            {"id":"graph.remove_nodes","params":{"ids":[2]}},
            {"id":"core.node_degree","params":{"target":"degrees"}},
            {"id":"core.attach_node_attr","params":{"source":"degrees","attr":"degree"}},
            {"id":"core.weighted_degree","params":{"weight_attr":"weight","target":"weights"}},
            {"id":"core.attach_node_attr","params":{"source":"weights","attr":"incidence"}},
            {"id":"core.load_node_attr","params":{"attr":"node_id","target":"ids"}},
            {"id":"core.neighbor_agg","params":{"source":"ids","target":"incoming","direction":"in","agg":"sum"}},
            {"id":"core.attach_node_attr","params":{"source":"incoming","attr":"incoming"}},
            {"id":"core.init_scalar","params":{"target":"start","value":0}},
            {"id":"core.shortest_path_map","params":{"source":"start","output":"paths","weight_attr":"weight"}},
            {"id":"core.attach_node_attr","params":{"source":"paths","attr":"distance"}},
            {"id":"graph.capture_staged","params":{"output":"final"}},
            {"id":"graph.emit","params":{"source":"final","name":"final"}}
        ]);
        pipeline(vec![program])
            .run_atomic_properties(&mut context, selected(&graph))
            .unwrap();
        assert_eq!(graph.borrow().current_state_id(), head + 1);
        let outputs = context.take_outputs();
        let AlgorithmOutput::GraphView(view) = &outputs["final"] else {
            panic!("view")
        };
        for (node, degree, weighted, incoming, distance) in [
            (0, if directed { 1 } else { 2 }, 3.5, 4.0, 0.0),
            (
                4,
                if directed { 2 } else { 3 },
                5.5,
                11.0,
                if directed { 5.0 } else { 1.0 },
            ),
            (7, 2 - i64::from(directed), 5.0, 0.0, 2.5),
        ] {
            assert_eq!(
                view.node_attr(node, &"degree".into()).unwrap(),
                Some(AttrValue::Int(degree))
            );
            assert_eq!(
                view.node_attr(node, &"incidence".into()).unwrap(),
                Some(AttrValue::Float64(weighted))
            );
            assert_eq!(
                view.node_attr(node, &"incoming".into()).unwrap(),
                Some(AttrValue::Float64(incoming))
            );
            assert_eq!(
                view.node_attr(node, &"distance".into()).unwrap(),
                Some(AttrValue::Float64(distance))
            );
        }
    }
}

#[test]
fn captured_computation_columns_attach_with_explicit_times_and_survive_recovery() {
    use groggy::state::publication::ChangeKey;
    let graph = fixture(false);
    let head = graph.borrow().current_state_id();
    let mut context = Context::new();
    let result = pipeline(vec![create_nodes(), json!([
        {"id":"graph.capture_staged","params":{"output":"pending"}},
        {"id":"graph.degrees_table","params":{"source":"pending","output":"degrees"}},
        {"id":"table.attach","params":{"source":"degrees","columns":{"degree":"computed_degree"},"effective_at":30}},
        {"id":"table.emit","params":{"source":"degrees","name":"degrees"}}
    ])]).run_atomic_properties(&mut context, selected(&graph)).unwrap();
    assert_eq!(result.node_count(), 4);
    assert_eq!(graph.borrow().current_state_id(), head + 1);
    assert_eq!(
        graph
            .borrow()
            .get_node_attr(7, &"computed_degree".into())
            .unwrap(),
        Some(AttrValue::Int(0))
    );
    let metadata = graph
        .borrow()
        .publication_metadata(head + 1)
        .unwrap()
        .unwrap();
    for node in [0, 2, 4, 7] {
        assert_eq!(
            metadata.change_times[&ChangeKey::NodeAttribute(node, "computed_degree".into())],
            30
        );
    }
    let outputs = context.take_outputs();
    let AlgorithmOutput::Table(table) = &outputs["degrees"] else {
        panic!("table")
    };
    assert_eq!(
        table.source_views()[0]
            .node_attr(7, &"computed_degree".into())
            .unwrap(),
        None
    );
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("captured-properties.archive");
    graph.borrow().save_archive(&path).unwrap();
    let recovered = Graph::load_archive(&path).unwrap();
    assert_eq!(
        recovered
            .get_node_attr(7, &"computed_degree".into())
            .unwrap(),
        Some(AttrValue::Int(0))
    );
    assert_eq!(
        recovered
            .read_view_at(29, None)
            .unwrap()
            .node_attr(7, &"computed_degree".into())
            .unwrap(),
        None
    );
    assert_eq!(
        recovered
            .read_view_at(30, None)
            .unwrap()
            .node_attr(7, &"computed_degree".into())
            .unwrap(),
        Some(AttrValue::Int(0))
    );
}

#[test]
fn nested_loops_refresh_pending_properties_and_keep_typed_capture_authority() {
    let graph = fixture(false);
    let head = graph.borrow().current_state_id();
    let mut context = Context::new();
    let increment = json!([
        {"id":"core.load_node_attr","params":{"target":"counter","attr":"counter","default":0}},
        {"id":"core.init_scalar","params":{"target":"one","value":1}},
        {"id":"core.add","params":{"left":"counter","right":"one","target":"next"}},
        {"id":"core.attach_node_attr","params":{"source":"next","attr":"counter"}},
        {"id":"core.load_node_attr","params":{"target":"reload","attr":"counter"}},
        {"id":"core.attach_node_attr","params":{"source":"reload","attr":"mirror"}}
    ]);
    pipeline(vec![create_nodes(), json!([
        {"id":"iter.loop","params":{"iterations":2,"body":[{"id":"iter.loop","params":{"iterations":3,"body":increment}}]}},
        {"id":"graph.capture_staged","params":{"output":"seed"}},
        {"id":"iter.loop","params":{"iterations":2,"loop_vars":[["seed","local"]],"body":[
            {"id":"graph.degrees_table","params":{"source":"local","output":"metrics"}},
            {"id":"table.attach","params":{"source":"metrics","columns":{"degree":"loop_degree"},"effective_at":30}}
        ]}}
    ])]).run_atomic_properties(&mut context, selected(&graph)).unwrap();
    assert_eq!(graph.borrow().current_state_id(), head + 1);
    for node in [0, 2, 4, 7] {
        assert_eq!(
            graph
                .borrow()
                .get_node_attr(node, &"counter".into())
                .unwrap(),
            Some(AttrValue::Int(6))
        );
        assert_eq!(
            graph
                .borrow()
                .get_node_attr(node, &"mirror".into())
                .unwrap(),
            Some(AttrValue::Int(6))
        );
    }
    assert_eq!(
        graph
            .borrow()
            .get_node_attr(7, &"loop_degree".into())
            .unwrap(),
        Some(AttrValue::Int(0))
    );
}

#[test]
fn pending_filters_fused_kernels_and_masks_share_one_selected_acceptance() {
    let graph = fixture(true);
    let head = graph.borrow().current_state_id();
    let old = graph.borrow().read_view().unwrap();
    pipeline(vec![create_nodes(), json!([
        {"id":"graph.add_edges","params":{"edges":[[0,{"added":0}],[{"added":0},4]],"output":"new_edges"}},
        {"id":"graph.set_edge_endpoints","params":{"revisions":[[0,4,0]]}},
        {"id":"graph.remove_nodes","params":{"ids":[2]}},
        {"id":"iter.loop","params":{"iterations":1,"body":[
            {"id":"core.load_node_attr","params":{"attr":"node_id","target":"ids"}},
            {"id":"core.init_nodes","params":{"target":"two","value":2}},
            {"id":"graph.fused_neighbor_mul_agg","params":{"values":"ids","scalars":"two","target":"fused","direction":"out"}},
            {"id":"core.attach_node_attr","params":{"source":"fused","attr":"neighbor_sum"}},
            {"id":"core.normalize_values","params":{"source":"ids","target":"normalized","method":"sum"}},
            {"id":"core.attach_node_attr","params":{"source":"normalized","attr":"normalized"}},
            {"id":"core.top_k","params":{"source":"ids","target":"top","k":2,"order":"desc"}},
            {"id":"core.attach_node_attr","params":{"source":"top","attr":"rank"}},
            {"id":"core.sample_nodes","params":{"target":"all","fraction":1.0,"seed":43}},
            {"id":"core.reservoir_sample","params":{"source":"all","target":"sample","k":2,"seed":43}},
            {"id":"core.attach_node_attr","params":{"source":"sample","attr":"sample"}},
            {"id":"core.community_seed","params":{"strategy":"singleton","target":"labels"}},
            {"id":"core.label_propagate_step","params":{"labels":"labels","target":"propagated"}},
            {"id":"core.attach_node_attr","params":{"source":"propagated","attr":"label"}}
        ]}}
    ])]).run_atomic_properties(&mut Context::new(), selected(&graph)).unwrap();
    let graph = graph.borrow();
    assert_eq!(graph.current_state_id(), head + 1);
    let mut sampled = 0;
    for (node, sum, label) in [(0, 14.0, 0), (4, 8.0, 4), (7, 8.0, 4)] {
        assert_eq!(
            graph.get_node_attr(node, &"neighbor_sum".into()).unwrap(),
            Some(AttrValue::Float64(sum))
        );
        assert_eq!(
            graph.get_node_attr(node, &"label".into()).unwrap(),
            Some(AttrValue::Int(label))
        );
        assert_eq!(
            graph.get_node_attr(node, &"normalized".into()).unwrap(),
            Some(AttrValue::Float64(node as f64 / 11.0))
        );
        sampled += usize::from(
            graph
                .get_node_attr(node, &"sample".into())
                .unwrap()
                .is_some(),
        );
    }
    assert_eq!(sampled, 2);
    assert_eq!(
        graph.get_node_attr(7, &"rank".into()).unwrap(),
        Some(AttrValue::Int(0))
    );
    assert_eq!(
        graph.get_node_attr(5, &"neighbor_sum".into()).unwrap(),
        None
    );
    assert_eq!(old.edge_endpoints(0).unwrap(), (0, 2));
    assert_eq!(old.node_attr(4, &"neighbor_sum".into()).unwrap(), None);
}
