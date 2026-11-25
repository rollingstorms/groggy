//! Owned aggregation inputs use identical kernels before one durable acceptance.
use groggy::{
    algorithms::{global_registry, AlgorithmOutput, Context, Pipeline, PipelineBuilder},
    subgraphs::Subgraph,
    AttrValue as V, BatchNodeRef as N, DurableGraph, Graph, GraphBatch,
};
use serde_json::{json, Value};
use std::fs;
fn pipeline(steps: Value) -> Pipeline {
    PipelineBuilder::new()
        .with_algorithm("builder.step_pipeline", |p| {
            p.set_text("steps", json!({"atomic":true,"steps":steps}).to_string());
        })
        .build(global_registry())
        .unwrap()
}
fn fixture() -> Graph {
    let mut g = Graph::new();
    g.apply_batch(
        GraphBatch {
            add_nodes: 6,
            node_attrs: [(
                "score".into(),
                [1, 999, 2, 3, 4, 999]
                    .into_iter()
                    .enumerate()
                    .map(|(i, v)| (N::Added(i), V::Int(v)))
                    .collect(),
            )]
            .into(),
            effective_at: Some(0),
            ..Default::default()
        },
        0,
        "fixture".into(),
        "test".into(),
    )
    .unwrap();
    g.apply_batch(
        GraphBatch {
            remove_nodes: vec![N::Existing(1)],
            effective_at: Some(0),
            ..Default::default()
        },
        g.current_state_id(),
        "sparse".into(),
        "test".into(),
    )
    .unwrap();
    g
}
fn statistic_steps(name: &str, params: Value) -> Value {
    let mut options = params;
    options["source"] = json!("values");
    options["target"] = json!("stat");
    json!([
        {"id":"core.load_node_attr","params":{"attr":"score","target":"values"}},
        {"id":"iter.loop","params":{"iterations":2,"body":[
            {"id":format!("core.{name}"),"params":options},
            {"id":"core.broadcast_scalar","params":{"scalar":"stat","reference":"values","target":"answer"}}
        ]}},
        {"id":"core.attach_node_attr","params":{"source":"answer","attr":"answer","effective_at":30}}
    ])
}
#[test]
fn every_aggregation_matches_expected_memory_worker_and_recovery_results() {
    for (name, options, expected) in [
        ("reduce_nodes", json!({"reducer":"sum"}), V::Float64(10.)),
        ("reduce_nodes", json!({"reducer":"mean"}), V::Float64(2.5)),
        ("reduce_nodes", json!({"reducer":"min"}), V::Float64(1.)),
        ("reduce_nodes", json!({"reducer":"max"}), V::Float64(4.)),
        ("std", json!({}), V::Float64((5.0f64 / 3.).sqrt())),
        ("median", json!({}), V::Float64(2.5)),
        ("quantile", json!({"q":0.25}), V::Float64(1.75)),
        ("mode", json!({}), V::Int(1)),
        ("entropy", json!({}), V::Float64(2.)),
        ("histogram", json!({"bins":2}), V::IntVec(vec![2, 2])),
    ] {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("aggregate.journal");
        let graph = fixture();
        let mut store = DurableGraph::create_from_graph(&path, &graph, None).unwrap();
        let source = store
            .read_view()
            .unwrap()
            .select_nodes(&[0, 2, 3, 4])
            .unwrap();
        let head = source.write_state_id().unwrap();
        let p = pipeline(statistic_steps(name, options));
        let plan = p.prepare_durable(&store, &source).unwrap();
        let mut ctx = Context::new();
        let pending = std::thread::scope(|threads| {
            threads
                .spawn(|| plan.execute(&mut ctx).unwrap())
                .join()
                .unwrap()
        });
        let accepted = pending.accept(&mut store).unwrap();
        assert_eq!(accepted.write_state_id(), Some(head + 1));
        let expected = if name == "std" {
            let Some(V::Float64(actual)) = accepted.node_attr(0, &"answer".into()).unwrap() else {
                panic!("floating sample standard deviation")
            };
            let V::Float64(reference) = expected else {
                panic!("floating reference")
            };
            assert!((actual - reference).abs() <= 4. * f64::EPSILON * reference.abs());
            V::Float64(actual)
        } else {
            expected
        };
        for node in [0, 2, 3, 4] {
            assert_eq!(
                accepted.node_attr(node, &"answer".into()).unwrap(),
                Some(expected.clone()),
                "{name}/{node}"
            );
        }
        assert_eq!(
            store
                .read_view()
                .unwrap()
                .node_attr(5, &"answer".into())
                .unwrap(),
            None
        );
        if name == "histogram" {
            let outputs = ctx.take_outputs();
            let AlgorithmOutput::Histogram(counts) = &outputs["histograms.stat"] else {
                panic!("histogram")
            };
            assert_eq!(counts, &vec![2, 2]);
        }
        let owner = graph.into_shared();
        let memory = Subgraph::from_frozen_view(
            owner.clone(),
            owner
                .borrow()
                .read_view()
                .unwrap()
                .select_nodes(&[0, 2, 3, 4])
                .unwrap(),
            "test".into(),
        );
        p.run(&mut Context::new(), memory).unwrap();
        assert_eq!(
            owner.borrow().get_node_attr(0, &"answer".into()).unwrap(),
            Some(expected.clone())
        );
        drop(store);
        let reopened = DurableGraph::open(&path).unwrap();
        assert_eq!(
            reopened
                .read_view_at(29, None)
                .unwrap()
                .node_attr(0, &"answer".into())
                .unwrap(),
            None
        );
        assert_eq!(
            reopened
                .read_view_at(30, None)
                .unwrap()
                .node_attr(0, &"answer".into())
                .unwrap(),
            Some(expected)
        );
    }
}
#[test]
fn read_only_histogram_returns_owned_counts_without_publishing_and_empty_maps_work() {
    for empty in [false, true] {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("readonly.journal");
        let mut store = DurableGraph::create_from_graph(&path, &fixture(), None).unwrap();
        let source = store
            .read_view()
            .unwrap()
            .select_nodes(if empty { &[] } else { &[0, 2, 3, 4] })
            .unwrap();
        let bytes = fs::read(&path).unwrap();
        let p = pipeline(json!([
            {"id":"core.load_node_attr","params":{"attr":"score","target":"values"}},
            {"id":"core.histogram","params":{"source":"values","target":"hist","bins":3}},
            {"id":"core.median","params":{"source":"values","target":"median"}},
            {"id":"core.mode","params":{"source":"values","target":"mode"}},
            {"id":"core.std","params":{"source":"values","target":"std"}},
            {"id":"core.entropy","params":{"source":"values","target":"entropy"}}
        ]));
        let mut ctx = Context::new();
        p.run_durable(&mut ctx, &mut store, &source).unwrap();
        let outputs = ctx.take_outputs();
        let AlgorithmOutput::Histogram(counts) = &outputs["histograms.hist"] else {
            panic!("histogram")
        };
        let expected = if empty { vec![0, 0, 0] } else { vec![1, 1, 2] };
        assert_eq!(counts, &expected);
        assert_eq!(fs::read(&path).unwrap(), bytes);
    }
}
#[test]
fn later_bad_input_restores_outputs_journal_and_allocations() {
    for name in ["reduce_nodes", "std", "median", "quantile", "histogram"] {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("reject.journal");
        let mut graph = fixture();
        graph
            .set_node_attr(0, "score".into(), V::Text("bad".into()))
            .unwrap();
        let mut store = DurableGraph::create_from_graph(&path, &graph, None).unwrap();
        let source = store.read_view().unwrap();
        let bytes = fs::read(&path).unwrap();
        let mut params = json!({"source":"bad","target":"answer"});
        if name == "reduce_nodes" {
            params["reducer"] = json!("sum");
        }
        if name == "quantile" {
            params["q"] = json!(0.5);
        }
        if name == "histogram" {
            params["bins"] = json!(2);
        }
        let p = pipeline(json!([
            {"id":"graph.add_nodes","params":{"count":1,"output":"new"}},
            {"id":"core.init_nodes","params":{"target":"good","value":1}},
            {"id":"core.histogram","params":{"source":"good","target":"discard","bins":2}},
            {"id":"core.load_node_attr","params":{"attr":"score","target":"bad","default":1}},
            {"id":format!("core.{name}"),"params":params}
        ]));
        let mut ctx = Context::new();
        ctx.add_output("retained", AlgorithmOutput::Histogram(vec![7]));
        assert!(
            p.run_durable(&mut ctx, &mut store, &source).is_err(),
            "{name}"
        );
        let outputs = ctx.take_outputs();
        assert_eq!(outputs.len(), 1);
        let AlgorithmOutput::Histogram(v) = &outputs["retained"] else {
            panic!("retained")
        };
        assert_eq!(v, &vec![7]);
        assert_eq!(fs::read(&path).unwrap(), bytes);
        assert_eq!(store.graph().unwrap().pool().statistics().node_count, 6);
    }
}
