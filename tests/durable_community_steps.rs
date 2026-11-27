//! Existing community helper kernels retain their definitions on detached reads.
use groggy::{
    algorithms::{global_registry, Context, Pipeline, PipelineBuilder},
    subgraphs::Subgraph,
    AttrValue as V, BatchNodeRef as N, DurableGraph, Graph, GraphBatch,
};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
};
fn pipeline(steps: Value) -> Pipeline {
    PipelineBuilder::new()
        .with_algorithm("builder.step_pipeline", |p| {
            p.set_text("steps", json!({"atomic":true,"steps":steps}).to_string());
        })
        .build(global_registry())
        .unwrap()
}
fn fixture(directed: bool) -> Graph {
    let mut g = if directed {
        Graph::new_directed()
    } else {
        Graph::new()
    };
    g.apply_batch(
        GraphBatch {
            add_nodes: 6,
            add_edges: [(0, 2), (0, 2), (2, 2), (2, 4), (4, 0), (4, 5)]
                .into_iter()
                .map(|(a, b)| (N::Added(a), N::Added(b)))
                .collect(),
            effective_at: Some(0),
            ..Default::default()
        },
        0,
        "fixture".into(),
        "test".into(),
    )
    .unwrap();
    g
}
fn objective(
    nodes: &[usize],
    edges: &[(usize, usize)],
    labels: &BTreeMap<usize, i64>,
    directed: bool,
) -> f64 {
    let m = edges.len() as f64;
    let mut out: BTreeMap<_, f64> = nodes.iter().map(|&n| (n, 0.0)).collect();
    let mut ins = out.clone();
    let mut internal: BTreeMap<i64, f64> = BTreeMap::new();
    for &(a, b) in edges {
        *out.get_mut(&a).unwrap() += 1.;
        *ins.get_mut(&b).unwrap() += 1.;
        if !directed {
            *out.get_mut(&b).unwrap() += 1.;
        }
        if labels[&a] == labels[&b] {
            *internal.entry(labels[&a]).or_default() += 1.;
        }
    }
    let mut totals: BTreeMap<i64, (f64, f64)> = BTreeMap::new();
    for &n in nodes {
        let t = totals.entry(labels[&n]).or_default();
        t.0 += out[&n];
        t.1 += ins[&n];
    }
    totals
        .into_iter()
        .map(|(label, (a, b))| {
            internal.get(&label).copied().unwrap_or(0.) / m
                - if directed {
                    a * b / (m * m)
                } else {
                    (a / (2. * m)).powi(2)
                }
        })
        .sum()
}
#[test]
fn seeded_labels_modularity_and_propagation_match_worker_memory_and_independent_definitions() {
    for directed in [false, true] {
        for strategy in ["singleton", "degree", "random"] {
            let graph = fixture(directed);
            let temp = tempfile::tempdir().unwrap();
            let path = temp.path().join("community.journal");
            let mut store = DurableGraph::create_from_graph(&path, &graph, None).unwrap();
            let selected = store.read_view().unwrap().select_nodes(&[0, 2, 4]).unwrap();
            let head = selected.write_state_id().unwrap();
            let p = pipeline(json!([
            {"id":"graph.add_nodes","params":{"count":1,"output":"new","effective_at":10}},
            {"id":"graph.add_edges","params":{"edges":[[{"added":0},0]],"output":"new_edge","effective_at":10}},
            {"id":"graph.set_edge_endpoints","params":{"revisions":[[0,4,2]],"effective_at":10}},
            {"id":"graph.remove_edges","params":{"ids":[1],"effective_at":10}},
            {"id":"core.community_seed","params":{"strategy":strategy,"k":3,"seed":42,"target":"labels"}},
            {"id":"alias","params":{"source":"labels","target":"saved"}},
            {"id":"iter.loop","params":{"iterations":2,"body":[{"id":"core.modularity_gain","params":{"partition":"labels","target":"gain"}},{"id":"core.label_propagate_step","params":{"labels":"labels","target":"next"}}]}},
            {"id":"core.attach_node_attr","params":{"source":"saved","attr":"seed","effective_at":20}},
            {"id":"core.attach_node_attr","params":{"source":"gain","attr":"gain","effective_at":20}},
            {"id":"core.attach_node_attr","params":{"source":"next","attr":"next","effective_at":20}}
            ]));
            let plan = p.prepare_durable(&store, &selected).unwrap();
            let pending = std::thread::scope(|s| {
                s.spawn(move || plan.execute(&mut Context::new()).unwrap())
                    .join()
                    .unwrap()
            });
            let result = pending.accept(&mut store).unwrap();
            assert_eq!(result.write_state_id(), Some(head + 1));
            let nodes = [0, 2, 4, 6];
            let edges = [(4, 2), (2, 2), (2, 4), (4, 0), (6, 0)];
            let labels: BTreeMap<_, _> = nodes
                .iter()
                .map(|&n| {
                    let Some(V::Int(label)) = result.node_attr(n, &"seed".into()).unwrap() else {
                        panic!("integer label")
                    };
                    (n, label)
                })
                .collect();
            let mut neighbors: BTreeMap<_, Vec<usize>> =
                nodes.iter().map(|&n| (n, vec![])).collect();
            for &(a, b) in &edges {
                neighbors.get_mut(&a).unwrap().push(b);
                if !directed && a != b {
                    neighbors.get_mut(&b).unwrap().push(a);
                }
            }
            for row in neighbors.values_mut() {
                row.sort_unstable();
            }
            if strategy == "singleton" {
                for &n in &nodes {
                    assert_eq!(labels[&n], n as i64);
                }
            }
            if strategy == "random" {
                assert!(labels.values().all(|&v| (0..3).contains(&v)));
            }
            if strategy == "degree" {
                let mut hubs = nodes.to_vec();
                hubs.sort_by_key(|n| (std::cmp::Reverse(neighbors[n].len()), *n));
                let mut assigned = BTreeMap::from([(hubs[0], hubs[0] as i64)]);
                for &n in &nodes {
                    let label = assigned
                        .get(&n)
                        .copied()
                        .or_else(|| neighbors[&n].iter().find_map(|m| assigned.get(m).copied()))
                        .unwrap_or(n as i64);
                    assigned.insert(n, label);
                    assert_eq!(labels[&n], label);
                }
            }
            let baseline = objective(&nodes, &edges, &labels, directed);
            for &n in &nodes {
                let mut candidates: BTreeSet<_> = neighbors[&n].iter().map(|m| labels[m]).collect();
                if directed {
                    for &(a, b) in &edges {
                        if b == n {
                            candidates.insert(labels[&a]);
                        }
                    }
                }
                let mut best = 0.0_f64;
                for label in candidates {
                    if label != labels[&n] {
                        let mut changed = labels.clone();
                        changed.insert(n, label);
                        best = best.max(objective(&nodes, &edges, &changed, directed) - baseline);
                    }
                }
                let Some(V::Float64(gain)) = result.node_attr(n, &"gain".into()).unwrap() else {
                    panic!("gain")
                };
                assert!((gain - best).abs() < 1e-12);
                let mut votes: BTreeMap<i64, usize> = BTreeMap::new();
                for neighbor in std::iter::once(&n).chain(neighbors[&n].iter()) {
                    *votes.entry(labels[neighbor]).or_default() += 1;
                }
                let expected = votes
                    .into_iter()
                    .min_by_key(|(label, count)| (std::cmp::Reverse(*count), *label))
                    .unwrap()
                    .0;
                assert_eq!(
                    result.node_attr(n, &"next".into()).unwrap(),
                    Some(V::Int(expected))
                );
            }
            assert_eq!(
                store
                    .read_view()
                    .unwrap()
                    .node_attr(5, &"seed".into())
                    .unwrap(),
                None
            );
            let owner = graph.into_shared();
            let memory = Subgraph::from_frozen_view(
                owner.clone(),
                owner
                    .borrow()
                    .read_view()
                    .unwrap()
                    .select_nodes(&[0, 2, 4])
                    .unwrap(),
                "test".into(),
            );
            p.run(&mut Context::new(), memory).unwrap();
            for n in nodes {
                for attr in ["seed", "gain", "next"] {
                    assert_eq!(
                        owner.borrow().get_node_attr(n, &attr.into()).unwrap(),
                        result.node_attr(n, &attr.into()).unwrap()
                    );
                }
            }
            drop(store);
            let reopened = DurableGraph::open(&path).unwrap();
            assert_eq!(
                reopened
                    .read_view_at(19, None)
                    .unwrap()
                    .node_attr(0, &"seed".into())
                    .unwrap(),
                None
            );
            assert_eq!(
                reopened
                    .read_view_at(20, None)
                    .unwrap()
                    .node_attr(0, &"seed".into())
                    .unwrap(),
                Some(V::Int(labels[&0]))
            );
        }
    }
}
#[test]
fn malformed_partition_rejects_staged_properties_and_allocations() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("reject.journal");
    let mut store = DurableGraph::create_from_graph(&path, &fixture(false), None).unwrap();
    let selected = store.read_view().unwrap();
    let before = fs::read(&path).unwrap();
    let p = pipeline(
        json!([{"id":"graph.add_nodes","params":{"count":1,"output":"new"}},{"id":"core.init_nodes","params":{"value":true,"target":"bad"}},{"id":"core.attach_node_attr","params":{"source":"bad","attr":"tentative"}},{"id":"core.modularity_gain","params":{"partition":"bad","target":"out"}}]),
    );
    assert!(p
        .run_durable(&mut Context::new(), &mut store, &selected)
        .is_err());
    assert_eq!(fs::read(&path).unwrap(), before);
    assert_eq!(store.graph().unwrap().pool().statistics().node_count, 6);
}
#[test]
fn read_only_community_helpers_do_not_publish() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("read.journal");
    let mut store = DurableGraph::create_from_graph(&path, &fixture(false), None).unwrap();
    let selected = store.read_view().unwrap();
    let before = fs::read(&path).unwrap();
    let p = pipeline(
        json!([{"id":"core.community_seed","params":{"strategy":"singleton","target":"labels"}},{"id":"core.modularity_gain","params":{"partition":"labels","target":"gain"}},{"id":"core.label_propagate_step","params":{"labels":"labels","target":"next"}}]),
    );
    let result = p
        .run_durable(&mut Context::new(), &mut store, &selected)
        .unwrap();
    assert_eq!(result.write_state_id(), selected.write_state_id());
    assert_eq!(fs::read(&path).unwrap(), before);
}
