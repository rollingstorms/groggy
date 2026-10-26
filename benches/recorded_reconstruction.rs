//! Sequential immutable replay versus borrowed single-buffer recorded reconstruction.
use groggy::{
    state::{change_tracker::ChangeSet, Delta, GraphSnapshot},
    AttrValue,
};
use serde_json::json;
use std::{
    alloc::{GlobalAlloc, Layout, System},
    hint::black_box,
    sync::atomic::{AtomicUsize, Ordering::Relaxed},
    time::Instant,
};
struct Counting;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static CALLS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
fn allocate(size: usize) {
    let live = LIVE.fetch_add(size, Relaxed) + size;
    PEAK.fetch_max(live, Relaxed);
    CALLS.fetch_add(1, Relaxed);
    BYTES.fetch_add(size, Relaxed);
}
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc(layout);
        if !p.is_null() {
            allocate(layout.size());
        }
        p
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc_zeroed(layout);
        if !p.is_null() {
            allocate(layout.size());
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        System.dealloc(p, layout);
        LIVE.fetch_sub(layout.size(), Relaxed);
    }
    unsafe fn realloc(&self, p: *mut u8, old: Layout, size: usize) -> *mut u8 {
        let p = System.realloc(p, old, size);
        if !p.is_null() {
            LIVE.fetch_sub(old.size(), Relaxed);
            allocate(size);
        }
        p
    }
}
#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for (nodes, versions, adds) in [
        (1_000, 1_000, false),
        (10_000, 1_000, false),
        (0, 5_000, true),
    ] {
        let mut base = GraphSnapshot::empty(0);
        base.active_nodes = (0..nodes).collect();
        base.node_attributes = (0..nodes)
            .map(|id| {
                (
                    id,
                    std::collections::HashMap::from([(
                        "x".into(),
                        AttrValue::Text(format!("payload-{id:08}")),
                    )]),
                )
            })
            .collect();
        let deltas: Vec<_> = (0..versions)
            .map(|i| {
                let mut change = ChangeSet::new();
                if adds {
                    change.nodes_added.push(i);
                } else {
                    change.node_attr_changes.push((
                        i % nodes,
                        "x".into(),
                        None,
                        Some(AttrValue::Int(i as i64)),
                    ));
                }
                (Delta::from_changes(change), i as u64 + 1)
            })
            .collect();
        let mut samples = Vec::new();
        for trial in 0..3 {
            for mode in if trial % 2 == 0 {
                ["sequential", "borrowed"]
            } else {
                ["borrowed", "sequential"]
            } {
                let baseline = LIVE.load(Relaxed);
                PEAK.store(baseline, Relaxed);
                CALLS.store(0, Relaxed);
                BYTES.store(0, Relaxed);
                let started = Instant::now();
                let snapshot = black_box(if mode == "sequential" {
                    deltas
                        .iter()
                        .fold(base.clone(), |s, (d, id)| s.apply_delta(d, *id))
                } else {
                    GraphSnapshot::reconstruct_from_deltas(Some(&base), &deltas)?
                });
                let seconds = started.elapsed().as_secs_f64();
                let calls = CALLS.load(Relaxed);
                let bytes = BYTES.load(Relaxed);
                let peak = PEAK.load(Relaxed).saturating_sub(baseline);
                assert_eq!(snapshot.state_id, versions as u64);
                assert_eq!(
                    snapshot.active_nodes.len(),
                    if adds { versions } else { nodes }
                );
                samples.push(json!({"trial":trial+1,"mode":mode,"seconds":seconds,"allocation_calls":calls,"requested_bytes":bytes,"peak_additional_requested_live_bytes":peak}));
                drop(snapshot);
            }
        }
        cases.push(
            json!({"initial_nodes":nodes,"versions":versions,"scalar_adds":adds,"samples":samples}),
        );
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"benchmark":"recorded_reconstruction","scope":"same inputs, alternating modes, release, process-wide Rust requested allocations; not allocator/RSS or FFI metrics; public immutable sequential replay is reference","cases":cases})
        )?
    );
    Ok(())
}
