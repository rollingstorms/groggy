//! Per-phase requested Rust allocations for the shared native workflow.
#[path = "support/core_workflow.rs"]
mod workflow;
use serde_json::json;
use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicUsize, Ordering::Relaxed},
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
    let mut phases = Vec::new();
    let mut baseline = LIVE.load(Relaxed);
    PEAK.store(baseline, Relaxed);
    let result = workflow::run(|name| {
        let live = LIVE.load(Relaxed);
        let peak = PEAK.load(Relaxed);
        let calls = CALLS.load(Relaxed);
        let bytes = BYTES.load(Relaxed);
        phases.push(json!({"phase":name,"live_requested_bytes":live,
            "start_live_requested_bytes":baseline,
            "peak_additional_requested_bytes":peak.saturating_sub(baseline),
            "allocation_calls":calls,"requested_bytes":bytes}));
        baseline = LIVE.load(Relaxed);
        PEAK.store(baseline, Relaxed);
        CALLS.store(0, Relaxed);
        BYTES.store(0, Relaxed);
    })?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "scope":"Instrumented shared fixture; process-wide requested Rust allocations, not allocator overhead/RSS. Counter overhead affects timing. Compare timings only with the same instrumented binary configuration.",
            "workflow":result,"phases":phases
        }))?
    );
    Ok(())
}
