//! Requested Rust allocations for the shared exact child-frame workflow.
#[path = "support/pending_frames.rs"]
mod frames;
#[path = "support/pending_overwrites.rs"]
mod overwrites;
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
    let baseline = std::cell::Cell::new(0usize);
    let start = || {
        let current = LIVE.load(Relaxed);
        baseline.set(current);
        PEAK.store(current, Relaxed);
        CALLS.store(0, Relaxed);
        BYTES.store(0, Relaxed);
    };
    let stop = || {
        let peak = PEAK.load(Relaxed);
        let live = LIVE.load(Relaxed);
        let calls = CALLS.load(Relaxed);
        let bytes = BYTES.load(Relaxed);
        json!({"peak_additional_requested_bytes":peak.saturating_sub(baseline.get()),"end_additional_requested_bytes":live.saturating_sub(baseline.get()),"allocation_calls":calls,"requested_bytes":bytes})
    };
    let mut receipt = if std::env::args().any(|arg| arg == "--overwritten-values") {
        overwrites::run(start, stop)?
    } else {
        frames::run(start, stop)?
    };
    receipt["instrumentation"]=json!("Process-wide System allocator requested-byte counters; excludes allocator overhead/RSS, includes counter overhead in timing. Same fixture and instrumentation must be used for comparisons.");
    let path = std::env::var("GROGGY_ATOMIC_BUILDER_PROFILE")
        .unwrap_or_else(|_| "pending_frame_allocations.json".into());
    std::fs::write(&path, serde_json::to_vec_pretty(&receipt)?)?;
    println!("Wrote pending frame allocation receipt to {path}");
    Ok(())
}
