//! Release numerical corpus with process-wide allocation receipts (single runner).
use groggy::{
    storage::{
        matrix::{MatrixStats, UnifiedMatrix},
        Axis, GraphMatrix, NumArray,
    },
    GraphResult,
};
use serde_json::{json, Value};
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
fn measure(
    name: &str,
    shape: &[usize],
    mut operation: impl FnMut() -> GraphResult<f64>,
) -> GraphResult<Value> {
    black_box(operation()?);
    let mut samples = Vec::new();
    for _ in 0..7 {
        let baseline = LIVE.load(Relaxed);
        PEAK.store(baseline, Relaxed);
        CALLS.store(0, Relaxed);
        BYTES.store(0, Relaxed);
        let start = Instant::now();
        let checksum = black_box(operation()?);
        let seconds = start.elapsed().as_secs_f64();
        let (calls, bytes, peak) = (
            CALLS.load(Relaxed),
            BYTES.load(Relaxed),
            PEAK.load(Relaxed).saturating_sub(baseline),
        );
        samples.push(json!({"seconds":seconds,"allocation_calls":calls,"requested_bytes":bytes,"peak_additional_live_bytes":peak,"checksum":checksum}));
    }
    let median = |key: &str| {
        let mut v: Vec<f64> = samples
            .iter()
            .map(|v| v[key].as_f64().unwrap_or(0.))
            .collect();
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    Ok(
        json!({"name":name,"shape":shape,"median_seconds":median("seconds"),"median_allocation_calls":median("allocation_calls"),"median_requested_bytes":median("requested_bytes"),"median_peak_additional_live_bytes":median("peak_additional_live_bytes"),"samples":samples}),
    )
}
fn matrix(n: usize) -> GraphMatrix<f64> {
    GraphMatrix::from_row_major_data(
        (0..n * n)
            .map(|i| ((i * 31 % 127) as f64 - 63.) / 71.)
            .collect(),
        n,
        n,
        None,
    )
    .expect("fixture")
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    if cfg!(debug_assertions) {
        return Err("Run this corpus with cargo bench --bench numerical_completion".into());
    }
    let mut cases = Vec::new();
    let a = matrix(64);
    let b = a.transpose()?;
    cases.push(measure("dense_fractional_matmul", &[64, 64, 64], || {
        a.multiply(&b)?.sum()
    })?);
    let ai = GraphMatrix::from_row_major_data(
        (0..64 * 64).map(|i| (i % 11) as i64 - 5).collect(),
        64,
        64,
        None,
    )?;
    cases.push(measure("dense_integer_matmul", &[64, 64, 64], || {
        Ok(ai.multiply(&ai)?.sum()? as f64)
    })?);
    let reductions = matrix(256);
    cases.push(measure("dense_axis_mean", &[256, 256], || {
        Ok(reductions.mean_axis(Axis::Columns)?.iter().sum())
    })?);
    cases.push(measure("dense_axis_std", &[256, 256], || {
        Ok(reductions.std_axis(Axis::Columns)?.iter().sum())
    })?);
    cases.push(measure("dense_norm", &[256, 256], || reductions.norm())?);
    let x = NumArray::new((0..100_000).map(|i| (i % 127) as f64 / 131.).collect());
    cases.push(measure("vector_fractional_add", &[100000], || {
        Ok(x.checked_add(&x)?.get(0).copied().unwrap_or(0.))
    })?);
    cases.push(measure("vector_fractional_dot", &[100000], || x.dot(&x))?);
    let x = NumArray::new((0..100_000).map(|i| i as i64).collect());
    cases.push(measure("vector_integer_dot", &[100000], || {
        Ok(x.dot(&x)? as f64)
    })?);
    let n = 10000;
    let sparse = GraphMatrix::from_csr(n, n, vec![4.; n], (0..n).collect(), (0..=n).collect())?;
    cases.push(measure("csr_diagonal_matmul", &[n, n], || {
        sparse.multiply(&sparse)?.sum()
    })?);
    cases.push(measure("csr_column_std", &[n, n], || {
        Ok(sparse.std_axis(Axis::Columns)?.iter().sum())
    })?);
    cases.push(measure("csr_transpose", &[n, n], || {
        sparse.transpose()?.sum()
    })?);
    cases.push(measure("csr_cholesky", &[n, n], || {
        sparse.cholesky_decomposition()?.sum()
    })?);
    cases.push(measure("csr_lu", &[n, n], || {
        sparse.lu_decomposition()?.2.sum()
    })?);
    let rhs = GraphMatrix::<f64>::ones(n, 1);
    cases.push(measure("csr_solve", &[n, n, 1], || {
        sparse.solve(&rhs)?.sum()
    })?);
    cases.push(measure(
        "csr_numerical_transfer_roundtrip",
        &[n, n],
        || sparse.numerical_data()?.into_matrix()?.sum(),
    )?);
    cases.push(measure(
        "dense_numerical_transfer_roundtrip",
        &[256, 256],
        || reductions.numerical_data()?.into_matrix()?.sum(),
    )?);
    let a = matrix(24);
    let b = a.transpose()?;
    let spd = a.multiply(&b)?.add(&GraphMatrix::identity(24)?)?;
    cases.push(measure("dense_qr", &[24, 24], || {
        a.qr_decomposition()?.1.sum()
    })?);
    cases.push(measure("dense_svd", &[24, 24], || {
        Ok(a.svd()?.1.iter().sum())
    })?);
    cases.push(measure("symmetric_eigen", &[24, 24], || {
        Ok(spd.eigenvalue_decomposition()?.0.iter().sum())
    })?);
    let low = UnifiedMatrix::from_data((0..65536).map(|i| i as i64).collect(), 256, 256)?;
    cases.push(measure(
        "lower_integer_population_std",
        &[256, 256],
        || MatrixStats::std(&low).map_err(|e| groggy::GraphError::InvalidInput(e.to_string())),
    )?);
    let rectangular = GraphMatrix::from_row_major_data(
        (0..512 * 4)
            .map(|i| ((i * 31 % 127) as f64 - 63.) / 71.)
            .collect(),
        512,
        4,
        None,
    )?;
    cases.push(measure("rectangular_full_svd", &[512, 4], || {
        Ok(rectangular.svd()?.1.iter().sum())
    })?);
    cases.push(measure("rectangular_thin_svd", &[512, 4], || {
        Ok(rectangular.svd_thin()?.1.iter().sum())
    })?);
    let tall = GraphMatrix::from_row_major_data(
        (0..10000 * 4)
            .map(|i| ((i * 31 % 127) as f64 - 63.) / 71.)
            .collect(),
        10000,
        4,
        None,
    )?;
    cases.push(measure("large_tall_thin_svd", &[10000, 4], || {
        Ok(tall.svd_thin()?.1.iter().sum())
    })?);
    cases.push(measure("large_tall_rank", &[10000, 4], || {
        Ok(tall.rank()? as f64)
    })?);
    let rhs = GraphMatrix::ones(10000, 1);
    cases.push(measure("large_tall_least_squares", &[10000, 4, 1], || {
        tall.least_squares(&rhs, None)?.sum()
    })?);
    let report = json!({"profile":"release","arch":std::env::consts::ARCH,"os":std::env::consts::OS,"samples_per_case":7,"allocation_scope":"process-wide System allocation/reallocation requests; additional live-byte peak during operation; excludes fixture setup; includes output creation/drop; not RSS","cases":cases});
    let output = std::env::var("GROGGY_NUMERICAL_PROFILE")
        .unwrap_or_else(|_| "/tmp/groggy-numerical-profile.json".into());
    std::fs::write(&output, serde_json::to_string_pretty(&report)?)?;
    println!(
        "Wrote {} numerical release cases to {output}",
        report["cases"].as_array().map_or(0, Vec::len)
    );
    Ok(())
}
