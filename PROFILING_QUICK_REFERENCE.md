# Profiling Quick Reference Card

## Enable Profiling

```bash
# Connected Components
export GROGGY_PROFILE_CC=1

# Run your script
python your_script.py
```

## Clean Reinstall (After Rust Changes)

```bash
# One-liner
pip uninstall groggy -y && cargo clean && maturin develop --release

# Verify
python -c "import groggy; print(groggy.__version__, groggy.__file__)"
```

## Add Profiling to Algorithm

```rust
// In your algorithm implementation
use std::time::Instant;

// Time a phase
let start = Instant::now();
// ... do work ...
ctx.record_call("algo.phase_name", start.elapsed());

// Time with closure
let result = ctx.with_counted_timer("algo.phase", || {
    // ... do work ...
    result_value
});

// Store statistics (as nanosecond duration)
ctx.record_call(
    "algo.count.nodes_processed",
    Duration::from_nanos(count as u64)
);

// Print report at end of execute()
if std::env::var("GROGGY_PROFILE_ALGO").is_ok() {
    ctx.print_profiling_report("Algorithm Name");
}
```

## Naming Convention

```
{algorithm}.{mode}.{phase}

Examples:
- cc.bfs.component_traversal
- cc.tarjan.scc_extraction
- cc.build_csr
- cc.count.nodes_processed
```

## Interpreting Results

```
Phase                                   Calls   Total (ms)   Avg (μs)
------------------------------------------------------------------------
cc.total_execution                          1      2.324     2324.167
cc.write_attributes                         1      1.234     1233.542  ← Bottleneck!
cc.build_csr                                1      0.398      397.708
cc.bfs.component_traversal                 61      0.017        0.276  ← 61 components
```

**Key Metrics:**
- **High Total + High Calls**: Hot path, optimize per-call cost
- **High Total + Low Calls**: Expensive operation, consider algorithmic improvement
- **High Calls + Low Avg**: Efficient, already optimized
- **Statistics section**: Counts stored as nanoseconds for unified format

## Common Commands

```bash
# Build with optimizations
maturin develop --release

# Build without optimizations (faster compile)
maturin develop

# Check syntax only
cargo check --all-targets

# Format code
cargo fmt --all

# Run tests
cargo test
pytest tests -q

# Clean everything
cargo clean && find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
```

## Files to Reference

- `AGENTS.md` - Full guidelines and profiling infrastructure docs
- `CONNECTED_COMPONENTS_PROFILING_GUIDE.md` - Detailed CC profiling guide
- `CC_PROFILING_IMPLEMENTATION_SUMMARY.md` - Implementation details
- `profile_cc_detailed.py` - Example profiling script

## Troubleshooting

**Python not seeing Rust changes?**
→ Run clean reinstall workflow

**Profiling not printing?**
→ Check environment variable is set: `echo $GROGGY_PROFILE_CC`

**Can't import groggy?**
→ `maturin develop --release` then check with `pip list | grep groggy`

**Want to add profiling to new algorithm?**
→ Copy pattern from `src/algorithms/community/components.rs`
