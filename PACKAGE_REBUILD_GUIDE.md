# Package Installation & Rebuild Guide

## Clean Reinstall Workflow

Use this workflow when profiling changes or Rust updates aren't reflected in Python.

### Step-by-Step Process

```bash
# 1. Uninstall existing package
pip uninstall groggy -y

# 2. Clean build artifacts (optional but recommended for major changes)
cargo clean  # Removes ~3-4GB of build artifacts

# 3. Clean Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null

# 4. Rebuild and install from source
maturin develop --release

# 5. Verify installation
python -c "import groggy; print(f'Version: {groggy.__version__}'); print(f'Location: {groggy.__file__}')"
```

### One-Liner Clean Reinstall

```bash
pip uninstall groggy -y && cargo clean && find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null && find . -name "*.pyc" -delete 2>/dev/null && maturin develop --release
```

## When to Clean Rebuild

Perform a full clean rebuild in these scenarios:

- ✅ After adding new profiling instrumentation to Rust algorithms
- ✅ When Context struct changes (new fields, methods)
- ✅ If Python can't see latest Rust changes (FFI binding updates)
- ✅ Before performance benchmarking (ensure optimized build)
- ✅ After switching branches with significant algorithm changes
- ✅ When you get unexpected behavior after Rust modifications
- ✅ After updating Rust dependencies in Cargo.toml
- ✅ When FFI bindings are modified

## Incremental Rebuild

For faster rebuilds when only small changes are made:

```bash
maturin develop --release  # Reuses cached dependencies
```

This is suitable when:
- Making small Rust code changes
- Modifying existing functions without changing signatures
- Updating documentation or comments
- No structural changes to Context or public APIs

## Development Iteration

For rapid development cycles with frequent changes:

```bash
# Build without release optimizations (faster compile, slower runtime)
maturin develop

# With cargo check for syntax validation only (no actual build)
cargo check --all-targets
```

**Note**: Use non-release builds during development to speed up compilation, then switch to `--release` for performance testing.

## Troubleshooting

### Issue: Python not seeing Rust changes

**Symptom**: Modified Rust code but Python behavior unchanged

**Solution**:
```bash
pip uninstall groggy -y
maturin develop --release
python -c "import groggy; print(groggy.__file__)"  # Verify location
```

### Issue: Import error or module not found

**Symptom**: `ModuleNotFoundError: No module named 'groggy'`

**Solution**:
```bash
# Check if installed
pip list | grep groggy

# If not found, rebuild
maturin develop --release

# Verify
python -c "import groggy"
```

### Issue: Getting old behavior after rebuild

**Symptom**: Code changes aren't reflected even after `maturin develop`

**Solution**: Full clean rebuild
```bash
cargo clean
pip uninstall groggy -y
maturin develop --release
```

### Issue: Build fails with dependency errors

**Symptom**: Compilation errors about missing dependencies

**Solution**:
```bash
# Update Cargo.lock
cargo update

# Clean and rebuild
cargo clean
maturin develop --release
```

### Issue: Profiling not showing new metrics

**Symptom**: Added profiling points but don't appear in output

**Solution**:
1. Verify environment variable: `echo $GROGGY_PROFILE_CC`
2. Clean rebuild to ensure new code is compiled
3. Check profiling is gated correctly in Rust code

## Why Rebuild is Necessary

`maturin develop` installs the package as editable but still requires full compilation:

```
Rust Source Code
    ↓ (rustc compile)
Native Extension (.so / .dylib / .dll)
    ↓ (PyO3 binding)
Python Module
    ↓ (pip install editable)
Available in Python
```

Changes to any of these layers require recompilation:
- **Rust code changes** → Need rustc recompile
- **FFI layer changes** → Need PyO3 rebinding
- **Context struct changes** → Need full rebuild
- **Algorithm implementations** → Need recompile

## Performance Considerations

### Build Times

| Command | Compile Time | Runtime Performance | Use Case |
|---------|--------------|---------------------|----------|
| `maturin develop` | Fast (~20-30s) | Slower (no optimizations) | Development |
| `maturin develop --release` | Slow (~1-2 min) | Fast (full optimizations) | Testing/Production |
| `cargo check` | Very fast (~5-10s) | N/A (syntax only) | Quick validation |
| `cargo clean + rebuild` | Very slow (~2-3 min) | Fast | Major changes |

### Disk Space

- **cargo clean** removes ~3-4GB of build artifacts
- Incremental builds reuse cached dependencies
- Consider cleaning periodically to free space

## Best Practices

1. **Use incremental builds** for small changes during development
2. **Clean rebuild** before committing or when switching contexts
3. **Always use `--release`** for performance benchmarking
4. **Verify installation** after every rebuild with version check
5. **Document breaking changes** that require users to rebuild
6. **Test in both debug and release** modes for different failure modes

## Related Documentation

- **AGENTS.md** - Full repository guidelines including profiling
- **PROFILING_QUICK_REFERENCE.md** - Quick commands and troubleshooting
- **CONNECTED_COMPONENTS_PROFILING_GUIDE.md** - Algorithm-specific profiling details

## Quick Command Reference

```bash
# Full clean rebuild
pip uninstall groggy -y && cargo clean && maturin develop --release

# Incremental rebuild
maturin develop --release

# Quick syntax check
cargo check --all-targets

# Verify installation
python -c "import groggy; print(groggy.__version__, groggy.__file__)"

# Check if package installed
pip list | grep groggy

# Format code before commit
cargo fmt --all
black .
isort .
```

---

**Last Updated**: 2024
**Applies to**: groggy 0.5.1+
**Build System**: maturin + PyO3
