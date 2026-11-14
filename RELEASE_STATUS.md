# Groggy 0.6.0 release qualification

Updated 2026-11-14. **Unreleased; qualification in progress.** No tag, upload or
publication has occurred as part of this release-preparation slice.

## Supported release scope

An embedded Rust/Python graph and analytics library. See the
[scope and migration notes](RELEASE_NOTES_0.6.0.md). Graphiti adapters/drivers are
future external plugin work. Core retrieval expansion and multi-client services
remain deferred; they are not silently included in this release.

## Evidence from this slice

- Root Maturin wheel builds offline with locked resolution on macOS arm64,
  CPython 3.9. An isolated subprocess with site-packages disabled imports the
  unpacked wheel, not the checkout:
  distribution/runtime/native versions agree at 0.6.0, all 104 declared exports
  resolve, and graph/snapshot/f64/numerical smoke assertions pass. Both archive
  and journal acceptance workflows also pass against each of the three wheels.
- Source distribution builds with the standard backend, with no extension import
  or stub regeneration during packaging. Rebuilding the extracted source and
  alternate entry-point qualification are recorded in the current receipt.
- Eight focused regressions pass: version/exports/display delegation and the
  cross-surface archive/journal workflow. See
  [the executable workflow](docs/api/core-release-workflow.md).
- Maintained Python gate: 2,684 passed, one skipped, 15 warnings; 266 executable
  documentation examples pass. The known writable PyGraph finalization warning
  remains among these warnings; passing assertions do not erase that defect.
- Native gates: 1,561 Rust tests passed, 54 ignored; strict all-target Clippy passed.
- Existing native workflow benchmark: three fresh processes each at 1k/100k
  nodes and edges. At 100k, median bulk acceptance is 0.600 seconds and archive
  save/recovery 1.418/0.841 seconds; whole-process peak RSS is 836–869 MiB.
  These are bounded fixture measurements, not universal budgets or a matched
  before/after comparison. Memory/recovery costs remain existing scale work.

## Remaining release decisions and checks

- **Ownership defect:** wrong-thread cyclic collection of a writable Graph can
  report an unraisable error and skip its native destructor. Existing Rc/RefCell
  ownership reaches into core pools/spaces and connected wrappers. No unsafe Send,
  global GC suppression or test-only workaround has been applied. A safe fix or
  explicit narrower release decision is required before declaring release-ready.
- Cross-platform wheel/import/test qualification remains open. Local CPython 3.9
  evidence does not certify all advertised Python versions or operating systems.
- Broad old-format migration and platform/power-loss guarantees remain unproven;
  preserve the supported storage boundaries in the release notes.
- Native tests, Clippy and workload measurements are recorded in the
  [current qualification receipt](notes/planning/release-060-qualification-2026-11-14.md).
  A public release still needs review of the resulting artifact and scope.

This scope does not require completing all 53 open checklist rows. It also does
not mark broad ownership, numerical, schema, scale or compatibility parents done.

---

# Archived 0.5.2 status (historical, not current blockers)

The following text is retained as historical context. Its CI run references,
completion claims and estimates do not describe the current branch.

# Release 0.5.2 Status Summary

**Date:** December 4, 2024  
**Status:** ❌ BLOCKED - CI/Test Failures

---

## Current Situation

Both GitHub Actions workflows (ci.yml and test.yml) are failing on the latest commit (`0b10a6f`) with Rust compilation errors in the JIT module.

### Failed Workflows
- **CI Workflow:** Run #99 - Failed at 21:59 UTC
- **Test Workflow:** Run #119 - Failed at 21:59 UTC
- **Branch:** `main`
- **Commit:** `0b10a6f24d00227e5e2a3a00a64df754cdbb8b08`

---

## Root Cause

The experimental **JIT compilation** feature (`src/algorithms/steps/loop_step.rs`) has thread-safety issues:

1. **Sync Trait Violation:**
   - `LoopStep` struct requires `Send + Sync` to implement the `Step` trait
   - Contains `Mutex<Option<JitManager>>` which wraps `JITModule` from Cranelift
   - Cranelift's `JITModule` contains non-`Send` function resolver callbacks
   - This violates the `Sync` requirement for `Mutex<T>`

2. **Type Mismatch Errors:**
   - Lines 152, 174: Pattern matching treats `Result` as `Option`
   - Line 187: Incorrect mutex guard dereferencing

### Compilation Errors
```
error[E0277]: `(dyn for<'a> Fn(&'a str) -> Option<*const u8> + 'static)` cannot be sent between threads safely
  --> src/algorithms/steps/loop_step.rs:36:12
   |
36 | pub struct LoopStep {
   |            ^^^^^^^^ within `JITModule`, the trait `Send` is not implemented
```

---

## Impact Assessment

### ✅ What Works
All core features and deliverables for v0.5.2 are functional:
- ✅ Native algorithms module (PageRank, LPA, Connected Components, etc.)
- ✅ Builder DSL with trait-based decorators
- ✅ Batch executor with slot allocation
- ✅ FFI bridge and Python API
- ✅ Documentation (algorithms, tutorials, API reference)
- ✅ Test coverage for all algorithms
- ✅ Release notes and changelog

### ❌ What's Broken
Only the experimental JIT feature:
- ❌ `LoopStep` JIT compilation path
- ❌ CI/Test workflows (can't compile)

### 🔍 User Impact
**NONE** - The JIT feature was part of the "Tier 2" roadmap and was never enabled or documented for users. All advertised features work correctly using the interpreted batch executor.

---

## Resolution Options

### Option A: Disable JIT Module (RECOMMENDED)
**Effort:** 5 minutes  
**Risk:** None  
**Tradeoff:** Delays JIT feature to v0.6.0

**Steps:**
1. Add `#[cfg(feature = "jit")]` gates around JIT code
2. Remove JIT imports from modules without the feature
3. Document as "Known Issue" in release notes ✅ (Already Done)
4. Push fix and verify CI passes

### Option B: Fix Thread Safety
**Effort:** 2-4 hours  
**Risk:** Medium (could introduce subtle bugs)  
**Tradeoff:** Blocks release today

**Approaches:**
- Wrap callbacks in `Arc<Mutex<>>` (might not work)
- Use thread-local storage for JIT context
- Refactor to avoid Sync requirement
- Wait for upstream Cranelift fix

### Option C: Remove JIT Code Entirely
**Effort:** 10 minutes  
**Risk:** Low  
**Tradeoff:** Cleaner but more work to re-add later

**Steps:**
1. Delete `src/algorithms/execution/jit/` directory
2. Remove JIT references from `loop_step.rs`
3. Remove from documentation
4. Re-add in dedicated v0.6.0 branch

---

## Recommendation

**Proceed with Option A** - Feature-gate the JIT code and ship v0.5.2 today.

### Rationale
1. **Zero user impact:** JIT was never public-facing
2. **All deliverables complete:** 11 algorithms, Builder DSL, batch executor
3. **Clean separation:** JIT belongs in its own release (v0.6.0)
4. **Fast resolution:** 5 minutes vs 2-4 hours
5. **Transparent:** Already documented in release notes

### Next Steps (5 minutes)
1. Add feature gate to `Cargo.toml`: `jit = ["cranelift-jit"]`
2. Guard JIT code: `#[cfg(feature = "jit")]`
3. Remove unconditional JIT imports
4. Test locally: `cargo build --all-targets`
5. Push and verify CI: `git push origin main`
6. Create release tag: `git tag v0.5.2 && git push origin v0.5.2`

---

## Release Notes Update

✅ **Already Added:**  
A "Known Issues" section documents the JIT block status, workaround (batch executor), and resolution timeline (v0.6.0).

---

## Post-Release Plan

### For v0.5.2
- Monitor PyPI deployment
- Update documentation website
- Announce release (blog, social media)

### For v0.6.0 (JIT Revival)
- Create dedicated `feature/jit-thread-safety` branch
- Research Cranelift thread-safety patterns
- Consider alternative JIT backends (LLVM, Wasmer)
- Implement comprehensive JIT test suite
- Benchmark JIT vs interpreter performance

---

## Files Modified Today

### Release Prep (Completed)
- ✅ `Cargo.toml` - Version bump to 0.5.2
- ✅ `python-groggy/Cargo.toml` - Version bump to 0.5.2
- ✅ `pyproject.toml` - Version bump to 0.5.2
- ✅ `CHANGELOG.md` - Full release history
- ✅ `RELEASE_NOTES_0.5.2.md` - Comprehensive release notes
- ✅ `docs/algorithms/` - Native algorithm documentation
- ✅ `docs/builder/` - Updated Builder DSL docs
- ✅ `docs/tutorials/` - Updated tutorials 1-4
- ✅ `mkdocs.yml` - Navigation updates

### Needs Fix (5 minutes)
- ❌ `Cargo.toml` - Add JIT feature flag
- ❌ `src/algorithms/steps/loop_step.rs` - Add cfg gates
- ❌ `src/algorithms/execution/mod.rs` - Conditional JIT exports

---

## Contacts & Links

- **GitHub Actions:** https://github.com/rollingstorms/groggy/actions
- **Latest Run (CI):** https://github.com/rollingstorms/groggy/actions/runs/19945247889
- **Latest Run (Test):** https://github.com/rollingstorms/groggy/actions/runs/19945247890
- **Commit:** https://github.com/rollingstorms/groggy/commit/0b10a6f

---

**Decision Required:** Approve Option A to proceed with release today?
