# Checkpoint recovery after abrupt process termination

Existing 10.4/31.4 failure-corpus scope. No new checklist rows; counts remain
**201 checked / 67 open**.

Native subprocess tests now stop a checkpoint helper at six precise boundaries:
before write, after the first large baseline fragment, after file sync, before
replacement, after replacement and after directory sync. The parent observes a
test-only handshake, sends SIGKILL, waits for process termination, then recovers
using a fresh owner. Test hooks do not exist in production builds.

Every case verifies selected state, original clocks/content hashes, complete
retained history, schema revision, checkpoint annotations, exclusive writer lock
reacquisition and successful append/reopen. Before replacement, the named original
journal is byte-identical. After replacement, the complete baseline is recovered.
The partial-fragment case exercises real 8 MiB framing with a 9 MiB property.
Orphan sibling temporaries from a killed process are ignored; they are not promoted
or automatically deleted. Test temporary directories clean up their own fixtures.

This tests process death, OS lock cleanup and ordinary filesystem recovery. It
is not sudden-power-loss/controller-cache/filesystem fault emulation, a Windows
replacement certification, or an automatic orphan-cleanup policy.

Verification: six checkpoint unit tests pass (including the child helper and
six-boundary parent matrix); full native **1,263 passed / 51 ignored**, strict
all-target Clippy. Production API/binary behavior is unchanged by the test hooks;
the preceding rebuilt Python **2,000 passed / 1 skipped** and **158 documentation
examples** remain the applicable Python surface gate.

## Serial subprocess gate

A later full-suite run exposed a real test interaction: fork/exec can briefly
retain unrelated parallel tests' flock descriptors until exec closes them. That
can make another test's immediate close/reopen fail transiently. The process-death
matrix is now an explicit ignored-by-default **serial gate**, not an unexecuted
check. Run it alongside the normal parallel suite:

```sh
cargo test --offline --locked --lib killed_checkpoint_process -- --ignored --test-threads=1
```

The normal unit suite keeps all other writer-lock tests parallel. Production lock
semantics are unchanged. The serial matrix must pass before claiming process-death
coverage; final totals distinguish this gate from default-suite ignored counts.

The dedicated serial gate passed locally (six termination boundaries in one matrix
test) and is now wired after the normal Rust test step in `.github/workflows/test.yml`.
The workflow itself was not run remotely in this slice.
