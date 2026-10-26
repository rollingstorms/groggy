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
