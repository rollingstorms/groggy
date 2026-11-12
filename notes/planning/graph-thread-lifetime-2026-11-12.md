# Thread-bound graph cyclic collection — 2026-11-12

Existing scope: 19.2. Investigation outcome, not a fixed lifetime contract.

The November 12 baseline repeats the November 11 unraisable `PyGraph is
unsendable, but is being dropped on another thread` warning: 2,612 maintained
Python tests pass, one skips, 15 warnings; 261 executable documentation examples
pass. The frozen A* file alone passes all nine tests without this warning.

`notes/scripts/diagnose_graph_thread_lifetime.py` isolates the mechanism in fresh
interpreters. Disable automatic GC; create a self-referencing Python list holding
a Graph; drop external references; explicitly collect from another thread. This
reliably emits the warning without invoking any graph algorithm. Collecting the
same graph cycle on its creating thread emits none. Replacing Graph with its
detached read_view, deleting the Graph on its creating thread, then collecting
the view cycle on a worker emits none. All three diagnostic assertions pass.

This establishes a general cyclic-GC/thread-bound-owner failure mechanism, but
**does not identify which earlier test allocated the particular graph observed
in the full suite**. Pytest's A* reporting location is where GC happened.

PyGraph is deliberately unsendable and owns Rc<RefCell<Graph>>. Python cycles
and exception tracebacks may delay final release until a worker runs GC. PyO3
rejects wrong-thread destruction before the native destructor runs. Removing the
unsendable guard, adding unsafe Send, globally disabling GC, or adding a test-only
collection workaround would not repair ownership. A complete fix requires an
owner-safe lifetime design across writable graph owners/accessors (with measured
costs), rather than an algorithm or renderer patch. Keep 19.2 open.

Supported boundary remains: use detached immutable GraphView values on workers;
keep writable Graph and its connected accessors on their creating thread. This
boundary alone does not promise arbitrary Python cycles containing writable
objects are safe to collect from workers.
