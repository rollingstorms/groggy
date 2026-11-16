# Native numerical memory ownership

`SharedBuffer` owns aligned dense storage. `data()` and `data_mut()` return lock
guards: keep the guard alive for the entire live read/write. They do not copy the
buffer. A successful mutable borrow advances its modification counter; a rejected
borrow does not.

`view_for_backend(BackendId::Native)` now captures an **owned read snapshot**.
Its `BackendView::data()` slice stays valid and unchanged while the original
buffer is edited or dropped, including edits on another thread. Capturing costs
O(n) time and one value buffer; requesting a snapshot is explicit. Normal matrix
arithmetic does not create backend views implicitly. The previous implementation
released its read lock before returning the slice and could not safely promise
live access through that interface.

```rust
use groggy::storage::matrix::{memory::BackendId, MatrixLayout, SharedBuffer};

let buffer = SharedBuffer::from_data(vec![1.0, 2.0], (1, 2), MatrixLayout::RowMajor)?;
let mut snapshot = buffer.view_for_backend(BackendId::Native)?;
{
    let mut write = buffer.data_mut()?;
    write.as_slice_mut()[0] = 9.0;
}
assert_eq!(snapshot.data(), &[1.0, 2.0]);
assert_eq!(buffer.data()?.as_slice(), &[9.0, 2.0]);

// Explicit refresh copies a same-length input into this owned snapshot.
snapshot.sync_from_main(buffer.data()?.as_slice())?;
assert_eq!(snapshot.data(), &[9.0, 2.0]);
assert!(snapshot.data_mut().is_none());
assert!(snapshot.sync_to_main().is_err());
assert!(buffer.sync_from_backend(BackendId::BLAS).is_err());
# Ok::<(), groggy::storage::matrix::MemoryError>(())
```

## Native compatibility correction

Method signatures and module imports remain stable. Backend views no longer
observe later source writes automatically; capture a new snapshot or explicitly
refresh the owned one. `sync_from_main` validates length before changing anything
and copies exact typed values. It does not write the original SharedBuffer.
`sync_to_main` now explicitly rejects: an owned read snapshot grants no writeback
authority. Mutate the source through its existing live write guard.

NumPy, BLAS, CUDA and OpenCL view/synchronization requests reject explicitly.
Native residency checks acquire the live read guard and report poisoned storage
as an error. No external backend transfer is implemented or claimed. Unused private
backend-status/cache placeholders have been removed; exported capability enums
are unchanged. Public Python APIs and retained graph/storage formats are unchanged.

## Safe generic initialization

`NumericType` is a safe trait and may be implemented by callers. An all-zero byte
pattern need not represent its mathematical zero or even a valid value. Aligned
allocation uses the established zeroed-allocation fast path only for the six
concrete built-ins (f32/f64/i32/i64/i128/bool). Other types are initialized with
`T::zero()` before exposing a slice. Zero-sized custom numerical types reject
explicitly before allocator access. Logically empty built-in buffers still expose
empty slices with valid aligned storage.

Snapshot copying and explicit refresh are measured separately from live guard
access in `numerical_completion --backend-views`. These have different ownership
semantics; comparing their timings does not establish a speedup or a universal
numerical throughput budget.
