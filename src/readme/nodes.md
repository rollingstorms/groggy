# Node Collection & Proxy (nodes)

Implements `NodeCollection` and `NodeProxy` for ergonomic, batch-friendly node access.

- `NodeCollection`: Batch add/remove, filter, iterate, get proxies.
- `NodeProxy`: Per-node attribute access, get/set attributes, fetch all attributes.

**Python API:**
- `NodeCollection` and `NodeProxy` are available as classes.
- Methods: `.add()`, `.remove()`, `.size()`, `.ids()`, `.attr()`, `.get()`, `.filter()`

**Rust:**
- See `nodes/collection.rs` and `nodes/proxy.rs` for details.
