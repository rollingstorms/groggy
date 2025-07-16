# Edge Collection & Proxy (edges)

Implements `EdgeCollection` and `EdgeProxy` for ergonomic, batch-friendly edge access.

- `EdgeCollection`: Batch add/remove, filter, iterate, get proxies.
- `EdgeProxy`: Per-edge attribute access, get/set attributes, fetch all attributes, endpoints.

**Python API:**
- `EdgeCollection` and `EdgeProxy` are available as classes.
- Methods: `.add()`, `.remove()`, `.size()`, `.ids()`, `.attr()`, `.get()`, `.filter()`

**Rust:**
- See `edges/collection.rs` and `edges/proxy.rs` for details.
