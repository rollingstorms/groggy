# Next Steps

## Immediate (Fix Warnings)
- [ ] Remove unused imports (HashSet, DefaultHasher, etc.)
- [ ] Prefix unused variables with underscore
- [ ] Clean up dead code warnings
- [ ] Run `cargo fix --lib -p groggy` to auto-fix

## Core Improvements
- [ ] Implement proper branch checkout with state isolation
- [ ] Add commit history retrieval (currently returns empty)
- [ ] Implement memory usage calculation in statistics
- [ ] Add query engine filtering capabilities (find_nodes, find_edges)

## Advanced Features
- [ ] Historical views and time-travel queries
- [ ] Graph merging and conflict resolution
- [ ] Persistence layer (save/load from disk)
- [ ] Advanced query patterns and filtering

## Performance
- [ ] Add adjacency lists for O(1) neighbor queries
- [ ] Implement graph compression
- [ ] Add indexing for attribute queries
- [ ] Benchmark with larger datasets

## Testing
- [ ] Add unit tests with `cargo test`
- [ ] Integration tests for edge cases
- [ ] Property-based testing
- [ ] Performance regression tests

## Documentation
- [ ] API documentation with examples
- [ ] Architecture guide
- [ ] Usage tutorials
- [ ] Performance characteristics guide