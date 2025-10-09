//! Phase 4: Trait-Based Delegation System
//!
//! # Purpose
//!
//! This module implements a universal method availability system through Rust traits,
//! enabling seamless method chaining across all object types in the Groggy ecosystem.
//! The system allows Python users to call methods on objects without worrying about
//! type conversions or intermediate steps.
//!
//! # Design Goals
//!
//! 1. **Universal Method Availability**: Any object should be able to call methods from
//!    related types (e.g., a Subgraph can call table methods, a Table can call array methods)
//!
//! 2. **Infinite Chaining**: Enable fluent API patterns where users can chain operations
//!    naturally: `graph.subgraph().filter().table().sort().head(10)`
//!
//! 3. **Type Safety**: Maintain Rust's type safety while providing Python's flexibility
//!
//! 4. **Performance**: Minimize FFI overhead through efficient delegation patterns
//!
//! # Architecture
//!
//! The system consists of five components:
//!
//! - `traits`: Core trait definitions (SubgraphOps, TableOps, ArrayOps, GraphOps)
//! - `implementations`: Trait implementations for existing FFI types
//! - `forwarding`: Generic forwarding containers with built-in delegation
//! - `error_handling`: Unified error handling for delegated operations
//! - `examples`: Usage examples and integration patterns
//!
//! # Current Status
//!
//! **Phase**: Design and prototyping (not yet integrated into Python API)
//!
//! This module is currently experimental. The traits and implementations are complete,
//! but they are not yet exposed through the Python FFI layer. Integration is planned
//! for a future release once the API design is validated.
//!
//! # Future Integration Plan
//!
//! 1. Add trait implementations to existing PyO3 classes
//! 2. Expose delegating methods through `#[pymethods]`
//! 3. Update Python type stubs to reflect new methods
//! 4. Add integration tests for cross-type method calls
//! 5. Document usage patterns in user guide
//!
//! # Example Future Usage
//!
//! ```python
//! # Once integrated, this would work:
//! subgraph = graph.nodes([1, 2, 3]).subgraph()  # SubgraphOps
//! table = subgraph.table()                       # TableOps delegation
//! filtered = table.filter("age > 25")            # TableOps
//! result = filtered.select(["name", "age"])      # Chaining
//! ```

// Allow dead_code for the entire delegation module as it's experimental
// and not yet integrated into the Python API
#![allow(dead_code)]

pub mod error_handling;
pub mod examples;
pub mod forwarding;
pub mod implementations;
pub mod traits;

// Re-export the main traits for easy access
