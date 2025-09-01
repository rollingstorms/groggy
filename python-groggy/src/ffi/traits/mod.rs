//! FFI Traits - Pure delegation to core trait implementations
//!
//! This module provides Python-accessible trait interfaces that delegate
//! to our efficient core trait implementations. All algorithm logic remains
//! in the core - this is pure translation layer.

// Subgraph operations moved to direct implementation in ffi/core/subgraph.rs
// No trait-based patterns needed - direct delegation is cleaner
