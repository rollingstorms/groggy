//! Phase 4: Trait-Based Delegation System
//!
//! This module implements universal method availability through traits,
//! enabling infinite method chaining across all object types in the Groggy ecosystem.

pub mod traits;
pub mod forwarding;
pub mod error_handling;
pub mod implementations;
pub mod examples;

// Re-export the main traits for easy access
pub use traits::{
    SubgraphOps, TableOps, GraphOps, 
    BaseArrayOps, StatsArrayOps,
    DelegatingIterator
};

pub use forwarding::{
    ForwardingArray, ForwardingIterator
};

pub use error_handling::{
    DelegationResult, DelegationError
};