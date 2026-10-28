//! Query and traversal functionality
//!
//! This module contains query processing and graph traversal:
//! - Query language
//! - Query parser
//! - Traversal algorithms

pub(crate) mod endpoints;
mod entrypoints;
pub(crate) mod filtering;
pub mod index;
pub mod ordering;
pub mod pattern;
pub mod query;
pub mod query_parser;
pub mod selection;
pub mod traversal;
pub(crate) mod weights;

pub use index::GraphQueryIndex;
pub use ordering::QueryOptions;
pub use pattern::{
    GraphPattern, PatternEdge, PatternMatch, PatternMatches, PatternNode, PatternOptions,
};
pub use query::*;
pub use query_parser::*;
pub use selection::TraversalDirection;
pub use traversal::*;

pub use crate::state::effective_read::{
    ActivationInterval, EffectiveTimeEvent, EffectiveTimeIndex, PropertyInterval,
};
