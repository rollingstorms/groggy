//! Subgraph types and operations
//!
//! This module contains different types of subgraphs and their operations:
//! - Basic subgraphs
//! - Hierarchical subgraphs  
//! - Neighborhood subgraphs
//! - Filtered subgraphs
//! - Component subgraphs

mod adjacency;
pub use adjacency::AdjacencyRead;
mod degrees;
pub(crate) mod sampling;
pub use degrees::{DegreeMode, SubgraphDegreeRead};

pub mod component;
pub mod composer;
pub(crate) mod composition;
pub mod filtered;
pub mod hierarchical;
pub mod neighborhood;
pub mod subgraph;
pub mod view;
mod view_algebra;
mod view_entities;
mod view_metrics;
mod view_projection;
mod view_traversal;
pub mod visualization;
pub use view_entities::{GraphViewEntities, GraphViewEntity, GraphViewEntityIterator};

pub use component::*;
pub use composer::*;
pub use filtered::*;
pub use hierarchical::*;
pub use neighborhood::*;
pub use subgraph::*;
pub use visualization::*;

pub use view::GraphView;
