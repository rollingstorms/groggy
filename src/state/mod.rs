//! State management
//!
//! This module contains state management and change tracking:
//! - Graph state
//! - History tracking
//! - Change tracking
//! - Delta operations
//! - Reference management
//! - Space management

pub mod change_tracker;
pub mod delta;
pub mod history;
pub mod ref_manager;
pub mod space;
pub mod state;
pub mod topology;

pub use change_tracker::*;
pub use delta::*;
pub use ref_manager::*;
pub use space::*;
pub use topology::*;

// Re-export history items except those that conflict with state module
pub use history::{
    Commit, CommitDiff, Delta, HistoricalView, HistoryForest, HistoryStatistics, ViewSummary,
};

// Re-export state items (including the canonical AttributeChange and EntityType)
pub use state::*;
