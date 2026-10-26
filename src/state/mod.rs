//! State management
//!
//! This module contains state management and change tracking:
//! - Graph state
//! - History tracking
//! - Change tracking
//! - Delta operations
//! - Reference management
//! - Space management

pub mod change;
pub mod change_tracker;
pub mod checkpoint;
pub mod delta;
pub mod effective_read;
pub mod history;
pub mod publication;
pub mod read_state;
pub mod ref_manager;
pub mod space;
pub mod state;
pub mod topology;

pub use change_tracker::*;
pub use delta::*;
pub use ref_manager::*;
pub use space::*;
pub use topology::*;

// Re-export history-specific containers; logical change types share one owner.
pub use history::{
    Commit, CommitDiff, CommitDiffPlan, Delta, HistoricalView, HistoryForest, HistoryStatistics,
    RecordedReadPlan, ViewSummary,
};

// Re-export state items and their compatibility exports of canonical change types.
pub use state::*;

pub use read_state::PinnedReadState;
