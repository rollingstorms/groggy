//! State management
//!
//! This module contains state management and change tracking:
//! - Graph state
//! - History tracking
//! - Change tracking
//! - Delta operations
//! - Reference management

pub mod state;
pub mod history;
pub mod change_tracker;
pub mod delta;
pub mod ref_manager;

pub use state::*;
pub use history::*;
pub use change_tracker::*;
pub use delta::*;
pub use ref_manager::*;