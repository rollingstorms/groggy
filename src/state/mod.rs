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

pub use change_tracker::*;
pub use delta::*;
pub use history::*;
pub use ref_manager::*;
pub use space::*;
pub use state::*;
