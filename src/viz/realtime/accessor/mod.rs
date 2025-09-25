//! Realtime Viz Accessor - Bridge from DataSource to Realtime Engine
//!
//! Provides a clean, testable bridge from DataSource → Realtime Engine,
//! independent of transport/UI. This is Phase 1 of the Realtime Viz Integration.

pub mod engine_messages;
pub mod realtime_viz_accessor;

pub use engine_messages::*;
pub use realtime_viz_accessor::*;
