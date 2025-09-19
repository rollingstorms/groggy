//! Unified visualization core engine
//! 
//! This module contains the core visualization engine that powers all visualization
//! backends (Jupyter widgets, streaming server, file export) with a single
//! physics simulation and rendering pipeline.
//!
//! Architecture:
//! - `engine.rs`: Main VizEngine that coordinates all components
//! - `physics.rs`: Physics simulation extracted from layouts
//! - `rendering.rs`: Unified rendering pipeline 
//! - `interaction.rs`: Interaction state management
//! - `frame.rs`: Frame data structures for output

pub mod engine;
pub mod physics;
pub mod rendering; 
pub mod interaction;
pub mod frame;

#[cfg(test)]
mod test_core;

pub use engine::*;
pub use physics::*;
pub use rendering::*;
pub use interaction::*;
pub use frame::*;