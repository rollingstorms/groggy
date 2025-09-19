//! Visualization adapters that wrap the unified core engine
//!
//! Adapters provide specific interfaces for different output formats while
//! all using the same underlying VizEngine core for consistency.
//!
//! Architecture:
//! - `streaming.rs`: StreamingAdapter for WebSocket server integration
//! - `jupyter.rs`: JupyterAdapter for notebook widget integration  
//! - `file.rs`: FileAdapter for static file export
//! - `traits.rs`: Common adapter traits and interfaces
//! - `tests.rs`: Compatibility tests ensuring adapters produce equivalent results

pub mod traits;
pub mod streaming;
pub mod jupyter;
pub mod file;

#[cfg(test)]
mod tests;

pub use traits::*;
pub use streaming::*;
pub use jupyter::*;
pub use file::*;