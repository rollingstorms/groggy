//! Realtime Server - Phase 2 of Realtime Viz Integration
//!
//! Provides HTTP + WebSocket transport layer for realtime visualization.
//! Serves HTML/JS client and bridges engine messages to WebSocket clients.

pub mod realtime_server;
pub mod ws_bridge;

pub use realtime_server::*;
pub use ws_bridge::*;
