//! Compatibility wrapper that bridges legacy streaming APIs onto the realtime server.
//!
//! Existing callers expect `StreamingServer` to spin up an HTTP/WebSocket endpoint for
//! interactive visualization. Rather than maintaining a second implementation, this
//! shim hands everything to the realtime pipeline while preserving the original type
//! surface.

use super::data_source::DataSource;
use super::types::{ServerHandle, StreamingConfig, StreamingError, StreamingResult};
use crate::viz::realtime::accessor::{DataSourceRealtimeAccessor, RealtimeVizAccessor};
use crate::viz::realtime::server::start_realtime_background;
use std::net::IpAddr;
use std::sync::Arc;

/// Legacy streaming server facade.
#[derive(Clone, Debug)]
pub struct StreamingServer {
    /// Data source driving the visualization.
    pub data_source: Arc<dyn DataSource>,
    /// Caller-provided configuration (mostly retained for API compatibility).
    pub config: StreamingConfig,
}

impl StreamingServer {
    /// Create a new streaming server wrapper.
    pub fn new(data_source: Arc<dyn DataSource>, config: StreamingConfig) -> Self {
        Self {
            data_source,
            config,
        }
    }

    /// Start the realtime server in the background and return a handle that matches the
    /// legacy interface. The `addr` parameter is ignored because the realtime stack binds
    /// to localhost; it is retained so existing call sites continue to compile.
    pub fn start_background(&self, _addr: IpAddr, port_hint: u16) -> StreamingResult<ServerHandle> {
        // Prefer explicit configuration, otherwise fall back to the caller hint (with zero
        // signalling "choose any available port").
        let desired_port = if self.config.port != 0 {
            self.config.port
        } else {
            port_hint
        };

        let accessor: Arc<dyn RealtimeVizAccessor> =
            Arc::new(DataSourceRealtimeAccessor::new(self.data_source.clone()));

        let handle = start_realtime_background(desired_port, accessor, 0).map_err(|e| {
            // Quiet for streaming
            StreamingError::Server(format!("failed to start realtime server: {}", e))
        })?;

        Ok(ServerHandle {
            port: handle.port,
            cancel: handle.cancel,
            thread: handle.thread,
        })
    }
}
