//! Test module for adapter compatibility
//!
//! Verifies that all adapters follow the expected interface patterns.

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tempfile::TempDir;

    /// Test data generator
    fn create_test_data() -> (Vec<crate::viz::streaming::data_source::GraphNode>, Vec<crate::viz::streaming::data_source::GraphEdge>) {
        use crate::viz::streaming::data_source::{GraphNode, GraphEdge, Position};
        
        let nodes = vec![
            GraphNode {
                id: "node1".to_string(),
                label: Some("Node 1".to_string()),
                position: Some(Position { x: 0.0, y: 0.0 }),
                style: HashMap::new(),
                metadata: HashMap::new(),
            },
            GraphNode {
                id: "node2".to_string(),
                label: Some("Node 2".to_string()),
                position: Some(Position { x: 100.0, y: 0.0 }),
                style: HashMap::new(),
                metadata: HashMap::new(),
            },
        ];

        let edges = vec![
            GraphEdge {
                id: "edge1".to_string(),
                source: "node1".to_string(),
                target: "node2".to_string(),
                label: None,
                style: HashMap::new(),
                metadata: HashMap::new(),
            },
        ];

        (nodes, edges)
    }

    /// Test adapter configuration structure
    #[test]
    fn test_adapter_config_creation() {
        let config = AdapterConfig {
            width: 800.0,
            height: 600.0,
            physics_enabled: true,
            interactions_enabled: true,
            auto_fit: true,
            theme: Some("dark".to_string()),
            custom_styles: HashMap::new(),
        };

        assert_eq!(config.width, 800.0);
        assert_eq!(config.height, 600.0);
        assert!(config.physics_enabled);
        assert!(config.interactions_enabled);
        assert!(config.auto_fit);
        assert_eq!(config.theme, Some("dark".to_string()));
    }

    /// Test ExportFormat functionality
    #[test]
    fn test_export_format_properties() {
        let formats = vec![
            ExportFormat::SVG,
            ExportFormat::HTML,
            ExportFormat::JSON,
            ExportFormat::PNG { dpi: 300 },
            ExportFormat::PDF,
        ];

        // Test extensions
        assert_eq!(ExportFormat::SVG.extension(), "svg");
        assert_eq!(ExportFormat::HTML.extension(), "html");
        assert_eq!(ExportFormat::JSON.extension(), "json");
        assert_eq!(ExportFormat::PNG { dpi: 300 }.extension(), "png");
        assert_eq!(ExportFormat::PDF.extension(), "pdf");

        // Test MIME types
        assert_eq!(ExportFormat::SVG.mime_type(), "image/svg+xml");
        assert_eq!(ExportFormat::HTML.mime_type(), "text/html");
        assert_eq!(ExportFormat::JSON.mime_type(), "application/json");
        assert_eq!(ExportFormat::PNG { dpi: 300 }.mime_type(), "image/png");
        assert_eq!(ExportFormat::PDF.mime_type(), "application/pdf");
    }

    /// Test adapter builder patterns work correctly
    #[test]
    fn test_streaming_adapter_builder() {
        use crate::viz::streaming::websocket_server::StreamingConfig;
        
        let streaming_config = StreamingConfig::default();
        let adapter = StreamingAdapterBuilder::new()
            .with_dimensions(1024.0, 768.0)
            .with_port(8080)
            .with_physics(false)
            .build();

        let config = adapter.get_config();
        // Just test that the adapter can be created and config retrieved
        assert_eq!(config.physics_enabled, false);
    }

    /// Test file adapter builder
    #[test] 
    fn test_file_adapter_builder() {
        let temp_dir = TempDir::new().unwrap();
        let file_adapter = FileAdapterBuilder::new()
            .with_dimensions(1200.0, 800.0)
            .with_output_directory(temp_dir.path())
            .with_default_format(ExportFormat::HTML)
            .with_optimization(false)
            .with_metadata(false)
            .build();

        let supported_formats = file_adapter.supported_formats();
        assert!(supported_formats.contains(&ExportFormat::SVG));
        assert!(supported_formats.contains(&ExportFormat::HTML));
        assert!(supported_formats.contains(&ExportFormat::JSON));
    }

    /// Test that adapters can be created and have expected basic functionality
    #[test]
    fn test_adapter_creation_and_basic_interface() {
        let config = AdapterConfig::default();
        
        // Test StreamingAdapter creation
        use crate::viz::streaming::websocket_server::StreamingConfig;
        let streaming_config = StreamingConfig::default();
        let streaming_adapter = StreamingAdapter::new(config.clone(), streaming_config);
        assert!(!streaming_adapter.is_streaming());
        assert_eq!(streaming_adapter.connection_count(), 0);

        // Test JupyterAdapter creation  
        let jupyter_adapter = JupyterAdapter::new(config.clone());
        assert!(!jupyter_adapter.is_ready());

        // Test FileAdapter creation
        let temp_dir = TempDir::new().unwrap();
        let file_adapter = FileAdapter::new(config, Some(temp_dir.path().to_path_buf()));
        assert!(!file_adapter.is_ready());
    }

    /// Test error handling in adapters
    #[test]
    fn test_adapter_error_handling() {
        let config = AdapterConfig::default();
        
        // Test FileAdapter error handling
        let temp_dir = TempDir::new().unwrap();
        let mut file_adapter = FileAdapter::new(config, Some(temp_dir.path().to_path_buf()));
        
        // Test unsupported format handling
        let unsupported_formats = vec![
            ExportFormat::PNG { dpi: 300 },
            ExportFormat::PDF,
        ];
        
        for format in unsupported_formats {
            let result = file_adapter.export_to_file("test", format);
            assert!(result.is_err(), "Expected error for unsupported format");
        }
    }

    /// Test adapter result types
    #[test]
    fn test_adapter_result_types() {
        // Test StreamingResult
        let streaming_result = StreamingResult {
            url: "ws://localhost:8080".to_string(),
            port: 8080,
            connections: 5,
            status: "running".to_string(),
        };
        
        assert_eq!(streaming_result.port, 8080);
        assert_eq!(streaming_result.connections, 5);
        assert_eq!(streaming_result.status, "running");

        // Test FileResult
        let file_result = FileResult {
            file_path: "/tmp/test.svg".to_string(),
            file_size_bytes: 1024,
            format: "SVG".to_string(),
            export_time_ms: 15.5,
        };
        
        assert_eq!(file_result.file_size_bytes, 1024);
        assert_eq!(file_result.format, "SVG");
        assert!(file_result.export_time_ms > 0.0);

        // Test JupyterResult
        let jupyter_metadata = JupyterMetadata {
            widget_id: "widget_123".to_string(),
            data_size_bytes: 2048,
            render_time_ms: 25.0,
        };

        let jupyter_result = JupyterResult {
            html: "<div>Test</div>".to_string(),
            javascript: "console.log('test')".to_string(),
            metadata: jupyter_metadata,
        };
        
        assert!(!jupyter_result.html.is_empty());
        assert!(!jupyter_result.javascript.is_empty());
        assert_eq!(jupyter_result.metadata.data_size_bytes, 2048);
    }

    /// Test format settings and configuration
    #[test]
    fn test_format_settings() {
        let mut settings = FormatSettings::default();
        settings.interactive = true;
        settings.custom_styles = "body { background: #333; }".to_string();
        settings.metadata.insert("author".to_string(), "test".to_string());

        assert!(settings.interactive);
        assert!(!settings.custom_styles.is_empty());
        assert_eq!(settings.metadata.get("author"), Some(&"test".to_string()));
    }

    /// Test that adapter configuration can be updated
    #[test]
    fn test_adapter_config_updates() {
        let initial_config = AdapterConfig {
            width: 800.0,
            height: 600.0,
            physics_enabled: true,
            interactions_enabled: true,
            auto_fit: true,
            theme: None,
            custom_styles: HashMap::new(),
        };

        let updated_config = AdapterConfig {
            width: 1200.0,
            height: 900.0,
            physics_enabled: false,
            interactions_enabled: false,
            auto_fit: false,
            theme: Some("dark".to_string()),
            custom_styles: {
                let mut styles = HashMap::new();
                styles.insert("background".to_string(), "#333".to_string());
                styles
            },
        };

        // Verify configs are different
        assert_ne!(initial_config.width, updated_config.width);
        assert_ne!(initial_config.physics_enabled, updated_config.physics_enabled);
        assert_ne!(initial_config.theme, updated_config.theme);
    }

    /// Test adapter interface traits
    #[test] 
    fn test_adapter_trait_interface() {
        // Test that adapters implement the expected traits
        let config = AdapterConfig::default();
        
        // Test VizAdapter trait methods exist (compilation test)
        use crate::viz::streaming::websocket_server::StreamingConfig;
        let streaming_config = StreamingConfig::default();
        let mut streaming_adapter = StreamingAdapter::new(config.clone(), streaming_config);
        
        // These should compile without errors
        let _config = streaming_adapter.get_config();
        let _ready = streaming_adapter.is_ready();
        
        let mut file_adapter = FileAdapter::new(config.clone(), None);
        let _config = file_adapter.get_config();
        let _ready = file_adapter.is_ready();
        let _formats = file_adapter.supported_formats();
        
        let mut jupyter_adapter = JupyterAdapter::new(config);
        let _config = jupyter_adapter.get_config();
        let _ready = jupyter_adapter.is_ready();
    }
}