//! Comprehensive tests for streaming infrastructure
//!
//! Tests the complete streaming system including:
//! - DataSource trait implementation
//! - VirtualScrollManager with LRU caching  
//! - WebSocket server functionality
//! - BaseTable streaming integration

use groggy::core::streaming::*;
use groggy::storage::table::BaseTable;
use groggy::types::AttrValue;
use std::collections::HashMap;
use std::sync::Arc;

/// Create a test table with sample data for streaming tests
fn create_test_table() -> BaseTable {
    let mut table = BaseTable::new();

    // Add sample data - 1000 rows for testing virtual scrolling
    let mut names = Vec::new();
    let mut ages = Vec::new();
    let mut scores = Vec::new();

    for i in 0..1000 {
        names.push(AttrValue::Text(format!("User_{}", i)));
        ages.push(AttrValue::Int(20 + (i % 60) as i64));
        scores.push(AttrValue::Float((i as f32 * 0.1) % 100.0));
    }

    table.add_column("name".to_string(), names).unwrap();
    table.add_column("age".to_string(), ages).unwrap();
    table.add_column("score".to_string(), scores).unwrap();

    table
}

#[cfg(test)]
mod data_source_tests {
    use super::*;

    #[test]
    fn test_data_source_basic_functionality() {
        let table = create_test_table();

        // Test basic DataSource implementation
        assert_eq!(table.total_rows(), 1000);
        assert_eq!(table.total_cols(), 3);
        assert!(table.supports_streaming());

        // Test column names and types
        let names = table.get_column_names();
        assert_eq!(names, vec!["name", "age", "score"]);

        let types = table.get_column_types();
        assert_eq!(types.len(), 3);
    }

    #[test]
    fn test_data_window_creation() {
        let table = create_test_table();

        // Test getting a data window
        let window = table.get_window(0, 10);

        assert_eq!(window.headers.len(), 3);
        assert_eq!(window.rows.len(), 10);
        assert_eq!(window.total_rows, 1000);
        assert_eq!(window.start_offset, 0);

        // Test window data content
        assert_eq!(window.headers, vec!["name", "age", "score"]);
        if let AttrValue::Text(name) = &window.rows[0][0] {
            assert_eq!(name, "User_0");
        } else {
            panic!("Expected text value for name");
        }
    }

    #[test]
    fn test_data_window_boundary_conditions() {
        let table = create_test_table();

        // Test empty window
        let empty_window = table.get_window(1000, 10);
        assert_eq!(empty_window.rows.len(), 0);
        assert_eq!(empty_window.start_offset, 1000);

        // Test partial window at end
        let partial_window = table.get_window(995, 10);
        assert_eq!(partial_window.rows.len(), 5);
        assert_eq!(partial_window.start_offset, 995);

        // Test window beyond bounds
        let beyond_window = table.get_window(2000, 10);
        assert_eq!(beyond_window.rows.len(), 0);
    }

    #[test]
    fn test_schema_information() {
        let table = create_test_table();
        let schema = table.get_schema();

        assert_eq!(schema.columns.len(), 3);
        assert_eq!(schema.source_type, "BaseTable");
        assert!(schema.primary_key.is_none());

        // Test column schemas
        assert_eq!(schema.columns[0].name, "name");
        assert_eq!(schema.columns[1].name, "age");
        assert_eq!(schema.columns[2].name, "score");
    }

    #[test]
    fn test_cache_key_generation() {
        let table = create_test_table();

        let key1 = table.get_cache_key(0, 10);
        let key2 = table.get_cache_key(0, 10);
        let key3 = table.get_cache_key(10, 10);

        // Same parameters should generate same key
        assert_eq!(key1.start, key2.start);
        assert_eq!(key1.count, key2.count);
        assert_eq!(key1.source_id, key2.source_id);

        // Different parameters should generate different keys
        assert_ne!(key1.start, key3.start);
    }
}

#[cfg(test)]
mod virtual_scroll_tests {
    use super::*;

    #[test]
    fn test_virtual_scroll_manager_creation() {
        let config = VirtualScrollConfig::default();
        let manager = VirtualScrollManager::new(config);

        assert_eq!(manager.get_config().window_size, 50);
        assert_eq!(manager.get_config().buffer_size, 200);
        assert_eq!(manager.get_config().cache_size, 100);
    }

    #[test]
    fn test_virtual_scroll_basic_operations() {
        let table = create_test_table();
        let config = VirtualScrollConfig::default();
        let manager = VirtualScrollManager::new(config);

        // Test getting visible window
        let window = manager.get_visible_window(&table).unwrap();
        assert_eq!(window.rows.len(), 50); // Default window size
        assert_eq!(window.start_offset, 0);

        // Test getting window at specific offset
        let offset_window = manager.get_window_at_offset(&table, 100).unwrap();
        assert_eq!(offset_window.rows.len(), 50);
        assert_eq!(offset_window.start_offset, 100);
    }

    #[test]
    fn test_virtual_scroll_caching() {
        let table = create_test_table();
        let config = VirtualScrollConfig::default();
        let mut manager = VirtualScrollManager::new(config);

        // First access - should be a cache miss
        let _window1 = manager.get_window_at_offset(&table, 0).unwrap();
        let stats1 = manager.get_cache_stats();
        assert_eq!(stats1.misses, 1);
        assert_eq!(stats1.hits, 0);

        // Second access to same window - should be a cache hit
        let _window2 = manager.get_window_at_offset(&table, 0).unwrap();
        let stats2 = manager.get_cache_stats();
        assert_eq!(stats2.hits, 1);
        assert_eq!(stats2.misses, 1);
    }

    #[test]
    fn test_virtual_scroll_handle_scroll() {
        let table = create_test_table();
        let config = VirtualScrollConfig::default();
        let mut manager = VirtualScrollManager::new(config);

        // Test scroll that doesn't require update (small movement)
        let result1 = manager.handle_scroll(5, &table).unwrap();
        assert!(!result1.updated); // Should not update for small scroll

        // Test scroll that requires update (large movement)
        let result2 = manager.handle_scroll(100, &table).unwrap();
        assert!(result2.updated); // Should update for large scroll
        assert!(result2.new_window.is_some());
        assert_eq!(result2.new_offset, 100);
    }

    #[test]
    fn test_virtual_scroll_preload_buffer() {
        let table = create_test_table();
        let config = VirtualScrollConfig::default();
        let manager = VirtualScrollManager::new(config);

        // Test preloading buffer around center offset
        let result = manager.preload_buffer(&table, 500);
        assert!(result.is_ok());

        // After preloading, cache should have multiple entries
        let stats = manager.get_cache_stats();
        assert!(stats.size > 1);
    }

    #[test]
    fn test_virtual_scroll_cache_overflow() {
        let table = create_test_table();
        let mut config = VirtualScrollConfig::default();
        config.cache_size = 2; // Very small cache to test overflow

        let manager = VirtualScrollManager::new(config);

        // Access multiple windows to trigger cache overflow
        let _w1 = manager.get_window_at_offset(&table, 0).unwrap();
        let _w2 = manager.get_window_at_offset(&table, 100).unwrap();
        let _w3 = manager.get_window_at_offset(&table, 200).unwrap();

        let stats = manager.get_cache_stats();
        assert!(stats.size <= 2); // Should not exceed cache size
    }
}

#[cfg(test)]
mod streaming_server_tests {
    use super::*;
    use tokio;

    #[tokio::test]
    async fn test_streaming_server_creation() {
        let table = create_test_table();
        let data_source: Arc<dyn DataSource> = Arc::new(table);
        let config = StreamingConfig::default();

        let server = StreamingServer::new(data_source, config);
        let stats = server.get_stats();

        assert_eq!(stats.active_connections, 0);
        assert_eq!(stats.total_rows, 1000);
        assert_eq!(stats.total_cols, 3);
    }

    #[tokio::test]
    async fn test_streaming_server_start() {
        let table = create_test_table();
        let data_source: Arc<dyn DataSource> = Arc::new(table);
        let config = StreamingConfig::default();

        let server = StreamingServer::new(data_source, config);

        // Test starting server on available port
        let handle = server.start(0).await; // Port 0 = any available port
        assert!(handle.is_ok());

        let server_handle = handle.unwrap();
        assert!(server_handle.is_running);
    }

    #[tokio::test]
    async fn test_broadcast_update() {
        let table = create_test_table();
        let data_source: Arc<dyn DataSource> = Arc::new(table);
        let config = StreamingConfig::default();

        let server = StreamingServer::new(data_source, config);

        let update = DataUpdate {
            update_type: UpdateType::Refresh,
            affected_rows: vec![0, 1, 2],
            new_data: None,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };

        // Test broadcasting update (should not error even with no clients)
        let result = server.broadcast_update(update).await;
        assert!(result.is_ok());
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_basetable_streaming_integration() {
        let table = create_test_table();

        // Test that BaseTable properly implements streaming
        assert!(table.supports_streaming());
        assert_eq!(table.total_rows(), 1000);
        assert_eq!(table.total_cols(), 3);

        // Test streaming configuration
        let streaming_config = table.streaming_config();
        assert_eq!(streaming_config.port, 8080); // Default port
        assert_eq!(streaming_config.max_connections, 100);
    }

    #[test]
    fn test_end_to_end_virtual_scrolling() {
        let table = create_test_table();
        let config = VirtualScrollConfig {
            window_size: 25,
            buffer_size: 50,
            cache_size: 10,
            auto_preload: true,
            cache_timeout_secs: 300,
        };

        let mut manager = VirtualScrollManager::new(config);

        // Simulate user scrolling through data
        let positions = vec![0, 25, 50, 100, 200, 500, 800, 900];

        for pos in positions {
            let result = manager.handle_scroll(pos, &table).unwrap();

            if result.updated {
                assert!(result.new_window.is_some());
                let window = result.new_window.unwrap();
                assert_eq!(window.start_offset, pos);
                assert!(window.rows.len() <= 25); // Window size or remaining rows
            }
        }

        // Check that cache is being utilized
        let final_stats = manager.get_cache_stats();
        assert!(final_stats.hits > 0);
        assert!(final_stats.hit_rate > 0.0);
    }

    #[test]
    fn test_websocket_message_serialization() {
        let table = create_test_table();
        let window = table.get_window(0, 10);

        // Test serializing WebSocket messages
        let msg = WSMessage::InitialData {
            window: window.clone(),
            total_rows: 1000,
        };

        let serialized = serde_json::to_string(&msg);
        assert!(serialized.is_ok());

        let json_str = serialized.unwrap();
        let deserialized: Result<WSMessage, _> = serde_json::from_str(&json_str);
        assert!(deserialized.is_ok());
    }

    #[test]
    fn test_cache_invalidation_on_data_changes() {
        let mut table = create_test_table();
        let config = VirtualScrollConfig::default();
        let manager = VirtualScrollManager::new(config);

        // Get initial cache key
        let key1 = table.get_cache_key(0, 10);
        let version1 = table.get_version();

        // Modify table (this should increment version)
        table.increment_version();

        // Get new cache key after modification
        let key2 = table.get_cache_key(0, 10);
        let version2 = table.get_version();

        // Version should have changed, invalidating cache
        assert_ne!(version1, version2);
        assert_ne!(key1.version, key2.version);
    }

    #[test]
    fn test_streaming_config_customization() {
        let mut table = create_test_table();

        let custom_config = StreamingConfig {
            scroll_config: VirtualScrollConfig {
                window_size: 100,
                buffer_size: 300,
                cache_size: 50,
                auto_preload: false,
                cache_timeout_secs: 600,
            },
            port: 9090,
            max_connections: 200,
            auto_broadcast: false,
            update_throttle_ms: 50,
        };

        table.set_streaming_config(custom_config.clone());

        let retrieved_config = table.streaming_config();
        assert_eq!(retrieved_config.port, 9090);
        assert_eq!(retrieved_config.max_connections, 200);
        assert_eq!(retrieved_config.scroll_config.window_size, 100);
    }
}

#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_data_window_performance() {
        let table = create_test_table();

        let start = Instant::now();

        // Test getting multiple windows
        for i in 0..100 {
            let _window = table.get_window(i * 10, 10);
        }

        let duration = start.elapsed();
        println!("100 data windows took: {:?}", duration);

        // Should be fast - under 100ms for 100 windows
        assert!(duration.as_millis() < 100);
    }

    #[test]
    fn test_cache_performance() {
        let table = create_test_table();
        let config = VirtualScrollConfig::default();
        let manager = VirtualScrollManager::new(config);

        let start = Instant::now();

        // Fill cache
        for i in 0..10 {
            let _window = manager.get_window_at_offset(&table, i * 50).unwrap();
        }

        // Test cached access performance
        let cache_start = Instant::now();
        for i in 0..10 {
            let _window = manager.get_window_at_offset(&table, i * 50).unwrap();
        }
        let cache_duration = cache_start.elapsed();

        let total_duration = start.elapsed();

        println!(
            "Initial load: {:?}, Cached access: {:?}",
            total_duration - cache_duration,
            cache_duration
        );

        // Cached access should be significantly faster
        assert!(cache_duration < total_duration / 2);
    }

    #[test]
    fn test_large_dataset_handling() {
        // Create larger test dataset
        let mut table = BaseTable::new();

        let mut data = Vec::new();
        for i in 0..10000 {
            data.push(AttrValue::Int(i as i64));
        }

        table.add_column("large_data".to_string(), data).unwrap();

        let config = VirtualScrollConfig::default();
        let manager = VirtualScrollManager::new(config);

        let start = Instant::now();

        // Test virtual scrolling with large dataset
        let _window = manager.get_window_at_offset(&table, 5000).unwrap();

        let duration = start.elapsed();
        println!("Large dataset window access took: {:?}", duration);

        // Should handle large datasets efficiently
        assert!(duration.as_millis() < 50);
    }
}
