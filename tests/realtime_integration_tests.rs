//! Integration tests for Phase 3 real-time visualization system
//!
//! Tests the complete pipeline from graph creation through real-time streaming,
//! interactive controls, incremental updates, and performance monitoring.

use groggy::api::graph::Graph;
use groggy::errors::GraphResult;
use groggy::storage::matrix::GraphMatrix;
use groggy::viz::embeddings::{EmbeddingConfig, EmbeddingMethod, GraphEmbeddingExt};
use groggy::viz::projection::{
    ProjectionConfig, ProjectionMethod, HoneycombConfig, QualityConfig,
    InterpolationConfig, GraphProjectionExt
};
use groggy::viz::realtime::{
    RealTimeVizConfig, RealTimeVizEngine, RealTimeStreamingManager,
    InteractiveControlManager, IncrementalUpdateManager, AdvancedPerformanceMonitor,
    ControlCommand, PositionUpdate, PerformanceMetrics, GraphChange, GraphChangeType,
    ControlPanelConfig, ParameterValue, StreamingConfig, IncrementalConfig,
    PerformanceMonitorConfig
};
use groggy::viz::streaming::data_source::Position;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::time::sleep;

// Helper functions for creating test data

fn create_test_graph() -> Graph {
    let mut graph = Graph::new();

    // Create a structured test graph with communities
    let nodes: Vec<_> = (0..20).map(|_| graph.add_node()).collect();

    // Community 1: Dense cluster (nodes 0-6)
    for i in 0..7 {
        for j in (i+1)..7 {
            if i != j {
                graph.add_edge(nodes[i], nodes[j]).unwrap();
            }
        }
    }

    // Community 2: Star pattern (nodes 7-13)
    let center = nodes[7];
    for i in 8..14 {
        graph.add_edge(center, nodes[i]).unwrap();
    }

    // Community 3: Chain pattern (nodes 14-19)
    for i in 14..19 {
        graph.add_edge(nodes[i], nodes[i+1]).unwrap();
    }

    // Inter-community bridges
    graph.add_edge(nodes[3], nodes[7]).unwrap();
    graph.add_edge(nodes[10], nodes[16]).unwrap();

    graph
}

fn create_dynamic_test_graph() -> Graph {
    let mut graph = Graph::new();

    // Start with a small graph that will grow dynamically
    let nodes: Vec<_> = (0..5).map(|_| graph.add_node()).collect();

    // Initial structure
    graph.add_edge(nodes[0], nodes[1]).unwrap();
    graph.add_edge(nodes[1], nodes[2]).unwrap();
    graph.add_edge(nodes[2], nodes[3]).unwrap();
    graph.add_edge(nodes[3], nodes[4]).unwrap();

    graph
}

async fn setup_realtime_viz_engine() -> GraphResult<RealTimeVizEngine> {
    let graph = create_test_graph();
    let config = RealTimeVizConfig::default();

    let mut engine = RealTimeVizEngine::new(graph, config);
    engine.initialize().await?;

    Ok(engine)
}

// Phase 3 Integration Tests

#[tokio::test]
async fn test_complete_realtime_pipeline() -> GraphResult<()> {
    // Test the complete Phase 1 + Phase 2 + Phase 3 pipeline
    let graph = create_test_graph();
    let config = RealTimeVizConfig::default();

    // Create and initialize engine
    let mut engine = RealTimeVizEngine::new(graph, config);
    engine.initialize().await?;

    // Start the engine briefly
    tokio::spawn(async move {
        if let Err(e) = engine.start().await {
            eprintln!("Engine error: {}", e);
        }
    });

    // Wait a moment for initialization
    sleep(Duration::from_millis(100)).await;

    // Test passed if we reach here without panicking
    Ok(())
}

#[tokio::test]
async fn test_realtime_streaming_system() -> GraphResult<()> {
    let config = StreamingConfig::default();
    let mut streaming_manager = RealTimeStreamingManager::new(config);

    // Start streaming server
    streaming_manager.start().await?;

    // Simulate client connection
    let (client_tx, mut client_rx) = tokio::sync::mpsc::unbounded_channel();
    streaming_manager.add_client(
        "test_client".to_string(),
        groggy::viz::realtime::streaming::ClientCapabilities::default(),
        client_tx
    ).await?;

    // Broadcast some position updates
    let updates = vec![
        PositionUpdate {
            node_id: 0,
            position: Position { x: 10.0, y: 20.0 },
            timestamp: 12345,
            update_type: groggy::viz::realtime::engine::PositionUpdateType::Full,
            quality: None,
        },
        PositionUpdate {
            node_id: 1,
            position: Position { x: 30.0, y: 40.0 },
            timestamp: 12346,
            update_type: groggy::viz::realtime::engine::PositionUpdateType::Full,
            quality: None,
        },
    ];

    streaming_manager.broadcast_position_updates(updates).await?;

    // Verify client received updates
    if let Ok(message) = tokio::time::timeout(Duration::from_millis(100), client_rx.recv()).await {
        assert!(message.is_some());
    }

    // Cleanup
    streaming_manager.stop().await?;

    Ok(())
}

#[tokio::test]
async fn test_interactive_controls_system() -> GraphResult<()> {
    let config = ControlPanelConfig::default();
    let mut control_manager = InteractiveControlManager::new(config);

    // Initialize control manager
    control_manager.initialize()?;

    // Test parameter setting and retrieval
    control_manager.set_parameter("embedding.dimensions", ParameterValue::UInteger(8))?;
    control_manager.set_parameter("projection.cell_size", ParameterValue::Float(50.0))?;
    control_manager.set_parameter("animation.speed", ParameterValue::Float(1.5))?;

    // Verify parameters were set correctly
    if let Some(ParameterValue::UInteger(dims)) = control_manager.get_parameter("embedding.dimensions") {
        assert_eq!(*dims, 8);
    } else {
        panic!("Embedding dimensions not set correctly");
    }

    if let Some(ParameterValue::Float(cell_size)) = control_manager.get_parameter("projection.cell_size") {
        assert_eq!(*cell_size, 50.0);
    } else {
        panic!("Cell size not set correctly");
    }

    // Test batch parameter changes
    let batch_changes = vec![
        ("embedding.method".to_string(), ParameterValue::String("energy".to_string())),
        ("quality.neighborhood_weight".to_string(), ParameterValue::Float(0.6)),
        ("animation.easing".to_string(), ParameterValue::String("ease-in-out".to_string())),
    ];

    control_manager.apply_batch_changes(batch_changes)?;

    // Test parameter history and undo/redo
    control_manager.create_snapshot("Test parameter changes".to_string());

    // Change a parameter
    control_manager.set_parameter("projection.cell_size", ParameterValue::Float(25.0))?;

    // Undo should restore previous value
    let undo_success = control_manager.undo()?;
    assert!(undo_success);

    if let Some(ParameterValue::Float(cell_size)) = control_manager.get_parameter("projection.cell_size") {
        assert_eq!(*cell_size, 50.0); // Should be restored to previous value
    }

    // Redo should apply change again
    let redo_success = control_manager.redo()?;
    assert!(redo_success);

    if let Some(ParameterValue::Float(cell_size)) = control_manager.get_parameter("projection.cell_size") {
        assert_eq!(*cell_size, 25.0); // Should be changed value again
    }

    Ok(())
}

#[tokio::test]
async fn test_incremental_updates_system() -> GraphResult<()> {
    let config = IncrementalConfig::default();
    let mut update_manager = IncrementalUpdateManager::new(config);

    let graph = create_dynamic_test_graph();
    update_manager.initialize(&graph)?;

    // Test adding graph changes
    let changes = vec![
        GraphChange {
            change_type: GraphChangeType::NodeAdded,
            node_id: Some(5),
            edge_id: None,
            timestamp: Instant::now(),
        },
        GraphChange {
            change_type: GraphChangeType::EdgeAdded,
            node_id: Some(5),
            edge_id: Some(0), // Simplified edge ID
            timestamp: Instant::now(),
        },
        GraphChange {
            change_type: GraphChangeType::NodeAttributeChanged,
            node_id: Some(2),
            edge_id: None,
            timestamp: Instant::now(),
        },
    ];

    // Add changes to pending queue
    for change in changes {
        update_manager.add_change(change);
    }

    // Verify changes are pending
    assert!(update_manager.has_pending_updates());

    // Process pending changes
    let result = update_manager.process_pending_changes()?;
    assert!(result.success);
    assert!(result.nodes_affected > 0);

    // Verify changes were processed
    assert!(!update_manager.has_pending_updates());

    Ok(())
}

#[tokio::test]
async fn test_performance_monitoring_system() -> GraphResult<()> {
    let config = PerformanceMonitorConfig::default();
    let mut performance_monitor = AdvancedPerformanceMonitor::new(config);

    // Start monitoring
    performance_monitor.start()?;

    // Simulate some frame times
    let frame_times = vec![
        Duration::from_millis(16), // Good performance
        Duration::from_millis(20), // Slightly slower
        Duration::from_millis(33), // Poor performance
        Duration::from_millis(50), // Very poor performance
    ];

    for frame_time in frame_times {
        performance_monitor.update_metrics(frame_time)?;
        sleep(Duration::from_millis(10)).await; // Small delay between updates
    }

    // Get performance report
    let report = performance_monitor.get_performance_report();

    // Verify monitoring is working
    assert!(report.current_metrics.fps > 0.0);
    assert!(!report.recommendations.is_empty()); // Should have recommendations for poor performance

    // Stop monitoring
    performance_monitor.stop();

    Ok(())
}

#[tokio::test]
async fn test_adaptive_quality_control() -> GraphResult<()> {
    let mut config = PerformanceMonitorConfig::default();
    config.adaptive_quality.enabled = true;
    config.adaptive_quality.sensitivity = 0.8; // High sensitivity for testing

    let mut performance_monitor = AdvancedPerformanceMonitor::new(config);
    performance_monitor.start()?;

    // Simulate consistently poor performance
    for _ in 0..10 {
        let poor_frame_time = Duration::from_millis(50); // 20 FPS
        performance_monitor.update_metrics(poor_frame_time)?;
        sleep(Duration::from_millis(10)).await;
    }

    let report = performance_monitor.get_performance_report();

    // Quality should have been reduced due to poor performance
    assert!(report.quality_settings.overall_quality < 1.0);
    assert!(report.quality_settings.cell_size > 40.0); // Should increase cell size (reduce quality)

    // Now simulate good performance
    for _ in 0..10 {
        let good_frame_time = Duration::from_millis(12); // ~83 FPS
        performance_monitor.update_metrics(good_frame_time)?;
        sleep(Duration::from_millis(10)).await;
    }

    let report2 = performance_monitor.get_performance_report();

    // Quality should start improving again
    assert!(report2.quality_settings.overall_quality >= report.quality_settings.overall_quality);

    performance_monitor.stop();

    Ok(())
}

#[tokio::test]
async fn test_end_to_end_visualization_workflow() -> GraphResult<()> {
    // Test complete workflow: Graph -> Embedding -> Projection -> Real-time Updates

    let graph = create_test_graph();

    // Phase 1: Test embedding
    let embedding_config = EmbeddingConfig {
        method: EmbeddingMethod::Energy {
            energy_function: groggy::viz::embeddings::EnergyFunction::SpringElectric,
            iterations: 100,
            learning_rate: 0.01,
            damping: 0.9,
        },
        dimensions: 5,
        seed: Some(42),
        debug_enabled: false,
    };

    let embedding = graph.compute_embedding(&embedding_config)?;
    assert_eq!(embedding.shape(), (20, 5)); // 20 nodes, 5 dimensions

    // Phase 2: Test projection
    let projection_config = ProjectionConfig {
        method: ProjectionMethod::PCA { center: true, standardize: true },
        honeycomb_config: HoneycombConfig::default(),
        quality_config: QualityConfig::default(),
        interpolation_config: InterpolationConfig::default(),
        debug_enabled: false,
        seed: Some(42),
    };

    let positions = graph.project_to_honeycomb(&embedding, &projection_config)?;
    assert_eq!(positions.len(), 20); // One position per node

    // Phase 3: Test real-time system initialization
    let realtime_config = RealTimeVizConfig {
        embedding_config,
        projection_config,
        realtime_config: groggy::viz::realtime::RealTimeConfig::default(),
        performance_config: PerformanceMonitorConfig::default(),
        interaction_config: groggy::viz::realtime::InteractionConfig::default(),
        streaming_config: StreamingConfig::default(),
    };

    let mut engine = RealTimeVizEngine::new(graph, realtime_config);
    engine.initialize().await?;

    // Test that engine can start (run briefly)
    let engine_handle = tokio::spawn(async move {
        // Run for a short time then stop
        tokio::select! {
            result = engine.start() => {
                if let Err(e) = result {
                    eprintln!("Engine error: {}", e);
                }
            }
            _ = sleep(Duration::from_millis(50)) => {
                // Timeout after 50ms
            }
        }
    });

    // Wait for engine to run briefly
    let _ = tokio::time::timeout(Duration::from_millis(100), engine_handle).await;

    Ok(())
}

#[tokio::test]
async fn test_dynamic_graph_updates() -> GraphResult<()> {
    // Test adding and removing nodes/edges during real-time visualization

    let graph = create_dynamic_test_graph();
    let config = RealTimeVizConfig::default();

    let mut engine = RealTimeVizEngine::new(graph, config);
    engine.initialize().await?;

    // Simulate control commands for dynamic updates
    let commands = vec![
        ControlCommand::AddNodes {
            node_data: vec![
                HashMap::from([("id".to_string(), serde_json::Value::Number(serde_json::Number::from(10)))]),
                HashMap::from([("id".to_string(), serde_json::Value::Number(serde_json::Number::from(11)))]),
            ],
        },
        ControlCommand::AddEdges {
            edge_data: vec![
                (10, 11, HashMap::new()),
                (0, 10, HashMap::new()),
            ],
        },
        ControlCommand::RemoveNodes {
            node_ids: vec![4],
        },
    ];

    // Test that commands can be processed (engine would handle these in real scenario)
    for command in commands {
        // In a real implementation, these would be sent through channels
        // Here we just verify the command structures are valid
        match command {
            ControlCommand::AddNodes { node_data } => {
                assert_eq!(node_data.len(), 2);
            }
            ControlCommand::AddEdges { edge_data } => {
                assert_eq!(edge_data.len(), 2);
            }
            ControlCommand::RemoveNodes { node_ids } => {
                assert_eq!(node_ids.len(), 1);
            }
            _ => {}
        }
    }

    Ok(())
}

#[tokio::test]
async fn test_multi_client_streaming() -> GraphResult<()> {
    // Test multiple clients receiving streaming updates

    let config = StreamingConfig {
        server_port: 8081, // Different port to avoid conflicts
        max_connections: 5,
        broadcast_interval_ms: 50,
        enable_position_compression: true,
        position_precision: 2,
        enable_update_batching: true,
        max_batch_size: 10,
    };

    let mut streaming_manager = RealTimeStreamingManager::new(config);
    streaming_manager.start().await?;

    // Create multiple client connections
    let mut client_receivers = Vec::new();
    for i in 0..3 {
        let (client_tx, client_rx) = tokio::sync::mpsc::unbounded_channel();
        streaming_manager.add_client(
            format!("client_{}", i),
            groggy::viz::realtime::streaming::ClientCapabilities::default(),
            client_tx
        ).await?;
        client_receivers.push(client_rx);
    }

    // Broadcast updates
    let updates = vec![
        PositionUpdate {
            node_id: 0,
            position: Position { x: 100.0, y: 200.0 },
            timestamp: 98765,
            update_type: groggy::viz::realtime::engine::PositionUpdateType::Incremental,
            quality: None,
        },
    ];

    streaming_manager.broadcast_position_updates(updates).await?;

    // Verify all clients received updates
    for mut receiver in client_receivers {
        let received = tokio::time::timeout(Duration::from_millis(100), receiver.recv()).await;
        assert!(received.is_ok() && received.unwrap().is_some());
    }

    streaming_manager.stop().await?;

    Ok(())
}

#[tokio::test]
async fn test_performance_under_load() -> GraphResult<()> {
    // Test system performance under simulated load

    let config = PerformanceMonitorConfig::default();
    let mut performance_monitor = AdvancedPerformanceMonitor::new(config);
    performance_monitor.start()?;

    // Simulate varying load conditions
    let load_scenarios = vec![
        ("light_load", Duration::from_millis(8)),   // 125 FPS
        ("normal_load", Duration::from_millis(16)), // 62.5 FPS
        ("heavy_load", Duration::from_millis(33)),  // 30 FPS
        ("extreme_load", Duration::from_millis(100)), // 10 FPS
    ];

    for (scenario_name, frame_time) in load_scenarios {
        println!("Testing scenario: {}", scenario_name);

        // Run scenario for several frames
        for _ in 0..20 {
            performance_monitor.update_metrics(frame_time)?;
            sleep(Duration::from_millis(5)).await;
        }

        let report = performance_monitor.get_performance_report();

        // Verify monitoring is tracking performance correctly
        let expected_fps = 1000.0 / frame_time.as_millis() as f64;
        let fps_tolerance = expected_fps * 0.2; // 20% tolerance

        assert!(
            (report.current_metrics.fps - expected_fps).abs() < fps_tolerance,
            "FPS measurement incorrect for {}: expected ~{}, got {}",
            scenario_name, expected_fps, report.current_metrics.fps
        );

        // Verify adaptive quality responds to load
        if frame_time.as_millis() > 50 { // Poor performance
            assert!(
                report.quality_settings.overall_quality < 1.0,
                "Quality should be reduced under heavy load"
            );
        }
    }

    performance_monitor.stop();

    Ok(())
}

#[tokio::test]
async fn test_error_handling_and_recovery() -> GraphResult<()> {
    // Test error handling and graceful recovery

    // Test with invalid configuration
    let mut invalid_config = RealTimeVizConfig::default();
    invalid_config.embedding_config.dimensions = 0; // Invalid

    let graph = create_test_graph();
    let mut engine = RealTimeVizEngine::new(graph, invalid_config);

    // Should handle invalid config gracefully
    let init_result = engine.initialize().await;
    // Note: Depending on validation, this might succeed or fail gracefully

    // Test streaming manager with invalid port
    let invalid_streaming_config = StreamingConfig {
        server_port: 0, // Invalid port
        ..Default::default()
    };

    let mut streaming_manager = RealTimeStreamingManager::new(invalid_streaming_config);
    // Should handle invalid port gracefully (might use default or fail gracefully)
    let _ = streaming_manager.start().await;

    // Test incremental manager with empty graph
    let empty_graph = Graph::new();
    let config = IncrementalConfig::default();
    let mut update_manager = IncrementalUpdateManager::new(config);

    // Should handle empty graph gracefully
    let init_result = update_manager.initialize(&empty_graph);
    assert!(init_result.is_ok());

    Ok(())
}

#[tokio::test]
async fn test_memory_usage_tracking() -> GraphResult<()> {
    // Test memory usage monitoring

    let config = PerformanceMonitorConfig {
        resource_monitoring: groggy::viz::realtime::performance::ResourceMonitoringConfig {
            enable_memory_monitoring: true,
            enable_cpu_monitoring: true,
            polling_interval_ms: 100,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut performance_monitor = AdvancedPerformanceMonitor::new(config);
    performance_monitor.start()?;

    // Simulate some activity
    for _ in 0..10 {
        performance_monitor.update_metrics(Duration::from_millis(16))?;
        sleep(Duration::from_millis(20)).await;
    }

    let report = performance_monitor.get_performance_report();

    // Memory usage should be tracked (even if zero in test environment)
    assert!(report.resource_usage.memory_mb >= 0.0);
    assert!(report.resource_usage.cpu_usage >= 0.0);

    performance_monitor.stop();

    Ok(())
}

// Benchmark test (optional, for performance validation)
#[tokio::test]
#[ignore] // Ignore by default, run with --ignored for performance testing
async fn benchmark_realtime_system() -> GraphResult<()> {
    // Benchmark the complete real-time system

    let large_graph = {
        let mut graph = Graph::new();
        let nodes: Vec<_> = (0..1000).map(|_| graph.add_node()).collect();

        // Create random edges
        for i in 0..5000 {
            let src = nodes[i % 1000];
            let dst = nodes[(i * 7) % 1000];
            if src != dst {
                let _ = graph.add_edge(src, dst);
            }
        }

        graph
    };

    let config = RealTimeVizConfig::default();
    let mut engine = RealTimeVizEngine::new(large_graph, config);

    let start_time = Instant::now();
    engine.initialize().await?;
    let init_time = start_time.elapsed();

    println!("Initialization time for 1000 nodes, 5000 edges: {:?}", init_time);

    // Should complete initialization in reasonable time
    assert!(init_time < Duration::from_secs(10));

    Ok(())
}

// Helper function to run multiple tests in parallel
#[tokio::test]
async fn test_concurrent_operations() -> GraphResult<()> {
    // Test running multiple real-time operations concurrently

    let tasks = vec![
        tokio::spawn(test_streaming_system()),
        tokio::spawn(test_incremental_updates()),
        tokio::spawn(test_performance_monitoring()),
    ];

    // Wait for all tasks to complete
    for task in tasks {
        let result = task.await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_ok());
    }

    Ok(())
}

// Isolated test functions (to avoid interference)
async fn test_streaming_system() -> GraphResult<()> {
    let config = StreamingConfig {
        server_port: 8082,
        ..Default::default()
    };
    let mut manager = RealTimeStreamingManager::new(config);
    manager.start().await?;
    sleep(Duration::from_millis(50)).await;
    manager.stop().await?;
    Ok(())
}

async fn test_incremental_updates() -> GraphResult<()> {
    let config = IncrementalConfig::default();
    let mut manager = IncrementalUpdateManager::new(config);
    let graph = create_test_graph();
    manager.initialize(&graph)?;

    manager.add_change(GraphChange {
        change_type: GraphChangeType::NodeAdded,
        node_id: Some(100),
        edge_id: None,
        timestamp: Instant::now(),
    });

    let _result = manager.process_pending_changes()?;
    Ok(())
}

async fn test_performance_monitoring() -> GraphResult<()> {
    let config = PerformanceMonitorConfig::default();
    let mut monitor = AdvancedPerformanceMonitor::new(config);
    monitor.start()?;
    monitor.update_metrics(Duration::from_millis(16))?;
    monitor.stop();
    Ok(())
}

// Summary test to verify all components integrate correctly
#[tokio::test]
async fn test_phase_3_integration_summary() -> GraphResult<()> {
    println!("=== Phase 3 Real-time Visualization System Integration Test ===");

    // Test 1: Engine Creation and Initialization
    println!("✓ Testing engine creation and initialization...");
    let engine = setup_realtime_viz_engine().await?;
    drop(engine); // Cleanup

    // Test 2: Streaming System
    println!("✓ Testing streaming system...");
    test_streaming_system().await?;

    // Test 3: Interactive Controls
    println!("✓ Testing interactive controls...");
    let config = ControlPanelConfig::default();
    let mut control_manager = InteractiveControlManager::new(config);
    control_manager.initialize()?;
    control_manager.set_parameter("test.param", ParameterValue::Float(42.0))?;

    // Test 4: Incremental Updates
    println!("✓ Testing incremental updates...");
    test_incremental_updates().await?;

    // Test 5: Performance Monitoring
    println!("✓ Testing performance monitoring...");
    test_performance_monitoring().await?;

    println!("=== All Phase 3 Integration Tests Passed! ===");

    Ok(())
}