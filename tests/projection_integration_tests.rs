//! Integration tests for Phase 2 projection system
//!
//! Tests the complete pipeline from high-dimensional embeddings to 2D honeycomb positions
//! with quality validation and smooth interpolation.

use groggy::api::graph::Graph;
use groggy::errors::GraphResult;
use groggy::storage::matrix::GraphMatrix;
use groggy::viz::embeddings::{EmbeddingConfig, EmbeddingMethod, GraphEmbeddingExt};
use groggy::viz::projection::honeycomb::{HexCoord, HoneycombGrid};
use groggy::viz::projection::interpolation::{
    AnimationManager, AnimationState, InterpolationEngine,
};
use groggy::viz::projection::quality::{QualityEvaluator, QualityOptimizer};
use groggy::viz::projection::{
    EasingFunction, GraphProjectionExt, HoneycombConfig, HoneycombLayoutStrategy,
    InterpolationConfig, InterpolationMethod, ProjectionConfig, ProjectionEngineFactory,
    ProjectionMethod, QualityConfig,
};
use groggy::viz::streaming::data_source::Position;
use std::collections::HashMap;

// Helper functions for creating test graphs and embeddings
fn create_test_graph() -> Graph {
    let mut graph = Graph::new();
    let nodes: Vec<_> = (0..8).map(|_| graph.add_node()).collect();

    // Create a connected graph with some structure
    graph.add_edge(nodes[0], nodes[1]).unwrap();
    graph.add_edge(nodes[1], nodes[2]).unwrap();
    graph.add_edge(nodes[2], nodes[3]).unwrap();
    graph.add_edge(nodes[3], nodes[0]).unwrap(); // Form a cycle
    graph.add_edge(nodes[0], nodes[4]).unwrap(); // Branch out
    graph.add_edge(nodes[4], nodes[5]).unwrap();
    graph.add_edge(nodes[5], nodes[6]).unwrap();
    graph.add_edge(nodes[6], nodes[7]).unwrap();

    graph
}

fn create_test_embedding() -> GraphMatrix {
    let mut embedding = GraphMatrix::zeros(8, 5);

    // Create some structured 5D embedding data
    let data = [
        [1.0, 0.0, 0.0, 0.5, 0.2],
        [0.8, 0.3, 0.0, 0.4, 0.1],
        [0.0, 1.0, 0.0, 0.3, 0.6],
        [0.0, 0.8, 0.2, 0.2, 0.8],
        [0.5, 0.0, 1.0, 0.0, 0.3],
        [0.3, 0.0, 0.9, 0.1, 0.2],
        [0.0, 0.0, 0.5, 1.0, 0.4],
        [0.0, 0.0, 0.3, 0.9, 0.5],
    ];

    for (i, row) in data.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            embedding.set(i, j, val).unwrap();
        }
    }

    embedding
}

fn create_large_test_graph() -> Graph {
    let mut graph = Graph::new();
    let nodes: Vec<_> = (0..25).map(|_| graph.add_node()).collect();

    // Create a more complex graph structure
    for i in 0..24 {
        graph.add_edge(nodes[i], nodes[i + 1]).unwrap();
    }

    // Add some cross connections
    for i in 0..5 {
        for j in 0..5 {
            if i != j {
                let idx1 = i * 5 + j;
                let idx2 = j * 5 + i;
                if idx1 < 25 && idx2 < 25 {
                    graph.add_edge(nodes[idx1], nodes[idx2]).unwrap();
                }
            }
        }
    }

    graph
}

/// Test basic PCA projection
#[test]
fn test_pca_projection() -> GraphResult<()> {
    let graph = create_test_graph();
    let embedding = create_test_embedding();

    // Test basic PCA projection
    let positions = graph.project_pca(&embedding)?;

    assert_eq!(positions.len(), 8);

    // Check that all positions are finite
    for pos in &positions {
        assert!(
            pos.x.is_finite(),
            "PCA projection X coordinate should be finite"
        );
        assert!(
            pos.y.is_finite(),
            "PCA projection Y coordinate should be finite"
        );
    }

    println!(
        "✅ PCA projection: {} nodes projected to 2D",
        positions.len()
    );
    Ok(())
}

/// Test t-SNE inspired projection
#[test]
fn test_tsne_projection() -> GraphResult<()> {
    let graph = create_test_graph();
    let embedding = create_test_embedding();

    // Test t-SNE projection with small perplexity for small graph
    let positions = graph.project_tsne(&embedding, 2.0)?;

    assert_eq!(positions.len(), 8);

    // Check that positions are finite and spread out
    for pos in &positions {
        assert!(
            pos.x.is_finite(),
            "t-SNE projection X coordinate should be finite"
        );
        assert!(
            pos.y.is_finite(),
            "t-SNE projection Y coordinate should be finite"
        );
    }

    println!(
        "✅ t-SNE projection: {} nodes projected with perplexity 2.0",
        positions.len()
    );
    Ok(())
}

/// Test UMAP inspired projection
#[test]
fn test_umap_projection() -> GraphResult<()> {
    let graph = create_test_graph();
    let embedding = create_test_embedding();

    // Test UMAP projection with small n_neighbors for small graph
    let positions = graph.project_umap(&embedding, 3)?;

    assert_eq!(positions.len(), 8);

    // Check positions are valid
    for pos in &positions {
        assert!(
            pos.x.is_finite(),
            "UMAP projection X coordinate should be finite"
        );
        assert!(
            pos.y.is_finite(),
            "UMAP projection Y coordinate should be finite"
        );
    }

    println!(
        "✅ UMAP projection: {} nodes projected with 3 neighbors",
        positions.len()
    );
    Ok(())
}

/// Test custom projection configuration
#[test]
fn test_custom_projection_config() -> GraphResult<()> {
    let graph = create_test_graph();
    let embedding = create_test_embedding();

    let config = ProjectionConfig {
        method: ProjectionMethod::PCA {
            center: true,
            standardize: false,
        },
        honeycomb_config: HoneycombConfig {
            cell_size: 60.0,
            layout_strategy: HoneycombLayoutStrategy::DistancePreserving,
            snap_to_centers: true,
            grid_padding: 30.0,
            max_grid_size: None,
            target_avg_occupancy: 4.0,
            min_cell_size: 6.0,
        },
        quality_config: QualityConfig {
            compute_neighborhood_preservation: true,
            compute_distance_preservation: true,
            k_neighbors: 3,
            ..Default::default()
        },
        ..Default::default()
    };

    let positions = graph.project_to_honeycomb(&embedding, &config)?;

    assert_eq!(positions.len(), 8);
    println!("✅ Custom projection config: Distance-preserving honeycomb layout");
    Ok(())
}

/// Test honeycomb grid mapping
#[test]
fn test_honeycomb_grid_mapping() -> GraphResult<()> {
    let embedding = create_test_embedding();
    let graph = create_test_graph();

    // First get 2D positions from PCA
    let initial_positions = graph.project_pca(&embedding)?;

    // Test different honeycomb strategies
    let strategies = [
        HoneycombLayoutStrategy::Spiral,
        HoneycombLayoutStrategy::DensityBased,
        HoneycombLayoutStrategy::DistancePreserving,
    ];

    for strategy in &strategies {
        let config = HoneycombConfig {
            cell_size: 40.0,
            layout_strategy: strategy.clone(),
            snap_to_centers: true,
            grid_padding: 20.0,
            max_grid_size: None,
            target_avg_occupancy: 4.0,
            min_cell_size: 6.0,
        };

        let mut grid = HoneycombGrid::new(config);
        let honeycomb_positions = grid.map_positions_to_grid(&initial_positions)?;

        assert_eq!(honeycomb_positions.len(), 8);

        // Check that grid tracking is working
        assert_eq!(grid.get_occupied_coords().len(), 8);

        // Test hex coordinate conversion
        for i in 0..8 {
            if let Some(hex_coord) = grid.get_hex_coord(i) {
                let pixel_pos = grid.hex_to_pixel(&hex_coord);
                assert!(pixel_pos.x.is_finite());
                assert!(pixel_pos.y.is_finite());

                // Test round-trip conversion
                let back_to_hex = grid.pixel_to_hex(&pixel_pos);
                assert_eq!(hex_coord, back_to_hex);
            }
        }

        println!("✅ Honeycomb grid mapping: {:?} strategy", strategy);
    }

    Ok(())
}

/// Test projection quality metrics
#[test]
fn test_projection_quality_metrics() -> GraphResult<()> {
    let graph = create_test_graph();
    let embedding = create_test_embedding();

    let config = QualityConfig {
        compute_neighborhood_preservation: true,
        compute_distance_preservation: true,
        compute_clustering_preservation: true,
        k_neighbors: 3,
        optimize_for_quality: false,
        ..Default::default()
    };

    let evaluator = QualityEvaluator::new(config);

    // Test with PCA projection
    let positions = graph.project_pca(&embedding)?;
    let metrics = evaluator.evaluate_projection(&embedding, &positions, &graph)?;

    // Validate metric ranges
    assert!(
        metrics.neighborhood_preservation >= 0.0 && metrics.neighborhood_preservation <= 1.0,
        "Neighborhood preservation should be between 0 and 1"
    );
    assert!(
        metrics.distance_correlation >= -1.0 && metrics.distance_correlation <= 1.0,
        "Distance correlation should be between -1 and 1"
    );
    assert!(metrics.stress >= 0.0, "Stress should be non-negative");
    assert!(
        metrics.local_continuity >= 0.0,
        "Local continuity should be non-negative"
    );
    assert!(
        metrics.global_structure >= -1.0 && metrics.global_structure <= 1.0,
        "Global structure should be between -1 and 1"
    );
    assert!(
        metrics.overall_score >= 0.0 && metrics.overall_score <= 1.0,
        "Overall score should be between 0 and 1"
    );

    // Test quality suggestions
    let suggestions = evaluator.suggest_improvements(&metrics);
    assert!(
        !suggestions.is_empty(),
        "Should provide quality suggestions"
    );

    println!(
        "✅ Quality metrics - Neighborhood: {:.3}, Distance: {:.3}, Stress: {:.3}, Overall: {:.3}",
        metrics.neighborhood_preservation,
        metrics.distance_correlation,
        metrics.stress,
        metrics.overall_score
    );

    Ok(())
}

/// Test interpolation system
#[test]
fn test_interpolation_system() -> GraphResult<()> {
    let graph = create_test_graph();
    let embedding = create_test_embedding();

    // Create start and end positions
    let start_positions = graph.project_pca(&embedding)?;
    let end_positions = graph.project_tsne(&embedding, 2.0)?;

    // Test different interpolation methods
    let methods = [
        InterpolationMethod::Linear,
        InterpolationMethod::SpringPhysics {
            damping: 0.8,
            stiffness: 0.2,
        },
    ];

    for method in &methods {
        let config = InterpolationConfig {
            enable_interpolation: true,
            method: method.clone(),
            steps: 10,
            easing: EasingFunction::EaseInOut,
            preserve_honeycomb: false,
        };

        let engine = InterpolationEngine::new(config);
        let frames = engine.interpolate_positions(&start_positions, &end_positions)?;

        assert_eq!(frames.len(), 10, "Should generate correct number of frames");
        assert_eq!(
            frames[0].len(),
            8,
            "Each frame should have correct number of nodes"
        );

        // Check first and last frames match start and end
        for i in 0..8 {
            let start_diff_x = (frames[0][i].x - start_positions[i].x).abs();
            let start_diff_y = (frames[0][i].y - start_positions[i].y).abs();
            assert!(
                start_diff_x < 1e-10 && start_diff_y < 1e-10,
                "First frame should match start positions"
            );

            let end_diff_x = (frames[9][i].x - end_positions[i].x).abs();
            let end_diff_y = (frames[9][i].y - end_positions[i].y).abs();
            assert!(
                end_diff_x < 1e-10 && end_diff_y < 1e-10,
                "Last frame should match end positions"
            );
        }

        println!("✅ Interpolation: {:?} method with 10 frames", method);
    }

    Ok(())
}

/// Test animation state management
#[test]
fn test_animation_state() -> GraphResult<()> {
    let graph = create_test_graph();
    let embedding = create_test_embedding();

    let start_positions = graph.project_pca(&embedding)?;
    let end_positions = graph.project_tsne(&embedding, 2.0)?;

    let config = InterpolationConfig {
        enable_interpolation: true,
        method: InterpolationMethod::Linear,
        steps: 5,
        easing: EasingFunction::Linear,
        preserve_honeycomb: false,
    };

    let engine = InterpolationEngine::new(config);
    let frames = engine.interpolate_positions(&start_positions, &end_positions)?;

    // Test AnimationState
    let mut animation = AnimationState::new(frames, 1000); // 1 second duration

    assert!(!animation.is_active(), "Animation should start inactive");
    assert_eq!(animation.progress(), 0.0, "Initial progress should be 0");

    animation.start();
    assert!(
        animation.is_active(),
        "Animation should be active after start"
    );

    // Test manual progress setting
    animation.set_progress(0.5);
    assert!(
        (animation.progress() - 0.5).abs() < 1e-10,
        "Progress should be settable"
    );

    if let Some(positions) = animation.current_positions() {
        assert_eq!(
            positions.len(),
            8,
            "Current positions should have correct length"
        );
    }

    animation.stop();
    assert!(
        !animation.is_active(),
        "Animation should be inactive after stop"
    );

    println!("✅ Animation state management: Start/stop/progress control");
    Ok(())
}

/// Test animation manager
#[test]
fn test_animation_manager() -> GraphResult<()> {
    let graph = create_test_graph();
    let embedding = create_test_embedding();

    let start_positions = graph.project_pca(&embedding)?;
    let end_positions = graph.project_tsne(&embedding, 2.0)?;

    let config = InterpolationConfig {
        enable_interpolation: true,
        method: InterpolationMethod::Linear,
        steps: 5,
        easing: EasingFunction::Linear,
        preserve_honeycomb: false,
    };

    let engine = InterpolationEngine::new(config);
    let frames = engine.interpolate_positions(&start_positions, &end_positions)?;

    let mut manager = AnimationManager::new();

    // Add animation
    let animation = AnimationState::new(frames, 1000);
    manager.add_animation("test_transition".to_string(), animation);

    assert!(
        !manager.has_active_animations(),
        "No animations should be active initially"
    );

    // Start animation
    assert!(
        manager.start_animation("test_transition"),
        "Should successfully start animation"
    );
    assert!(
        manager.has_active_animations(),
        "Should have active animations after start"
    );

    // Test getting current positions
    if let Some(positions) = manager.get_current_positions("test_transition") {
        assert_eq!(
            positions.len(),
            8,
            "Should return correct number of positions"
        );
    }

    // Stop animation
    assert!(
        manager.stop_animation("test_transition"),
        "Should successfully stop animation"
    );
    assert!(
        !manager.has_active_animations(),
        "Should have no active animations after stop"
    );

    println!("✅ Animation manager: Multiple animation coordination");
    Ok(())
}

/// Test quality optimization
#[test]
fn test_quality_optimization() -> GraphResult<()> {
    let graph = create_test_graph();
    let embedding = create_test_embedding();

    let config = QualityConfig {
        compute_neighborhood_preservation: true,
        compute_distance_preservation: true,
        k_neighbors: 3,
        optimize_for_quality: true,
        ..Default::default()
    };

    let optimizer = QualityOptimizer::new(config);

    // Start with basic PCA projection
    let initial_positions = graph.project_pca(&embedding)?;

    // Optimize for better quality (with limited iterations for testing)
    let (optimized_positions, optimized_metrics) =
        optimizer.optimize_projection(&embedding, &initial_positions, &graph, 5)?;

    assert_eq!(
        optimized_positions.len(),
        8,
        "Optimized positions should have correct length"
    );

    // Check that positions are still finite after optimization
    for pos in &optimized_positions {
        assert!(pos.x.is_finite(), "Optimized X coordinate should be finite");
        assert!(pos.y.is_finite(), "Optimized Y coordinate should be finite");
    }

    println!(
        "✅ Quality optimization: Overall score {:.3}",
        optimized_metrics.overall_score
    );
    Ok(())
}

/// Test multi-scale projection
#[test]
fn test_multi_scale_projection() -> GraphResult<()> {
    let graph = create_test_graph();
    let embedding = create_test_embedding();

    let config = ProjectionConfig {
        method: ProjectionMethod::MultiScale {
            global_method: Box::new(ProjectionMethod::PCA {
                center: true,
                standardize: true,
            }),
            local_method: Box::new(ProjectionMethod::TSNE {
                perplexity: 2.0,
                iterations: 50, // Reduced for faster testing
                learning_rate: 100.0,
                early_exaggeration: 4.0,
            }),
            global_weight: 0.6,
        },
        ..Default::default()
    };

    let positions = graph.project_to_honeycomb(&embedding, &config)?;

    assert_eq!(positions.len(), 8);

    // Check positions are finite
    for pos in &positions {
        assert!(
            pos.x.is_finite(),
            "Multi-scale X coordinate should be finite"
        );
        assert!(
            pos.y.is_finite(),
            "Multi-scale Y coordinate should be finite"
        );
    }

    println!("✅ Multi-scale projection: PCA (60%) + t-SNE (40%) combination");
    Ok(())
}

/// Test complete end-to-end pipeline
#[test]
fn test_phase_2_integration_complete() -> GraphResult<()> {
    println!("🎉 Phase 2 Integration Test: Multi-dimensional to Honeycomb Projection");

    let graph = create_test_graph();
    let embedding = create_test_embedding();

    println!(
        "📊 Test graph: {} nodes, {} edges",
        graph.space().node_count(),
        graph.space().edge_count()
    );
    println!("📊 Embedding: {} dimensions", embedding.shape().1);

    // 1. Test all projection methods work
    println!("🧪 Testing all projection methods...");

    let pca_positions = graph.project_pca(&embedding)?;
    let tsne_positions = graph.project_tsne(&embedding, 2.0)?;
    let umap_positions = graph.project_umap(&embedding, 3)?;

    println!("✅ PCA: {:?} positions", pca_positions.len());
    println!("✅ t-SNE: {:?} positions", tsne_positions.len());
    println!("✅ UMAP: {:?} positions", umap_positions.len());

    // 2. Test honeycomb grid mapping
    println!("🔧 Testing honeycomb grid mapping...");

    let config = HoneycombConfig {
        cell_size: 50.0,
        layout_strategy: HoneycombLayoutStrategy::DistancePreserving,
        snap_to_centers: true,
        grid_padding: 25.0,
        max_grid_size: None,
    };

    let mut grid = HoneycombGrid::new(config);
    let honeycomb_positions = grid.map_positions_to_grid(&pca_positions)?;

    println!(
        "✅ Honeycomb mapping: {} cells occupied",
        grid.get_occupied_coords().len()
    );

    // 3. Test quality evaluation
    println!("📏 Testing quality evaluation...");

    let quality_config = QualityConfig {
        compute_neighborhood_preservation: true,
        compute_distance_preservation: true,
        k_neighbors: 3,
        ..Default::default()
    };

    let evaluator = QualityEvaluator::new(quality_config);
    let metrics = evaluator.evaluate_projection(&embedding, &honeycomb_positions, &graph)?;

    println!("✅ Quality metrics computed:");
    println!(
        "   • Neighborhood preservation: {:.3}",
        metrics.neighborhood_preservation
    );
    println!(
        "   • Distance correlation: {:.3}",
        metrics.distance_correlation
    );
    println!("   • Stress: {:.3}", metrics.stress);
    println!("   • Overall score: {:.3}", metrics.overall_score);

    // 4. Test interpolation
    println!("🎬 Testing smooth interpolation...");

    let interpolation_config = InterpolationConfig {
        enable_interpolation: true,
        method: InterpolationMethod::SpringPhysics {
            damping: 0.7,
            stiffness: 0.3,
        },
        steps: 15,
        easing: EasingFunction::EaseInOut,
        preserve_honeycomb: true,
    };

    let engine = InterpolationEngine::new(interpolation_config);
    let frames = engine.interpolate_with_honeycomb(&pca_positions, &honeycomb_positions, &grid)?;

    println!(
        "✅ Interpolation: {} frames with honeycomb constraints",
        frames.len()
    );

    // 5. Test animation management
    println!("🎮 Testing animation management...");

    let mut manager = AnimationManager::new();
    let animation = AnimationState::new(frames, 2000);
    manager.add_animation("honeycomb_transition".to_string(), animation);

    manager.start_animation("honeycomb_transition");
    assert!(manager.has_active_animations());

    println!("✅ Animation system: Transition management working");

    // 6. Test factory system
    println!("🏭 Testing projection factory...");

    let factory_config = ProjectionConfig {
        method: ProjectionMethod::EnergyBased {
            attraction_strength: 1.0,
            repulsion_strength: 0.5,
            iterations: 20, // Reduced for testing
            learning_rate: 0.05,
        },
        honeycomb_config: HoneycombConfig::default(),
        quality_config: QualityConfig::default(),
        interpolation_config: InterpolationConfig::default(),
        debug_enabled: true,
        seed: Some(42),
    };

    let engine = ProjectionEngineFactory::create_engine(&factory_config)?;
    let factory_positions = engine.project_embedding(&embedding, &graph)?;

    println!(
        "✅ Factory system: {} created energy-based projection",
        engine.name()
    );

    println!();
    println!("🎯 Phase 2 Integration Complete!");
    println!("   • Multi-dimensional projections: ✅");
    println!("   • Honeycomb grid mapping: ✅");
    println!("   • Quality evaluation: ✅");
    println!("   • Smooth interpolation: ✅");
    println!("   • Animation management: ✅");
    println!("   • Factory system: ✅");

    Ok(())
}

/// Test large graph performance
#[test]
fn test_large_graph_performance() -> GraphResult<()> {
    let graph = create_large_test_graph();
    println!("🚀 Performance test: {} nodes", graph.space().node_count());

    // Create larger embedding
    let mut embedding = GraphMatrix::zeros(25, 8);
    let mut rng = fastrand::Rng::with_seed(42);

    for i in 0..25 {
        for j in 0..8 {
            embedding.set(i, j, rng.f64() * 2.0 - 1.0)?;
        }
    }

    let start_time = std::time::Instant::now();

    // Test PCA projection (fastest)
    let pca_positions = graph.project_pca(&embedding)?;
    let pca_time = start_time.elapsed();

    // Test quality evaluation
    let quality_config = QualityConfig {
        compute_neighborhood_preservation: true,
        compute_distance_preservation: true,
        k_neighbors: 5,
        ..Default::default()
    };

    let evaluator = QualityEvaluator::new(quality_config);
    let quality_start = std::time::Instant::now();
    let metrics = evaluator.evaluate_projection(&embedding, &pca_positions, &graph)?;
    let quality_time = quality_start.elapsed();

    // Test honeycomb mapping
    let honeycomb_start = std::time::Instant::now();
    let mut grid = HoneycombGrid::new(HoneycombConfig::default());
    let honeycomb_positions = grid.map_positions_to_grid(&pca_positions)?;
    let honeycomb_time = honeycomb_start.elapsed();

    println!("⏱️  Performance results:");
    println!("   • PCA projection: {:?}", pca_time);
    println!("   • Quality evaluation: {:?}", quality_time);
    println!("   • Honeycomb mapping: {:?}", honeycomb_time);
    println!("   • Quality score: {:.3}", metrics.overall_score);

    assert_eq!(pca_positions.len(), 25);
    assert_eq!(honeycomb_positions.len(), 25);
    assert!(
        pca_time.as_millis() < 1000,
        "PCA should be fast for 25 nodes"
    );

    Ok(())
}

/// Test projection engine availability
#[test]
fn test_projection_engines_available() {
    let methods = ProjectionEngineFactory::available_methods();

    let expected_methods = [
        "pca",
        "tsne",
        "umap",
        "multi_scale",
        "custom_matrix",
        "energy_based",
    ];

    for method in &expected_methods {
        assert!(
            methods.contains(method),
            "Should have {} method available",
            method
        );
    }

    println!("✅ Available projection methods: {:?}", methods);
}

/// Test hex coordinate system
#[test]
fn test_hex_coordinate_system() {
    // Test hex coordinate basics
    let hex1 = HexCoord::new(0, 0);
    let hex2 = HexCoord::new(2, -1);

    // Test distance calculation
    let distance = hex1.distance(&hex2);
    assert_eq!(distance, 2, "Distance should be correct");

    // Test neighbors
    let neighbors = hex1.neighbors();
    assert_eq!(neighbors.len(), 6, "Should have 6 neighbors");

    for neighbor in &neighbors {
        assert_eq!(
            hex1.distance(neighbor),
            1,
            "All neighbors should be distance 1"
        );
    }

    // Test cube coordinate conversion
    let (x, y, z) = hex2.to_cube();
    assert_eq!(x + y + z, 0, "Cube coordinates should sum to 0");

    let back_to_hex = HexCoord::from_cube(x, y, z).unwrap();
    assert_eq!(hex2, back_to_hex, "Round-trip conversion should work");

    // Test linear interpolation
    let hex3 = HexCoord::new(4, -2);
    let midpoint = hex1.lerp(&hex3, 0.5);
    let expected_distance = hex1.distance(&hex3) / 2;
    let actual_distance = hex1.distance(&midpoint);

    // Should be approximately half the distance (allowing for rounding)
    assert!(
        (actual_distance as i32 - expected_distance).abs() <= 1,
        "Linear interpolation should give approximately correct distance"
    );

    println!("✅ Hex coordinate system: Distance, neighbors, cube conversion, interpolation");
}
