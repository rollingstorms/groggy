# N-Dimensional Honeycomb Layout Planning Document

## Overview

Implementation of a sophisticated multi-dimensional graph layout system that projects high-dimensional node embeddings onto a 2D honeycomb grid with real-time topological transformations via user interaction.

## Core Concept

Unlike traditional static layouts, this system:
- Embeds nodes in high-dimensional space (2D to 20D+)
- Projects different 2D views from the n-dimensional embedding
- Snaps projected positions to hexagonal grid cells
- Enables topological exploration through interactive transformations

## Architecture Components

### 1. High-Dimensional Embeddings (`EmbeddingEngine`)

**Matrix Ecosystem Integration:**
- Embeddings are `GraphMatrix` instances, enabling all matrix operations
- Custom embeddings can be created via matrix transformations
- Leverage existing matrix backends (NumPy, BLAS, GPU) for performance
- Support matrix slicing, concatenation, and mathematical operations

**Embedding Types:**
- `EnergyND`: n-dimensional energy-based optimization with custom energy functions
- `Spectral`: Laplacian eigenvector embeddings (graph structure-based)
- `RandomND`: Random high-dimensional embeddings (for testing)
- `ForceDirectedND`: n-dimensional force-directed layout
- `CustomMatrix`: User-provided matrix embeddings
- `CompositeEmbedding`: Combination of multiple embedding methods

**Interface:**
```rust
trait EmbeddingEngine {
    fn compute_embedding(&self, graph: &Graph, dimensions: usize) -> GraphResult<GraphMatrix>;
    fn supports_incremental(&self) -> bool;
    fn supports_streaming(&self) -> bool;
    fn name(&str) -> &str;
}

// EmbeddingMatrix is now just a GraphMatrix with node metadata
type EmbeddingMatrix = GraphMatrix; // [node_count x dimensions]

impl EmbeddingMatrix {
    fn node_embedding(&self, node_id: &str) -> Option<&[f64]>;
    fn dimension_slice(&self, dims: &[usize]) -> GraphMatrix;
    fn apply_transformation(&self, transform: &TransformMatrix) -> GraphMatrix;
    fn compute_pairwise_distances(&self) -> GraphMatrix;
    fn project_to_subspace(&self, basis: &GraphMatrix) -> GraphMatrix;
}
```

### 2. Energy-Based Projection System (`ProjectionEngine`)

**Core Concept:**
- Projections are energy-driven matrix transformations
- Rotation sequences are programmable transformation chains
- Node dragging manipulates the projection's center/focus in embedding space
- Canvas interactions map to specific axis rotations via programmable schemes

**Projection as Matrix Operations:**
```rust
trait ProjectionEngine {
    fn project_to_2d(&self, embedding: &GraphMatrix) -> GraphMatrix; // [nodes x 2]
    fn get_projection_matrix(&self) -> GraphMatrix; // [dimensions x 2]
    fn update_projection(&mut self, update: ProjectionUpdate) -> GraphResult<()>;
    fn set_focus_node(&mut self, node_id: &str, embedding: &GraphMatrix) -> GraphResult<()>;
}

// All transformations are matrix operations
type ProjectionMatrix = GraphMatrix; // [dimensions x 2]
type TransformMatrix = GraphMatrix;  // [dimensions x dimensions]

struct EnergyProjection {
    projection_matrix: ProjectionMatrix,    // Current 2D projection basis
    focus_embedding: Option<Vec<f64>>,      // Center point in n-D space
    rotation_sequence: RotationSequence,   // Programmable rotation pattern
    interaction_scheme: InteractionScheme, // How user input maps to rotations
}
```

**Programmable Rotation Sequences:**
```rust
trait RotationSequence {
    fn get_transform_at_time(&self, t: f64) -> TransformMatrix;
    fn combine(sequences: &[Box<dyn RotationSequence>]) -> CompositeSequence;
}

// Built-in rotation templates
struct SpiralRotation { axes: (usize, usize), spiral_rate: f64 }
struct OscillatingRotation { axes: (usize, usize), amplitude: f64, frequency: f64 }
struct NDCycleRotation { axis_pairs: Vec<(usize, usize)>, cycle_time: f64 }

// Programmable custom sequences
struct CustomRotation {
    script: String, // Rust/Lua script for transformation
    compiled_fn: Box<dyn Fn(f64) -> TransformMatrix>,
}

// Example: "spiral through axes 0-2, then oscillate 3-5"
let sequence = RotationSequence::chain(vec![
    SpiralRotation { axes: (0, 2), spiral_rate: 1.0 }.for_duration(3.0),
    OscillatingRotation { axes: (3, 5), amplitude: 0.5, frequency: 2.0 }.for_duration(2.0),
]);
```

**Programmable Interaction Schemes:**
```rust
trait InteractionScheme {
    fn handle_node_drag(&self, node_id: &str, delta: Vec2, embedding: &GraphMatrix) -> ProjectionUpdate;
    fn handle_canvas_drag(&self, delta: Vec2) -> ProjectionUpdate;
    fn handle_scroll(&self, delta: f64) -> ProjectionUpdate;
}

struct ProjectionUpdate {
    rotation_delta: Option<TransformMatrix>,
    focus_shift: Option<Vec<f64>>,
    scale_factor: Option<f64>,
}

// Example interaction schemes:
struct NodeCentricScheme; // Dragging node brings its embedding to projection center
struct AxisRotationScheme { primary_axes: (usize, usize), secondary_axes: (usize, usize) };
struct EnergyGradientScheme; // Drag follows energy gradients in embedding space

impl InteractionScheme for NodeCentricScheme {
    fn handle_node_drag(&self, node_id: &str, delta: Vec2, embedding: &GraphMatrix) -> ProjectionUpdate {
        // Shift projection focus toward dragged node's embedding
        let node_embedding = embedding.node_embedding(node_id)?;
        let focus_shift = node_embedding * (delta.magnitude() * 0.1);
        ProjectionUpdate { focus_shift: Some(focus_shift), ..Default::default() }
    }

    fn handle_canvas_drag(&self, delta: Vec2) -> ProjectionUpdate {
        // Rotate primary axes based on drag direction
        let rotation_angle = delta.y * 0.01;
        let rotation = TransformMatrix::givens_rotation(0, 1, rotation_angle);
        ProjectionUpdate { rotation_delta: Some(rotation), ..Default::default() }
    }
}
```

### 3. Honeycomb Quantization (`HoneycombGrid`)

**Grid Management:**
```rust
struct HoneycombGrid {
    cell_size: f64,
    radius: f64,
    centers: Vec<Position>,
    assignment_cache: HashMap<u64, Vec<usize>>, // spatial hash -> cell indices
}

impl HoneycombGrid {
    fn assign_nodes_to_cells(&self, positions: &[Position]) -> Vec<usize>;
    fn get_optimal_cell_size(&self, node_count: usize) -> f64;
    fn generate_hex_centers_in_circle(&mut self, radius: f64);
}
```

### 4. Interactive Transformation System

**User Input Mapping:**
- Mouse drag → Rotation in selected 2-plane
- Mouse wheel → Zoom/scale projection
- Keyboard modifiers → Change transformation type
- Animation → Time-based parameter evolution

**Real-time Updates:**
```rust
struct InteractiveHoneycomb {
    embedding: EmbeddingMatrix,
    projection: Box<dyn ProjectionEngine>,
    grid: HoneycombGrid,
    animation_state: AnimationState,
}

impl InteractiveHoneycomb {
    fn handle_user_input(&mut self, input: UserInput) -> bool;
    fn update_animation(&mut self, delta_time: f64) -> bool;
    fn get_current_positions(&self) -> Vec<(String, Position)>;
}
```

## Implementation Phases

### Phase 1: Multi-Dimensional Embeddings
- [ ] Implement `SpectralEmbedding` engine
- [ ] Implement `RandomNDEmbedding` engine
- [ ] Extend existing `ForceDirectedLayout` to n-dimensions
- [ ] Add embedding dimension parameter to layout options

### Phase 2: Projection System
- [ ] Implement basic `OrthogonalProjection`
- [ ] Implement `RotationProjection` with Givens rotations
- [ ] Add time-based animation parameters
- [ ] Support for different transformation types

### Phase 3: Interactive Controls
- [ ] Map mouse/keyboard input to projection parameters
- [ ] Real-time position updates in web viewer
- [ ] Smooth animation transitions
- [ ] Performance optimization for real-time updates

### Phase 4: Advanced Transformations
- [ ] Implement all transformation types from Python script
- [ ] Add custom transformation support
- [ ] Multi-axis rotation sequences
- [ ] Energy-aware projections that preserve important structures

## Technical Challenges

### 1. Performance Optimization
- **Real-time constraint**: Must update at 30+ FPS
- **Memory efficiency**: Large embedding matrices (1000+ nodes × 20D)
- **Incremental updates**: Only recompute changed projections
- **Spatial indexing**: Fast hex cell assignment

### 2. Numerical Stability
- **Orthogonalization**: Prevent drift in rotation matrices
- **Normalization**: Keep embeddings bounded
- **Singularities**: Handle degenerate cases gracefully

### 3. User Experience
- **Intuitive controls**: Natural mapping from input to topological changes
- **Visual feedback**: Show which transformation is active
- **Parameter adjustment**: Real-time sliders for cell size, speed, etc.

## Web Interface Integration

### Layout Dropdown Enhancement
```html
<select id="layout-select">
    <option value="honeycomb-2d">Honeycomb (2D Energy)</option>
    <option value="honeycomb-spectral">Honeycomb (Spectral)</option>
    <option value="honeycomb-nd">Honeycomb (N-Dimensional)</option>
</select>

<div id="honeycomb-controls" style="display: none;">
    <label>Dimensions: <input type="range" min="2" max="20" value="10" id="dimensions-slider"></label>
    <label>Cell Size: <input type="range" min="10" max="100" value="40" id="cell-size-slider"></label>
    <label>Transform:
        <select id="transform-select">
            <option value="rotate">Rotate</option>
            <option value="spiral">Spiral</option>
            <option value="oscillate">Oscillate</option>
            <option value="nd_rotate">N-D Rotate</option>
        </select>
    </label>
    <label>Speed: <input type="range" min="0.1" max="3.0" step="0.1" value="1.0" id="speed-slider"></label>
</div>
```

### JavaScript Event Handling
```javascript
// Map mouse movement to projection parameters
canvas.addEventListener('mousemove', (e) => {
    if (mouseDown && currentLayout === 'honeycomb-nd') {
        const deltaX = e.clientX - lastMouseX;
        const deltaY = e.clientY - lastMouseY;

        sendLayoutUpdate({
            type: 'projection_update',
            rotation_delta: { x: deltaX * 0.01, y: deltaY * 0.01 },
            transform_type: currentTransform
        });
    }
});
```

## Matrix-Based Programming Interface

**Custom Embedding Creation:**
```rust
// Create custom embeddings via matrix transformations
let embedding = graph.nodes().spectral_embedding(10)?  // 10D spectral embedding
    .transform(|m| m.apply_pca(8))?                    // Reduce to 8D via PCA
    .add_noise(0.1)?                                   // Add slight noise
    .normalize_rows()?;                                // L2 normalize each node

// Combine multiple embeddings
let combined = GraphMatrix::concatenate_columns(vec![
    graph.spectral_embedding(5)?,
    graph.force_directed_embedding(3)?,
    graph.random_embedding(2)?,
])?; // Results in 10D embedding

// Apply custom energy functions
let energy_embedding = graph.energy_embedding()
    .with_repulsion_function(|d| 1.0 / (d + 0.1).powi(2))
    .with_attraction_function(|d| d.min(2.0))
    .optimize()?;
```

**Programmable Rotation Sequences:**
```rust
// Define rotation sequences via builder pattern or scripting
let rotation = RotationSequence::builder()
    .spiral(axes: (0, 1), rate: 1.0, duration: 3.0)
    .then_oscillate(axes: (2, 3), amplitude: 0.8, frequency: 2.0, duration: 2.0)
    .then_nd_cycle(axis_pairs: vec![(4,5), (6,7), (8,9)], cycle_time: 4.0)
    .loop_forever()
    .build();

// Or via scripting (Rust-like syntax)
let script = r#"
    spiral((0,1), rate=1.5, duration=2.0);
    parallel {
        oscillate((2,3), amp=0.5, freq=1.0);
        spiral((4,5), rate=0.8);
    };
    nd_rotate_cycle([(0,2), (1,3), (4,6)], time=3.0);
"#;
let rotation = RotationSequence::from_script(script)?;
```

**Interaction Scheme Programming:**
```rust
// Program how user interactions map to projections
let interaction = InteractionScheme::builder()
    .on_node_drag(|node_id, delta, embedding| {
        // Bring dragged node's embedding toward projection center
        let node_emb = embedding.node_embedding(node_id)?;
        let focus_shift = node_emb * delta.magnitude() * 0.1;
        ProjectionUpdate::focus_shift(focus_shift)
    })
    .on_canvas_drag(|delta| {
        // X-drag rotates axes 0-1, Y-drag rotates axes 2-3
        let rot_01 = TransformMatrix::givens_rotation(0, 1, delta.x * 0.01);
        let rot_23 = TransformMatrix::givens_rotation(2, 3, delta.y * 0.01);
        ProjectionUpdate::rotation(rot_01.multiply(&rot_23)?)
    })
    .on_scroll(|delta| {
        // Zoom by scaling projection matrix
        ProjectionUpdate::scale(1.0 + delta * 0.1)
    })
    .build();

// Or load interaction schemes from configuration
let interaction = InteractionScheme::load_from_config("node_centric_scheme.toml")?;
```

## Configuration Options

```rust
#[derive(Debug, Clone)]
pub struct HoneycombLayoutConfig {
    // Matrix-based embedding configuration
    pub embedding: EmbeddingConfig,
    pub custom_embedding_matrix: Option<GraphMatrix>,

    // Grid parameters
    pub cell_size: f64,
    pub auto_adjust_cell_size: bool,
    pub grid_radius: f64,

    // Programmable projection
    pub rotation_sequence: RotationSequence,
    pub interaction_scheme: InteractionScheme,
    pub projection_update_rate: f64, // Hz

    // Matrix computation backend
    pub matrix_backend: MatrixBackend, // NumPy, BLAS, GPU, etc.
    pub performance_profile: PerformanceProfile,
}

#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    pub method: EmbeddingMethod,
    pub dimensions: usize,
    pub energy_function: Option<EnergyFunction>,
    pub preprocessing: Vec<MatrixTransform>,
    pub postprocessing: Vec<MatrixTransform>,
}
```

## Success Metrics

1. **Performance**: Maintains 30+ FPS with 500+ nodes in 10D space
2. **Interactivity**: Sub-100ms response to user input
3. **Visual Quality**: Smooth animations, clear topological structure
4. **Usability**: Intuitive controls, discoverable transformations

## Future Extensions

1. **Graph-aware projections**: Preserve important structural features
2. **Collaborative exploration**: Multiple users exploring same n-D graph
3. **Semantic embeddings**: Use graph neural networks for embeddings
4. **VR/AR support**: 3D projections in immersive environments
5. **Time-series graphs**: Temporal dimension exploration

## Testing & Debugging Strategy

### 1. Matrix Operations Validation

**Embedding Quality Tests:**
```rust
#[cfg(test)]
mod embedding_tests {
    use super::*;

    #[test]
    fn test_spectral_embedding_preserves_graph_structure() {
        let graph = create_test_graph_with_known_communities();
        let embedding = graph.spectral_embedding(10)?;

        // Test: nodes in same community should be closer in embedding space
        let community_distances = compute_intra_community_distances(&embedding, &graph);
        let inter_community_distances = compute_inter_community_distances(&embedding, &graph);

        assert!(community_distances.mean() < inter_community_distances.mean());

        // Test: embedding dimensions should have decreasing variance (spectral property)
        let variances = embedding.column_variances();
        for i in 1..variances.len() {
            assert!(variances[i-1] >= variances[i], "Spectral embedding should have decreasing variance");
        }
    }

    #[test]
    fn test_energy_embedding_convergence() {
        let graph = create_test_graph();
        let mut embedding = graph.energy_embedding()
            .with_debug_tracking(true)
            .optimize()?;

        // Access debug data
        let energy_history = embedding.debug_data().energy_history();
        let gradient_norms = embedding.debug_data().gradient_norms();

        // Test: energy should decrease over time
        assert!(energy_history.is_decreasing());

        // Test: gradients should approach zero
        assert!(gradient_norms.last().unwrap() < 0.01);

        // Export debug data for visualization
        embedding.debug_data().export_to_file("energy_convergence.json")?;
    }
}
```

**Matrix Transformation Tests:**
```rust
#[test]
fn test_projection_matrix_orthogonality() {
    let projection = ProjectionMatrix::random_orthogonal(10, 2);

    // Test: columns should be orthonormal
    let gram_matrix = projection.transpose().multiply(&projection)?;
    let identity = GraphMatrix::identity(2);

    assert_matrices_approximately_equal(&gram_matrix, &identity, 1e-10);
}

#[test]
fn test_givens_rotation_properties() {
    let rotation = TransformMatrix::givens_rotation(0, 1, std::f64::consts::PI / 4.0);

    // Test: should preserve vector norms
    let test_vector = vec![1.0, 2.0, 3.0, 4.0];
    let rotated = rotation.multiply_vector(&test_vector)?;

    assert_eq!(
        vector_norm(&test_vector),
        vector_norm(&rotated),
        "Rotation should preserve vector norms"
    );

    // Test: rotation matrix should be orthogonal
    let should_be_identity = rotation.transpose().multiply(&rotation)?;
    assert_matrix_is_identity(&should_be_identity, 1e-12);
}
```

### 2. Interactive Behavior Validation

**Projection Response Tests:**
```rust
#[test]
fn test_node_drag_brings_node_to_center() {
    let graph = create_test_graph();
    let embedding = graph.spectral_embedding(10)?;
    let mut projection = EnergyProjection::new(embedding.clone());

    // Drag a specific node
    let target_node = "node_5";
    let drag_delta = Vec2::new(100.0, 50.0); // Large drag toward center

    let update = projection.interaction_scheme().handle_node_drag(
        target_node, drag_delta, &embedding
    )?;
    projection.update_projection(update)?;

    // Test: target node should move toward projection center
    let initial_pos = projection.project_node(target_node)?;
    let final_pos = projection.project_node(target_node)?;

    assert!(
        final_pos.distance_to_origin() < initial_pos.distance_to_origin(),
        "Dragged node should move closer to center"
    );

    // Export visualization data
    projection.export_debug_state(&format!("node_drag_test_{}.json", target_node))?;
}

#[test]
fn test_canvas_drag_rotates_specified_axes() {
    let mut projection = EnergyProjection::with_interaction_scheme(
        AxisRotationScheme { primary_axes: (0, 1), secondary_axes: (2, 3) }
    );

    let initial_projection_matrix = projection.get_projection_matrix().clone();

    // Simulate horizontal drag (should rotate primary axes)
    let drag_delta = Vec2::new(50.0, 0.0);
    let update = projection.interaction_scheme().handle_canvas_drag(drag_delta)?;
    projection.update_projection(update)?;

    // Test: projection matrix should have changed in expected way
    let final_projection_matrix = projection.get_projection_matrix();
    let change = final_projection_matrix.subtract(&initial_projection_matrix)?;

    // The change should primarily affect dimensions 0 and 1
    assert!(change.row(0).norm() > 0.01, "Dimension 0 should change");
    assert!(change.row(1).norm() > 0.01, "Dimension 1 should change");
    assert!(change.row(2).norm() < 0.001, "Dimension 2 should not change much");
}
```

### 3. Performance & Numerical Stability

**Real-time Performance Tests:**
```rust
#[test]
fn test_real_time_update_performance() {
    let large_graph = create_large_test_graph(1000); // 1000 nodes
    let embedding = large_graph.spectral_embedding(20)?; // 20D
    let mut projection = EnergyProjection::new(embedding);

    let start_time = std::time::Instant::now();

    // Simulate 60 FPS for 1 second = 60 updates
    for frame in 0..60 {
        let t = frame as f64 / 60.0;
        let rotation_update = projection.rotation_sequence().get_transform_at_time(t);
        projection.update_projection(ProjectionUpdate::rotation(rotation_update))?;

        let positions = projection.get_current_positions();
        assert_eq!(positions.len(), 1000);
    }

    let elapsed = start_time.elapsed();
    assert!(elapsed.as_millis() < 1000, "Should maintain 60 FPS");

    println!("Performance: {} ms per frame", elapsed.as_millis() / 60);
}

#[test]
fn test_numerical_stability_over_time() {
    let mut projection = EnergyProjection::new(create_test_embedding());
    let initial_matrix = projection.get_projection_matrix().clone();

    // Apply many small rotations
    for _ in 0..10000 {
        let small_rotation = TransformMatrix::givens_rotation(0, 1, 0.001); // 0.001 radians
        projection.update_projection(ProjectionUpdate::rotation(small_rotation))?;
    }

    let final_matrix = projection.get_projection_matrix();

    // Test: projection matrix should still be orthogonal
    let gram = final_matrix.transpose().multiply(&final_matrix)?;
    let identity = GraphMatrix::identity(2);

    assert_matrices_approximately_equal(&gram, &identity, 1e-8,
        "Projection matrix should remain orthogonal after many updates");

    // Test: determinant should still be ±1
    let det = final_matrix.determinant()?;
    assert!((det.abs() - 1.0).abs() < 1e-8, "Determinant should remain ±1");
}
```

### 4. Data Export & Visualization Tools

**Debug Data Collection:**
```rust
pub struct DebugDataCollector {
    embedding_snapshots: Vec<(f64, GraphMatrix)>, // (time, embedding)
    projection_snapshots: Vec<(f64, ProjectionMatrix)>, // (time, projection)
    interaction_events: Vec<InteractionEvent>,
    performance_metrics: PerformanceMetrics,
    energy_history: Vec<f64>,
}

impl DebugDataCollector {
    pub fn export_full_session(&self, path: &str) -> Result<(), Box<dyn Error>> {
        let data = serde_json::json!({
            "embeddings": self.embedding_snapshots,
            "projections": self.projection_snapshots,
            "interactions": self.interaction_events,
            "performance": self.performance_metrics,
            "energy_history": self.energy_history,
            "metadata": {
                "timestamp": chrono::Utc::now(),
                "graph_info": self.get_graph_metadata(),
                "system_info": self.get_system_info()
            }
        });

        std::fs::write(path, serde_json::to_string_pretty(&data)?)?;
        Ok(())
    }

    pub fn create_visualization_html(&self, path: &str) -> Result<(), Box<dyn Error>> {
        let html = format!(r#"
<!DOCTYPE html>
<html>
<head>
    <title>Honeycomb Layout Debug Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <div id="embedding-evolution"></div>
    <div id="energy-plot"></div>
    <div id="projection-trace"></div>
    <script>
        const debugData = {};
        // Generate interactive plots showing:
        // 1. How embedding evolves over time
        // 2. Energy convergence plots
        // 3. Projection matrix evolution
        // 4. Performance metrics timeline
    </script>
</body>
</html>
        "#, serde_json::to_string(&self)?);

        std::fs::write(path, html)?;
        Ok(())
    }
}
```

**Live Debug Interface:**
```rust
pub struct LiveDebugInterface {
    web_server: Option<tokio::task::JoinHandle<()>>,
    debug_collector: Arc<Mutex<DebugDataCollector>>,
}

impl LiveDebugInterface {
    pub async fn start_debug_server(&mut self, port: u16) -> Result<(), Box<dyn Error>> {
        let collector = self.debug_collector.clone();

        let handle = tokio::spawn(async move {
            // Start web server that serves real-time debug data
            // Endpoints:
            // GET /current_state - current embedding & projection state
            // GET /energy_history - energy convergence data
            // GET /performance - real-time performance metrics
            // WebSocket /live_updates - streaming updates
        });

        self.web_server = Some(handle);
        println!("Debug server started at http://localhost:{}", port);
        Ok(())
    }
}
```

### 5. Integration Tests with Web Interface

**End-to-End Workflow Tests:**
```rust
#[tokio::test]
async fn test_full_honeycomb_workflow() {
    // 1. Create graph and embedding
    let graph = create_karate_club_graph();
    let layout_config = HoneycombLayoutConfig {
        embedding: EmbeddingConfig {
            method: EmbeddingMethod::Spectral,
            dimensions: 8,
            ..Default::default()
        },
        rotation_sequence: RotationSequence::builder()
            .spiral(axes: (0, 1), rate: 1.0, duration: 2.0)
            .build(),
        ..Default::default()
    };

    // 2. Start visualization server
    let server = start_visualization_server(&layout_config).await?;

    // 3. Simulate user interactions
    let interactions = vec![
        UserInteraction::NodeDrag { node_id: "node_1", delta: Vec2::new(50.0, 30.0) },
        UserInteraction::CanvasDrag { delta: Vec2::new(-20.0, 40.0) },
        UserInteraction::Scroll { delta: 0.1 },
    ];

    for interaction in interactions {
        server.handle_interaction(interaction).await?;

        // Verify server state after each interaction
        let state = server.get_current_state().await?;
        assert!(state.positions.len() == graph.node_count());
        assert!(state.performance_metrics.frame_time < 33.0); // < 33ms = 30+ FPS
    }

    // 4. Export final debug data
    server.export_debug_session("end_to_end_test.json").await?;
}
```

## Implementation Timeline

- **Week 1**: Phase 1 - Multi-dimensional embeddings + comprehensive testing framework
- **Week 2**: Phase 2 - Projection system + matrix operation validation
- **Week 3**: Phase 3 - Interactive controls + behavior validation tests
- **Week 4**: Phase 4 - Performance optimization + debug tools + end-to-end validation

## Open Questions

1. Should we support custom embedding algorithms via plugins?
2. How to handle very large graphs (10K+ nodes) efficiently?
3. What's the optimal default dimension count for different graph types?
4. Should transformations be graph-aware (preserve clusters, paths, etc.)?
5. How to visualize which part of n-D space is currently visible?

---

*This document will be updated as implementation progresses and new requirements emerge.*