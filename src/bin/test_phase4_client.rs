//! Test Phase 4 Client UI Features
//!
//! This test demonstrates the advanced client features including:
//! - N-dimensional embedding controls
//! - Real-time filtering and search
//! - Attribute panels and node selection
//! - Performance monitoring

use groggy::api::graph::{Graph, GraphDataSource};
use groggy::types::AttrValue;
use groggy::viz::realtime::accessor::{DataSourceRealtimeAccessor, RealtimeVizAccessor};
use groggy::viz::realtime::server::start_realtime_background;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Phase 4 Client UI Test");
    println!("{}", "=".repeat(60));

    // Create test graph with sample data
    println!("📊 Creating test graph...");
    let mut graph = Graph::new();

    // Add some nodes with attributes
    for i in 0..10 {
        let node_id = graph.add_node();
        // Add some sample attributes
        graph.set_node_attr(
            node_id,
            "name".to_string(),
            AttrValue::Text(format!("Node_{}", i)),
        )?;
        graph.set_node_attr(
            node_id,
            "type".to_string(),
            AttrValue::Text(if i % 2 == 0 {
                "primary".to_string()
            } else {
                "secondary".to_string()
            }),
        )?;
        graph.set_node_attr(node_id, "value".to_string(), AttrValue::Int(i * 10))?;
    }

    // Add some edges
    for i in 0..8 {
        graph.add_edge(i, i + 1)?;
    }
    graph.add_edge(0, 5)?; // Create a cycle
    graph.add_edge(3, 7)?; // Cross connection

    println!("✅ Created graph with {} nodes and {} edges", 10, 11);

    // Create data source and accessor
    println!("🔗 Setting up realtime accessor...");
    let data_source = Arc::new(GraphDataSource::new(&graph));
    let accessor: Arc<dyn RealtimeVizAccessor> =
        Arc::new(DataSourceRealtimeAccessor::new(data_source));

    // Start realtime server in background with proper cancellation support
    println!("🌐 Starting realtime server...");
    let port = 8081; // Use different port to avoid conflicts
    let server_handle = start_realtime_background(port, accessor)?;
    let actual_port = server_handle.port;

    println!("✅ Realtime server started on port {}", actual_port);
    println!();
    println!("🎮 Phase 4 Features Available:");
    println!("   • N-Dimensional Embedding Controls (2D-10D)");
    println!("   • Real-time Method Selection (PCA, UMAP, t-SNE, Force-Directed)");
    println!("   • Advanced Filtering & Search");
    println!("   • Node Selection with Attribute Panels");
    println!("   • Performance Monitoring (FPS, Update Rate, Latency)");
    println!("   • Play/Pause Simulation Controls");
    println!();
    println!("🌐 Open your browser and navigate to:");
    println!("   http://127.0.0.1:{}/realtime/", actual_port);
    println!();
    println!("🧪 Test Instructions:");
    println!("1. 🧠 Try changing embedding methods and dimensions");
    println!("2. 🔍 Use the search box to find nodes (try 'Node_5' or 'primary')");
    println!("3. 🎯 Click on nodes to view their attributes");
    println!("4. 📊 Use degree range filters");
    println!("5. ⏸️  Try the pause/play simulation button");
    println!("6. 👀 Watch the real-time performance stats");
    println!();
    println!(
        "📡 WebSocket endpoint: ws://127.0.0.1:{}/realtime/ws",
        actual_port
    );
    println!();
    println!("Press Ctrl+C to stop the server...");

    println!("Press Ctrl+C to stop the server...");

    // Set up Ctrl+C handler
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();

    ctrlc::set_handler(move || {
        println!("\n🛑 Received Ctrl+C, shutting down server...");
        r.store(false, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl-C handler");

    // Keep running until Ctrl+C
    while running.load(Ordering::SeqCst) {
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    // Stop the server cleanly
    server_handle.stop();
    println!("✅ Server stopped cleanly");

    Ok(())
}
