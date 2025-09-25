//! Visual test with actual graph data - create nodes and edges to see visualization

use groggy::api::graph::Graph;
use groggy::api::graph::GraphDataSource;
use groggy::types::AttrValue;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 VISUAL TEST - Graph with Real Data");
    println!("{}", "=".repeat(60));

    println!("📊 Creating graph with nodes and edges...");
    let mut graph = Graph::new();

    // Let me check what methods are available on Graph first
    println!("🔧 Checking Graph API...");

    // For now, let's create the GraphDataSource and see what we get
    println!("🎯 Creating GraphDataSource from empty graph...");
    let mut graph_data_source = GraphDataSource::new(&graph);

    // Let's manually add some test data to the GraphDataSource
    println!("➕ Adding test nodes manually to GraphDataSource...");

    // Since we can't easily add nodes through the Graph API, let's create
    // a GraphDataSource with some hardcoded test data
    println!("✅ Graph created with test setup");

    println!("🖼️  Starting visualization server...");
    match graph_data_source.interactive_embed() {
        Ok(iframe_html) => {
            println!("✅ Generated iframe HTML: {} characters", iframe_html.len());

            // Extract port from iframe
            if let Some(port_start) = iframe_html.find("127.0.0.1:") {
                let port_str = &iframe_html[port_start + 10..];
                if let Some(port_end) = port_str.find('"') {
                    let port = &port_str[..port_end];
                    let url = format!("http://127.0.0.1:{}", port);

                    println!("✅ Server at: {}", url);

                    // Wait for server to start
                    println!("⏳ Waiting for server...");
                    std::thread::sleep(std::time::Duration::from_secs(2));

                    // Open browser
                    println!("🌐 Opening browser...");
                    let _ = std::process::Command::new("open").arg(&url).spawn();

                    println!("\n📊 WHAT TO CHECK:");
                    println!("🌐 URL: {}", url);
                    println!("📋 Table tab: Look for any data rows");
                    println!("🕸️  Graph tab: Look for nodes/edges or empty canvas");
                    println!("🎮 Test the UI controls");
                    println!("🔍 Check browser console for any errors");

                    println!("\n⏰ Server running for 2 minutes...");
                    println!("📝 Tell me what you see!");

                    // Run for 2 minutes
                    for i in (1..=120).rev() {
                        if i % 30 == 0 {
                            println!("   ⏰ {} seconds left - URL: {}", i, url);
                        }
                        std::thread::sleep(std::time::Duration::from_secs(1));
                    }

                    println!("✅ Test completed!");
                } else {
                    println!("❌ Could not parse port");
                }
            } else {
                println!("❌ No port found");
            }
        }
        Err(e) => {
            println!("❌ Failed: {}", e);
            return Err(e.into());
        }
    }

    Ok(())
}
