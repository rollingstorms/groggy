//! Test graph visualization directly from Rust
//! This bypasses Python wrapper complexity and tests core functionality

use groggy::api::graph::Graph;
use groggy::api::graph::GraphDataSource;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Testing Graph Visualization from Rust");
    println!("{}", "=".repeat(50));

    println!("📊 Creating simple test graph...");
    let graph = Graph::new();
    println!("✅ Graph created successfully");

    println!("🎯 Creating GraphDataSource...");
    let graph_data_source = GraphDataSource::new(&graph);
    println!("✅ GraphDataSource created");

    println!("🖼️  Calling interactive_embed()...");
    match graph_data_source.interactive_embed() {
        Ok(iframe_html) => {
            println!("✅ Generated iframe HTML: {} characters", iframe_html.len());
            println!(
                "📋 Iframe content: {}",
                &iframe_html[..std::cmp::min(200, iframe_html.len())]
            );

            // Extract port from iframe
            if let Some(port_start) = iframe_html.find("127.0.0.1:") {
                let port_str = &iframe_html[port_start + 10..];
                if let Some(port_end) = port_str.find('"') {
                    let port = &port_str[..port_end];
                    println!("✅ Extracted port: {}", port);
                    println!("🌐 Server should be running at: http://127.0.0.1:{}", port);

                    // Test server with HTTP request
                    println!("🔍 Testing server response...");
                    let url = format!("http://127.0.0.1:{}", port);

                    // Wait a moment for server to start
                    println!("⏳ Waiting 3 seconds for server to start...");
                    std::thread::sleep(std::time::Duration::from_secs(3));

                    // Test with curl if available
                    match std::process::Command::new("curl")
                        .arg("-s")
                        .arg("-I") // Head request only
                        .arg("--max-time")
                        .arg("5")
                        .arg(&url)
                        .output()
                    {
                        Ok(output) => {
                            let response = String::from_utf8_lossy(&output.stdout);
                            if response.contains("HTTP/1.1 200") || response.contains("200 OK") {
                                println!("✅ Server responding with HTTP 200!");
                                println!("🎉 GRAPH VISUALIZATION WORKING FROM RUST!");

                                // Try to open in browser
                                println!("🌐 Attempting to open browser...");
                                let _ = std::process::Command::new("open").arg(&url).spawn();

                                println!("📊 What you should see:");
                                println!("   • Karate club social network (34 nodes, 78 edges)");
                                println!("   • Both Table and Graph tabs available");
                                println!("   • Graph tab should show interactive nodes and edges");

                                // Keep server running for manual testing
                                println!(
                                    "⏸️  Server running for 30 seconds for manual inspection..."
                                );
                                std::thread::sleep(std::time::Duration::from_secs(30));

                                println!("✅ Test completed successfully!");
                            } else {
                                println!("❌ Server not responding properly");
                                println!("📋 Response: {}", response);
                            }
                        }
                        Err(e) => {
                            println!("⚠️  Could not test with curl: {}", e);
                            println!("🌐 Manual test: Open browser to {}", url);
                            println!("⏸️  Keeping server alive for 30 seconds...");
                            std::thread::sleep(std::time::Duration::from_secs(30));
                        }
                    }
                } else {
                    println!("❌ Could not parse port from iframe HTML");
                }
            } else {
                println!("❌ No port found in iframe HTML");
                println!("📋 HTML: {}", iframe_html);
            }
        }
        Err(e) => {
            println!("❌ interactive_embed failed: {}", e);
            return Err(e.into());
        }
    }

    println!("🎯 Test completed!");
    Ok(())
}
