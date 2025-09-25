//! Visual test for graph visualization - opens browser for manual inspection

use groggy::api::graph::Graph;
use groggy::api::graph::GraphDataSource;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 VISUAL TEST - Graph Visualization from Rust");
    println!("{}", "=".repeat(60));

    println!("📊 Creating empty test graph...");
    let graph = Graph::new();
    println!("✅ Graph created successfully");

    println!("🎯 Creating GraphDataSource...");
    let graph_data_source = GraphDataSource::new(&graph);
    println!("✅ GraphDataSource created");

    println!("🖼️  Calling interactive_embed() to start visualization server...");
    match graph_data_source.interactive_embed() {
        Ok(iframe_html) => {
            println!("✅ Generated iframe HTML: {} characters", iframe_html.len());

            // Extract port from iframe
            if let Some(port_start) = iframe_html.find("127.0.0.1:") {
                let port_str = &iframe_html[port_start + 10..];
                if let Some(port_end) = port_str.find('"') {
                    let port = &port_str[..port_end];
                    let url = format!("http://127.0.0.1:{}/", port);

                    println!("✅ Extracted port: {}", port);
                    println!("🌐 Server running at: {}", url);

                    // Wait for server to fully start
                    println!("⏳ Waiting 3 seconds for server to start...");
                    std::thread::sleep(std::time::Duration::from_secs(3));

                    // Test server quickly
                    match std::process::Command::new("curl")
                        .arg("-s")
                        .arg("-o")
                        .arg("/dev/null")
                        .arg("-w")
                        .arg("%{http_code}")
                        .arg("--max-time")
                        .arg("3")
                        .arg(&url)
                        .output()
                    {
                        Ok(output) => {
                            let status_code = String::from_utf8_lossy(&output.stdout);
                            if status_code.trim() == "200" {
                                println!("✅ Server confirmed responding with HTTP 200");
                            } else {
                                println!("⚠️  Server status: {}", status_code.trim());
                            }
                        }
                        Err(_) => {
                            println!("⚠️  Could not test with curl (this is OK)");
                        }
                    }

                    // Open browser
                    println!("\n🌐 Opening browser for VISUAL INSPECTION...");
                    let open_result = std::process::Command::new("open").arg(&url).spawn();

                    match open_result {
                        Ok(_) => {
                            println!("✅ Browser should be opening...");
                            println!("🎯 If browser didn't open, manually go to: {}", url);
                        }
                        Err(e) => {
                            println!("⚠️  Could not auto-open browser: {}", e);
                            println!("🌐 Please manually open: {}", url);
                        }
                    }

                    println!("\n📊 WHAT TO LOOK FOR IN THE BROWSER:");
                    println!("{}", "-".repeat(50));
                    println!("🌐 URL: {}", url);
                    println!("📋 Two tabs: 'Table' and 'Graph'");
                    println!("📊 Table tab: Should show column headers and empty data");
                    println!("🕸️  Graph tab: Should show canvas with controls");
                    println!("🎮 Controls: 'Reset View', 'Change Layout' buttons");
                    println!("📱 Interactive: Try clicking, dragging, scrolling");
                    println!("🎨 Styling: Modern UI with rounded borders");

                    println!("\n🔍 EXPECTED RESULTS:");
                    println!("✅ Page loads successfully");
                    println!("✅ Both tabs are clickable and functional");
                    println!("✅ Table shows structure (headers: node_id, index, name, etc.)");
                    println!("✅ Graph shows canvas element (even if empty)");
                    println!("✅ No JavaScript errors in browser console");
                    println!("✅ WebSocket connection established");

                    println!("\n⏰ SERVER RUNNING FOR 3 MINUTES...");
                    println!("🔴 Press Ctrl+C to stop early");
                    println!("🌐 URL: {}", url);

                    // Run for 3 minutes with periodic reminders
                    for minute in 1..=3 {
                        for second in 1..=60 {
                            if second == 1 {
                                println!("   📍 Minute {} - URL: {}", minute, url);
                            }
                            std::thread::sleep(std::time::Duration::from_secs(1));
                        }
                    }

                    println!("\n⏰ 3 minutes completed!");
                    println!("✅ Visual test finished!");
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

    println!("\n{}", "=".repeat(60));
    println!("🎯 VISUAL TEST COMPLETED!");
    println!("📝 Please report what you saw in the browser.");
    Ok(())
}
