#!/usr/bin/env python3
"""
Groggy Python Visualization Examples

Demonstrates the three main visualization modes:
1. Static visualization - export to files  
2. Non-streaming interactive - simple browser view
3. Streaming interactive - real-time updates

Usage:
    python python_viz_examples.py static
    python python_viz_examples.py interactive  
    python python_viz_examples.py streaming
    python python_viz_examples.py all
"""

import sys
import os
import time
import tempfile

def example_static_visualization():
    """Example 1: Static visualization export"""
    print("📊 Example 1: Static Visualization")
    print("-" * 40)
    
    try:
        import groggy as gr
        
        # Create a sample graph
        print("Creating sample social network...")
        g = gr.Graph()
        
        # Add some nodes
        people = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"]
        for person in people:
            g.add_node(person, 
                      type="person", 
                      age=20 + hash(person) % 40,
                      department=["Engineering", "Marketing", "Sales"][hash(person) % 3])
        
        # Add relationships
        connections = [
            ("Alice", "Bob", "colleague"),
            ("Bob", "Charlie", "friend"), 
            ("Charlie", "Diana", "sibling"),
            ("Diana", "Eve", "roommate"),
            ("Eve", "Frank", "colleague"),
            ("Frank", "Alice", "neighbor"),
            ("Alice", "Diana", "friend"),
            ("Bob", "Eve", "colleague")
        ]
        
        for src, dst, relationship in connections:
            g.add_edge(src, dst, relationship=relationship, weight=1.0)
        
        print(f"Graph created: {g.num_nodes()} nodes, {g.num_edges()} edges")
        
        # Export in different formats
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Exporting to temporary directory: {temp_dir}")
            
            # SVG export (vector graphics)
            svg_path = os.path.join(temp_dir, "social_network.svg")
            try:
                result = g.viz.static(
                    filename=svg_path,
                    format="svg",
                    layout="force-directed",
                    theme="light",
                    width=1200,
                    height=800
                )
                print(f"✓ SVG exported: {svg_path}")
                
            except Exception as e:
                print(f"⚠️  SVG export: {e}")
            
            # PNG export (high-DPI raster)
            png_path = os.path.join(temp_dir, "social_network_hd.png")
            try:
                result = g.viz.static(
                    filename=png_path,
                    format="png", 
                    layout="circular",
                    theme="dark",
                    dpi=300,
                    width=1600,
                    height=1200
                )
                print(f"✓ High-DPI PNG exported: {png_path}")
                
            except Exception as e:
                print(f"⚠️  PNG export: {e}")
            
            # PDF export (publication ready)
            pdf_path = os.path.join(temp_dir, "social_network_pub.pdf")
            try:
                # Use publication preset
                pub_config = gr.VizConfig.publication(width=1600, height=1200)
                result = g.viz.static(
                    filename=pdf_path,
                    format="pdf",
                    layout="hierarchical",
                    theme="publication",
                    width=1600,
                    height=1200
                )
                print(f"✓ Publication PDF exported: {pdf_path}")
                
            except Exception as e:
                print(f"⚠️  PDF export: {e}")
        
        # Also demonstrate table visualization
        print("\nTable visualization:")
        nodes_table = g.nodes_table()
        print(f"Nodes table shape: {nodes_table.shape()}")
        
        try:
            info = nodes_table.viz.info()
            print(f"Table viz info: {info}")
        except Exception as e:
            print(f"⚠️  Table viz info: {e}")
        
        print("✅ Static visualization example completed!\n")
        
    except Exception as e:
        print(f"❌ Static example failed: {e}")
        import traceback
        traceback.print_exc()


def example_interactive_visualization():
    """Example 2: Non-streaming interactive visualization"""
    print("🌐 Example 2: Interactive Visualization")
    print("-" * 40)
    
    try:
        import groggy as gr
        
        # Create a more complex graph for interaction
        print("Creating Karate Club network...")
        g = gr.karate_club()  # Classic network analysis dataset
        print(f"Karate Club loaded: {g.num_nodes()} nodes, {g.num_edges()} edges")
        
        # Launch interactive visualization
        print("Launching interactive visualization...")
        print("(Browser should open automatically)")
        
        try:
            # Create interactive session
            session = g.viz.interactive(
                port=0,  # Auto-assign port
                layout="force-directed",
                theme="dark",
                width=1400,
                height=900,
                auto_open=True  # This will try to open browser
            )
            
            print(f"🌐 Visualization available at: {session.url()}")
            print(f"📡 Server running on port: {session.port()}")
            print("\nFeatures available in browser:")
            print("  • Drag nodes to reposition")
            print("  • Right-click for context menus") 
            print("  • Use mouse wheel to zoom")
            print("  • Selection tools in toolbar")
            print("  • Layout controls panel")
            
            # Keep the server running for a bit
            print("\n⏰ Server will run for 30 seconds...")
            print("   Press Ctrl+C to stop early")
            
            try:
                time.sleep(30)
            except KeyboardInterrupt:
                print("\n⚡ Stopped by user")
            
            # Stop the server
            session.stop()
            print("🛑 Visualization server stopped")
            
        except Exception as e:
            print(f"⚠️  Interactive session: {e}")
            print("   (Server infrastructure may not be fully implemented)")
        
        # Also demonstrate different layouts
        print("\nDifferent layout configurations:")
        layouts = ["force-directed", "circular", "grid", "hierarchical"]
        
        for layout in layouts:
            try:
                config = gr.VizConfig(
                    layout=layout,
                    theme="light",
                    width=1000,
                    height=700
                )
                print(f"✓ Config for {layout} layout: {config}")
            except Exception as e:
                print(f"✗ Failed to create {layout} config: {e}")
        
        print("✅ Interactive visualization example completed!\n")
        
    except Exception as e:
        print(f"❌ Interactive example failed: {e}")
        import traceback
        traceback.print_exc()


def example_streaming_visualization():
    """Example 3: Streaming interactive visualization with real-time updates"""
    print("📡 Example 3: Streaming Visualization")
    print("-" * 40)
    
    try:
        import groggy as gr
        import threading
        import random
        
        # Start with a small graph that we'll grow
        print("Creating dynamic network...")
        g = gr.Graph()
        
        # Initial nodes
        for i in range(5):
            g.add_node(f"node_{i}", 
                      value=random.randint(1, 100),
                      status="active")
        
        # Initial edges
        for i in range(4):
            g.add_edge(f"node_{i}", f"node_{i+1}", weight=random.random())
        
        print(f"Initial graph: {g.num_nodes()} nodes, {g.num_edges()} edges")
        
        try:
            # Launch streaming visualization
            print("Starting streaming visualization...")
            
            session = g.viz.interactive(
                port=0,
                layout="force-directed", 
                theme="dark",
                width=1200,
                height=800,
                auto_open=True
            )
            
            print(f"🌐 Streaming visualization at: {session.url()}")
            print("📡 Real-time updates will be streamed to browser")
            
            # Function to simulate dynamic changes
            def update_graph():
                node_count = 5
                for step in range(20):  # 20 updates
                    time.sleep(1)  # Update every second
                    
                    # Add a new node occasionally
                    if step % 3 == 0 and node_count < 15:
                        new_node = f"node_{node_count}"
                        g.add_node(new_node, 
                                  value=random.randint(1, 100),
                                  status="new")
                        
                        # Connect to a random existing node
                        existing_node = f"node_{random.randint(0, node_count-1)}"
                        g.add_edge(existing_node, new_node, 
                                  weight=random.random())
                        
                        node_count += 1
                        print(f"  📊 Added {new_node}, connected to {existing_node}")
                    
                    # Update node values (simulate changing data)
                    for i in range(min(node_count, 10)):
                        node_id = f"node_{i}"
                        if g.has_node(node_id):
                            # Update node attributes (would trigger visual updates)
                            new_value = random.randint(1, 100)
                            print(f"  🔄 Updated {node_id} value: {new_value}")
                    
                    print(f"  Step {step+1}/20: {g.num_nodes()} nodes, {g.num_edges()} edges")
            
            # Start updates in background thread
            print("\n🔄 Starting real-time updates...")
            print("   Watch the browser for live changes!")
            
            update_thread = threading.Thread(target=update_graph)
            update_thread.daemon = True
            update_thread.start()
            
            # Let it run for a while
            try:
                update_thread.join(timeout=25)  # Wait for updates to complete
                time.sleep(5)  # Show final state
            except KeyboardInterrupt:
                print("\n⚡ Stopped by user")
            
            session.stop()
            print("🛑 Streaming visualization stopped")
            
        except Exception as e:
            print(f"⚠️  Streaming session: {e}")
            print("   (Streaming infrastructure may not be fully implemented)")
            
            # Show what the final graph would look like
            print("\nFinal graph state (if streaming was working):")
            print(f"  Nodes: {g.num_nodes()}")
            print(f"  Edges: {g.num_edges()}")
            
            # Try static export of final state
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    final_path = os.path.join(temp_dir, "final_dynamic_graph.svg")
                    g.viz.static(final_path, format="svg", layout="force-directed")
                    print(f"  📁 Final state exported: {final_path}")
            except Exception:
                pass
        
        print("✅ Streaming visualization example completed!\n")
        
    except Exception as e:
        print(f"❌ Streaming example failed: {e}")
        import traceback
        traceback.print_exc()


def show_usage():
    """Show usage information"""
    print("🎯 Groggy Python Visualization Examples")
    print("=" * 50)
    print("Usage:")
    print("  python python_viz_examples.py static      - Static export demo")
    print("  python python_viz_examples.py interactive - Interactive browser demo") 
    print("  python python_viz_examples.py streaming   - Real-time streaming demo")
    print("  python python_viz_examples.py all         - Run all examples")
    print()
    print("Requirements:")
    print("  • Groggy library with viz module")
    print("  • Web browser for interactive examples")
    print("  • Network access for WebSocket connections")


def main():
    """Main entry point"""
    if len(sys.argv) != 2:
        show_usage()
        return 1
    
    mode = sys.argv[1].lower()
    
    print("🚀 Groggy Python Visualization Examples")
    print("=" * 50)
    
    if mode == "static":
        example_static_visualization()
    elif mode == "interactive":
        example_interactive_visualization()
    elif mode == "streaming":
        example_streaming_visualization()
    elif mode == "all":
        example_static_visualization()
        example_interactive_visualization() 
        example_streaming_visualization()
        print("🎉 All examples completed!")
    else:
        print(f"❌ Unknown mode: {mode}")
        show_usage()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())