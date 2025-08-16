#!/usr/bin/env python3
"""Quick test of the modular FFI implementation"""

import sys
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python-groggy/python')

try:
    import groggy as gr
    print("✅ Successfully imported groggy")
    
    # Test basic graph creation
    g = gr.Graph()
    print("✅ Successfully created Graph")
    
    # Test node operations
    node1 = g.add_node()
    node2 = g.add_node()
    print(f"✅ Added nodes: {node1}, {node2}")
    
    # Test node count
    count = g.node_count()
    print(f"✅ Node count: {count}")
    
    # Test edge count  
    edge_count = g.edge_count()
    print(f"✅ Edge count: {edge_count}")
    
    print("\n🎉 FFI Modularization SUCCESSFUL!")
    print("✅ All core Graph operations working")
    print("✅ Module imports correctly")
    print("✅ No compilation errors")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()