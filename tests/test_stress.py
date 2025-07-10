"""
Quick stress test for Groggy graph engine.
Tests performance with 1k nodes, 1k edges, and batch operations for demo purposes.
"""

import time
import random
import groggy as gr


def test_quick_graph_creation():
    """Test creating a graph with 1k nodes and 1k edges using batch operations."""
    print("🚀 Starting quick graph creation test...")
    
    g = gr.Graph()
    num_nodes = 10_000
    num_edges = 10_000
    
    # Create nodes efficiently using batch operation
    print(f"📝 Adding {num_nodes:,} nodes in batch...")
    start_time = time.time()
    
    nodes_data = []
    for i in range(num_nodes):
        nodes_data.append({
            'id': f"node_{i}",
            'value': random.randint(1, 1000),
            'category': random.choice(["A", "B", "C", "D"]),
            'active': random.choice([True, False]),
            'weight': random.uniform(0.1, 10.0)
        })
    
    g.add_nodes(nodes_data)
    node_creation_time = time.time() - start_time
    print(f"✅ Node creation completed in {node_creation_time:.3f} seconds")
    print(f"   Rate: {num_nodes / node_creation_time:.0f} nodes/second")
    
    # Verify nodes were added
    assert len(g.nodes) == num_nodes
    
    # Create edges efficiently using batch operation
    print(f"🔗 Adding {num_edges:,} edges in batch...")
    start_time = time.time()
    
    edges_data = []
    for i in range(num_edges):
        src_idx = random.randint(0, num_nodes - 1)
        dst_idx = random.randint(0, num_nodes - 1)
        
        if src_idx != dst_idx:
            edges_data.append({
                'source': f"node_{src_idx}",
                'target': f"node_{dst_idx}",
                'weight': random.uniform(0.1, 1.0),
                'relationship': random.choice(["connects", "links", "relates"])
            })
    
    g.add_edges(edges_data)
    edge_creation_time = time.time() - start_time
    print(f"✅ Edge creation completed in {edge_creation_time:.3f} seconds")
    print(f"   Rate: {len(edges_data) / edge_creation_time:.0f} edges/second")
    print(f"   Created {len(edges_data):,} edges")
    
    total_time = node_creation_time + edge_creation_time
    print(f"🎯 Total graph creation: {total_time:.3f} seconds")


def test_batch_updates():
    """Test batch updating node attributes."""
    print("\n🔄 Starting batch updates test...")
    
    g = gr.Graph()
    num_nodes = 500
    
    # Create nodes
    nodes_data = [
        {
            'id': f"user_{i}",
            'score': 100,
            'level': 1
        }
        for i in range(num_nodes)
    ]
    g.add_nodes(nodes_data)
    
    # Batch update nodes
    update_count = num_nodes // 2
    print(f"🔄 Batch updating {update_count:,} nodes...")
    start_time = time.time()
    
    updates = {}
    for i in range(update_count):
        updates[f"user_{i}"] = {
            'score': random.randint(200, 500),
            'level': random.randint(2, 10),
            'updated': True
        }
    
    g.update_nodes(updates)
    update_time = time.time() - start_time
    print(f"✅ Batch update completed in {update_time:.3f} seconds")
    if update_time > 0:
        print(f"   Rate: {update_count / update_time:.0f} updates/second")
    else:
        print(f"   Rate: Very fast (< 0.001 seconds)")
    
    # Verify updates
    sample_node = g.get_node("user_0")
    assert sample_node["score"] >= 200


def run_quick_stress_test():
    """Run quick stress tests."""
    print("=" * 50)
    print("🚀 GROGGY QUICK STRESS TEST")
    print("=" * 50)
    
    overall_start = time.time()
    
    try:
        test_quick_graph_creation()
        test_batch_updates()
        
        overall_time = time.time() - overall_start
        print("\n" + "=" * 50)
        print(f"✅ ALL TESTS COMPLETED!")
        print(f"🕒 Total time: {overall_time:.3f} seconds")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    run_quick_stress_test()
