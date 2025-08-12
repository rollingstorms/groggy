#!/usr/bin/env python3
"""
Python API stress test - identical operations to Rust version
"""
import time
import random
import groggy as gr

def stress_test():
    print("🐍 PYTHON API STRESS TEST")
    print("==========================")
    
    test_sizes = [(10000, 5000), (50000, 25000), (100000, 50000)]
    
    for num_nodes, num_edges in test_sizes:
        print(f"\n📊 Testing {num_nodes} nodes, {num_edges} edges")
        
        # === GRAPH CREATION ===
        start = time.time()
        graph = gr.Graph()
        
        # Create nodes with bulk operations
        nodes = graph.add_nodes(num_nodes)
        
        # Set attributes with bulk operations
        departments = ["Engineering", "Marketing", "Sales", "HR", "Finance", "Operations"]
        
        # Prepare bulk attribute data
        attrs_dict = {
            "type": [(node_id, gr.AttrValue("user")) for node_id in nodes],
            "department": [(node_id, gr.AttrValue(departments[i % 6])) for i, node_id in enumerate(nodes)],
            "active": [(node_id, gr.AttrValue(i % 3 != 0)) for i, node_id in enumerate(nodes)],
            "age": [(node_id, gr.AttrValue(25 + (i % 40))) for i, node_id in enumerate(nodes)],
            "salary": [(node_id, gr.AttrValue(50000 + (i % 100000))) for i, node_id in enumerate(nodes)]
        }
        
        graph.set_node_attributes(attrs_dict)
        
        # Create edges with bulk operations
        edge_specs = []
        for _ in range(num_edges):
            from_idx = random.randint(0, num_nodes - 1)
            to_idx = random.randint(0, num_nodes - 1)
            if from_idx != to_idx:
                edge_specs.append((nodes[from_idx], nodes[to_idx]))
        
        if edge_specs:
            graph.add_edges(edge_specs)
        
        creation_time = time.time() - start
        print(f"   ✅ Graph creation: {creation_time:.3f}s")
        
        # === FILTERING TESTS ===
        print("   🔍 Filtering tests:")
        
        # Single attribute filter
        start = time.time()
        type_filter = gr.NodeFilter.attribute_equals("type", gr.AttrValue("user"))
        type_results = graph.filter_nodes(type_filter)
        type_time = time.time() - start
        print(f"      Single attribute: {type_time:.3f}s ({len(type_results)} results)")
        
        # Complex AND filter
        start = time.time()
        and_filter = gr.NodeFilter.and_filters([
            gr.NodeFilter.attribute_equals("type", gr.AttrValue("user")),
            gr.NodeFilter.attribute_equals("department", gr.AttrValue("Engineering")),
            gr.NodeFilter.attribute_equals("active", gr.AttrValue(True))
        ])
        and_results = graph.filter_nodes(and_filter)
        and_time = time.time() - start
        print(f"      Complex AND: {and_time:.3f}s ({len(and_results)} results)")
        
        # OR filter
        start = time.time()
        or_filter = gr.NodeFilter.or_filters([
            gr.NodeFilter.attribute_equals("department", gr.AttrValue("Engineering")),
            gr.NodeFilter.attribute_equals("department", gr.AttrValue("Marketing"))
        ])
        or_results = graph.filter_nodes(or_filter)
        or_time = time.time() - start
        print(f"      Complex OR: {or_time:.3f}s ({len(or_results)} results)")
        
        # === TRAVERSAL TESTS ===
        print("   🌐 Traversal tests:")
        
        # Connected components
        start = time.time()
        components = graph.find_connected_components()
        cc_time = time.time() - start
        print(f"      Connected components: {cc_time:.3f}s ({len(components)} components)")
        
        # BFS from random node  
        if nodes:
            start = time.time()
            start_node = random.choice(nodes)
            bfs_result = graph.traverse_bfs(start_node, max_depth=None, node_filter=None, edge_filter=None)
            bfs_time = time.time() - start
            print(f"      BFS traversal: {bfs_time:.3f}s ({len(bfs_result)} nodes)")
        
        # === AGGREGATION TESTS ===
        print("   📊 Aggregation tests:")
        
        # Basic statistics (aggregate age)
        start = time.time()
        try:
            avg_age = graph.aggregate_node_attribute("age", "average")
            count_nodes = graph.aggregate_node_attribute("age", "count")
            stats_time = time.time() - start
            print(f"      Basic statistics: {stats_time:.3f}s")
            print(f"         Avg age: {avg_age.value:.1f}, Count: {count_nodes.value}")
        except Exception as e:
            stats_time = time.time() - start
            print(f"      Basic statistics: {stats_time:.3f}s (error: {e})")
        
        # Performance summary
        if creation_time > 0:
            nodes_per_ms = num_nodes / (creation_time * 1000)
            print(f"   📈 Performance: {nodes_per_ms:.1f} nodes/ms creation")

if __name__ == "__main__":
    stress_test()