#!/usr/bin/env python3
"""
Hierarchical Graph System Example
=================================

This example demonstrates groggy's powerful hierarchical graph capabilities,
including meta-node creation, edge aggregation control, and multi-level 
graph analysis.

Features demonstrated:
- Creating complex organizational hierarchies
- Configurable edge aggregation strategies
- Meta-node attribute aggregation
- Multi-level graph analysis and navigation
- Advanced edge filtering and control

Use case: Corporate organizational analysis with departments, teams, 
and cross-functional collaboration patterns.
"""

import groggy

def create_corporate_graph():
    """Create a sample corporate organizational graph"""
    print("🏢 Creating Corporate Organizational Graph...")
    
    g = groggy.Graph()
    
    # === EMPLOYEES ===
    # Engineering Department
    alice = g.add_node(name="Alice Chen", dept="Engineering", role="Senior Engineer", 
                      salary=95000, experience=5, location="SF")
    bob = g.add_node(name="Bob Smith", dept="Engineering", role="Tech Lead", 
                    salary=110000, experience=7, location="SF")
    carol = g.add_node(name="Carol Davis", dept="Engineering", role="Engineer", 
                      salary=80000, experience=3, location="Remote")
    
    # Marketing Department  
    david = g.add_node(name="David Wilson", dept="Marketing", role="Marketing Manager",
                      salary=85000, experience=6, location="NYC")
    eve = g.add_node(name="Eve Johnson", dept="Marketing", role="Content Specialist",
                    salary=65000, experience=2, location="NYC")
    
    # Sales Department
    frank = g.add_node(name="Frank Brown", dept="Sales", role="Sales Director",
                      salary=120000, experience=8, location="LA")
    grace = g.add_node(name="Grace Miller", dept="Sales", role="Sales Rep",
                      salary=70000, experience=4, location="LA")
    
    # === COLLABORATION EDGES ===
    # Engineering collaborations
    g.add_edge(alice, bob, type="mentorship", frequency=5, strength=0.9, project="Platform")
    g.add_edge(bob, carol, type="code_review", frequency=8, strength=0.8, project="API")
    g.add_edge(alice, carol, type="pair_programming", frequency=3, strength=0.7, project="Frontend")
    
    # Cross-department collaborations
    g.add_edge(bob, david, type="product_planning", frequency=2, strength=0.6, project="Launch")
    g.add_edge(alice, david, type="technical_consultation", frequency=1, strength=0.4, project="Integration")
    g.add_edge(david, frank, type="campaign_coordination", frequency=4, strength=0.8, project="Q4_Campaign")
    g.add_edge(eve, grace, type="content_collaboration", frequency=2, strength=0.5, project="Demo_Materials")
    
    # Sales team collaboration
    g.add_edge(frank, grace, type="training", frequency=3, strength=0.7, project="Sales_Methodology")
    
    print(f"✅ Created graph with {len(g.node_ids)} employees and {len(g.edge_ids)} collaborations")
    return g, {
        'engineering': [alice, bob, carol],
        'marketing': [david, eve], 
        'sales': [frank, grace]
    }

def demonstrate_basic_hierarchical_operations(g, employees):
    """Demonstrate basic meta-node creation and attribute aggregation"""
    print("\n🔧 Basic Hierarchical Operations")
    print("=" * 50)
    
    # Create department meta-nodes with simple aggregation
    engineering_team = g.nodes[employees['engineering']]
    marketing_team = g.nodes[employees['marketing']]
    sales_team = g.nodes[employees['sales']]
    
    print("\n📊 Creating Department Meta-Nodes...")
    
    # Engineering department aggregation
    eng_meta = engineering_team.add_to_graph({
        "total_salary": ("sum", "salary"),
        "avg_experience": ("mean", "experience"),
        "headcount": ("count", None),
        "dept_name": ("first", "dept")  # Take first dept name
    })
    
    print(f"✅ Engineering Meta-Node: {eng_meta.node_id}")
    print(f"   Total Salary: ${g.get_node_attr(eng_meta.node_id, 'total_salary'):,}")
    print(f"   Avg Experience: {g.get_node_attr(eng_meta.node_id, 'avg_experience'):.1f} years")
    print(f"   Headcount: {g.get_node_attr(eng_meta.node_id, 'headcount')} people")
    
    # Marketing department aggregation
    mkt_meta = marketing_team.add_to_graph({
        "total_salary": ("sum", "salary"),
        "avg_experience": ("mean", "experience"), 
        "headcount": ("count", None),
        "dept_name": ("first", "dept")
    })
    
    print(f"✅ Marketing Meta-Node: {mkt_meta.node_id}")
    print(f"   Total Salary: ${g.get_node_attr(mkt_meta.node_id, 'total_salary'):,}")
    print(f"   Avg Experience: {g.get_node_attr(mkt_meta.node_id, 'avg_experience'):.1f} years")
    print(f"   Headcount: {g.get_node_attr(mkt_meta.node_id, 'headcount')} people")
    
    # Sales department aggregation  
    sales_meta = sales_team.add_to_graph({
        "total_salary": ("sum", "salary"),
        "avg_experience": ("mean", "experience"),
        "headcount": ("count", None), 
        "dept_name": ("first", "dept")
    })
    
    print(f"✅ Sales Meta-Node: {sales_meta.node_id}")
    print(f"   Total Salary: ${g.get_node_attr(sales_meta.node_id, 'total_salary'):,}")
    print(f"   Avg Experience: {g.get_node_attr(sales_meta.node_id, 'avg_experience'):.1f} years")
    print(f"   Headcount: {g.get_node_attr(sales_meta.node_id, 'headcount')} people")
    
    return {'eng': eng_meta, 'mkt': mkt_meta, 'sales': sales_meta}

def demonstrate_advanced_edge_aggregation(g, employees):
    """Demonstrate advanced edge aggregation control"""
    print("\n🔗 Advanced Edge Aggregation Control")
    print("=" * 50)
    
    # Create a cross-functional team with specific edge aggregation requirements
    cross_functional = g.nodes[employees['engineering'][:2] + employees['marketing'][:1]]
    
    print("\n🎯 Cross-Functional Team Analysis")
    print("Team members: Alice (Eng), Bob (Eng), David (Mkt)")
    
    # Note: This demonstrates the API design, but Python FFI linking may have limitations
    try:
        # Attempt to use enhanced edge configuration
        # This would be the ideal API once Python bindings are fully linked:
        """
        cross_meta = cross_functional.add_to_graph_with_edge_config(
            agg_spec={
                "avg_salary": ("mean", "salary"),
                "team_size": ("count", None),
                "skill_diversity": ("concat_unique", "role")
            },
            edge_config={
                "edge_to_external": "aggregate",  # Combine parallel edges
                "edge_aggregation": {
                    "frequency": "sum",      # Sum collaboration frequency
                    "strength": "mean",      # Average relationship strength
                    "project": "concat_unique"  # List unique projects
                },
                "min_edge_count": 1,        # Include all external connections
                "mark_entity_type": True    # Mark as meta-edges
            }
        )
        """
        
        # For now, use basic aggregation (enhanced version available in Rust)
        cross_meta = cross_functional.add_to_graph({
            "avg_salary": ("mean", "salary"),
            "team_size": ("count", None),
            "primary_dept": ("first", "dept")
        })
        
        print(f"✅ Cross-Functional Meta-Node: {cross_meta.node_id}")
        print(f"   Average Salary: ${g.get_node_attr(cross_meta.node_id, 'avg_salary'):,.0f}")
        print(f"   Team Size: {g.get_node_attr(cross_meta.node_id, 'team_size')} members")
        
        # Analyze meta-edges created
        meta_edges = []
        for edge_id in g.edge_ids:
            if cross_meta.node_id in g.edge_endpoints(edge_id):
                meta_edges.append(edge_id)
        
        print(f"   Meta-Edges: {len(meta_edges)} connections to other parts of org")
        
        # Show edge aggregation in action
        if meta_edges:
            example_edge = meta_edges[0]
            frequency = g.get_edge_attr(example_edge, 'frequency')
            edge_count = g.get_edge_attr(example_edge, 'edge_count')
            print(f"   Example Meta-Edge: frequency={frequency}, edge_count={edge_count}")
            
    except Exception as e:
        print(f"⚠️  Enhanced API demonstration: {e}")
        print("   (Full Python bindings may need additional linking)")

def demonstrate_hierarchical_analysis(g):
    """Demonstrate multi-level graph analysis capabilities"""
    print("\n📈 Hierarchical Graph Analysis")
    print("=" * 50)
    
    # Analyze the hierarchical structure
    print("\n🔍 Entity Type Analysis")
    
    # Count different types of nodes
    base_nodes = []
    meta_nodes = []
    for node_id in g.node_ids:
        entity_type = g.get_node_attr(node_id, 'entity_type')
        if entity_type == 'base':
            base_nodes.append(node_id)
        elif entity_type == 'meta':
            meta_nodes.append(node_id)
    
    print(f"Base Nodes (Employees): {len(base_nodes)}")
    print(f"Meta Nodes (Departments): {len(meta_nodes)}")
    
    # Count different types of edges
    base_edges = []
    meta_edges = []
    for edge_id in g.edge_ids:
        entity_type = g.get_edge_attr(edge_id, 'entity_type')
        if entity_type == 'base':
            base_edges.append(edge_id)
        elif entity_type == 'meta':
            meta_edges.append(edge_id)
    
    print(f"Base Edges (Direct Collaborations): {len(base_edges)}")
    print(f"Meta Edges (Department Connections): {len(meta_edges)}")
    
    print(f"\n📊 Graph Statistics")
    print(f"Total Nodes: {len(g.node_ids)}")
    print(f"Total Edges: {len(g.edge_ids)}")
    
    if len(base_nodes) > 0:
        print(f"Hierarchical Ratio: {len(meta_nodes)}/{len(base_nodes)} = {len(meta_nodes)/len(base_nodes):.1%} compression")
    else:
        print(f"Complete Hierarchical Transformation: {len(meta_nodes)} meta-nodes created")
        print(f"   (Original nodes collapsed into departmental meta-nodes)")
    
    # Demonstrate attribute analysis at different levels
    print(f"\n💰 Multi-Level Salary Analysis")
    
    # Individual level (now stored in meta-nodes)
    print("Note: Individual employees have been aggregated into department meta-nodes")
    
    # Department level (meta-nodes)
    if meta_nodes:
        dept_totals = []
        for node_id in meta_nodes:
            total_sal = g.get_node_attr(node_id, 'total_salary')
            if total_sal:
                dept_totals.append(total_sal)
        
        if dept_totals:
            dept_total = sum(dept_totals)
            print(f"Department Level Total: ${dept_total:,}")
            
            # Expected total from original setup
            expected_total = 95000 + 110000 + 80000 + 85000 + 65000 + 120000 + 70000  # Sum of all employee salaries
            print(f"Expected Total: ${expected_total:,}")
            print(f"Aggregation Verification: {'✅ Match' if abs(dept_total - expected_total) < 0.01 else '❌ Mismatch'}")

def demonstrate_edge_filtering_and_analysis(g):
    """Demonstrate edge filtering and meta-edge analysis"""
    print("\n🔍 Edge Analysis and Filtering") 
    print("=" * 50)
    
    # Analyze collaboration patterns
    print("\n🤝 Collaboration Pattern Analysis")
    
    # Get all collaboration types
    collab_types = g.edges.type
    unique_types = list(set(t for t in collab_types if t))
    
    print("Collaboration Types:")
    for ctype in unique_types:
        count = len([t for t in collab_types if t == ctype])
        print(f"  {ctype}: {count} instances")
    
    # Analyze collaboration frequency
    frequencies = g.edges.frequency
    valid_freqs = [f for f in frequencies if f is not None]
    if valid_freqs:
        avg_freq = sum(valid_freqs) / len(valid_freqs)
        print(f"\nCollaboration Frequency:")
        print(f"  Average: {avg_freq:.1f} interactions/period")
        print(f"  Range: {min(valid_freqs)} - {max(valid_freqs)} interactions/period")
    
    # Analyze relationship strength
    strengths = g.edges.strength  
    valid_strengths = [s for s in strengths if s is not None]
    if valid_strengths:
        avg_strength = sum(valid_strengths) / len(valid_strengths)
        print(f"\nRelationship Strength:")
        print(f"  Average: {avg_strength:.2f}")
        print(f"  Range: {min(valid_strengths):.2f} - {max(valid_strengths):.2f}")
    
    print(f"\n🎯 Meta-Edge Analysis")
    
    # Focus on meta-edges specifically  
    meta_edge_list = []
    for edge_id in g.edge_ids:
        entity_type = g.get_edge_attr(edge_id, 'entity_type')
        if entity_type == 'meta':
            meta_edge_list.append(edge_id)
    
    if meta_edge_list:
        print(f"Meta-edges found: {len(meta_edge_list)}")
        
        # Analyze meta-edge attributes
        meta_edge_counts = []
        for edge_id in meta_edge_list:
            edge_count = g.get_edge_attr(edge_id, 'edge_count')
            if edge_count is not None:
                meta_edge_counts.append(edge_count)
        
        if meta_edge_counts:
            total_collapsed = sum(meta_edge_counts)
            print(f"Total collapsed edges: {total_collapsed}")
            print(f"Average edges per meta-edge: {total_collapsed/len(meta_edge_counts):.1f}")
    else:
        print("No meta-edges found (departments may not have external connections)")

def main():
    """Run the complete hierarchical graph demonstration"""
    print("🚀 Groggy Hierarchical Graph System Example")
    print("=" * 60)
    
    # Create the corporate graph
    g, employees = create_corporate_graph()
    
    # Demonstrate basic hierarchical operations
    dept_metas = demonstrate_basic_hierarchical_operations(g, employees)
    
    # Demonstrate advanced edge aggregation
    demonstrate_advanced_edge_aggregation(g, employees)
    
    # Demonstrate hierarchical analysis
    demonstrate_hierarchical_analysis(g)
    
    # Demonstrate edge analysis
    demonstrate_edge_filtering_and_analysis(g)
    
    print("\n🎉 Hierarchical Graph System Demonstration Complete!")
    print("\n💡 Key Takeaways:")
    print("   • Meta-nodes enable multi-level graph analysis")
    print("   • Configurable attribute aggregation (sum, mean, count, etc.)")
    print("   • Edge aggregation control for complex relationship modeling")
    print("   • Hierarchical compression while preserving analytical capabilities")
    print("   • Seamless navigation between individual and aggregate levels")
    
    print(f"\n📚 For more advanced edge aggregation control, see:")
    print(f"   • EdgeAggregationConfig for fine-tuned edge handling")
    print(f"   • Custom aggregation functions (sum, mean, max, min, concat)")
    print(f"   • Edge filtering by count, attributes, and entity types")
    
    return g

if __name__ == "__main__":
    graph = main()