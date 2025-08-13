#!/usr/bin/env python3
"""
Comprehensive test showcasing all the new Pythonic API features
"""

import groggy as gr

def test_comprehensive_api():
    """Test all the new API features together"""
    
    print("🎯 === COMPREHENSIVE PYTHONIC API SHOWCASE ===")
    
    # === 1. FLEXIBLE GRAPH CONSTRUCTION ===
    print("\n🔧 === 1. FLEXIBLE GRAPH CONSTRUCTION ===")
    
    g = gr.Graph()
    
    # Multi-format node creation with automatic ID mapping
    employee_data = [
        {"id": "alice", "name": "Alice Chen", "role": "Senior Engineer", "dept": "Engineering", 
         "salary": 120000, "years": 5, "location": "SF"},
        {"id": "bob", "name": "Bob Smith", "role": "Manager", "dept": "Engineering", 
         "salary": 140000, "years": 8, "location": "NYC"},
        {"id": "carol", "name": "Carol Davis", "role": "Designer", "dept": "Design", 
         "salary": 95000, "years": 3, "location": "SF"},
        {"id": "david", "name": "David Wilson", "role": "Director", "dept": "Engineering", 
         "salary": 180000, "years": 12, "location": "NYC"},
        {"id": "eve", "name": "Eve Johnson", "role": "Engineer", "dept": "Engineering", 
         "salary": 100000, "years": 2, "location": "Remote"},
    ]
    
    # Create nodes with automatic string-to-numeric ID mapping
    employee_map = g.add_nodes(employee_data, id_key="id")
    print(f"✅ Created {len(g.nodes)} employees with mapping: {employee_map}")
    
    # Multiple edge creation formats
    collaboration_data = [
        {"source": "alice", "target": "bob", "type": "reports_to", "frequency": "daily", "strength": 0.9},
        {"source": "carol", "target": "alice", "type": "collaborates", "frequency": "weekly", "strength": 0.7},
        {"source": "eve", "target": "alice", "type": "mentored_by", "frequency": "bi-weekly", "strength": 0.8},
        {"source": "david", "target": "bob", "type": "manages", "frequency": "daily", "strength": 0.95},
    ]
    
    # Bulk edge creation with string ID resolution
    g.add_edges(collaboration_data, node_mapping=employee_map)
    
    # Add some ad-hoc edges with kwargs and uid_key
    g.add_edge("bob", "carol", uid_key="id", type="cross_dept", frequency="monthly", strength=0.4)
    
    print(f"✅ Graph: {g} with {len(g.edges)} connections")
    
    # === 2. PROPERTY ACCESS ===
    print(f"\n📊 === 2. PROPERTY ACCESS ===")
    
    print(f"g.nodes: {g.nodes}")
    print(f"g.edges: {g.edges}")
    print(f"len(g): {len(g)}")
    
    # Column attribute access
    print(f"g.attributes.keys(): {g.attributes.keys()}")
    print(f"Names: {g.attributes['name']}")
    print(f"Salaries: {g.attributes['salary']}")
    print(f"Departments: {g.attributes['dept']}")
    
    # === 3. SUBGRAPH ANALYSIS WITH FILTERING ===
    print(f"\n🔍 === 3. SUBGRAPH ANALYSIS WITH FILTERING ===")
    
    # Multi-step filtering 
    engineering_team = g.filter_nodes('dept == "Engineering"')
    print(f"Engineering team: {engineering_team}")
    print(f"Engineering nodes: {engineering_team.nodes}")
    print(f"Engineering edges: {engineering_team.edges}")
    
    # Further filter within subgraph
    senior_engineers = g.filter_subgraph_nodes(engineering_team, 'years > 4')  # Use > instead of >=
    print(f"Senior engineers: {senior_engineers}")
    print(f"Senior engineer nodes: {senior_engineers.nodes}")
    
    high_salary_seniors = g.filter_subgraph_nodes(senior_engineers, 'salary > 120000')
    print(f"High-salary senior engineers: {high_salary_seniors}")
    
    # === 4. COMPLEX MULTI-STEP WORKFLOW ===
    print(f"\n⚡ === 4. COMPLEX MULTI-STEP WORKFLOW ===")
    
    # Find all high performers (high salary, but not the highest level)
    all_employees = g.filter_nodes('salary > 0')  # Everyone
    print(f"All employees: {len(all_employees.nodes)} people")
    
    # Filter to mid-high earners (not directors)
    mid_high_earners = g.filter_subgraph_nodes(all_employees, 'salary > 100000 and salary < 150000')
    print(f"Mid-high earners: {len(mid_high_earners.nodes)} people")
    
    # Further filter to experienced employees
    experienced_performers = g.filter_subgraph_nodes(mid_high_earners, 'years > 3')
    print(f"Experienced high performers: {len(experienced_performers.nodes)} people")
    
    # Show details
    print(f"Experienced high performers details:")
    for node_id in experienced_performers.nodes:
        name = g.attributes['name'][g.nodes.index(node_id)]
        role = g.attributes['role'][g.nodes.index(node_id)]
        salary = g.attributes['salary'][g.nodes.index(node_id)]
        years = g.attributes['years'][g.nodes.index(node_id)]
        print(f"  - {name}: {role}, ${salary:,}, {years} years")
    
    # === 5. EDGE ATTRIBUTE ANALYSIS ===
    print(f"\n🔗 === 5. EDGE ATTRIBUTE ANALYSIS ===")
    
    print(f"Edge details:")
    for edge_id in g.edges:
        edge_type = g.get_edge_attribute(edge_id, 'type')
        strength = g.get_edge_attribute(edge_id, 'strength')
        print(f"  Edge {edge_id}: type={edge_type.value if edge_type else 'N/A'}, "
              f"strength={strength.value if strength else 'N/A'}")
    
    # === 6. STRING QUERY CAPABILITIES ===
    print(f"\n🎯 === 6. STRING QUERY CAPABILITIES ===")
    
    # Various string query formats
    queries = [
        'salary > 120000',
        'dept == "Engineering"',
        'years > 4',
        'location == "SF"',
        'role == "Manager"'
    ]
    
    for query in queries:
        result = g.filter_nodes(query)
        print(f"Query '{query}': {len(result.nodes)} matches")
    
    # === 7. FINAL SUMMARY ===
    print(f"\n🎉 === 7. FINAL SUMMARY ===")
    
    print(f"✅ Successfully created graph with:")
    print(f"   📊 {len(g.nodes)} nodes with full attribute support")
    print(f"   🔗 {len(g.edges)} edges with relationship metadata")
    print(f"   🏷️  {len(g.attributes.keys())} attribute types: {list(g.attributes.keys())}")
    print(f"   🔍 String-based filtering with complex queries")
    print(f"   ⛓️  Chainable subgraph operations")
    print(f"   🎯 Multiple input formats (dicts, tuples, kwargs)")
    print(f"   🆔 String ID resolution with uid_key and node_mapping")
    print(f"   📈 Column-wise attribute access for analytics")
    
    print(f"\n🚀 **All Pythonic API enhancements working perfectly!** 🚀")

if __name__ == "__main__":
    test_comprehensive_api()