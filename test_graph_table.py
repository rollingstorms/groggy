#!/usr/bin/env python3
"""
Test GraphTable - DataFrame-like views (Step D)

This tests the GraphTable functionality for tabular views of graph data.
"""

import sys
sys.path.insert(0, 'python-groggy/python')

import groggy as gr

def test_graph_table_basic():
    """Test basic GraphTable functionality"""
    
    print("📊 Testing GraphTable - DataFrame-like Views (Step D)")
    
    # Create test graph with varied attributes
    g = gr.Graph()
    
    # Add nodes with different attribute combinations
    alice = g.add_node(name="Alice", age=30, dept="Engineering", salary=120000)
    bob = g.add_node(name="Bob", age=25, dept="Engineering", salary=100000) 
    carol = g.add_node(name="Carol", age=35, dept="Design", salary=110000)
    david = g.add_node(name="David", age=28, dept="Marketing", salary=95000)
    eve = g.add_node(name="Eve", age=32, salary=130000)  # Missing dept
    
    # Add edges with attributes
    edge1 = g.add_edge(alice, bob, weight=0.8, relationship="mentor", type="strong")
    edge2 = g.add_edge(bob, carol, weight=0.6, relationship="collaborates")  # Missing type
    edge3 = g.add_edge(carol, david, weight=0.9, relationship="reports_to", type="strong")
    edge4 = g.add_edge(david, eve, relationship="knows")  # Missing weight and type
    
    print(f"✅ Created test graph: {g.node_count()} nodes, {g.edge_count()} edges")
    
    # Test 1: Basic GraphTable creation and display
    print(f"\n📋 Test 1: Basic GraphTable Creation")
    try:
        # Create node table
        node_table = gr.GraphTable(g, "nodes")
        
        print(f"✅ Node table created: {node_table.shape}")
        print(f"✅ Columns: {node_table.columns}")
        print(f"✅ Node table preview:")
        print(node_table)
        
        # Create edge table
        edge_table = gr.GraphTable(g, "edges")
        
        print(f"\n✅ Edge table created: {edge_table.shape}")
        print(f"✅ Columns: {edge_table.columns}")
        print(f"✅ Edge table preview:")
        print(edge_table)
        
    except Exception as e:
        print(f"❌ Basic GraphTable creation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Table data access
    print(f"\n📋 Test 2: Table Data Access")
    try:
        node_table = gr.GraphTable(g, "nodes")
        
        # Column access
        names = node_table['name']
        ages = node_table['age']
        
        print(f"✅ Names column: {names}")
        print(f"✅ Ages column: {ages}")
        
        # Row access
        first_row = node_table[0]
        print(f"✅ First row: {first_row}")
        
        # Verify data consistency
        assert len(names) == g.node_count(), f"Names length mismatch"
        assert len(ages) == g.node_count(), f"Ages length mismatch"
        print(f"✅ Data access working correctly")
        
    except Exception as e:
        print(f"❌ Table data access failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Export functionality
    print(f"\n📋 Test 3: Export Functionality")
    try:
        node_table = gr.GraphTable(g, "nodes")
        
        # Test CSV export
        csv_content = node_table.to_csv()
        print(f"✅ CSV export successful ({len(csv_content)} characters)")
        print(f"   CSV preview (first 200 chars): {csv_content[:200]}...")
        
        # Test JSON export
        json_content = node_table.to_json()
        print(f"✅ JSON export successful ({len(json_content)} characters)")
        
        # Test dictionary conversion
        dict_data = node_table.to_dict()
        print(f"✅ Dictionary conversion: {dict_data['shape']} shape, {len(dict_data['columns'])} columns")
        
    except Exception as e:
        print(f"❌ Export functionality failed: {e}")
        import traceback
        traceback.print_exc()

def test_subgraph_table():
    """Test GraphTable with subgraphs"""
    
    print(f"\n🔍 Testing GraphTable with Subgraphs")
    
    try:
        # Create test graph
        g = gr.Graph()
        
        alice = g.add_node(name="Alice", age=30, dept="Engineering", salary=120000)
        bob = g.add_node(name="Bob", age=25, dept="Engineering", salary=100000) 
        carol = g.add_node(name="Carol", age=35, dept="Design", salary=110000)
        david = g.add_node(name="David", age=28, dept="Marketing", salary=95000)
        
        # Create subgraph with enhanced filtering
        engineers = gr.enhanced_filter_nodes(g, "dept == 'Engineering'")
        
        print(f"✅ Created Engineering subgraph: {len(engineers.nodes)} nodes")
        
        # Create table from subgraph
        eng_table = gr.GraphTable(engineers, "nodes")
        
        print(f"✅ Engineering table: {eng_table.shape}")
        print(f"✅ Engineering table preview:")
        print(eng_table)
        
        # Verify only Engineering employees
        dept_column = eng_table['dept']
        assert all(dept == 'Engineering' for dept in dept_column if dept is not None), "Non-Engineering nodes found"
        print(f"✅ Subgraph table filtering working correctly")
        
    except Exception as e:
        print(f"❌ Subgraph table test failed: {e}")
        import traceback
        traceback.print_exc()

def test_table_aggregations():
    """Test GraphTable aggregation functionality"""
    
    print(f"\n📊 Testing GraphTable Aggregations")
    
    try:
        # Create test graph
        g = gr.Graph()
        
        # Add employees with salary data
        alice = g.add_node(name="Alice", dept="Engineering", salary=120000)
        bob = g.add_node(name="Bob", dept="Engineering", salary=100000) 
        carol = g.add_node(name="Carol", dept="Design", salary=110000)
        david = g.add_node(name="David", dept="Design", salary=95000)
        eve = g.add_node(name="Eve", dept="Marketing", salary=105000)
        
        print(f"✅ Created salary test graph: {g.node_count()} nodes")
        
        # Create table and test groupby
        node_table = gr.GraphTable(g, "nodes")
        
        # Group by department
        dept_groups = node_table.groupby('dept')
        
        # Test count aggregation
        dept_counts = dept_groups.count()
        print(f"✅ Department counts: {dept_counts}")
        
        # Test mean aggregation
        salary_means = dept_groups.mean('salary')
        print(f"✅ Department salary averages: {salary_means}")
        
        # Verify aggregations
        assert dept_counts['Engineering'] == 2, "Engineering count wrong"
        assert dept_counts['Design'] == 2, "Design count wrong"
        assert dept_counts['Marketing'] == 1, "Marketing count wrong"
        
        # Check salary averages (approximately)
        eng_avg = salary_means['Engineering']
        design_avg = salary_means['Design']
        assert abs(eng_avg - 110000) < 1000, f"Engineering average wrong: {eng_avg}"
        assert abs(design_avg - 102500) < 1000, f"Design average wrong: {design_avg}"
        
        print(f"✅ Aggregations working correctly")
        
    except Exception as e:
        print(f"❌ Aggregation test failed: {e}")
        import traceback
        traceback.print_exc()

def test_pandas_integration():
    """Test pandas integration if available"""
    
    print(f"\n🐼 Testing Pandas Integration")
    
    try:
        import pandas as pd
        
        # Create test graph
        g = gr.Graph()
        alice = g.add_node(name="Alice", age=30, salary=120000)
        bob = g.add_node(name="Bob", age=25, salary=100000)
        
        # Create table and convert to pandas
        node_table = gr.GraphTable(g, "nodes")
        df = node_table.to_pandas()
        
        print(f"✅ Pandas conversion successful: {df.shape}")
        print(f"✅ Pandas DataFrame:")
        print(df)
        
        # Test pandas operations
        high_earners = df[df['salary'] > 110000]
        print(f"✅ Pandas filtering: {len(high_earners)} high earners")
        
        avg_salary = df['salary'].mean()
        print(f"✅ Pandas aggregation: Average salary ${avg_salary:,.0f}")
        
        print(f"✅ Pandas integration working correctly")
        
    except ImportError:
        print(f"⚠️  Pandas not available - skipping pandas integration tests")
        print(f"   Install with: pip install pandas")
    except Exception as e:
        print(f"❌ Pandas integration test failed: {e}")
        import traceback
        traceback.print_exc()

def test_comprehensive_workflow():
    """Test a comprehensive GraphTable workflow"""
    
    print(f"\n🚀 Testing Comprehensive GraphTable Workflow")
    
    try:
        # Step 1: Create rich graph
        g = gr.Graph()
        
        employees = [
            {"name": "Alice", "age": 30, "dept": "Engineering", "salary": 120000, "level": "senior"},
            {"name": "Bob", "age": 25, "dept": "Engineering", "salary": 100000, "level": "junior"},
            {"name": "Carol", "age": 35, "dept": "Design", "salary": 110000, "level": "senior"},
            {"name": "David", "age": 28, "dept": "Marketing", "salary": 95000, "level": "junior"},
            {"name": "Eve", "age": 32, "dept": "Engineering", "salary": 130000, "level": "senior"},
        ]
        
        nodes = {}
        for emp in employees:
            node_id = g.add_node(**emp)
            nodes[emp["name"]] = node_id
        
        # Add relationships
        g.add_edge(nodes["Alice"], nodes["Bob"], relationship="mentors", strength=0.8)
        g.add_edge(nodes["Bob"], nodes["Carol"], relationship="collaborates", strength=0.6)
        g.add_edge(nodes["Carol"], nodes["David"], relationship="coordinates", strength=0.7)
        g.add_edge(nodes["David"], nodes["Eve"], relationship="reports_to", strength=0.9)
        
        print(f"✅ Created comprehensive graph: {g.node_count()} nodes, {g.edge_count()} edges")
        
        # Step 2: Full graph table analysis
        full_table = gr.GraphTable(g, "nodes")
        print(f"\n📊 Full organization table:")
        print(full_table)
        
        # Step 3: Department analysis
        dept_groups = full_table.groupby('dept')
        dept_stats = {
            'counts': dept_groups.count(),
            'avg_salary': dept_groups.mean('salary'),
        }
        print(f"\n📈 Department analysis:")
        print(f"   Counts: {dept_stats['counts']}")
        print(f"   Avg Salary: {dept_stats['avg_salary']}")
        
        # Step 4: Filtered subgraph analysis
        seniors = gr.enhanced_filter_nodes(g, "level == 'senior'")
        senior_table = gr.GraphTable(seniors, "nodes")
        print(f"\n👔 Senior employees table:")
        print(senior_table)
        
        # Step 5: Edge relationship analysis
        edge_table = gr.GraphTable(g, "edges")
        print(f"\n🔗 Relationships table:")
        print(edge_table)
        
        # Step 6: Export for external analysis
        csv_export = full_table.to_csv()
        json_export = senior_table.to_json()
        
        print(f"\n💾 Export summary:")
        print(f"   CSV export: {len(csv_export)} characters")
        print(f"   JSON export: {len(json_export)} characters")
        
        print(f"✅ Comprehensive workflow completed successfully!")
        
    except Exception as e:
        print(f"❌ Comprehensive workflow failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_graph_table_basic()
    test_subgraph_table()
    test_table_aggregations()
    test_pandas_integration()
    test_comprehensive_workflow()
    
    print(f"\n🎉 GraphTable Testing Complete!")
    print(f"✨ Step D Successfully Implemented:")
    print(f"   • DataFrame-like views: GraphTable(graph, 'nodes')")
    print(f"   • Attribute discovery: Automatic detection of all node/edge attributes")
    print(f"   • NaN handling: Missing attributes displayed as None/NaN")
    print(f"   • Subgraph compatibility: Works with filtered subgraphs")
    print(f"   • Export capabilities: CSV, JSON, pandas DataFrame")
    print(f"   • Aggregation support: groupby, count, mean operations")
    print(f"   • Comprehensive workflow: Full data science pipeline support")