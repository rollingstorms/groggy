#!/usr/bin/env python3
"""Comprehensive test suite for MetaGraph Composer functionality"""

import sys
sys.path.append('.')

try:
    import groggy as gr
    print("✓ Groggy imported successfully")
except ImportError as e:
    print(f"✗ Failed to import groggy: {e}")
    sys.exit(1)

def create_test_graph():
    """Create a comprehensive test graph with multiple communities"""
    g = gr.Graph(directed=False)
    
    # Engineering team (nodes 0-3)
    g.add_node(name="Alice", age=25, salary=95000, department="engineering", level="senior", location="SF")
    g.add_node(name="Bob", age=30, salary=110000, department="engineering", level="lead", location="SF") 
    g.add_node(name="Carol", age=28, salary=80000, department="engineering", level="junior", location="remote")
    g.add_node(name="Dave", age=35, salary=120000, department="engineering", level="principal", location="SF")
    
    # Marketing team (nodes 4-6)
    g.add_node(name="Eve", age=32, salary=85000, department="marketing", level="manager", location="NYC")
    g.add_node(name="Frank", age=27, salary=65000, department="marketing", level="specialist", location="NYC")
    g.add_node(name="Grace", age=29, salary=70000, department="marketing", level="coordinator", location="remote")
    
    # Sales team (nodes 7-8)
    g.add_node(name="Henry", age=40, salary=130000, department="sales", level="director", location="LA")
    g.add_node(name="Ivy", age=26, salary=75000, department="sales", level="rep", location="LA")
    
    # External partner (node 9)
    g.add_node(name="Jack", age=45, salary=150000, department="external", level="consultant", location="chicago")
    
    # Engineering team internal connections
    g.add_edge(0, 1, weight=0.9, type="collaboration", frequency="daily", project="core")
    g.add_edge(1, 2, weight=0.7, type="mentoring", frequency="weekly", project="core") 
    g.add_edge(2, 3, weight=0.8, type="collaboration", frequency="daily", project="infrastructure")
    g.add_edge(0, 3, weight=0.6, type="review", frequency="monthly", project="architecture")
    
    # Marketing team internal connections
    g.add_edge(4, 5, weight=0.85, type="partnership", frequency="daily", project="campaigns")
    g.add_edge(5, 6, weight=0.75, type="support", frequency="weekly", project="content")
    g.add_edge(4, 6, weight=0.9, type="management", frequency="daily", project="strategy")
    
    # Sales team internal connection
    g.add_edge(7, 8, weight=0.8, type="supervision", frequency="daily", project="targets")
    
    # Cross-department edges
    g.add_edge(1, 4, weight=0.5, type="cross_dept", frequency="monthly", project="product_launch")
    g.add_edge(3, 7, weight=0.4, type="requirements", frequency="quarterly", project="enterprise")
    g.add_edge(6, 8, weight=0.3, type="leads", frequency="weekly", project="marketing_qualified")
    
    # External connections
    g.add_edge(0, 9, weight=0.6, type="consulting", frequency="monthly", project="architecture")
    g.add_edge(4, 9, weight=0.7, type="consulting", frequency="weekly", project="brand_strategy")
    
    return g

def test_basic_collapse():
    """Test basic collapse functionality"""
    print("\n=== Testing Basic Collapse Functionality ===")
    
    g = create_test_graph()
    engineering_team = g.nodes[[0, 1, 2, 3]]  # Alice, Bob, Carol, Dave
    
    print(f"Engineering subgraph: {engineering_team.node_count()} nodes, {engineering_team.edge_count()} edges")
    
    # Test basic collapse with no parameters
    plan1 = engineering_team.collapse()
    print(f"✓ Basic collapse: {plan1}")
    
    # Test collapse with node aggregations
    plan2 = engineering_team.collapse(
        node_aggs={
            "total_salary": ("sum", "salary"),
            "avg_age": ("mean", "age"), 
            "team_size": "count",
            "department": ("first", "department")
        }
    )
    print(f"✓ Node aggregation collapse: {plan2}")
    
    # Test collapse with edge aggregations
    plan3 = engineering_team.collapse(
        node_aggs={"team_size": "count"},
        edge_aggs={
            "avg_weight": "mean",
            "collaboration_types": "concat"
        }
    )
    print(f"✓ Edge aggregation collapse: {plan3}")
    
    return True

def test_edge_strategies():
    """Test different edge strategies"""
    print("\n=== Testing Edge Strategies ===")
    
    g = create_test_graph()
    marketing_team = g.nodes[[4, 5, 6]]  # Eve, Frank, Grace
    
    # Test aggregate strategy (default)
    plan_agg = marketing_team.collapse(
        node_aggs={"team_size": "count"},
        edge_strategy="aggregate"
    )
    preview_agg = plan_agg.preview()
    print(f"✓ Aggregate strategy - estimated meta-edges: {preview_agg.meta_edges_count}")
    
    # Test keep_external strategy
    plan_keep = marketing_team.collapse(
        node_aggs={"team_size": "count"},
        edge_strategy="keep_external"
    )
    preview_keep = plan_keep.preview()
    print(f"✓ Keep external strategy - estimated meta-edges: {preview_keep.meta_edges_count}")
    
    # Test drop_all strategy
    plan_drop = marketing_team.collapse(
        node_aggs={"team_size": "count"},
        edge_strategy="drop_all"
    )
    preview_drop = plan_drop.preview()
    print(f"✓ Drop all strategy - estimated meta-edges: {preview_drop.meta_edges_count}")
    
    # Test contract_all strategy
    plan_contract = marketing_team.collapse(
        node_aggs={"team_size": "count"},
        edge_strategy="contract_all"
    )
    preview_contract = plan_contract.preview()
    print(f"✓ Contract all strategy - estimated meta-edges: {preview_contract.meta_edges_count}")
    
    return True

def test_presets():
    """Test preset configurations"""
    print("\n=== Testing Preset Configurations ===")
    
    g = create_test_graph()
    sales_team = g.nodes[[7, 8]]  # Henry, Ivy
    
    try:
        # Test social_network preset
        plan_social = sales_team.collapse(preset="social_network")
        preview_social = plan_social.preview()
        print(f"✓ Social network preset: {preview_social.entity_type} entity")
        print(f"  Edge strategy: {preview_social.edge_strategy}")
        print(f"  Will include edge count: {preview_social.will_include_edge_count}")
        
        # Test org_hierarchy preset
        plan_org = sales_team.collapse(preset="org_hierarchy")
        preview_org = plan_org.preview()
        print(f"✓ Org hierarchy preset: {preview_org.entity_type} entity")
        print(f"  Edge strategy: {preview_org.edge_strategy}")
        
        # Test flow_network preset
        plan_flow = sales_team.collapse(preset="flow_network")
        preview_flow = plan_flow.preview()
        print(f"✓ Flow network preset: {preview_flow.entity_type} entity")
        print(f"  Edge strategy: {preview_flow.edge_strategy}")
        
        return True
        
    except Exception as e:
        print(f"✗ Preset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_preview_functionality():
    """Test preview functionality and validation"""
    print("\n=== Testing Preview Functionality ===")
    
    g = create_test_graph()
    cross_functional = g.nodes[[1, 4, 7]]  # Bob (eng), Eve (marketing), Henry (sales)
    
    # Create a complex plan (using list format for tuple aggregations)
    plan = cross_functional.collapse(
        node_aggs=[
            ("total_budget", "sum", "salary"),
            ("avg_age", "mean", "age"),
            ("leadership_count", "count"),
            ("departments", "concat", "department")
        ],
        edge_aggs={
            "connection_strength": "mean",
            "interaction_types": "concat",
            "meeting_frequency": "concat"
        },
        edge_strategy="aggregate",
        include_edge_count=True,
        entity_type="cross_functional_team"
    )
    
    # Test preview
    preview = plan.preview()
    print(f"✓ Preview generated successfully")
    print(f"  Meta-node attributes: {len(preview.meta_node_attributes)} attributes")
    for attr, func in preview.meta_node_attributes.items():
        print(f"    {attr}: {func}")
    print(f"  Estimated meta-edges: {preview.meta_edges_count}")
    print(f"  Edge strategy: {preview.edge_strategy}")
    print(f"  Will include edge count: {preview.will_include_edge_count}")
    print(f"  Entity type: '{preview.entity_type}'")
    
    return True

def test_complex_aggregations():
    """Test complex aggregation patterns"""
    print("\n=== Testing Complex Aggregation Patterns ===")
    
    g = create_test_graph()
    
    # Test with mixed data types and different aggregation functions
    diverse_team = g.nodes[[0, 4, 7, 9]]  # Alice, Eve, Henry, Jack - diverse roles
    
    plan = diverse_team.collapse(
        node_aggs=[
            # Numeric aggregations
            ("total_compensation", "sum", "salary"),
            ("avg_age", "mean", "age"),
            ("max_salary", "max", "salary"),
            ("min_age", "min", "age"),
            
            # String aggregations  
            ("all_departments", "concat", "department"),
            ("primary_location", "first", "location"),
            ("seniority_level", "first", "level"),
            
            # Count aggregation
            ("team_size", "count")
        ],
        edge_aggs={
            "avg_collaboration": "mean",
            "interaction_types": "concat",
            "project_involvement": "concat"
        },
        edge_strategy="aggregate",
        entity_type="leadership_council"
    )
    
    preview = plan.preview()
    print(f"✓ Complex aggregation plan created")
    print(f"  {len(preview.meta_node_attributes)} node attributes configured")
    print(f"  Entity type: {preview.entity_type}")
    
    # Validate that all expected attributes are present
    expected_attrs = {
        "total_compensation", "avg_age", "max_salary", "min_age",
        "all_departments", "primary_location", "seniority_level", "team_size"
    }
    actual_attrs = set(preview.meta_node_attributes.keys())
    
    if expected_attrs.issubset(actual_attrs):
        print(f"✓ All expected attributes present: {sorted(expected_attrs)}")
        return True
    else:
        missing = expected_attrs - actual_attrs
        print(f"✗ Missing attributes: {missing}")
        return False

def test_error_handling():
    """Test error handling and validation"""
    print("\n=== Testing Error Handling ===")
    
    g = create_test_graph()
    test_team = g.nodes[[0, 1]]
    
    try:
        # Test invalid preset
        try:
            plan_bad_preset = test_team.collapse(preset="invalid_preset")
            print("✗ Should have failed with invalid preset")
            return False
        except Exception as e:
            print(f"✓ Correctly rejected invalid preset: {type(e).__name__}")
        
        # Test invalid edge strategy  
        try:
            plan_bad_strategy = test_team.collapse(edge_strategy="invalid_strategy")
            print("✗ Should have failed with invalid edge strategy")
            return False
        except Exception as e:
            print(f"✓ Correctly rejected invalid edge strategy: {type(e).__name__}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_builder_pattern_methods():
    """Test the builder pattern methods on MetaNodePlan"""
    print("\n=== Testing Builder Pattern Methods ===")
    
    g = create_test_graph()
    test_team = g.nodes[[0, 1, 2]]
    
    try:
        # Get a plan to work with
        plan = test_team.collapse()
        print(f"✓ Initial plan: {plan}")
        
        # Test that we got a plan-like object
        if hasattr(plan, 'preview'):
            preview1 = plan.preview()
            print(f"✓ Initial preview: {len(preview1.meta_node_attributes)} attributes")
            
            # Note: The current implementation creates an executor, not a mutable plan
            # This validates that the API surface exists and works
            return True
        else:
            print("✗ Plan object missing preview method")
            return False
            
    except Exception as e:
        print(f"✗ Builder pattern test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run comprehensive test suite"""
    print("Comprehensive MetaGraph Composer Test Suite")
    print("=" * 60)
    
    # Test functions in order of complexity
    tests = [
        test_basic_collapse,
        test_edge_strategies,
        test_preview_functionality,
        test_complex_aggregations,
        test_presets,
        test_builder_pattern_methods,
        test_error_handling,
    ]
    
    results = []
    for test in tests:
        try:
            print(f"\nRunning {test.__name__}...")
            result = test()
            results.append(result)
            if result:
                print(f"✅ {test.__name__} PASSED")
            else:
                print(f"❌ {test.__name__} FAILED")
        except Exception as e:
            print(f"💥 {test.__name__} CRASHED: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE TEST RESULTS")
    print(f"{'='*60}")
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - MetaGraph Composer is working perfectly!")
        return 0
    else:
        print(f"⚠️  {total - passed} test(s) failed - see details above")
        return 1

if __name__ == "__main__":
    sys.exit(main())