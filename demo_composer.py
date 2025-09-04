#!/usr/bin/env python3
"""
MetaGraph Composer Demo
=======================

This script demonstrates the powerful MetaGraph Composer API with real-world examples
showing how to collapse subgraphs into meta-nodes with various aggregation strategies.
"""

import sys
sys.path.append('.')
import groggy as gr

def create_company_graph():
    """Create a realistic company organizational graph"""
    print("🏢 Building Company Organizational Graph...")
    
    g = gr.Graph(directed=False)
    
    # Engineering Department
    alice = g.add_node(name="Alice Chen", role="Senior Engineer", department="engineering", 
                      salary=95000, years_exp=5, location="SF", team="backend")
    bob = g.add_node(name="Bob Smith", role="Tech Lead", department="engineering",
                    salary=110000, years_exp=7, location="SF", team="backend")
    carol = g.add_node(name="Carol Wang", role="Frontend Engineer", department="engineering",
                      salary=88000, years_exp=4, location="Remote", team="frontend")
    dave = g.add_node(name="Dave Brown", role="DevOps Engineer", department="engineering",
                     salary=92000, years_exp=6, location="NYC", team="infrastructure")
    
    # Marketing Department
    eve = g.add_node(name="Eve Johnson", role="Marketing Manager", department="marketing",
                    salary=85000, years_exp=6, location="NYC", team="growth")
    frank = g.add_node(name="Frank Miller", role="Content Writer", department="marketing", 
                      salary=65000, years_exp=3, location="Remote", team="content")
    grace = g.add_node(name="Grace Davis", role="Social Media Specialist", department="marketing",
                      salary=62000, years_exp=2, location="LA", team="social")
    
    # Sales Department  
    henry = g.add_node(name="Henry Wilson", role="Sales Director", department="sales",
                      salary=120000, years_exp=8, location="NYC", team="enterprise")
    ivy = g.add_node(name="Ivy Taylor", role="Account Executive", department="sales",
                    salary=75000, years_exp=4, location="SF", team="smb")
    
    # Add collaboration edges within teams
    # Engineering team connections
    g.add_edge(alice, bob, strength=0.9, type="daily_standup", project="api_v2", hours_per_week=10)
    g.add_edge(bob, carol, strength=0.7, type="code_review", project="ui_refresh", hours_per_week=5)
    g.add_edge(alice, dave, strength=0.8, type="deployment", project="infrastructure", hours_per_week=8)
    g.add_edge(carol, dave, strength=0.6, type="frontend_deploy", project="ci_cd", hours_per_week=3)
    
    # Marketing team connections
    g.add_edge(eve, frank, strength=0.85, type="content_planning", project="blog_strategy", hours_per_week=6)
    g.add_edge(frank, grace, strength=0.75, type="content_distribution", project="social_media", hours_per_week=4)
    g.add_edge(eve, grace, strength=0.7, type="campaign_coordination", project="product_launch", hours_per_week=5)
    
    # Sales team connection
    g.add_edge(henry, ivy, strength=0.9, type="mentoring", project="sales_training", hours_per_week=3)
    
    # Cross-department collaborations
    g.add_edge(bob, eve, strength=0.5, type="product_feedback", project="feature_prioritization", hours_per_week=2)
    g.add_edge(alice, henry, strength=0.4, type="technical_sales", project="enterprise_demos", hours_per_week=1)
    g.add_edge(grace, ivy, strength=0.3, type="lead_qualification", project="marketing_qualified_leads", hours_per_week=2)
    
    print(f"✅ Created company graph: {g.node_count()} employees, {g.edge_count()} collaborations")
    return g

def demo_basic_department_collapse():
    """Demonstrate basic department-level collapse"""
    print("\n" + "="*60)
    print("🎯 DEMO 1: Basic Department Collapse")
    print("="*60)
    
    g = create_company_graph()
    
    # Get engineering team  
    eng_nodes = [n for n in g.node_ids if g.get_node_attr(n, 'department') == 'engineering']
    engineering = g.nodes[eng_nodes]
    
    print(f"\n📊 Engineering Department Analysis")
    print(f"   Team size: {engineering.node_count()} engineers")
    print(f"   Internal collaborations: {engineering.edge_count()}")
    
    # Create meta-node plan with comprehensive aggregations
    plan = engineering.collapse(
        node_aggs=[
            ("total_salary_budget", "sum", "salary"),
            ("avg_experience", "mean", "years_exp"), 
            ("team_size", "count"),
            ("department_name", "first", "department"),
            ("tech_stack_expertise", "concat", "team"),
            ("office_locations", "concat", "location")
        ],
        edge_aggs={
            "avg_collaboration_strength": "mean",
            "collaboration_types": "concat", 
            "total_weekly_hours": "sum"
        },
        edge_strategy="aggregate",
        entity_type="engineering_department"
    )
    
    # Preview the plan
    preview = plan.preview()
    print(f"\n🔍 Plan Preview:")
    print(f"   📋 Meta-node will have {len(preview.meta_node_attributes)} aggregated attributes")
    for attr, func in preview.meta_node_attributes.items():
        print(f"      • {attr}: {func}")
    print(f"   🔗 Estimated meta-edges: {preview.meta_edges_count}")
    print(f"   ⚙️  Edge strategy: {preview.edge_strategy}")
    print(f"   🏷️  Entity type: '{preview.entity_type}'")
    
    print(f"\n✅ Engineering department successfully modeled as meta-node")

def demo_cross_functional_team():
    """Demonstrate cross-functional team collapse"""
    print("\n" + "="*60)
    print("🎯 DEMO 2: Cross-Functional Team Collapse") 
    print("="*60)
    
    g = create_company_graph()
    
    # Create cross-functional product team (one from each dept)
    product_team_nodes = [1, 4, 7]  # Bob (eng), Eve (marketing), Henry (sales)
    product_team = g.nodes[product_team_nodes]
    
    print(f"\n🚀 Product Leadership Team")
    print(f"   Cross-functional team: {product_team.node_count()} leaders")
    for node_id in product_team_nodes:
        name = g.get_node_attr(node_id, 'name')
        role = g.get_node_attr(node_id, 'role') 
        dept = g.get_node_attr(node_id, 'department')
        print(f"      • {name} - {role} ({dept})")
    
    # Use preset for organizational hierarchy
    plan = product_team.collapse(preset="org_hierarchy")
    preview = plan.preview()
    
    print(f"\n🎯 Using '{preview.entity_type}' preset configuration:")
    print(f"   📊 Edge strategy: {preview.edge_strategy}")
    print(f"   🔢 Include edge count: {preview.will_include_edge_count}")
    print(f"   📈 Meta-edges estimated: {preview.meta_edges_count}")
    
    # Also show a custom configuration
    custom_plan = product_team.collapse(
        node_aggs=[
            ("leadership_budget", "sum", "salary"),
            ("combined_experience", "sum", "years_exp"),
            ("avg_seniority", "mean", "years_exp"),
            ("represented_departments", "concat", "department"),
            ("geographic_reach", "concat", "location")
        ],
        edge_strategy="keep_external",
        entity_type="product_council"
    )
    
    custom_preview = custom_plan.preview()
    print(f"\n🛠️  Custom Configuration Preview:")
    print(f"   📋 Attributes: {list(custom_preview.meta_node_attributes.keys())}")
    print(f"   🔗 Meta-edges (keep_external): {custom_preview.meta_edges_count}")
    
    print(f"\n✅ Cross-functional team successfully modeled with multiple strategies")

def demo_edge_strategies():
    """Demonstrate different edge strategies"""
    print("\n" + "="*60)
    print("🎯 DEMO 3: Edge Strategy Comparison")
    print("="*60)
    
    g = create_company_graph()
    
    # Get marketing team for comparison
    mkt_nodes = [n for n in g.node_ids if g.get_node_attr(n, 'department') == 'marketing']
    marketing = g.nodes[mkt_nodes]
    
    print(f"\n📱 Marketing Department: {marketing.node_count()} team members")
    print(f"   Internal edges: {marketing.edge_count()}")
    
    strategies = ["aggregate", "keep_external", "drop_all", "contract_all"]
    
    print(f"\n🔄 Comparing Edge Strategies:")
    for strategy in strategies:
        plan = marketing.collapse(
            node_aggs=[("team_size", "count")],
            edge_strategy=strategy
        )
        preview = plan.preview()
        
        print(f"   📊 {strategy:15} → {preview.meta_edges_count:2} meta-edges")
        
        if strategy == "aggregate":
            print(f"      💡 Combines parallel edges into single aggregated meta-edge")
        elif strategy == "keep_external": 
            print(f"      💡 Preserves all external edges as-is to meta-node")
        elif strategy == "drop_all":
            print(f"      💡 Isolates meta-node completely (no external edges)")
        elif strategy == "contract_all":
            print(f"      💡 Routes external edges through meta-node (flow network)")
    
    print(f"\n✅ All edge strategies demonstrated successfully")

def demo_preset_configurations():
    """Demonstrate preset configurations"""
    print("\n" + "="*60)
    print("🎯 DEMO 4: Preset Configurations")
    print("="*60)
    
    g = create_company_graph()
    sales_nodes = [n for n in g.node_ids if g.get_node_attr(n, 'department') == 'sales']
    sales_team = g.nodes[sales_nodes]
    
    print(f"\n💼 Sales Team: {sales_team.node_count()} members")
    
    presets = ["social_network", "org_hierarchy", "flow_network"]
    
    print(f"\n🎨 Available Preset Configurations:")
    for preset in presets:
        plan = sales_team.collapse(preset=preset)
        preview = plan.preview()
        
        print(f"\n   🔧 {preset.upper().replace('_', ' ')} PRESET:")
        print(f"      🏷️  Entity type: {preview.entity_type}")
        print(f"      ⚙️  Edge strategy: {preview.edge_strategy}")
        print(f"      🔢 Include edge count: {preview.will_include_edge_count}")
        print(f"      📈 Meta-edges: {preview.meta_edges_count}")
        
        if preset == "social_network":
            print(f"      💡 Optimized for community detection and social analysis")
        elif preset == "org_hierarchy":
            print(f"      💡 Designed for organizational structure modeling")
        elif preset == "flow_network":
            print(f"      💡 Configured for flow analysis and bottleneck detection")
    
    print(f"\n✅ All preset configurations demonstrated successfully")

def main():
    """Run the comprehensive MetaGraph Composer demonstration"""
    print("🌟 MetaGraph Composer API Demonstration")
    print("🔬 Showcasing Advanced Graph Hierarchical Modeling")
    print("=" * 80)
    
    try:
        # Run all demonstrations
        demo_basic_department_collapse()
        demo_cross_functional_team() 
        demo_edge_strategies()
        demo_preset_configurations()
        
        print("\n" + "=" * 80)
        print("🎉 DEMONSTRATION COMPLETE!")
        print("=" * 80)
        print("✨ MetaGraph Composer Features Demonstrated:")
        print("   ✅ Flexible node aggregation with multiple functions")
        print("   ✅ Configurable edge strategies for different use cases")
        print("   ✅ Preview functionality for plan validation")  
        print("   ✅ Preset configurations for common patterns")
        print("   ✅ Cross-functional team modeling")
        print("   ✅ Department-level organizational analysis")
        print("\n💡 The MetaGraph Composer provides a clean, intuitive API for")
        print("   creating hierarchical graph structures while maintaining full")
        print("   control over aggregation strategies and meta-edge behavior.")
        
        return 0
        
    except Exception as e:
        print(f"\n💥 Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())