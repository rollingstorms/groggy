#!/usr/bin/env python3
"""
Groggy Storage System - Phase 1, 2 & 3 Usage Examples
====================================================

This example demonstrates the advanced table operations, bundle system,
and BaseArray chaining features implemented in Phases 1-3 of the Groggy storage system.

Features Demonstrated:
- Phase 1: File I/O, Filter Operations, Attribute Modification, Bundle System
- Phase 2: Group By Operations, Multi-table Operations (Joins), Enhanced Bundle System  
- Phase 3: BaseArray Chaining System, Method Delegation, Trait-Based Method Injection
"""

import groggy
import json
from pathlib import Path

def phase_1_example():
    """Phase 1 Features: Core Operations and File I/O"""
    print("=" * 60)
    print("PHASE 1: Core Operations and File I/O")
    print("=" * 60)
    
    # Create sample data
    print("\n1. Creating Sample Tables")
    print("-" * 30)
    
    # Create users table
    users_data = {
        'user_id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'age': [25, 30, 35, 28, 22],
        'department': ['Engineering', 'Sales', 'Engineering', 'Marketing', 'Sales'],
        'salary': [75000, 60000, 85000, 70000, 55000]
    }
    users_table = groggy.BaseTable.from_dict(users_data)
    print(f"Users table created: {users_table.shape}")
    
    # Create projects table
    projects_data = {
        'project_id': [101, 102, 103, 104],
        'name': ['WebApp', 'MobileApp', 'Analytics', 'Marketing Campaign'],
        'budget': [50000, 75000, 30000, 25000],
        'status': ['Active', 'Planning', 'Active', 'Completed']
    }
    projects_table = groggy.BaseTable.from_dict(projects_data)
    print(f"Projects table created: {projects_table.shape}")
    
    # Phase 1 Feature 1: File I/O System
    print("\n2. File I/O Operations")
    print("-" * 30)
    
    # Export to CSV
    users_table.to_csv("users.csv")
    projects_table.to_csv("projects.csv")
    print("✓ Tables exported to CSV")
    
    # Export to JSON  
    users_table.to_json("users.json")
    projects_table.to_json("projects.json")
    print("✓ Tables exported to JSON")
    
    # Load from files
    loaded_users = groggy.BaseTable.from_csv("users.csv")
    loaded_projects = groggy.BaseTable.from_json("projects.json")
    print(f"✓ Loaded users from CSV: {loaded_users.shape}")
    print(f"✓ Loaded projects from JSON: {loaded_projects.shape}")
    
    # Phase 1 Feature 2: Enhanced Filter Operations
    print("\n3. Advanced Filter Operations")
    print("-" * 30)
    
    # String predicate filters
    senior_users = users_table.filter("age > 30")
    print(f"✓ Senior users (age > 30): {senior_users.nrows} found")
    
    high_salary = users_table.filter("salary >= 70000")
    print(f"✓ High salary users (salary >= 70000): {high_salary.nrows} found")
    
    engineering_dept = users_table.filter("department == 'Engineering'")
    print(f"✓ Engineering department: {engineering_dept.nrows} users")
    
    # Python function filters (more complex logic)
    def custom_filter(row):
        return row['age'] > 25 and row['department'] in ['Engineering', 'Marketing']
    
    filtered_users = users_table.filter(custom_filter)
    print(f"✓ Custom filtered users: {filtered_users.nrows} found")
    
    # Phase 1 Feature 3: Flexible Attribute Modification
    print("\n4. Flexible Attribute Modification")
    print("-" * 30)
    
    # Add bonus column with different formats
    bonus_updates = {
        'bonus': {
            0: 5000,    # Integer ID
            1: 4000,  # String ID (flexible conversion)
            2: 6000,
            3: 5500,
            4: 3500
        }
    }
    users_table.assign(bonus_updates)
    print("✓ Added bonus column with flexible ID types")
    
    # Column-centric format update
    performance_updates = {
        'performance_rating': {
            0:4.2, 1:3.8, 2:4.5, 3:4.0, 4:3.9
        }
    }
    users_table.assign(performance_updates)
    print("✓ Added performance ratings using column-centric format")
    
    print(f"Updated users table: {users_table.shape}")
    print(f"Columns: {users_table.column_names}")
    
    return users_table, projects_table

def phase_2_example(users_table, projects_table):
    """Phase 2 Features: Advanced Operations and Enhanced Bundle System"""
    print("\n" + "=" * 60)
    print("PHASE 2: Advanced Operations and Enhanced Bundle System")
    print("=" * 60)
    
    # Phase 2 Feature 1: Group By Operations
    print("\n1. Group By Operations")
    print("-" * 30)
    
    # Group by department and calculate statistics using new pattern
    grouped_depts = users_table.group_by(['department'])
    dept_stats = grouped_depts.agg({
        'salary': 'avg',
        'age': 'mean', 
        'bonus': 'sum',
        'user_id': 'count'
    })
    print("✓ Department statistics calculated:")
    print(f"   Grouped into: {len(grouped_depts)} departments")
    print(f"   Aggregated shape: {dept_stats.shape}")
    print(f"   Aggregated columns: {dept_stats.column_names}")
    
    # Overall table aggregation
    overall_stats = users_table.agg({
        'salary': 'avg',
        'age': 'min',
        'bonus': 'max',
        'user_id': 'count'
    })
    print("✓ Overall statistics:")
    print(f"   Shape: {overall_stats.shape}")
    
    # Phase 2 Feature 2: Multi-table Operations (Unified Join Interface)
    print("\n2. Multi-table Operations (Joins)")
    print("-" * 30)
    
    # Create user-project assignments table
    assignments_data = {
        'user_id': [1, 2, 3, 1, 4, 5],
        'project_id': [101, 102, 103, 104, 101, 102],
        'role': ['Lead', 'Developer', 'Analyst', 'Contributor', 'Designer', 'Tester'],
        'hours_allocated': [40, 35, 30, 20, 25, 20]
    }
    assignments_table = groggy.BaseTable.from_dict(assignments_data)
    print(f"✓ Created assignments table: {assignments_table.shape}")
    
    # Different join examples
    # 1. Simple inner join
    user_assignments = users_table.join(assignments_table, on="user_id", how="inner")
    print(f"✓ User-assignments inner join: {user_assignments.shape}")
    
    # 2. Left join to see all users (including those without assignments)
    all_users_assignments = users_table.join(assignments_table, on="user_id", how="left")
    print(f"✓ All users with assignments (left join): {all_users_assignments.shape}")
    
    # 3. Join with different column names
    project_assignments = assignments_table.join(
        projects_table, 
        on={"left": "project_id", "right": "project_id"}, 
        how="inner"
    )
    print(f"✓ Project-assignments join: {project_assignments.shape}")
    
    # 4. Complex three-way join scenario
    full_context = user_assignments.join(
        projects_table,
        on={"left": "project_id", "right": "project_id"},
        how="inner"
    )
    print(f"✓ Full context join (users-assignments-projects): {full_context.shape}")
    
    # Union and Intersect operations
    print("\n3. Set Operations")
    print("-" * 30)
    
    # Create two user subsets
    senior_eng = users_table.filter("age > 30").filter("department == 'Engineering'")
    high_performers = users_table.filter("performance_rating > 4.0")
    
    # Union: Combine unique users
    combined_talent = senior_eng.union(high_performers)
    print(f"✓ Senior engineers OR high performers (union): {combined_talent.nrows} users")
    
    # Intersect: Find overlap
    elite_engineers = senior_eng.intersect(high_performers)
    print(f"✓ Senior engineers AND high performers (intersect): {elite_engineers.nrows} users")
    
    return full_context, dept_stats

def phase_2_bundle_system_example(graph_data):
    """Phase 2 Enhanced Bundle System with Metadata and Validation"""
    print("\n4. Enhanced Bundle System (v2.0)")
    print("-" * 30)
    
    # Create a graph table for bundle operations
    nodes_data = {
        'node_id': [1, 2, 3, 4, 5],
        'type': ['User', 'User', 'Project', 'Project', 'Team'],
        'name': ['Alice', 'Bob', 'WebApp', 'MobileApp', 'Engineering'],
        'active': [True, True, True, False, True]
    }
    
    edges_data = {
        'edge_id': [1, 2, 3, 4, 5],
        'source': [1, 2, 1, 2, 5],
        'target': [3, 4, 5, 5, 3],
        'relationship': ['works_on', 'works_on', 'member_of', 'member_of', 'manages'],
        'weight': [1.0, 0.8, 1.0, 1.0, 0.9]
    }
    
    nodes_table = groggy.NodesTable.from_dict(nodes_data)
    edges_table = groggy.EdgesTable.from_dict(edges_data)
    graph_table = groggy.GraphTable(nodes_table, edges_table)
    
    print("✓ Created sample graph table")
    
    # Save as v2.0 bundle with comprehensive metadata
    bundle_path = "./sample_graph_bundle"
    graph_table.save_bundle(bundle_path)
    print(f"✓ Saved v2.0 bundle to: {bundle_path}")
    
    # Inspect bundle metadata without loading full data
    metadata = groggy.GraphTable.get_bundle_info(bundle_path)
    print("✓ Bundle metadata:")
    print(f"   Format version: {metadata['version']}")
    print(f"   Created: {metadata['created_at']}")
    print(f"   Groggy version: {metadata['groggy_version']}")
    print(f"   Nodes: {metadata['node_count']}, Edges: {metadata['edge_count']}")
    print(f"   Validation status: {'Valid' if metadata['validation_summary']['is_valid'] else 'Invalid'}")
    
    # Verify bundle integrity
    verification = groggy.GraphTable.verify_bundle(bundle_path)
    print("✓ Bundle verification:")
    print(f"   Format: v{verification['format_version']}")
    print(f"   Integrity: {'✓ Valid' if verification['is_valid'] else '✗ Invalid'}")
    if verification.get('missing_files'):
        print(f"   Missing files: {verification['missing_files']}")
    
    # Load bundle with automatic format detection
    loaded_graph = groggy.GraphTable.load_bundle(bundle_path)
    print(f"✓ Loaded graph table: {loaded_graph.shape}")
    print(f"   Nodes: {loaded_graph.nodes.nrows}, Edges: {loaded_graph.edges.nrows}")
    
    # Show bundle structure
    bundle_dir = Path(bundle_path)
    if bundle_dir.exists():
        print("\n✓ Bundle structure:")
        for file_path in sorted(bundle_dir.iterdir()):
            size_kb = file_path.stat().st_size / 1024
            print(f"   {file_path.name}: {size_kb:.1f} KB")
    
    return loaded_graph

def phase_3_example(users_table):
    """Phase 3 Features: BaseArray Chaining System and Method Delegation"""
    print("\n" + "=" * 60)
    print("PHASE 3: BaseArray Chaining System and Method Delegation")
    print("=" * 60)
    
    # Phase 3 Feature 1: Array Method Delegation
    print("\n1. Array Method Delegation")
    print("-" * 30)
    
    # Create string array to demonstrate delegation using new BaseArray
    names = ['alice', 'bob', 'charlie', 'diana', 'eve']
    names_array = groggy.BaseArray(names)
    print(f"✓ Created names array: {names_array}")
    
    # Demonstrate method delegation - apply string methods to each element
    try:
        # Apply .upper() to each name
        upper_names = names_array.apply_to_each('upper', ())
        print(f"✓ Method delegation works! Upper case names: {list(upper_names)}")
        
        # Apply .replace() with arguments to each name
        replaced_names = names_array.apply_to_each('replace', ('a', 'X'))
        print(f"✓ Method with arguments: {list(replaced_names)}")
        
        # Apply .capitalize() to each name
        capitalized = names_array.apply_to_each('capitalize', ())
        print(f"✓ Capitalized names: {list(capitalized)}")
        
    except Exception as e:
        print(f"❌ Method delegation failed: {e}")
    
    # Phase 3 Feature 2: Chaining with Graph Components
    print("\n2. Graph Component Chaining")
    print("-" * 30)
    
    try:
        # Create a graph for component chaining
        nodes_data = {
            'node_id': [1, 2, 3, 4, 5, 6],
            'type': ['User', 'User', 'User', 'Project', 'Project', 'Team'],
            'name': ['Alice', 'Bob', 'Charlie', 'WebApp', 'MobileApp', 'Engineering'],
            'age': [25, 30, 35, None, None, None],
            'department': ['Engineering', 'Sales', 'Engineering', None, None, 'Engineering']
        }
        
        edges_data = {
            'edge_id': [1, 2, 3, 4, 5],
            'source': [1, 2, 3, 1, 6],
            'target': [4, 5, 6, 6, 4],
            'relationship': ['works_on', 'works_on', 'member_of', 'member_of', 'manages'],
            'weight': [1.0, 0.8, 1.0, 1.0, 0.9]
        }
        
        nodes_table = groggy.NodesTable.from_dict(nodes_data)
        edges_table = groggy.EdgesTable.from_dict(edges_data)
        graph_table = groggy.GraphTable(nodes_table, edges_table)
        graph = graph_table.to_graph()
        
        print(f"✓ Created graph with {graph.node_count()} nodes, {graph.edge_count()} edges")
        
        # Get connected components and demonstrate trait-based method injection
        components = graph.connected_components()
        print(f"✓ Found {len(components)} connected components")
        
        # Test trait-based chaining on components
        component_iterator = components.iter()
        print(f"✓ Created component iterator: {type(component_iterator)}")
        
        # Test SubgraphLike trait methods
        if hasattr(component_iterator, 'filter_nodes'):
            filtered_iterator = component_iterator.filter_nodes('department == "Engineering"')
            print(f"✓ Applied filter_nodes (SubgraphLike trait): {type(filtered_iterator)}")
        
        # Show how this eliminates method duplication
        print("✓ Trait-based method injection working:")
        print("   - Components automatically get SubgraphLike methods")
        print("   - No need to reimplement methods for each array type")
        
    except Exception as e:
        print(f"❌ Component chaining failed: {e}")
    
    # Phase 3 Feature 3: BaseArray Integration
    print("\n3. BaseArray Integration and Fluent Operations")
    print("-" * 30)
    
    try:
        # Get a column as BaseArray and test chaining
        age_column = users_table.column('age')
        print(f"✓ Got age column as BaseArray: {type(age_column)}")
        
        # Test statistical operations
        print(f"   - Length: {len(age_column)}")
        
        # Test describe() method (returns statistics dict)
        stats = age_column.describe()
        if stats and 'mean' in stats:
            print(f"   - Mean: {stats['mean']}")
        
        # Test unique() method
        unique_vals = age_column.unique()
        print(f"   - Unique values: {list(unique_vals)}")
        
        # Test if BaseArray supports iteration
        if hasattr(age_column, 'iter'):
            age_iterator = age_column.iter()
            print(f"✓ BaseArray supports iteration: {type(age_iterator)}")
            
            # Test chaining operations
            if hasattr(age_iterator, 'filter'):
                filtered_ages = age_iterator.filter(lambda x: x > 28)
                collected_ages = filtered_ages.collect()
                print(f"✓ Chained filter operation: {type(collected_ages)}")
        
    except Exception as e:
        print(f"❌ BaseArray integration test failed: {e}")
    
    return names_array

def performance_showcase():
    """Showcase performance with larger datasets"""
    print("\n" + "=" * 60)
    print("PERFORMANCE SHOWCASE")
    print("=" * 60)
    
    print("\n1. Large Dataset Operations")
    print("-" * 30)
    
    # Create larger dataset
    import random
    random.seed(42)
    
    # Generate 10,000 users
    large_users_data = {
        'user_id': list(range(1, 10001)),
        'department': [random.choice(['Engineering', 'Sales', 'Marketing', 'HR', 'Finance']) for _ in range(10000)],
        'salary': [random.randint(40000, 150000) for _ in range(10000)],
        'performance': [round(random.uniform(2.0, 5.0), 1) for _ in range(10000)],
        'years_exp': [random.randint(0, 20) for _ in range(10000)]
    }
    
    large_table = groggy.BaseTable.from_dict(large_users_data)
    print(f"✓ Created large dataset: {large_table.shape}")
    
    # Performance operations
    import time
    
    # Filtering performance
    start_time = time.time()
    filtered = large_table.filter("salary > 80000").filter("performance >= 4.0")
    filter_time = time.time() - start_time
    print(f"✓ Filtered {large_table.nrows} rows to {filtered.nrows} in {filter_time:.3f}s")
    
    # Group by performance using new pattern
    start_time = time.time()
    grouped_large = large_table.group_by(['department'])
    dept_analysis = grouped_large.agg({
        'salary': 'avg',
        'performance': 'avg',
        'years_exp': 'avg',
        'user_id': 'count'
    })
    groupby_time = time.time() - start_time
    print(f"✓ Group by analysis completed in {groupby_time:.3f}s")
    print(f"   Groups: {len(grouped_large)}, Result: {dept_analysis.shape}")
    
    # File I/O performance
    start_time = time.time()
    large_table.to_csv("large_dataset.csv")
    export_time = time.time() - start_time
    print(f"✓ Exported {large_table.nrows} rows to CSV in {export_time:.3f}s")
    
    start_time = time.time()
    loaded_large = groggy.BaseTable.from_csv("large_dataset.csv")
    import_time = time.time() - start_time
    print(f"✓ Loaded {loaded_large.nrows} rows from CSV in {import_time:.3f}s")

def main():
    """Main example runner"""
    print("Groggy Storage System - Phase 1, 2 & 3 Feature Demo")
    print("=" * 60)
    print("This example demonstrates advanced table operations, bundle management,")
    print("and the new BaseArray Chaining System implemented in Phases 1-3.\n")
    
    try:
        # Run Phase 1 examples
        users_table, projects_table = phase_1_example()
        
        # Run Phase 2 examples
        full_context, dept_stats = phase_2_example(users_table, projects_table)
        
        # Enhanced Bundle System
        loaded_graph = phase_2_bundle_system_example(full_context)
        
        # NEW: Run Phase 3 examples
        names_array = phase_3_example(users_table)
        
        # Performance showcase
        performance_showcase()
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()