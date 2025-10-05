#!/usr/bin/env python3
"""
Demo of the pandas-like ArrayArray GroupBy API in Groggy

This demonstrates the new functionality:
- g.nodes.group_by('column')['attribute'].aggregation()
- Multiple aggregations with .agg()
- Direct chaining syntax
"""

import groggy as gr

def demo_basic_groupby():
    """Basic pandas-like groupby syntax"""
    print("=" * 60)
    print("Demo: Basic GroupBy with Aggregation")
    print("=" * 60)

    # Create a simple organizational graph
    g = gr.Graph()

    # Add employees with department and salary
    employees = [
        ("Alice", "Engineering", 120000),
        ("Bob", "Engineering", 110000),
        ("Carol", "Sales", 95000),
        ("Dave", "Sales", 105000),
        ("Eve", "Engineering", 130000),
        ("Frank", "Sales", 100000),
    ]

    for name, dept, salary in employees:
        g.add_node(name=name, department=dept, salary=salary)

    print(f"\nCreated graph with {g.node_count()} nodes")
    print("\nEmployees by department:")
    for name, dept, salary in employees:
        print(f"  {name:10s} | {dept:12s} | ${salary:,}")

    # Group by department and calculate mean salary
    print("\n--- Calculating mean salary by department ---")

    dept_groups = g.nodes.group_by("department")
    print(f"Created {len(dept_groups)} groups")

    salary_arrays = dept_groups["salary"]
    print(f"Extracted salary arrays: {salary_arrays}")

    mean_salaries = salary_arrays.mean()
    print(f"\nMean salaries by department:")
    print(mean_salaries)

    # Chained syntax (pandas-like)
    print("\n--- Same thing with chained syntax ---")
    result = g.nodes.group_by("department")["salary"].mean()
    print(result)


def demo_multi_aggregation():
    """Multiple aggregations at once with .agg()"""
    print("\n\n" + "=" * 60)
    print("Demo: Multiple Aggregations with .agg()")
    print("=" * 60)

    # Create a graph with project data
    g = gr.Graph()

    projects = [
        ("ProjectA", "Engineering", 8.5, 12),
        ("ProjectB", "Engineering", 9.2, 10),
        ("ProjectC", "Engineering", 7.8, 15),
        ("ProjectD", "Sales", 9.5, 8),
        ("ProjectE", "Sales", 8.8, 9),
        ("ProjectF", "Marketing", 9.0, 11),
        ("ProjectG", "Marketing", 8.5, 10),
    ]

    for name, dept, score, weeks in projects:
        g.add_node(project=name, department=dept, score=score, duration_weeks=weeks)

    print(f"\nCreated graph with {g.node_count()} projects")

    # Multi-aggregation: mean score, total duration, project count per department
    print("\n--- Applying multiple aggregations ---")

    score_arrays = g.nodes.group_by("department")["score"]

    stats = score_arrays.agg({
        "mean_score": "mean",
        "min_score": "min",
        "max_score": "max",
        "project_count": "count",
    })

    print("\nDepartment Statistics:")
    print(stats)

    # Also get duration stats
    print("\n--- Duration statistics ---")
    duration_arrays = g.nodes.group_by("department")["duration_weeks"]
    duration_stats = duration_arrays.agg({
        "avg_duration": "mean",
        "total_duration": "sum",
    })
    print(duration_stats)


def demo_all_aggregations():
    """Show all available aggregation methods"""
    print("\n\n" + "=" * 60)
    print("Demo: All Aggregation Methods")
    print("=" * 60)

    g = gr.Graph()

    # Create data with clear patterns
    values = [
        ("Group A", [1, 2, 3, 4, 5]),      # mean=3, sum=15, min=1, max=5, std≈1.58
        ("Group B", [10, 20, 30, 40, 50]), # mean=30, sum=150, min=10, max=50, std≈15.8
    ]

    for group_name, vals in values:
        for val in vals:
            g.add_node(group=group_name, value=val)

    print(f"\nCreated graph with {g.node_count()} nodes")
    print("Data:")
    for group_name, vals in values:
        print(f"  {group_name}: {vals}")

    # Extract arrays and show all aggregations
    value_arrays = g.nodes.group_by("group")["value"]

    print("\n--- All Aggregation Methods ---")
    print(f"mean():  {value_arrays.mean()}")
    print(f"sum():   {value_arrays.sum()}")
    print(f"min():   {value_arrays.min()}")
    print(f"max():   {value_arrays.max()}")
    print(f"std():   {value_arrays.std()}")
    print(f"count(): {value_arrays.count()}")


def demo_real_world_example():
    """A more realistic example with network analysis"""
    print("\n\n" + "=" * 60)
    print("Demo: Real-World Network Analysis")
    print("=" * 60)

    # Create a social network with influence scores
    g = gr.Graph()

    people = [
        # (name, community, influence_score, connections)
        ("Alice", "Tech", 85, 12),
        ("Bob", "Tech", 92, 15),
        ("Carol", "Tech", 78, 10),
        ("Dave", "Business", 88, 20),
        ("Eve", "Business", 95, 18),
        ("Frank", "Business", 82, 16),
        ("Grace", "Creative", 90, 14),
        ("Henry", "Creative", 87, 13),
    ]

    nodes = {}
    for name, community, influence, connections in people:
        node_id = g.add_node(
            name=name,
            community=community,
            influence_score=influence,
            connection_count=connections
        )
        nodes[name] = node_id

    # Add some connections
    g.add_edge(nodes["Alice"], nodes["Bob"])
    g.add_edge(nodes["Bob"], nodes["Carol"])
    g.add_edge(nodes["Dave"], nodes["Eve"])
    g.add_edge(nodes["Eve"], nodes["Frank"])
    g.add_edge(nodes["Grace"], nodes["Henry"])

    print(f"\nCreated social network with {g.node_count()} people and {g.edge_count()} connections")

    # Analyze by community
    print("\n--- Community Analysis ---")

    # Get comprehensive stats per community
    community_groups = g.nodes.group_by("community")

    influence_arrays = community_groups["influence_score"]
    influence_stats = influence_arrays.agg({
        "avg_influence": "mean",
        "top_influence": "max",
        "member_count": "count",
    })

    print("\nInfluence Statistics by Community:")
    print(influence_stats)

    connection_arrays = community_groups["connection_count"]
    connection_stats = connection_arrays.agg({
        "avg_connections": "mean",
        "total_connections": "sum",
    })

    print("\nConnection Statistics by Community:")
    print(connection_stats)

    print("\n--- Finding High-Influence Communities ---")
    mean_influence = influence_arrays.mean()
    print(f"Mean influence scores:\n{mean_influence}")


if __name__ == "__main__":
    print("\n")
    print("*" * 60)
    print("Groggy ArrayArray API Demo")
    print("Pandas-like GroupBy Operations for Graph Data")
    print("*" * 60)

    demo_basic_groupby()
    demo_multi_aggregation()
    demo_all_aggregations()
    demo_real_world_example()

    print("\n\n" + "*" * 60)
    print("Demo Complete!")
    print("*" * 60)
    print("\nKey Features Demonstrated:")
    print("  ✓ g.nodes.group_by('column')['attribute'].mean()")
    print("  ✓ Multiple aggregations with .agg()")
    print("  ✓ All aggregation methods: mean, sum, min, max, std, count")
    print("  ✓ Automatic packaging with group keys into BaseTable")
    print("  ✓ Chainable, pandas-like syntax")
    print()
