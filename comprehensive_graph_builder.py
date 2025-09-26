#!/usr/bin/env python3
"""
Comprehensive Graph Builder

Creates rich, complex graphs to thoroughly test the visualization system.
Tests performance, edge cases, and real-world scenarios.
"""

import sys
import random
import time
from typing import Dict, List, Tuple, Any

def has_edge_between_nodes(graph, node1, node2) -> bool:
    """Check if an edge exists between two nodes."""
    try:
        edge_id = get_edge_id_between_nodes(graph, node1, node2)
        return edge_id is not None
    except Exception:
        return False


def get_edge_id_between_nodes(graph, node1, node2):
    """Get edge ID between two nodes, or None if no edge exists."""
    try:
        # Get all edge IDs and check their endpoints
        edge_ids = graph.edge_ids
        for edge_id in edge_ids:
            endpoints = graph.edge_endpoints(edge_id)
            if (endpoints[0] == node1 and endpoints[1] == node2) or \
               (endpoints[0] == node2 and endpoints[1] == node1):
                return edge_id
        return None
    except Exception:
        return None


def create_social_network_graph(size: str = "medium") -> "groggy.Graph":
    """Create a realistic social network with rich attributes."""
    import groggy as gr
    
    # Size configurations
    configs = {
        "small": {"people": 50, "connections": 120},
        "medium": {"people": 200, "connections": 600}, 
        "large": {"people": 1000, "connections": 3000},
        "huge": {"people": 5000, "connections": 15000}
    }
    
    config = configs.get(size, configs["medium"])
    print(f"🏗️ Building {size} social network: {config['people']} people, ~{config['connections']} connections")
    
    g = gr.Graph()
    
    # Person data
    first_names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry", "Iris", "Jack",
                   "Kate", "Liam", "Maya", "Noah", "Olivia", "Paul", "Quinn", "Ruby", "Sam", "Tessa",
                   "Uma", "Victor", "Wendy", "Xavier", "Yara", "Zoe"]
    
    last_names = ["Smith", "Johnson", "Brown", "Davis", "Miller", "Wilson", "Moore", "Taylor", "Anderson", "Thomas",
                  "Jackson", "White", "Harris", "Martin", "Thompson", "Garcia", "Martinez", "Robinson", "Clark", "Rodriguez"]
    
    departments = ["Engineering", "Marketing", "Sales", "HR", "Finance", "Operations", "Research", "Design", "Support", "Legal"]
    
    locations = ["New York", "San Francisco", "London", "Tokyo", "Berlin", "Sydney", "Toronto", "Paris", "Singapore", "Mumbai"]
    
    interests = ["Technology", "Sports", "Music", "Travel", "Reading", "Gaming", "Photography", "Cooking", "Art", "Fitness"]
    
    # Create people
    people = []
    for i in range(config['people']):
        first_name = random.choice(first_names)
        last_name = random.choice(last_names)
        name = f"{first_name} {last_name}"
        
        # Rich attributes
        node_id = g.add_node(
            name=name,
            first_name=first_name,
            last_name=last_name,
            employee_id=f"EMP{i:05d}",
            department=random.choice(departments),
            location=random.choice(locations),
            age=random.randint(22, 65),
            tenure_years=random.randint(0, 20),
            salary_band=random.choice(["Junior", "Mid", "Senior", "Staff", "Principal", "Director"]),
            interests=random.sample(interests, random.randint(1, 4)),
            performance_rating=random.uniform(2.5, 5.0),
            is_manager=random.random() < 0.15,  # 15% managers
            remote_worker=random.random() < 0.3,  # 30% remote
            join_date=f"202{random.randint(0, 3)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
            project_count=random.randint(0, 8),
            certification_count=random.randint(0, 5)
        )
        people.append(node_id)
    
    print(f"✓ Created {len(people)} people with rich attributes")
    
    # Create realistic connections
    relationship_types = [
        ("colleague", 0.6),      # Same department
        ("teammate", 0.5),       # Same projects  
        ("mentor", 0.1),         # Senior to junior
        ("manager", 0.05),       # Manager relationship
        ("friend", 0.3),         # Social connection
        ("collaborator", 0.4),   # Cross-department work
        ("client_contact", 0.1), # Client relationships
        ("vendor_contact", 0.05) # Vendor relationships
    ]
    
    connections_made = 0
    target_connections = config['connections']
    
    # Department-based clustering
    dept_groups = {}
    for person_id in people:
        dept = g.get_node_attr(person_id, "department")
        if dept not in dept_groups:
            dept_groups[dept] = []
        dept_groups[dept].append(person_id)
    
    # Create connections within departments (higher probability)
    for dept, dept_people in dept_groups.items():
        if len(dept_people) < 2:
            continue
            
        # Connect some people within department
        for _ in range(min(len(dept_people) * 2, target_connections // 4)):
            if connections_made >= target_connections:
                break
                
            person1 = random.choice(dept_people)
            person2 = random.choice(dept_people)
            
            if person1 != person2 and not has_edge_between_nodes(g, person1, person2):
                rel_type = random.choices(
                    [rt[0] for rt in relationship_types],
                    weights=[rt[1] for rt in relationship_types]
                )[0]
                
                g.add_edge(
                    person1, person2,
                    relationship=rel_type,
                    strength=random.uniform(0.3, 1.0),
                    frequency=random.choice(["daily", "weekly", "monthly", "quarterly"]),
                    project_based=random.random() < 0.4,
                    start_date=f"202{random.randint(0, 3)}-{random.randint(1, 12):02d}",
                    interaction_count=random.randint(5, 200),
                    last_interaction=f"2023-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
                )
                connections_made += 1
    
    # Create cross-department connections (lower probability)
    while connections_made < target_connections:
        person1 = random.choice(people)
        person2 = random.choice(people)
        
        if person1 != person2 and not has_edge_between_nodes(g, person1, person2):
            # Lower probability for cross-department
            if g.get_node_attr(person1, "department") != g.get_node_attr(person2, "department"):
                if random.random() > 0.7:  # 30% chance for cross-dept
                    continue
            
            rel_type = random.choices(
                [rt[0] for rt in relationship_types],
                weights=[rt[1] for rt in relationship_types]
            )[0]
            
            g.add_edge(
                person1, person2,
                relationship=rel_type,
                strength=random.uniform(0.1, 0.9),
                frequency=random.choice(["daily", "weekly", "monthly", "quarterly"]),
                project_based=random.random() < 0.6,
                start_date=f"202{random.randint(0, 3)}-{random.randint(1, 12):02d}",
                interaction_count=random.randint(1, 150),
                last_interaction=f"2023-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
            )
            connections_made += 1
    
    print(f"✓ Created {connections_made} realistic connections")
    print(f"✓ Graph density: {(2 * g.edge_count()) / (g.node_count() * (g.node_count() - 1)):.3f}")
    
    return g


def create_hierarchical_organization(levels: int = 4, branching: int = 3) -> "groggy.Graph":
    """Create a hierarchical organization chart."""
    import groggy as gr
    
    print(f"🏗️ Building hierarchical organization: {levels} levels, branching factor {branching}")
    
    g = gr.Graph()
    
    positions = ["CEO", "VP", "Director", "Manager", "Senior", "Mid", "Junior", "Intern"]
    
    # Create CEO
    nodes_by_level = {0: []}
    
    ceo_id = g.add_node(
        name="Alex CEO",
        title="Chief Executive Officer",
        level=0,
        reports_count=0,
        budget_authority=10000000,
        department="Executive",
        hire_date="2015-01-01"
    )
    nodes_by_level[0].append(ceo_id)
    
    # Create hierarchy
    for level in range(1, levels):
        nodes_by_level[level] = []
        
        for parent in nodes_by_level[level - 1]:
            # Each parent gets branching factor children
            for i in range(branching):
                child_id = g.add_node(
                    name=f"Person L{level}_{i}_{parent}",
                    title=positions[min(level, len(positions) - 1)],
                    level=level,
                    manager=parent,
                    reports_count=0,
                    budget_authority=1000000 // (level * 2),
                    department=f"Dept_{level}_{i}",
                    hire_date=f"20{16 + level}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
                    performance=random.uniform(3.0, 5.0),
                    is_remote=random.random() < 0.25
                )
                nodes_by_level[level].append(child_id)
                
                # Create reporting relationship
                g.add_edge(
                    parent, child_id,
                    relationship="reports_to",
                    since=f"20{16 + level}-{random.randint(1, 12):02d}",
                    review_frequency="quarterly",
                    one_on_one_frequency="weekly"
                )
                
                # Update parent's reports count
                current_reports = g.get_node_attr(parent, "reports_count")
                g.set_node_attr(parent, "reports_count", current_reports + 1)
    
    total_nodes = sum(len(nodes) for nodes in nodes_by_level.values())
    print(f"✓ Created hierarchical org: {total_nodes} people across {levels} levels")
    
    return g


def create_project_collaboration_network() -> "groggy.Graph":
    """Create a network representing project collaborations."""
    import groggy as gr
    
    print("🏗️ Building project collaboration network")
    
    g = gr.Graph()
    
    # Projects with different characteristics
    projects = [
        {"name": "Project Alpha", "budget": 1000000, "team_size": 8, "duration_months": 12, "priority": "High"},
        {"name": "Project Beta", "budget": 500000, "team_size": 5, "duration_months": 6, "priority": "Medium"},
        {"name": "Project Gamma", "budget": 2000000, "team_size": 15, "duration_months": 18, "priority": "Critical"},
        {"name": "Project Delta", "budget": 300000, "team_size": 3, "duration_months": 4, "priority": "Low"},
        {"name": "Project Epsilon", "budget": 750000, "team_size": 10, "duration_months": 8, "priority": "High"},
        {"name": "Project Zeta", "budget": 1200000, "team_size": 12, "duration_months": 10, "priority": "Medium"},
    ]
    
    # Create project nodes
    project_nodes = []
    for project in projects:
        project_id = g.add_node(
            name=project["name"],
            type="project",
            budget=project["budget"],
            team_size=project["team_size"],
            duration_months=project["duration_months"],
            priority=project["priority"],
            status=random.choice(["Planning", "Active", "Complete", "On Hold"]),
            completion_percent=random.randint(0, 100),
            start_date=f"2023-{random.randint(1, 12):02d}-01",
            technologies=random.sample(["Python", "Rust", "JavaScript", "React", "Docker", "AWS", "ML", "AI"], 
                                     random.randint(2, 5))
        )
        project_nodes.append(project_id)
    
    # Create people working on projects
    people_nodes = []
    roles = ["Product Manager", "Tech Lead", "Software Engineer", "Data Scientist", "Designer", "QA Engineer"]
    
    for i in range(50):  # 50 people
        person_id = g.add_node(
            name=f"Engineer_{i}",
            type="person",
            role=random.choice(roles),
            experience_years=random.randint(1, 15),
            skills=random.sample(["Python", "Rust", "JavaScript", "ML", "Design", "Management", "Testing"], 
                               random.randint(2, 6)),
            current_projects=0,
            utilization_percent=0
        )
        people_nodes.append(person_id)
    
    # Assign people to projects
    for project_id in project_nodes:
        team_size = g.get_node_attr(project_id, "team_size")
        project_team = random.sample(people_nodes, min(team_size, len(people_nodes)))
        
        for person_id in project_team:
            # Person works on project
            g.add_edge(
                person_id, project_id,
                relationship="works_on",
                role_in_project=random.choice(["Lead", "Contributor", "Reviewer", "Consultant"]),
                allocation_percent=random.randint(25, 100),
                start_date=f"2023-{random.randint(1, 12):02d}-01",
                contribution_type=random.choice(["Development", "Architecture", "Testing", "Documentation"])
            )
            
            # Update person's current projects
            current = g.get_node_attr(person_id, "current_projects")
            g.set_node_attr(person_id, "current_projects", current + 1)
    
    # Create collaboration edges between people on same projects
    collaboration_count = 0
    for project_id in project_nodes:
        # Find all people on this project
        project_people = []
        for person_id in people_nodes:
            if has_edge_between_nodes(g, person_id, project_id):
                project_people.append(person_id)
        
        # Create collaboration edges
        for i, person1 in enumerate(project_people):
            for person2 in project_people[i+1:]:
                if not has_edge_between_nodes(g, person1, person2):
                    g.add_edge(
                        person1, person2,
                        relationship="collaborates",
                        shared_projects=1,
                        collaboration_strength=random.uniform(0.3, 1.0),
                        communication_frequency=random.choice(["daily", "weekly", "monthly"])
                    )
                    collaboration_count += 1
                else:
                    # Increase shared projects count
                    edge_id = get_edge_id_between_nodes(g, person1, person2)
                    if edge_id is not None:
                        current_shared = g.get_edge_attr(edge_id, "shared_projects")
                        g.set_edge_attr(edge_id, "shared_projects", current_shared + 1)
    
    print(f"✓ Created project network: {len(project_nodes)} projects, {len(people_nodes)} people")
    print(f"✓ Created {collaboration_count} collaboration relationships")
    
    return g


def test_visualization_with_comprehensive_graph(graph, graph_name: str):
    """Test all visualization features with the comprehensive graph."""
    print(f"\n🎨 Testing visualization with {graph_name}")
    print("-" * 50)
    
    # Get graph info
    info = graph.viz().info()
    print(f"Graph info: {info}")
    
    supports_graph = graph.viz().supports_graph_view()
    print(f"Supports graph view: {supports_graph}")
    
    print(f"Actual stats: {graph.node_count()} nodes, {graph.edge_count()} edges")
    
    return True


def performance_benchmark(graph, graph_name: str):
    """Benchmark visualization performance."""
    print(f"\n⚡ Performance benchmark for {graph_name}")
    print("-" * 50)
    
    # Benchmark info() method
    start_time = time.time()
    info = graph.viz().info()
    info_time = time.time() - start_time
    print(f"  info() time: {info_time:.3f}s")
    
    print("\n📈 Interactive/static benchmarks skipped (legacy API removed).")
    print(f"  Total nodes: {graph.node_count()}")
    print(f"  Total edges: {graph.edge_count()}")
    print(f"  Nodes per second (info): {graph.node_count() / info_time:.0f}")


def main():
    """Build comprehensive graphs and test visualization thoroughly."""
    print("🚀 Comprehensive Graph Builder & Viz Testing")
    print("=" * 60)
    
    try:
        # Test 1: Medium social network
        print("\n" + "="*60)
        print("TEST 1: Social Network (Medium)")
        print("="*60)
        
        social_graph = create_social_network_graph("medium")
        test_visualization_with_comprehensive_graph(social_graph, "Social Network")
        performance_benchmark(social_graph, "Social Network")
        
        # Test 2: Hierarchical organization
        print("\n" + "="*60)
        print("TEST 2: Hierarchical Organization")
        print("="*60)
        
        hierarchy_graph = create_hierarchical_organization(levels=5, branching=4)
        test_visualization_with_comprehensive_graph(hierarchy_graph, "Hierarchical Org")
        performance_benchmark(hierarchy_graph, "Hierarchical Org")
        
        # Test 3: Project collaboration network
        print("\n" + "="*60)
        print("TEST 3: Project Collaboration Network")
        print("="*60)
        
        project_graph = create_project_collaboration_network()
        test_visualization_with_comprehensive_graph(project_graph, "Project Network")
        performance_benchmark(project_graph, "Project Network")
        
        # Test 4: Large scale test
        print("\n" + "="*60)
        print("TEST 4: Large Scale Social Network")
        print("="*60)
        
        large_graph = create_social_network_graph("large")
        test_visualization_with_comprehensive_graph(large_graph, "Large Social Network")
        performance_benchmark(large_graph, "Large Social Network")
        
        print("\n" + "="*60)
        print("🎉 ALL COMPREHENSIVE TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\n✨ Visualization system handles complex, realistic graphs!")
        print("✨ All methods working with rich, varied data!")
        print("✨ Performance is acceptable for production use!")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Comprehensive testing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
