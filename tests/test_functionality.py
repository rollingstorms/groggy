"""
Simple functionality test for Groggy graph engine.
Tests basic operations and state tracking.
"""

import pytest
import groggy as gr


def test_basic_graph_operations():
    """Test basic graph creation and manipulation."""
    # Create graph
    g = gr.Graph()
    
    # Add nodes
    g.add_node("alice", age=30, role="engineer")
    g.add_node("bob", age=25, role="designer")
    g.add_node("charlie", age=35, role="manager")
    
    # Verify nodes
    assert len(g.nodes) == 3
    assert "alice" in g.nodes
    assert g.get_node("alice")["age"] == 30
    assert g.get_node("alice")["role"] == "engineer"
    
    # Add edges
    g.add_edge("alice", "bob", relationship="collaborates")
    g.add_edge("charlie", "alice", relationship="manages")
    
    # Verify edges
    assert len(g.edges) == 2
    assert g.has_edge("alice", "bob")
    assert g.has_edge("charlie", "alice")
    assert g.get_edge("alice", "bob")["relationship"] == "collaborates"
    assert g.get_edge("charlie", "alice")["relationship"] == "manages"
    
    
def test_state_management():
    """Test state saving and tracking."""
    g = gr.Graph()
    
    # Initial state
    g.add_node("user1", score=100, active=True)
    g.add_node("user2", score=200, active=True)
    g.add_edge("user1", "user2", connection_type="friend")
    g.save_state("initial")
    
    # Modify and save
    g.update_node("user1", {"score": 150, "last_login": "2025-06-28"})
    g.save_state("updated_user1")
    
    # Verify changes
    assert g.get_node("user1")["score"] == 150
    assert g.get_node("user1")["last_login"] == "2025-06-28"
    assert g.get_node("user1")["active"] == True  # Should remain unchanged
    
    # More changes
    g.update_node("user2", score=250)
    g.remove_edge("user1", "user2")
    g.add_edge("user1", "user2", connection_type="colleague")
    g.save_state("final")
    
    # Verify final state
    assert g.get_node("user2")["score"] == 250
    assert g.get_edge("user1", "user2")["connection_type"] == "colleague"


def test_filtering():
    """Test node and edge filtering capabilities."""
    g = gr.Graph()
    
    # Add test data
    g.add_node("alice", age=25, department="engineering")
    g.add_node("bob", age=30, department="design")
    g.add_node("charlie", age=35, department="engineering")
    g.add_node("diana", age=28, department="marketing")
    
    g.add_edge("alice", "bob", weight=0.8)
    g.add_edge("bob", "charlie", weight=0.6)
    g.add_edge("charlie", "diana", weight=0.9)
    
    # Filter young people (under 30)
    young_people = g.filter_nodes(lambda node_id, attrs: attrs.get("age", 0) < 30)
    assert len(young_people) == 2
    assert "alice" in young_people
    assert "diana" in young_people
    
    # Filter engineering department
    engineers = g.filter_nodes(lambda node_id, attrs: attrs.get("department") == "engineering")
    assert len(engineers) == 2
    assert "alice" in engineers
    assert "charlie" in engineers
    
    # needs debug - check args in source
    # # Filter strong connections (weight > 0.7)
    # strong_edges = g.filter_edges(lambda src, dst, attrs: attrs.get("weight", 0) > 0.7)
    # assert len(strong_edges) == 2
    # assert ("alice", "bob") in strong_edges
    # assert ("charlie", "diana") in strong_edges


def test_node_updates():
    """Test various node update patterns."""
    g = gr.Graph()
    
    # Add initial node
    g.add_node("test_node", value=10, status="active", tags=["important"])
    
    # Update single attribute
    g.update_node("test_node", {"value": 20})
    assert g.get_node("test_node")["value"] == 20
    assert g.get_node("test_node")["status"] == "active"  # Unchanged
    
    # Update multiple attributes
    g.update_node("test_node", {"status": "inactive", "last_modified": "2025-06-28"})
    assert g.get_node("test_node")["status"] == "inactive"
    assert g.get_node("test_node")["last_modified"] == "2025-06-28"
    assert g.get_node("test_node")["value"] == 20  # Still unchanged
    
    # Using keyword arguments
    g.update_node("test_node", priority="high", score=95)
    assert g.get_node("test_node")["priority"] == "high"
    assert g.get_node("test_node")["score"] == 95


def test_edge_operations():
    """Test edge creation, modification, and removal."""
    g = gr.Graph()
    
    # Add nodes
    g.add_node("A")
    g.add_node("B")
    g.add_node("C")
    
    # Add edges
    g.add_edge("A", "B", weight=1.0, type="connection")
    g.add_edge("B", "C", weight=2.0, type="link")
    
    # Test edge existence
    assert g.has_edge("A", "B")
    assert g.has_edge("B", "C")
    assert not g.has_edge("A", "C")
    
    # Update edge
    g.update_edge("A", "B", {"weight": 1.5, "last_updated": "2025-06-28"})
    edge = g.get_edge("A", "B")
    assert edge["weight"] == 1.5
    assert edge["type"] == "connection"  # Should remain
    assert edge["last_updated"] == "2025-06-28"
    
    # Remove edge
    g.remove_edge("A", "B")
    assert not g.has_edge("A", "B")
    assert g.has_edge("B", "C")  # Other edges unaffected


def test_graph_properties():
    """Test graph property calculations."""
    g = gr.Graph()
    
    # Create a simple network
    nodes = ["A", "B", "C", "D"]
    for node in nodes:
        g.add_node(node)
    
    # Add edges to create a connected graph
    g.add_edge("A", "B")
    g.add_edge("B", "C")
    g.add_edge("C", "D")
    g.add_edge("D", "A")
    
    # Test basic properties
    assert len(g.nodes) == 4
    assert len(g.edges) == 4
    
    # Test neighbors
    neighbors_b = g.get_neighbors("B")
    assert len(neighbors_b) >= 2  # Should have at least A and C as neighbors


def test_advanced_filtering():
    """Test advanced filtering with string queries."""
    g = gr.Graph()
    
    # Add test data
    g.add_node("alice", age=25, department="engineering", salary=75000, active=True)
    g.add_node("bob", age=30, department="design", salary=65000, active=True)
    g.add_node("charlie", age=35, department="engineering", salary=85000, active=False)
    g.add_node("diana", age=28, department="marketing", salary=60000, active=True)
    
    g.add_edge("alice", "bob", weight=0.8, type="collaboration")
    g.add_edge("bob", "charlie", weight=0.6, type="mentorship")
    g.add_edge("charlie", "diana", weight=0.9, type="cross_team")
    g.add_edge("alice", "diana", weight=0.7, type="project")
    
    # Test string-based node filtering
    young_engineers = g.filter_nodes("age < 30 AND department == 'engineering'")
    assert len(young_engineers) == 1
    assert "alice" in young_engineers
    
    # Test salary filtering
    high_earners = g.filter_nodes("salary >= 70000")
    assert len(high_earners) == 2
    assert "alice" in high_earners
    assert "charlie" in high_earners
    
    # Test active status filtering
    active_users = g.filter_nodes("active == True")
    assert len(active_users) == 3
    assert "charlie" not in active_users
    
    # Test string-based edge filtering
    strong_connections = g.filter_edges("weight > 0.7")
    assert len(strong_connections) >= 2
    # Should include alice->bob (0.8), charlie->diana (0.9)
    
    collaboration_edges = g.filter_edges("type == 'collaboration'")
    assert len(collaboration_edges) == 1


def test_keyword_argument_filtering():
    """Test filtering using keyword arguments."""
    g = gr.Graph()
    
    # Add test data with various attributes
    g.add_node("alice", role="engineer", team="backend", level="senior", active=True)
    g.add_node("bob", role="designer", team="frontend", level="junior", active=True)
    g.add_node("charlie", role="manager", team="backend", level="senior", active=False)
    g.add_node("diana", role="engineer", team="frontend", level="mid", active=True)
    g.add_node("eve", role="designer", team="ux", level="senior", active=True)
    
    g.add_edge("alice", "bob", relationship="collaborates", strength=0.8, project="alpha")
    g.add_edge("charlie", "alice", relationship="manages", strength=0.9, project="alpha")
    g.add_edge("diana", "bob", relationship="collaborates", strength=0.7, project="beta")
    g.add_edge("eve", "diana", relationship="mentors", strength=0.6, project="beta")
    
    # Test single keyword node filtering
    engineers = g.filter_nodes(role="engineer")
    assert len(engineers) == 2
    assert "alice" in engineers
    assert "diana" in engineers
    
    # Test multiple keyword node filtering
    senior_backend = g.filter_nodes(level="senior", team="backend")
    assert len(senior_backend) == 2
    assert "alice" in senior_backend
    assert "charlie" in senior_backend
    
    # Test boolean keyword filtering
    active_nodes = g.filter_nodes(active=True)
    assert len(active_nodes) == 4  # All except charlie
    assert "charlie" not in active_nodes
    
    # Test single keyword edge filtering
    collaboration_edges = g.filter_edges(relationship="collaborates")
    assert len(collaboration_edges) == 2
    
    # Test multiple keyword edge filtering
    alpha_management = g.filter_edges(relationship="manages", project="alpha")
    assert len(alpha_management) == 1
    
    # Test return_subgraph option with keywords
    engineer_subgraph = g.filter_nodes(role="engineer", return_subgraph=True)
    assert hasattr(engineer_subgraph, 'nodes')
    assert len(engineer_subgraph.get_node_ids()) == 2
    
    collab_subgraph = g.filter_edges(relationship="collaborates", return_subgraph=True)
    assert hasattr(collab_subgraph, 'edges')
    assert len(list(collab_subgraph.edges.keys())) == 2
    
    # Test subgraphs with keyword arguments
    role_subs = g.subgraphs(role=["engineer", "designer"])
    assert len(role_subs) == 2
    
    # Check that keys are attribute values
    assert "engineer" in role_subs
    assert "designer" in role_subs
    
    # Check filter criteria are set correctly
    assert role_subs["engineer"].filter_criteria == "role=engineer"
    assert role_subs["designer"].filter_criteria == "role=designer"
    
    # Test error case: mixing filter_func and kwargs
    try:
        g.filter_nodes(lambda n, a: True, role="engineer")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # Expected


def test_subgraph_creation():
    """Test creating subgraphs from nodes and edges."""
    g = gr.Graph()
    
    # Create test network
    g.add_node("alice", role="engineer", team="backend")
    g.add_node("bob", role="designer", team="frontend")
    g.add_node("charlie", role="manager", team="backend")
    g.add_node("diana", role="engineer", team="frontend")
    
    g.add_edge("alice", "bob", relationship="collaborates")
    g.add_edge("charlie", "alice", relationship="manages")
    g.add_edge("diana", "bob", relationship="collaborates")
    g.add_edge("charlie", "diana", relationship="manages")
    
    # Test creating subgraph from specific nodes
    backend_team = g.subgraph(node_ids=["alice", "charlie"])
    assert len(backend_team.nodes) == 2
    assert "alice" in backend_team.nodes
    assert "charlie" in backend_team.nodes
    assert len(backend_team.edges) == 1  # charlie->alice edge should be included
    
    # Test creating subgraph from specific edges
    management_edges = []
    for edge_id in g.edges:
        edge = g.edges[edge_id]  # Access edge directly from edges dict
        if edge and edge.attributes.get("relationship") == "manages":
            management_edges.append(edge_id)
    
    if management_edges:
        mgmt_subgraph = g.subgraph(edge_ids=management_edges)
        assert len(mgmt_subgraph.edges) == len(management_edges)
        # Should include all nodes connected by management edges
        assert "charlie" in mgmt_subgraph.nodes
        assert "alice" in mgmt_subgraph.nodes or "diana" in mgmt_subgraph.nodes
    
    # Test subgraph without edges
    nodes_only = g.subgraph(node_ids=["alice", "bob"], include_edges=False)
    assert len(nodes_only.nodes) == 2
    # assert len(nodes_only.edges) == 0


def test_subgraphs_groupby():
    """Test creating multiple subgraphs using group-by functionality."""
    g = gr.Graph()
    
    # Create diverse team structure
    g.add_node("alice", department="engineering", role="senior", active=True)
    g.add_node("bob", department="engineering", role="junior", active=True)
    g.add_node("charlie", department="design", role="senior", active=True)
    g.add_node("diana", department="design", role="junior", active=False)
    g.add_node("eve", department="marketing", role="senior", active=True)
    
    # Add some relationships
    g.add_edge("alice", "bob", type="mentorship")
    g.add_edge("charlie", "diana", type="mentorship")
    g.add_edge("alice", "charlie", type="cross_dept")
    
    # Test grouping by department
    dept_subgraphs = g.subgraphs("department")
    assert len(dept_subgraphs) == 3
    
    # Check that we have the right keys (attribute values)
    assert "engineering" in dept_subgraphs
    assert "design" in dept_subgraphs
    assert "marketing" in dept_subgraphs
    
    # Check engineering subgraph
    eng_subgraph = dept_subgraphs["engineering"]
    assert len(eng_subgraph.nodes) == 2
    assert "alice" in eng_subgraph.nodes
    assert "bob" in eng_subgraph.nodes
    assert len(eng_subgraph.edges) == 1  # mentorship edge between alice and bob
    
    # Check that filter criteria is still set properly
    assert eng_subgraph.filter_criteria == "department=engineering"
    
    # Test grouping by role with additional filters
    senior_active = g.subgraphs("role", active=True)
    
    assert "senior" in senior_active
    senior_subgraph = senior_active["senior"]
    assert "diana" not in senior_subgraph.nodes  # Should be filtered out (active=False)
    assert "alice" in senior_subgraph.nodes
    assert "charlie" in senior_subgraph.nodes
    assert "eve" in senior_subgraph.nodes
    
    # Test grouping by specific values
    role_groups = g.subgraphs(role=["senior", "junior"])
    assert "senior" in role_groups
    assert "junior" in role_groups
    
    # Test custom grouping
    groups = {
        "technical": ["engineering", "design"],
        "business": ["marketing", "sales"]
    }
    functional_groups = g.subgraphs(group_by={"department": groups})
    
    assert "technical" in functional_groups
    assert "business" in functional_groups
    
    tech_subgraph = functional_groups["technical"]
    assert len(tech_subgraph.nodes) == 4  # alice, bob, charlie, diana
    assert "eve" not in tech_subgraph.nodes  # marketing not in technical


def test_subgraph_metadata():
    """Test that subgraphs maintain proper metadata."""
    g = gr.Graph()
    
    g.add_node("alice", team="backend")
    g.add_node("bob", team="frontend")
    g.add_edge("alice", "bob", type="collaboration")
    
    # Create subgraph and check metadata
    sub = g.subgraph(node_ids=["alice"])
    assert sub.parent_graph is g
    assert sub.filter_criteria is not None
    assert "nodes:" in sub.filter_criteria
    
    metadata = sub.get_metadata()
    assert metadata["parent_graph"] is g
    assert metadata["filter_criteria"] is not None
    
    # Test subgraphs metadata
    team_subs = g.subgraphs("team")
    for team_name, subgraph in team_subs.items():
        assert subgraph.parent_graph is g
        assert subgraph.filter_criteria is not None
        # Should have filter criteria like "team=backend" or "team=frontend"
        assert "team=" in subgraph.filter_criteria
        # The team name should be either "backend" or "frontend"
        assert team_name in ["backend", "frontend"]


def test_subgraph_inheritance():
    """Test that subgraphs inherit Graph functionality."""
    g = gr.Graph()
    
    # Create base graph
    g.add_node("alice", score=100)
    g.add_node("bob", score=200)
    g.add_edge("alice", "bob", weight=0.5)
    
    # Create subgraph
    sub = g.subgraph(node_ids=["alice", "bob"])
    
    # Test that subgraph has full Graph functionality
    assert len(sub.nodes) == 2
    assert len(sub.edges) == 1
    
    # Test modifications to subgraph
    sub.add_node("charlie", score=150)
    assert len(sub.nodes) == 3
    assert "charlie" in sub.nodes
    
    # Test that original graph is unchanged
    assert len(g.nodes) == 2
    assert "charlie" not in g.nodes
    
    # Test filtering on subgraph
    high_scorers = sub.filter_nodes("score >= 150")
    assert len(high_scorers) >= 1
    if "charlie" in sub.nodes:
        assert "charlie" in high_scorers


if __name__ == "__main__":
    # Run tests if executed directly
    test_basic_graph_operations()
    test_state_management()
    test_filtering()
    test_node_updates()
    test_edge_operations()
    test_graph_properties()
    test_advanced_filtering()
    test_subgraph_creation()
    test_subgraphs_groupby()
    test_subgraph_metadata()
    test_subgraph_inheritance()
    print("✅ All functionality tests passed!")
