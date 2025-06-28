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


if __name__ == "__main__":
    # Run tests if executed directly
    test_basic_graph_operations()
    test_state_management()
    test_filtering()
    test_node_updates()
    test_edge_operations()
    test_graph_properties()
    print("✅ All functionality tests passed!")
