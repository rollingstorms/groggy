"""
Clean state analysis functions for groggy.

These functions analyze actual saved states in the graph storage,
not in-memory tracking. They provide historical delta analysis.

This replaces the old in-memory StateTracker system which was:
- Redundant (Rust backend already stores deltas)
- Memory intensive (duplicated data)
- Complex (manual state sync required)
- Error-prone (easy to get out of sync)

New approach: Direct analysis of stored state deltas.
"""

from typing import Optional, Dict, List, Any, Union
import json
from datetime import datetime

def show_changes(graph_or_entity, graph=None, branch_name: str = None):
    """
    Analyze changes across saved states.
    
    Usage:
        show_changes(g)                          # All changes in graph, main branch
        show_changes(g, branch_name='feature')   # All changes in specific branch
        show_changes('alice', g)                 # Entity changes in current branch
        show_changes('alice', g, branch_name='feature') # Entity changes in specific branch
    """
    # determine the current branch of the graph
    if branch_name is None:
        if hasattr(graph, 'current_branch'):
            branch_name = graph.current_branch
        else:
            branch_name = 'main'

    # Determine if first arg is entity or graph
    if isinstance(graph_or_entity, str):
        # Entity analysis
        entity_id = graph_or_entity
        target_graph = graph
        if target_graph is None:
            raise ValueError("Graph must be provided when analyzing entity")
        _show_entity_changes(entity_id, target_graph, branch_name)
    else:
        # Graph analysis
        target_graph = graph_or_entity
        _show_graph_changes(target_graph, branch_name)

def _show_graph_changes(graph, branch_name: str):
    """Show all changes across states in a branch."""
    print(f"=== Changes Analysis for Branch '{branch_name}' ===")
    
    if not hasattr(graph, '_rust_store') or not graph.use_rust:
        print("❌ State analysis only supported with Rust backend")
        return
    
    # Get all branches
    branches = graph.branches
    if branch_name not in branches:
        print(f"❌ Branch '{branch_name}' not found. Available: {list(branches.keys())}")
        return
    
    # Switch to the target branch to analyze its states
    original_branch = graph.current_branch
    graph.switch_branch(branch_name)
    
    try:
        # Get all saved states for this branch
        saved_states = _get_saved_states(graph)
        
        if not saved_states:
            print(f"No saved states found in branch '{branch_name}'")
            return
            
        print(f"Found {len(saved_states)} saved states:")
        
        for i, state_info in enumerate(saved_states):
            state_hash = state_info.get('hash', 'unknown')
            metadata = state_info.get('metadata', {})
            description = metadata.get('description', 'No description')
            timestamp = state_info.get('timestamp', 'unknown')
            
            print(f"\n{i+1}. State {state_hash[:8]}... ({description})")
            print(f"   Timestamp: {timestamp}")
            
            # Analyze changes between consecutive states
            if i > 0:
                prev_state = saved_states[i-1]
                changes = _analyze_state_diff(graph, prev_state['hash'], state_hash)
                _print_changes(changes)
    
    finally:
        # Restore original branch
        if original_branch != branch_name:
            graph.switch_branch(original_branch)

def _show_entity_changes(entity_id: str, graph, branch_name: str):
    """Show changes for a specific entity across states."""
    print(f"=== Changes for Entity '{entity_id}' in Branch '{branch_name}' ===")
    
    if not hasattr(graph, '_rust_store') or not graph.use_rust:
        print("❌ State analysis only supported with Rust backend")
        return
    
    # Get all branches
    branches = graph.branches
    if branch_name not in branches:
        print(f"❌ Branch '{branch_name}' not found. Available: {list(branches.keys())}")
        return
    
    # Switch to target branch
    original_branch = graph.current_branch
    graph.switch_branch(branch_name)
    
    try:
        saved_states = _get_saved_states(graph)
        
        if not saved_states:
            print(f"No saved states found in branch '{branch_name}'")
            return
        
        entity_timeline = []
        
        # Track entity across all states by actually loading and comparing them
        for i, state_info in enumerate(saved_states):
            state_hash = state_info['hash']
            
            # Save current state to restore later
            current_state_backup = None
            if hasattr(graph, 'current_hash') and graph.current_hash:
                current_state_backup = graph.current_hash
            
            try:
                # Load this specific state
                success = graph.load_state(state_hash)
                if not success:
                    print(f"Warning: Could not load state {state_hash[:8]}...")
                    continue
                
                # Check if entity exists in this state
                entity_data = None
                if entity_id in graph.nodes:
                    node = graph.nodes[entity_id]
                    entity_data = {
                        'type': 'node',
                        'data': dict(node.attributes)
                    }
                elif entity_id in graph.edges:
                    edge = graph.edges[entity_id]
                    entity_data = {
                        'type': 'edge', 
                        'data': {
                            'source': edge.source,
                            'target': edge.target,
                            **edge.attributes
                        }
                    }
                
                timeline_entry = {
                    'state': state_info,
                    'exists': entity_data is not None,
                    'data': entity_data
                }
                
                entity_timeline.append(timeline_entry)
                
            except Exception as e:
                print(f"Error loading state {state_hash[:8]}...: {e}")
                # Add entry showing we couldn't load this state
                timeline_entry = {
                    'state': state_info,
                    'exists': False,
                    'data': None,
                    'error': str(e)
                }
                entity_timeline.append(timeline_entry)
            
            finally:
                # Restore original state if we had one
                if current_state_backup:
                    graph.load_state(current_state_backup)
        
        # Analyze timeline
        _print_entity_timeline(entity_id, entity_timeline)
        
    finally:
        # Restore original branch
        if original_branch != branch_name:
            graph.switch_branch(original_branch)

def _get_saved_states(graph) -> List[Dict]:
    """Get list of saved states from the graph storage.""" 
    
    # Use auto_states if available
    if hasattr(graph, 'auto_states') and graph.auto_states:
        states = []
        for i, state_hash in enumerate(graph.auto_states):
            states.append({
                'hash': state_hash,
                'timestamp': datetime.now().isoformat(),
                'metadata': {'description': f'State {i+1}'}
            })
        return states
    
    # Fallback: no saved states
    return []

def _analyze_state_diff(graph, prev_hash: str, curr_hash: str) -> Dict:
    """Analyze differences between two states by loading and comparing them."""
    
    # Save current state to restore later
    current_state_backup = None
    if hasattr(graph, 'current_hash') and graph.current_hash:
        current_state_backup = graph.current_hash
    
    try:
        # Load previous state and capture its data
        prev_success = graph.load_state(prev_hash)
        if not prev_success:
            print(f"Warning: Could not load previous state {prev_hash[:8]}...")
            return _empty_diff()
        
        prev_nodes = {}
        prev_edges = {}
        
        # Capture previous state nodes
        for node_id in graph.nodes:
            node = graph.nodes[node_id]
            prev_nodes[node_id] = dict(node.attributes)
        
        # Capture previous state edges
        for edge_id in graph.edges:
            edge = graph.edges[edge_id]
            prev_edges[edge_id] = {
                'source': edge.source,
                'target': edge.target,
                **edge.attributes
            }
        
        # Load current state and capture its data
        curr_success = graph.load_state(curr_hash)
        if not curr_success:
            print(f"Warning: Could not load current state {curr_hash[:8]}...")
            return _empty_diff()
        
        curr_nodes = {}
        curr_edges = {}
        
        # Capture current state nodes
        for node_id in graph.nodes:
            node = graph.nodes[node_id]
            curr_nodes[node_id] = dict(node.attributes)
        
        # Capture current state edges
        for edge_id in graph.edges:
            edge = graph.edges[edge_id]
            curr_edges[edge_id] = {
                'source': edge.source,
                'target': edge.target,
                **edge.attributes
            }
        
        # Compare and find differences
        return _compare_graph_states(prev_nodes, prev_edges, curr_nodes, curr_edges)
    
    except Exception as e:
        print(f"Error analyzing state diff: {e}")
        return _empty_diff()
    
    finally:
        # Restore original state if we had one
        if current_state_backup:
            try:
                graph.load_state(current_state_backup)
            except Exception as e:
                print(f"Warning: Could not restore original state: {e}")

def _empty_diff() -> Dict:
    """Return an empty diff structure."""
    return {
        'nodes_added': [],
        'nodes_removed': [],
        'nodes_modified': [],
        'edges_added': [],
        'edges_removed': [],
        'edges_modified': []
    }

def _compare_graph_states(prev_nodes: Dict, prev_edges: Dict, curr_nodes: Dict, curr_edges: Dict) -> Dict:
    """Compare two graph states and return the differences."""
    
    # Node changes
    prev_node_ids = set(prev_nodes.keys())
    curr_node_ids = set(curr_nodes.keys())
    
    nodes_added = list(curr_node_ids - prev_node_ids)
    nodes_removed = list(prev_node_ids - curr_node_ids)
    nodes_modified = []
    
    # Check for modified nodes (same ID, different attributes)
    for node_id in prev_node_ids & curr_node_ids:
        if prev_nodes[node_id] != curr_nodes[node_id]:
            nodes_modified.append(node_id)
    
    # Edge changes
    prev_edge_ids = set(prev_edges.keys())
    curr_edge_ids = set(curr_edges.keys())
    
    edges_added = list(curr_edge_ids - prev_edge_ids)
    edges_removed = list(prev_edge_ids - curr_edge_ids)
    edges_modified = []
    
    # Check for modified edges (same ID, different attributes or endpoints)
    for edge_id in prev_edge_ids & curr_edge_ids:
        if prev_edges[edge_id] != curr_edges[edge_id]:
            edges_modified.append(edge_id)
    
    return {
        'nodes_added': sorted(nodes_added),
        'nodes_removed': sorted(nodes_removed),
        'nodes_modified': sorted(nodes_modified),
        'edges_added': sorted(edges_added),
        'edges_removed': sorted(edges_removed),
        'edges_modified': sorted(edges_modified)
    }

def _print_changes(changes: Dict):
    """Print formatted change summary."""
    total_changes = sum(len(v) for v in changes.values() if isinstance(v, list))
    
    if total_changes == 0:
        print("   No changes detected")
        return
        
    print(f"   Changes ({total_changes} total):")
    
    for change_type, items in changes.items():
        if items:
            action = change_type.replace('_', ' ').title()
            print(f"     {action}: {items}")

def _print_entity_timeline(entity_id: str, timeline: List[Dict]):
    """Print formatted entity timeline."""
    print(f"\nTimeline for '{entity_id}':")
    
    for i, entry in enumerate(timeline):
        state_info = entry['state']
        exists = entry['exists']
        data = entry['data']
        
        state_hash = state_info['hash'][:8]
        description = state_info['metadata'].get('description', 'No description')
        
        if exists:
            if i == 0:
                print(f"  {i+1}. {state_hash}... - CREATED ({description})")
                print(f"     Type: {data['type']}")
                print(f"     Data: {data['data']}")
            else:
                # Compare with previous state
                prev_entry = timeline[i-1]
                if not prev_entry['exists']:
                    print(f"  {i+1}. {state_hash}... - CREATED ({description})")
                    print(f"     Type: {data['type']}")
                    print(f"     Data: {data['data']}")
                elif prev_entry['data']['data'] != data['data']:
                    print(f"  {i+1}. {state_hash}... - MODIFIED ({description})")
                    print(f"     From: {prev_entry['data']['data']}")
                    print(f"     To:   {data['data']}")
                else:
                    print(f"  {i+1}. {state_hash}... - NO CHANGE ({description})")
        else:
            if i > 0 and timeline[i-1]['exists']:
                print(f"  {i+1}. {state_hash}... - DELETED ({description})")
            # If it doesn't exist and didn't exist before, skip

# Convenience aliases
def show_entity_changes(entity_id: str, graph, branch_name: str = 'main'):
    """Convenience function for entity-specific analysis."""
    show_changes(entity_id, graph, branch_name)

def show_graph_changes(graph, branch_name: str = 'main'):
    """Convenience function for graph-wide analysis."""
    show_changes(graph, branch_name)

def track_attribute_changes(graph, attribute_name: str, branch_name: str = 'main'):
    """Track changes to a specific attribute across all entities over time
    
    Args:
        graph: The graph to analyze
        attribute_name: Name of the attribute to track
        branch_name: Branch to analyze (default: 'main')
    """
    print(f"=== Tracking '{attribute_name}' Changes in Branch '{branch_name}' ===")
    
    if not hasattr(graph, '_rust_store') or not graph.use_rust:
        print("❌ Attribute tracking only supported with Rust backend")
        return
    
    # Get all saved states
    original_branch = graph.current_branch
    graph.switch_branch(branch_name)
    
    try:
        saved_states = _get_saved_states(graph)
        
        if not saved_states:
            print(f"No saved states found in branch '{branch_name}'")
            return
        
        # Track attribute changes across states
        attribute_timeline = {}
        
        for i, state_info in enumerate(saved_states):
            state_hash = state_info['hash']
            
            # Save current state to restore later
            current_state_backup = graph.current_hash
            
            try:
                # Load this specific state
                success = graph.load_state(state_hash)
                if not success:
                    continue
                
                # Collect all entities with this attribute in this state
                entities_with_attr = {}
                
                # Check nodes
                for node_id in graph.nodes:
                    node = graph.nodes[node_id]
                    if attribute_name in node.attributes:
                        entities_with_attr[f"node:{node_id}"] = node.attributes[attribute_name]
                
                # Check edges  
                for edge_id in graph.edges:
                    edge = graph.edges[edge_id]
                    if attribute_name in edge.attributes:
                        entities_with_attr[f"edge:{edge_id}"] = edge.attributes[attribute_name]
                
                # Record this state
                for entity_id, value in entities_with_attr.items():
                    if entity_id not in attribute_timeline:
                        attribute_timeline[entity_id] = []
                    
                    attribute_timeline[entity_id].append({
                        'state': state_info,
                        'value': value
                    })
            
            except Exception as e:
                print(f"Error loading state {state_hash[:8]}...: {e}")
            
            finally:
                # Restore state if needed
                if current_state_backup:
                    graph.load_state(current_state_backup)
        
        # Print timeline summary
        print(f"\nFound {len(attribute_timeline)} entities with attribute '{attribute_name}':")
        print(f"Across {len(saved_states)} states")
        
        # Show changes for entities that actually changed
        entities_with_changes = 0
        for entity_id, timeline in attribute_timeline.items():
            if len(timeline) > 1:
                # Check if values actually changed
                values = [entry['value'] for entry in timeline]
                if len(set(str(v) for v in values)) > 1:  # Convert to string for comparison
                    entities_with_changes += 1
                    if entities_with_changes <= 10:  # Limit output
                        print(f"\n{entity_id}:")
                        for entry in timeline:
                            state_hash = entry['state']['hash'][:8]
                            print(f"  {state_hash}... -> {entry['value']}")
        
        print(f"\n📊 Summary: {entities_with_changes} entities had '{attribute_name}' changes")
        
    finally:
        # Restore original branch
        if original_branch != branch_name:
            graph.switch_branch(original_branch)
