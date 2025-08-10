#!/usr/bin/env python3
"""
Create a visual dependency graph showing the cleaned architecture
"""

def create_simplified_dependency_graph():
    """Create a simplified, clean dependency graph in DOT format"""
    
    dot_content = '''digraph groggy_architecture {
    // Graph settings
    rankdir=TB;
    nodesep=0.5;
    ranksep=0.8;
    
    // Node styles
    node [shape=box, style=filled, fontname="Arial", fontsize=10];
    
    // Color scheme
    subgraph cluster_api {
        label="Public API Layer";
        style=filled;
        color=lightblue;
        
        Graph [fillcolor=lightgreen, label="Graph\\n(Main API)\\n44 functions"];
    }
    
    subgraph cluster_core {
        label="Core Architecture";
        style=filled;
        color=lightgray;
        
        Space [fillcolor=lightcyan, label="GraphSpace\\n(Active State)\\n39 functions\\n72% complete"];
        Pool [fillcolor=lightcyan, label="GraphPool\\n(Data Storage)\\n46 functions\\n63% complete"];
        ChangeTracker [fillcolor=lightcyan, label="ChangeTracker\\n(Change Mgmt)\\n69 functions\\n38% complete"];
        Strategies [fillcolor=lightgreen, label="Strategies\\n(Temporal Storage)\\n42 functions\\n88% complete"];
        History [fillcolor=lightyellow, label="HistoryForest\\n(Version Control)\\n94 functions\\n1% complete"];
        Query [fillcolor=lightyellow, label="QueryEngine\\n(Analytics)\\n41 functions\\n2% complete"];
        RefManager [fillcolor=lightyellow, label="RefManager\\n(Branches/Tags)\\n67 functions\\n1% complete"];
    }
    
    subgraph cluster_support {
        label="Supporting Components";
        style=filled;
        color=lightgray;
        
        State [fillcolor=lightcyan, label="StateObject\\n(State Mgmt)\\n69 functions\\n68% complete"];
        Delta [fillcolor=lightgreen, label="DeltaObject\\n(Change Deltas)\\n37 functions\\n95% complete"];
        Config [fillcolor=lightyellow, label="GraphConfig\\n(Configuration)\\n28 functions\\n11% complete"];
        Types [fillcolor=lightcoral, label="Types\\n(Core Types)\\n12 functions\\n0% complete"];
        Errors [fillcolor=lightgreen, label="Errors\\n(Error Handling)\\n32 functions\\n88% complete"];
    }
    
    // Dependencies (main flow)
    Graph -> Space [label="manages active state"];
    Graph -> Pool [label="coordinates data ops"];
    Graph -> History [label="version control"];
    Graph -> Query [label="analytics"];
    Graph -> Config [label="configuration"];
    
    // Core component interactions
    Space -> Pool [label="attribute storage"];
    Space -> ChangeTracker [label="tracks changes"];
    ChangeTracker -> Strategies [label="storage strategy"];
    ChangeTracker -> Delta [label="creates deltas"];
    History -> State [label="manages states"];
    History -> Delta [label="applies changes"];
    History -> RefManager [label="branch/tag mgmt"];
    
    // Supporting dependencies
    Pool -> Types [label="uses core types"];
    Space -> Types [label="uses core types"];
    ChangeTracker -> Types [label="uses core types"];
    Graph -> Errors [label="error handling"];
    Pool -> Errors [label="error handling"];
    
    // Strategy pattern
    Strategies -> Delta [label="creates deltas", style=dashed];
    
    // Legend
    subgraph cluster_legend {
        label="Implementation Status";
        style=filled;
        color=white;
        
        legend1 [shape=box, fillcolor=lightgreen, label="80%+ Complete"];
        legend2 [shape=box, fillcolor=lightcyan, label="40-80% Complete"];
        legend3 [shape=box, fillcolor=lightyellow, label="10-40% Complete"];
        legend4 [shape=box, fillcolor=lightcoral, label="<10% Complete"];
        
        legend1 -> legend2 -> legend3 -> legend4 [style=invis];
    }
}'''
    
    return dot_content

def create_api_focus_graph():
    """Create a graph focused on the cleaned Graph API"""
    
    dot_content = '''digraph graph_api_structure {
    rankdir=TB;
    node [shape=box, style=filled, fontname="Arial"];
    
    // Main API
    Graph [fillcolor=lightblue, label="Graph API\\n(Clean, No Duplicates)"];
    
    // API Categories
    subgraph cluster_construction {
        label="Construction (3 functions)";
        style=filled;
        color=lightgreen;
        
        new [label="new()"];
        with_config [label="with_config()"];
        load_from_path [label="load_from_path()"];
    }
    
    subgraph cluster_entities {
        label="Entity Operations (6 functions)";
        style=filled;
        color=lightcyan;
        
        add_node [label="add_node()"];
        add_nodes [label="add_nodes()"];
        add_edge [label="add_edge()"];
        add_edges [label="add_edges()"];
        remove_node [label="remove_node()"];
        remove_edge [label="remove_edge()"];
    }
    
    subgraph cluster_attributes {
        label="Attribute Operations (10 functions)\\nCLEANED: No more duplicates!";
        style=filled;
        color=gold;
        
        // Single operations
        set_node_attr [label="set_node_attr()\\n(single)"];
        set_edge_attr [label="set_edge_attr()\\n(single)"];
        
        // Bulk operations (simplified!)
        set_node_attrs [label="set_node_attrs()\\n(bulk - handles all cases)", fillcolor=lightgreen];
        set_edge_attrs [label="set_edge_attrs()\\n(bulk - handles all cases)", fillcolor=lightgreen];
        
        // Getters
        get_node_attr [label="get_node_attr()"];
        get_edge_attr [label="get_edge_attr()"];
        get_node_attrs [label="get_node_attrs()"];
        get_edge_attrs [label="get_edge_attrs()"];
        get_nodes_attrs [label="get_nodes_attrs()"];
        get_edges_attrs [label="get_edges_attrs()"];
    }
    
    subgraph cluster_topology {
        label="Topology Queries (6 functions)";
        style=filled;
        color=lightcyan;
        
        contains_node [label="contains_node()"];
        contains_edge [label="contains_edge()"];
        node_ids [label="node_ids()"];
        edge_ids [label="edge_ids()"];
        neighbors [label="neighbors()"];
        degree [label="degree()"];
    }
    
    subgraph cluster_version_control {
        label="Version Control (2 functions)";
        style=filled;
        color=lightyellow;
        
        commit [label="commit()"];
        create_branch [label="create_branch()"];
    }
    
    // Connections
    Graph -> new;
    Graph -> add_node;
    Graph -> set_node_attrs;
    Graph -> contains_node;
    Graph -> commit;
    
    // Highlight cleaned functions
    removed1 [shape=ellipse, fillcolor=lightcoral, label="❌ set_node_attr_bulk\\n(REMOVED)"];
    removed2 [shape=ellipse, fillcolor=lightcoral, label="❌ set_multiple_node_attrs\\n(REMOVED)"];
    removed3 [shape=ellipse, fillcolor=lightcoral, label="❌ set_edge_attr_bulk\\n(REMOVED)"];
    removed4 [shape=ellipse, fillcolor=lightcoral, label="❌ set_multiple_edge_attrs\\n(REMOVED)"];
    
    // Show what was cleaned up
    removed1 -> set_node_attrs [label="consolidated into", style=dashed, color=red];
    removed2 -> set_node_attrs [label="consolidated into", style=dashed, color=red];
    removed3 -> set_edge_attrs [label="consolidated into", style=dashed, color=red];
    removed4 -> set_edge_attrs [label="consolidated into", style=dashed, color=red];
}'''
    
    return dot_content

def main():
    # Create architecture overview
    with open("groggy_architecture_clean.dot", "w") as f:
        f.write(create_simplified_dependency_graph())
    
    # Create API focus graph
    with open("graph_api_cleaned.dot", "w") as f:
        f.write(create_api_focus_graph())
    
    print("✅ Dependency graphs created:")
    print("   📊 groggy_architecture_clean.dot - Overall architecture")
    print("   🎯 graph_api_cleaned.dot - Cleaned Graph API focus")
    print()
    print("Generate visualizations with:")
    print("   dot -Tpng groggy_architecture_clean.dot -o architecture_clean.png")
    print("   dot -Tpng graph_api_cleaned.dot -o api_cleaned.png")

if __name__ == "__main__":
    main()