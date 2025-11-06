"""
Similarity Matrix Embedding Explorer

Experiment with different graph structures and similarity matrices to understand
how they produce different embeddings via normalized Laplacian spectral methods.

L = D^(-1/2) * S * D^(-1/2)

where S is the similarity/adjacency matrix and D is the degree matrix.
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import networkx as nx
import numpy as np
from scipy.linalg import eigh
from sklearn.cluster import KMeans
import umap

# -------------------------
# Core Embedding Functions
# -------------------------

def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple (0-255)."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb):
    """Convert RGB tuple (0-255) to hex color."""
    return '#%02x%02x%02x' % rgb


def color_similarity(color1, color2, method="euclidean"):
    """
    Compute similarity between two colors.
    
    Methods:
    - euclidean: 1 - normalized Euclidean distance in RGB space
    - cosine: Cosine similarity of RGB vectors
    """
    rgb1 = np.array(hex_to_rgb(color1), dtype=float)
    rgb2 = np.array(hex_to_rgb(color2), dtype=float)
    
    if method == "euclidean":
        # Normalize to [0, 1] range
        max_dist = np.sqrt(3 * 255**2)  # Maximum possible distance
        dist = np.linalg.norm(rgb1 - rgb2)
        return 1.0 - (dist / max_dist)
    elif method == "cosine":
        return np.dot(rgb1, rgb2) / (np.linalg.norm(rgb1) * np.linalg.norm(rgb2) + 1e-10)
    
    return 0.0


def compute_similarity_matrix(G, nodes, method="adjacency"):
    """
    Compute similarity matrix S from graph.
    
    Methods:
    - adjacency: Binary adjacency (1 if connected, 0 otherwise)
    - weighted: Use edge weights if available
    - gaussian: Gaussian similarity based on node positions
    - cosine: Cosine similarity of node features
    - color: Color similarity based on node hex colors (ignores edges)
    """
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    S = np.zeros((n, n))
    
    if method == "adjacency":
        for u, v in G.edges():
            i, j = node_to_idx[u], node_to_idx[v]
            S[i, j] = S[j, i] = 1.0
            
    elif method == "weighted":
        for u, v, data in G.edges(data=True):
            i, j = node_to_idx[u], node_to_idx[v]
            weight = data.get('weight', 1.0)
            S[i, j] = S[j, i] = weight
            
    elif method == "gaussian":
        # Gaussian similarity based on Euclidean distance
        positions = np.array([[G.nodes[n].get('x', 0), G.nodes[n].get('y', 0)] 
                             for n in nodes])
        sigma = 1.0  # bandwidth parameter
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(positions[i] - positions[j])
                sim = np.exp(-dist**2 / (2 * sigma**2))
                S[i, j] = S[j, i] = sim
                
    elif method == "cosine":
        # Cosine similarity of node feature vectors
        features = np.array([[G.nodes[n].get('x', 0), 
                             G.nodes[n].get('y', 0),
                             G.nodes[n].get('extra', 0)] for n in nodes])
        for i in range(n):
            for j in range(i+1, n):
                sim = np.dot(features[i], features[j]) / (
                    np.linalg.norm(features[i]) * np.linalg.norm(features[j]) + 1e-10)
                S[i, j] = S[j, i] = max(0, sim)  # Keep non-negative
    
    elif method == "color":
        # Color similarity based on hex colors (all-to-all, ignores graph edges)
        for i in range(n):
            for j in range(i+1, n):
                color_i = G.nodes[nodes[i]].get('hex_color', '#808080')
                color_j = G.nodes[nodes[j]].get('hex_color', '#808080')
                sim = color_similarity(color_i, color_j, method="euclidean")
                S[i, j] = S[j, i] = sim
        # Set diagonal to 1 (self-similarity)
        np.fill_diagonal(S, 1.0)
    
    return S


def compute_normalized_laplacian_embeddings(S, k=3):
    """
    Compute embeddings from normalized Laplacian.
    
    L = D^(-1/2) * S * D^(-1/2)
    
    Returns the top k eigenvectors (excluding the trivial one).
    """
    # Compute degree matrix
    D = np.diag(S.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / (np.sqrt(np.diag(D)) + 1e-10))
    
    # Normalized Laplacian (using similarity instead of Laplacian convention)
    L = D_inv_sqrt @ S @ D_inv_sqrt
    
    # Compute eigendecomposition
    eigenvalues, eigenvectors = eigh(L)
    
    # Sort by eigenvalue (descending) and take top k+1, skip the first (trivial)
    idx = np.argsort(eigenvalues)[::-1]
    embeddings = eigenvectors[:, idx[1:k+1]]
    
    return embeddings, eigenvalues[idx]


def power_iteration_embeddings(S, k=3, num_iterations=20):
    """
    Approximate top eigenvectors using block power iteration.
    
    This is a simpler, more interpretable method than full eigendecomposition.
    """
    n = S.shape[0]
    # Initialize with random vectors
    Q = np.random.randn(n, k)
    
    # Power iteration
    for _ in range(num_iterations):
        Q = S @ Q
        Q, _ = np.linalg.qr(Q)  # Orthonormalize
    
    return Q


# -------------------------
# Graph Generation
# -------------------------

def generate_graph(family, num_nodes, **params):
    """Generate various graph families with different structures."""
    if family == "ER":
        p = params.get('p', 0.3)
        G = nx.erdos_renyi_graph(num_nodes, p)
    elif family == "BA":
        m = min(params.get('m', 2), num_nodes - 1)
        G = nx.barabasi_albert_graph(num_nodes, m)
    elif family == "WS":
        k = min(params.get('k', 4), num_nodes - 1)
        if k % 2 == 1:
            k -= 1
        p = params.get('p', 0.1)
        G = nx.watts_strogatz_graph(num_nodes, k, p)
    elif family == "Grid":
        m = int(np.sqrt(num_nodes))
        G = nx.grid_2d_graph(m, m)
        G = nx.convert_node_labels_to_integers(G)
    elif family == "Complete":
        G = nx.complete_graph(num_nodes)
    elif family == "Star":
        G = nx.star_graph(num_nodes - 1)
    elif family == "Cycle":
        G = nx.cycle_graph(num_nodes)
    elif family == "Path":
        G = nx.path_graph(num_nodes)
    else:
        G = nx.erdos_renyi_graph(num_nodes, 0.3)
    
    # Add positions and random features
    pos = nx.spring_layout(G, k=2, iterations=50)
    for node in G.nodes():
        G.nodes[node]['x'] = float(pos[node][0]) * 10
        G.nodes[node]['y'] = float(pos[node][1]) * 10
        G.nodes[node]['z'] = float(np.random.uniform(-2, 2))
        G.nodes[node]['extra'] = float(np.random.uniform(0, 10))
        # Generate random hex color
        r, g, b = np.random.randint(0, 256, 3)
        G.nodes[node]['hex_color'] = rgb_to_hex((r, g, b))
    
    return G


def graph_to_data(G):
    """Convert NetworkX graph to our data format."""
    data = {
        "nodes": {},
        "edges": []
    }
    for node in G.nodes():
        data["nodes"][str(node)] = {
            "x": G.nodes[node].get('x', 0),
            "y": G.nodes[node].get('y', 0),
            "z": G.nodes[node].get('z', 0),
            "extra": G.nodes[node].get('extra', 0),
            "hex_color": G.nodes[node].get('hex_color', '#808080')
        }
    for u, v in G.edges():
        data["edges"].append((str(u), str(v)))
    return data


# -------------------------
# Dash App
# -------------------------

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.Div([
        html.H1("Similarity Matrix Embedding Explorer"),
        html.P("Explore how different graphs and similarity matrices produce different embeddings via L = D^(-1/2) * S * D^(-1/2)")
    ], style={"textAlign": "center", "padding": "20px"}),
    
    html.Div([
        # Left: Controls
        html.Div([
            html.H3("Graph Generation"),
            html.Label("Graph Family:"),
            dcc.Dropdown(
                id="graph-family",
                options=[
                    {"label": "Erdős-Rényi (Random)", "value": "ER"},
                    {"label": "Barabási-Albert (Scale-Free)", "value": "BA"},
                    {"label": "Watts-Strogatz (Small World)", "value": "WS"},
                    {"label": "Grid", "value": "Grid"},
                    {"label": "Complete", "value": "Complete"},
                    {"label": "Star", "value": "Star"},
                    {"label": "Cycle", "value": "Cycle"},
                    {"label": "Path", "value": "Path"},
                ],
                value="ER"
            ),
            html.Br(),
            html.Label("Number of Nodes:"),
            dcc.Slider(id="num-nodes", min=5, max=50, step=5, value=20, 
                      marks={i: str(i) for i in range(5, 51, 5)}),
            html.Br(),
            html.Button("+ Add Graph", id="generate-btn", n_clicks=0,
                       style={"width": "100%", "padding": "10px", "fontSize": "16px", "backgroundColor": "#4CAF50", "color": "white", "border": "none"}),
            
            html.Hr(),
            
            html.H3("Similarity Method"),
            dcc.Checklist(
                id="similarity-methods",
                options=[
                    {"label": "Adjacency (Binary)", "value": "adjacency"},
                    {"label": "Gaussian (Distance-based)", "value": "gaussian"},
                    {"label": "Cosine (Feature-based)", "value": "cosine"},
                    {"label": "Color (RGB Similarity)", "value": "color"},
                ],
                value=["adjacency"],
                labelStyle={'display': 'block', 'margin': '10px'}
            ),
            
            html.Hr(),
            
            html.H3("Embedding Method"),
            dcc.Checklist(
                id="embedding-methods",
                options=[
                    {"label": "Full Eigendecomposition", "value": "eigen"},
                    {"label": "Power Iteration", "value": "power"},
                ],
                value=["eigen"],
                labelStyle={'display': 'block', 'margin': '10px'}
            ),
            
            html.Br(),
            html.Label("Number of Dimensions:"),
            dcc.Slider(id="embed-dims", min=2, max=5, step=1, value=3,
                      marks={i: str(i) for i in range(2, 6)}),
            
            html.Br(),
            html.Button("+ Add Embeddings", id="compute-btn", n_clicks=0,
                       style={"width": "100%", "padding": "10px", "fontSize": "16px", "backgroundColor": "#2196F3", "color": "white", "border": "none"}),
            
            html.Hr(),
            
            html.H3("Clustering"),
            html.Label("Number of Clusters:"),
            dcc.Slider(id="num-clusters", min=2, max=8, step=1, value=3,
                      marks={i: str(i) for i in range(2, 9)}),
            html.Br(),
            html.Button("+ Add Clusters", id="cluster-btn", n_clicks=0,
                       style={"width": "100%", "padding": "10px", "fontSize": "16px", "backgroundColor": "#FF9800", "color": "white", "border": "none"}),
            
            html.Hr(),
            html.Button("Clear All", id="clear-btn", n_clicks=0,
                       style={"width": "100%", "padding": "10px", "fontSize": "14px", "backgroundColor": "#f44336", "color": "white", "border": "none"}),
            
            html.Hr(),
            html.Div(id="stats-output", style={"fontSize": "12px", "marginTop": "20px"})
            
        ], style={"width": "25%", "padding": "20px", "overflowY": "auto", "height": "90vh"}),
        
        # Right: Dynamic component grid
        html.Div([
            html.Div(id="components-grid", style={"display": "flex", "flexWrap": "wrap", "gap": "10px"})
        ], style={"width": "75%", "padding": "10px", "overflowY": "auto", "height": "90vh"}),
        
    ], style={"display": "flex"}),
    
    # Storage
    dcc.Store(id="components-store", data=[]),
])


def create_component_card(component_id, component_data):
    """Create a visual card for a component."""
    comp_type = component_data["type"]
    title = component_data.get("title", f"{comp_type} #{component_id}")
    
    card_style = {
        "border": "1px solid #ddd",
        "borderRadius": "5px",
        "padding": "10px",
        "margin": "5px",
        "backgroundColor": "#f9f9f9",
        "minWidth": "400px",
        "maxWidth": "600px",
        "flex": "1 1 45%"
    }
    
    return html.Div([
        html.Div([
            html.H4(title, style={"margin": "0", "display": "inline-block"}),
            html.Button("×", id={"type": "remove-component", "index": component_id},
                       style={"float": "right", "border": "none", "background": "none",
                             "fontSize": "24px", "cursor": "pointer", "color": "#999"})
        ]),
        dcc.Graph(id={"type": "component-graph", "index": component_id},
                 figure=component_data.get("figure", go.Figure()),
                 style={"height": "400px"}),
        html.Div(component_data.get("stats", ""), style={"fontSize": "11px", "marginTop": "5px"})
    ], style=card_style)


@app.callback(
    Output("components-store", "data"),
    Output("stats-output", "children"),
    Input("generate-btn", "n_clicks"),
    Input("compute-btn", "n_clicks"),
    Input("cluster-btn", "n_clicks"),
    Input("clear-btn", "n_clicks"),
    Input({"type": "remove-component", "index": dash.dependencies.ALL}, "n_clicks"),
    State("components-store", "data"),
    State("graph-family", "value"),
    State("num-nodes", "value"),
    State("similarity-methods", "value"),
    State("embedding-methods", "value"),
    State("embed-dims", "value"),
    State("num-clusters", "value"),
    prevent_initial_call=True
)
def manage_components(gen_clicks, comp_clicks, clust_clicks, clear_clicks, remove_clicks,
                     components, graph_family, num_nodes, sim_methods, embed_methods, k, num_clusters):
    ctx = callback_context
    if not ctx.triggered:
        return components, ""
    
    trigger_id = ctx.triggered[0]["prop_id"]
    
    # Clear all components
    if "clear-btn" in trigger_id:
        return [], "All components cleared"
    
    # Remove a specific component
    if "remove-component" in trigger_id:
        button_id = eval(trigger_id.split(".")[0])
        idx = button_id["index"]
        components = [c for c in components if c["id"] != idx]
        return components, f"Removed component {idx}"
    
    # Add new graph(s)
    if "generate-btn" in trigger_id:
        G = generate_graph(graph_family, num_nodes)
        graph_data = graph_to_data(G)
        
        comp_id = len(components)
        fig = create_graph_figure(graph_data)
        
        new_component = {
            "id": comp_id,
            "type": "graph",
            "title": f"{graph_family} Graph (n={num_nodes})",
            "graph_data": graph_data,
            "figure": fig,
            "stats": f"Nodes: {len(graph_data['nodes'])}, Edges: {len(graph_data['edges'])}"
        }
        components.append(new_component)
        return components, f"Added {graph_family} graph"
    
    # Add new embedding(s)
    if "compute-btn" in trigger_id:
        # Find most recent graph
        graph_components = [c for c in components if c["type"] == "graph"]
        if not graph_components:
            return components, "No graph available. Generate a graph first."
        
        latest_graph = graph_components[-1]
        graph_data = latest_graph["graph_data"]
        
        # Build NetworkX graph
        G = nx.Graph()
        for node_id, props in graph_data["nodes"].items():
            G.add_node(node_id, **props)
        for u, v in graph_data["edges"]:
            G.add_edge(u, v)
        nodes = sorted(G.nodes())
        
        # Generate embeddings for each selected method combination
        for sim_method in sim_methods:
            for embed_method in embed_methods:
                S = compute_similarity_matrix(G, nodes, method=sim_method)
                
                if embed_method == "eigen":
                    embeddings, eigenvalues = compute_normalized_laplacian_embeddings(S, k)
                    eig_info = f", λ₁={eigenvalues[1]:.3f}" if len(eigenvalues) > 1 else ""
                else:
                    embeddings = power_iteration_embeddings(S, k)
                    eig_info = ""
                
                embedding_data = {node: embeddings[i].tolist() for i, node in enumerate(nodes)}
                
                comp_id = len(components)
                fig = create_embedding_figure(embedding_data, graph_data, k)
                
                new_component = {
                    "id": comp_id,
                    "type": "embedding",
                    "title": f"{sim_method.capitalize()} + {embed_method.capitalize()} (k={k})",
                    "graph_data": graph_data,
                    "embedding_data": embedding_data,
                    "figure": fig,
                    "stats": f"Similarity: {sim_method}, Method: {embed_method}, Avg sim: {S.mean():.3f}{eig_info}"
                }
                components.append(new_component)
        
        return components, f"Added {len(sim_methods) * len(embed_methods)} embedding(s)"
    
    # Add clustering
    if "cluster-btn" in trigger_id:
        # Find most recent embedding
        embedding_components = [c for c in components if c["type"] == "embedding"]
        if not embedding_components:
            return components, "No embeddings available. Compute embeddings first."
        
        latest_embedding = embedding_components[-1]
        embedding_data = latest_embedding["embedding_data"]
        graph_data = latest_embedding["graph_data"]
        
        nodes = sorted(embedding_data.keys())
        X = np.array([embedding_data[node] for node in nodes])
        
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        labels = kmeans.fit_predict(X)
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#bcbd22']
        
        clustered_graph_data = {k: v.copy() if isinstance(v, dict) else v for k, v in graph_data.items()}
        clustered_graph_data["nodes"] = {k: v.copy() for k, v in graph_data["nodes"].items()}
        
        for i, node in enumerate(nodes):
            clustered_graph_data["nodes"][node]["cluster"] = int(labels[i])
            clustered_graph_data["nodes"][node]["color"] = colors[labels[i] % len(colors)]
        
        comp_id = len(components)
        fig = create_embedding_figure(embedding_data, clustered_graph_data, len(X[0]))
        
        new_component = {
            "id": comp_id,
            "type": "clustering",
            "title": f"K-Means Clustering (k={num_clusters})",
            "graph_data": clustered_graph_data,
            "embedding_data": embedding_data,
            "figure": fig,
            "stats": f"Clusters: {num_clusters}, Inertia: {kmeans.inertia_:.2f}"
        }
        components.append(new_component)
        return components, f"Added clustering with {num_clusters} clusters"
    
    return components, ""


@app.callback(
    Output("components-grid", "children"),
    Input("components-store", "data")
)
def render_components(components):
    if not components:
        return html.Div("No components yet. Use the controls to add graphs, embeddings, and clusters.",
                       style={"padding": "40px", "textAlign": "center", "color": "#999"})
    
    return [create_component_card(comp["id"], comp) for comp in components]


def create_graph_figure(graph_data):
    """Create a figure for a graph component."""
    nodes = graph_data.get("nodes", {})
    edges = graph_data.get("edges", [])
    
    if not nodes:
        return go.Figure()
    
    edge_x, edge_y = [], []
    for u, v in edges:
        if u in nodes and v in nodes:
            edge_x.extend([nodes[u]["x"], nodes[v]["x"], None])
            edge_y.extend([nodes[u]["y"], nodes[v]["y"], None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    node_x = [nodes[n]["x"] for n in nodes]
    node_y = [nodes[n]["y"] for n in nodes]
    node_text = [f"Node {n}<br>Color: {nodes[n].get('hex_color', '#808080')}" for n in nodes]
    # Use cluster color if available, otherwise use hex_color
    node_colors = [nodes[n].get("color", nodes[n].get("hex_color", "lightblue")) for n in nodes]
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        text=node_text,
        hoverinfo='text',
        marker=dict(size=8, color=node_colors, line=dict(width=1, color='white'))
    )
    
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=10, l=10, r=10, t=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white'
    )
    
    return fig


def create_embedding_figure(embedding_data, graph_data, k):
    """Create a figure for an embedding component."""
    if not embedding_data:
        return go.Figure()
    
    nodes = sorted(embedding_data.keys())
    embeddings = np.array([embedding_data[n] for n in nodes])
    
    if embeddings.shape[1] < 2:
        return go.Figure()
    
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    # Use cluster color if available, otherwise use hex_color
    node_colors = [graph_data["nodes"][n].get("color", graph_data["nodes"][n].get("hex_color", "lightblue")) for n in nodes]
    node_text = [f"Node {n}<br>Color: {graph_data['nodes'][n].get('hex_color', '#808080')}" for n in nodes]
    edges = graph_data.get("edges", [])
    
    if embeddings.shape[1] >= 3:
        edge_x, edge_y, edge_z = [], [], []
        for u, v in edges:
            if u in node_to_idx and v in node_to_idx:
                u_idx, v_idx = node_to_idx[u], node_to_idx[v]
                edge_x.extend([embeddings[u_idx, 0], embeddings[v_idx, 0], None])
                edge_y.extend([embeddings[u_idx, 1], embeddings[v_idx, 1], None])
                edge_z.extend([embeddings[u_idx, 2], embeddings[v_idx, 2], None])
        
        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        node_trace = go.Scatter3d(
            x=embeddings[:, 0], y=embeddings[:, 1], z=embeddings[:, 2],
            mode='markers',
            text=node_text,
            hoverinfo='text',
            marker=dict(size=5, color=node_colors, line=dict(width=0.5, color='white'))
        )
        
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            scene=dict(xaxis_title='Dim 1', yaxis_title='Dim 2', zaxis_title='Dim 3', bgcolor='white'),
            margin=dict(l=0, r=0, b=0, t=0),
            showlegend=False
        )
    else:
        edge_x, edge_y = [], []
        for u, v in edges:
            if u in node_to_idx and v in node_to_idx:
                u_idx, v_idx = node_to_idx[u], node_to_idx[v]
                edge_x.extend([embeddings[u_idx, 0], embeddings[v_idx, 0], None])
                edge_y.extend([embeddings[u_idx, 1], embeddings[v_idx, 1], None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        node_trace = go.Scatter(
            x=embeddings[:, 0], y=embeddings[:, 1],
            mode='markers',
            text=node_text,
            hoverinfo='text',
            marker=dict(size=8, color=node_colors, line=dict(width=1, color='white'))
        )
        
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            showlegend=False,
            plot_bgcolor='white',
            margin=dict(l=30, r=10, b=30, t=10)
        )
    
    return fig


if __name__ == '__main__':
    app.run(debug=True, port=8051)
