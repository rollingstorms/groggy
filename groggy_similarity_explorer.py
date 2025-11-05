"""
Groggy-Based Similarity Matrix Embedding Explorer

Uses groggy's graph primitives to compute similarity matrices and embeddings.
Demonstrates how different graphs produce different embeddings via:
    L = D^(-1/2) * S * D^(-1/2)

This version leverages groggy's columnar operations for efficient computation.
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import numpy as np
from scipy.linalg import eigh
from sklearn.cluster import KMeans

try:
    import groggy as gg
except ImportError:
    print("Error: groggy not installed. Run: maturin develop --release")
    exit(1)


# -------------------------
# Groggy Graph Generation
# -------------------------

def generate_groggy_graph(family, num_nodes, **params):
    """Generate various graph families using groggy."""
    
    if family == "ER":
        # Erdős-Rényi: random edges with probability p
        p = params.get('p', 0.3)
        g = gg.Graph()
        
        # Add nodes
        for i in range(num_nodes):
            g.add_node(i)
        
        # Add random edges
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if np.random.random() < p:
                    g.add_edge(i, j)
                    
    elif family == "BA":
        # Barabási-Albert: preferential attachment
        m = min(params.get('m', 2), num_nodes - 1)
        g = gg.Graph()
        
        # Start with complete graph of m nodes
        for i in range(m):
            g.add_node(i)
            for j in range(i):
                g.add_edge(i, j)
        
        # Add remaining nodes with preferential attachment
        for i in range(m, num_nodes):
            g.add_node(i)
            degrees = np.array([g.degree(j) for j in range(i)])
            probs = degrees / degrees.sum()
            targets = np.random.choice(i, size=min(m, i), replace=False, p=probs)
            for target in targets:
                g.add_edge(i, int(target))
                
    elif family == "WS":
        # Watts-Strogatz: small world
        k = min(params.get('k', 4), num_nodes - 1)
        if k % 2 == 1:
            k -= 1
        p = params.get('p', 0.1)
        
        g = gg.Graph()
        for i in range(num_nodes):
            g.add_node(i)
        
        # Create ring lattice
        for i in range(num_nodes):
            for j in range(1, k // 2 + 1):
                target = (i + j) % num_nodes
                g.add_edge(i, target)
        
        # Rewire edges
        edges_to_rewire = list(g.edges())
        for u, v in edges_to_rewire:
            if np.random.random() < p:
                g.remove_edge(u, v)
                new_target = np.random.randint(0, num_nodes)
                while new_target == u or g.has_edge(u, new_target):
                    new_target = np.random.randint(0, num_nodes)
                g.add_edge(u, new_target)
                
    elif family == "Grid":
        # 2D Grid
        m = int(np.sqrt(num_nodes))
        g = gg.Graph()
        
        for i in range(m):
            for j in range(m):
                node_id = i * m + j
                g.add_node(node_id)
                
                # Connect to right neighbor
                if j < m - 1:
                    g.add_edge(node_id, node_id + 1)
                
                # Connect to bottom neighbor
                if i < m - 1:
                    g.add_edge(node_id, node_id + m)
                    
    elif family == "Complete":
        # Complete graph
        g = gg.Graph()
        for i in range(num_nodes):
            g.add_node(i)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                g.add_edge(i, j)
                
    elif family == "Star":
        # Star graph
        g = gg.Graph()
        for i in range(num_nodes):
            g.add_node(i)
        for i in range(1, num_nodes):
            g.add_edge(0, i)
            
    elif family == "Cycle":
        # Cycle graph
        g = gg.Graph()
        for i in range(num_nodes):
            g.add_node(i)
        for i in range(num_nodes):
            g.add_edge(i, (i + 1) % num_nodes)
            
    elif family == "Path":
        # Path graph
        g = gg.Graph()
        for i in range(num_nodes):
            g.add_node(i)
        for i in range(num_nodes - 1):
            g.add_edge(i, i + 1)
    else:
        # Default to ER
        return generate_groggy_graph("ER", num_nodes, p=0.3)
    
    return g


def add_node_positions(g, layout="spring"):
    """Add x, y, z positions to nodes using layout algorithm."""
    nodes = list(g.nodes())
    n = len(nodes)
    
    if layout == "spring":
        # Simple force-directed layout
        pos = np.random.randn(n, 2) * 2
        
        # Run iterations
        for _ in range(50):
            forces = np.zeros((n, 2))
            
            # Repulsion between all nodes
            for i in range(n):
                for j in range(i + 1, n):
                    diff = pos[i] - pos[j]
                    dist = np.linalg.norm(diff) + 0.01
                    force = diff / (dist ** 2)
                    forces[i] += force * 0.1
                    forces[j] -= force * 0.1
            
            # Attraction along edges
            for u, v in g.edges():
                u_idx = nodes.index(u)
                v_idx = nodes.index(v)
                diff = pos[u_idx] - pos[v_idx]
                dist = np.linalg.norm(diff)
                force = diff * dist * 0.01
                forces[u_idx] -= force
                forces[v_idx] += force
            
            pos += forces * 0.1
    
    elif layout == "circular":
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pos = np.column_stack([np.cos(angles), np.sin(angles)]) * 5
    
    else:
        pos = np.random.randn(n, 2) * 5
    
    # Set node attributes
    for i, node in enumerate(nodes):
        g.set_node_attr(node, "x", float(pos[i, 0]))
        g.set_node_attr(node, "y", float(pos[i, 1]))
        g.set_node_attr(node, "z", float(np.random.uniform(-2, 2)))
        g.set_node_attr(node, "extra", float(np.random.uniform(0, 10)))
    
    return g


# -------------------------
# Similarity Matrix Computation (using groggy)
# -------------------------

def compute_similarity_matrix_groggy(g, method="adjacency"):
    """
    Compute similarity matrix using groggy graph.
    
    Methods:
    - adjacency: Binary adjacency matrix
    - gaussian: Gaussian similarity based on positions
    - cosine: Cosine similarity of node features
    """
    nodes = sorted(g.nodes())
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    S = np.zeros((n, n))
    
    if method == "adjacency":
        # Use groggy's edge list
        for u, v in g.edges():
            i, j = node_to_idx[u], node_to_idx[v]
            S[i, j] = S[j, i] = 1.0
            
    elif method == "gaussian":
        # Extract positions from node attributes
        positions = np.zeros((n, 2))
        for i, node in enumerate(nodes):
            positions[i, 0] = g.get_node_attr(node, "x")
            positions[i, 1] = g.get_node_attr(node, "y")
        
        # Compute Gaussian similarity
        sigma = 2.0
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(positions[i] - positions[j])
                sim = np.exp(-dist**2 / (2 * sigma**2))
                S[i, j] = S[j, i] = sim
                
    elif method == "cosine":
        # Extract features from node attributes
        features = np.zeros((n, 3))
        for i, node in enumerate(nodes):
            features[i, 0] = g.get_node_attr(node, "x")
            features[i, 1] = g.get_node_attr(node, "y")
            features[i, 2] = g.get_node_attr(node, "extra")
        
        # Compute cosine similarity
        for i in range(n):
            for j in range(i + 1, n):
                sim = np.dot(features[i], features[j]) / (
                    np.linalg.norm(features[i]) * np.linalg.norm(features[j]) + 1e-10
                )
                S[i, j] = S[j, i] = max(0, sim)
    
    return S, nodes


# -------------------------
# Embedding Computation
# -------------------------

def compute_normalized_laplacian_embeddings(S, k=3):
    """
    Compute embeddings from normalized Laplacian.
    L = D^(-1/2) * S * D^(-1/2)
    """
    D = np.diag(S.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / (np.sqrt(np.diag(D)) + 1e-10))
    L = D_inv_sqrt @ S @ D_inv_sqrt
    
    eigenvalues, eigenvectors = eigh(L)
    idx = np.argsort(eigenvalues)[::-1]
    embeddings = eigenvectors[:, idx[1:k+1]]
    
    return embeddings, eigenvalues[idx]


def power_iteration_embeddings(S, k=3, num_iterations=20):
    """Block power iteration for approximate embeddings."""
    n = S.shape[0]
    Q = np.random.randn(n, k)
    
    for _ in range(num_iterations):
        Q = S @ Q
        Q, _ = np.linalg.qr(Q)
    
    return Q


# -------------------------
# Conversion Functions
# -------------------------

def groggy_to_data(g):
    """Convert groggy graph to dict format for storage."""
    data = {"nodes": {}, "edges": []}
    
    for node in g.nodes():
        data["nodes"][str(node)] = {
            "x": g.get_node_attr(node, "x"),
            "y": g.get_node_attr(node, "y"),
            "z": g.get_node_attr(node, "z"),
            "extra": g.get_node_attr(node, "extra"),
        }
    
    for u, v in g.edges():
        data["edges"].append((str(u), str(v)))
    
    return data


def data_to_groggy(data):
    """Convert dict format back to groggy graph."""
    g = gg.Graph()
    
    for node_id, attrs in data["nodes"].items():
        node_id = int(node_id)
        g.add_node(node_id)
        for key, val in attrs.items():
            g.set_node_attr(node_id, key, val)
    
    for u, v in data["edges"]:
        g.add_edge(int(u), int(v))
    
    return g


# -------------------------
# Dash App
# -------------------------

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.Div([
        html.H1("Groggy Similarity Embedding Explorer"),
        html.P("Using groggy for efficient graph operations and similarity computations"),
        html.P("L = D^(-1/2) * S * D^(-1/2)", style={"fontFamily": "monospace"})
    ], style={"textAlign": "center", "padding": "20px", "backgroundColor": "#f0f0f0"}),
    
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
            dcc.Slider(id="num-nodes", min=5, max=100, step=5, value=30,
                      marks={i: str(i) for i in [5, 25, 50, 75, 100]}),
            html.Br(),
            
            html.Button("Generate Graph", id="generate-btn", n_clicks=0,
                       style={"width": "100%", "padding": "10px", "fontSize": "16px",
                              "backgroundColor": "#4CAF50", "color": "white", "border": "none"}),
            
            html.Hr(),
            
            html.H3("Similarity Method"),
            dcc.RadioItems(
                id="similarity-method",
                options=[
                    {"label": "Adjacency (Binary)", "value": "adjacency"},
                    {"label": "Gaussian (Distance-based)", "value": "gaussian"},
                    {"label": "Cosine (Feature-based)", "value": "cosine"},
                ],
                value="adjacency",
                labelStyle={'display': 'block', 'margin': '10px'}
            ),
            
            html.Hr(),
            
            html.H3("Embedding Method"),
            dcc.RadioItems(
                id="embedding-method",
                options=[
                    {"label": "Full Eigendecomposition", "value": "eigen"},
                    {"label": "Power Iteration", "value": "power"},
                ],
                value="eigen",
                labelStyle={'display': 'block', 'margin': '10px'}
            ),
            
            html.Br(),
            html.Label("Number of Dimensions:"),
            dcc.Slider(id="embed-dims", min=2, max=5, step=1, value=3,
                      marks={i: str(i) for i in range(2, 6)}),
            
            html.Br(),
            html.Button("Compute Embeddings", id="compute-btn", n_clicks=0,
                       style={"width": "100%", "padding": "10px", "fontSize": "16px",
                              "backgroundColor": "#2196F3", "color": "white", "border": "none"}),
            
            html.Hr(),
            
            html.H3("Clustering"),
            html.Label("Number of Clusters:"),
            dcc.Slider(id="num-clusters", min=2, max=8, step=1, value=3,
                      marks={i: str(i) for i in range(2, 9)}),
            html.Br(),
            html.Button("Cluster Embeddings", id="cluster-btn", n_clicks=0,
                       style={"width": "100%", "padding": "10px", "fontSize": "16px",
                              "backgroundColor": "#FF9800", "color": "white", "border": "none"}),
            
            html.Hr(),
            html.Div(id="stats-output", style={"fontSize": "12px", "marginTop": "20px",
                                               "padding": "10px", "backgroundColor": "#f9f9f9",
                                               "borderRadius": "5px"})
            
        ], style={"width": "25%", "padding": "20px", "overflowY": "auto", "height": "90vh",
                 "borderRight": "2px solid #ddd"}),
        
        # Middle: Original Graph
        html.Div([
            html.H3("Original Graph", style={"textAlign": "center"}),
            dcc.Graph(id="graph-viz", style={"height": "85vh"})
        ], style={"width": "37.5%", "padding": "10px"}),
        
        # Right: Embedding Space
        html.Div([
            html.H3("Embedding Space", style={"textAlign": "center"}),
            dcc.Graph(id="embedding-viz", style={"height": "85vh"})
        ], style={"width": "37.5%", "padding": "10px"}),
        
    ], style={"display": "flex"}),
    
    # Storage
    dcc.Store(id="graph-store", data={"nodes": {}, "edges": []}),
    dcc.Store(id="embedding-store", data={}),
])


@app.callback(
    Output("graph-store", "data"),
    Input("generate-btn", "n_clicks"),
    State("graph-family", "value"),
    State("num-nodes", "value"),
    prevent_initial_call=True
)
def generate_graph_callback(n_clicks, family, num_nodes):
    g = generate_groggy_graph(family, num_nodes)
    g = add_node_positions(g, layout="spring")
    return groggy_to_data(g)


@app.callback(
    Output("embedding-store", "data"),
    Output("stats-output", "children"),
    Input("compute-btn", "n_clicks"),
    State("graph-store", "data"),
    State("similarity-method", "value"),
    State("embedding-method", "value"),
    State("embed-dims", "value"),
    prevent_initial_call=True
)
def compute_embeddings_callback(n_clicks, graph_data, sim_method, embed_method, k):
    if not graph_data.get("nodes"):
        return {}, "No graph to embed."
    
    # Convert to groggy graph
    g = data_to_groggy(graph_data)
    
    # Compute similarity matrix using groggy
    S, nodes = compute_similarity_matrix_groggy(g, method=sim_method)
    
    # Compute embeddings
    if embed_method == "eigen":
        embeddings, eigenvalues = compute_normalized_laplacian_embeddings(S, k)
    else:
        embeddings = power_iteration_embeddings(S, k)
        eigenvalues = np.array([])
    
    # Store embeddings
    embedding_data = {
        str(node): embeddings[i].tolist() for i, node in enumerate(nodes)
    }
    
    # Compute stats using groggy
    num_nodes = g.num_nodes()
    num_edges = g.num_edges()
    degrees = [g.degree(node) for node in g.nodes()]
    avg_degree = np.mean(degrees)
    
    # Stats display
    stats = html.Div([
        html.P(f"Nodes: {num_nodes}", style={"margin": "5px"}),
        html.P(f"Edges: {num_edges}", style={"margin": "5px"}),
        html.P(f"Avg Degree: {avg_degree:.2f}", style={"margin": "5px"}),
        html.P(f"Similarity: {sim_method}", style={"margin": "5px"}),
        html.P(f"Method: {embed_method}", style={"margin": "5px"}),
        html.P(f"Dimensions: {k}", style={"margin": "5px"}),
        html.P(f"Avg Similarity: {S.mean():.3f}", style={"margin": "5px"}),
        html.P(f"Sparsity: {(S > 0).mean():.3f}", style={"margin": "5px"}),
    ])
    
    if len(eigenvalues) > 0:
        top_eigs = eigenvalues[:min(5, len(eigenvalues))]
        eig_str = ", ".join([f"{e:.3f}" for e in top_eigs])
        stats.children.append(html.P(f"Top eigenvalues: {eig_str}", style={"margin": "5px"}))
    
    return embedding_data, stats


@app.callback(
    Output("graph-store", "data", allow_duplicate=True),
    Input("cluster-btn", "n_clicks"),
    State("graph-store", "data"),
    State("embedding-store", "data"),
    State("num-clusters", "value"),
    prevent_initial_call=True
)
def cluster_callback(n_clicks, graph_data, embedding_data, k):
    if not embedding_data:
        return graph_data
    
    nodes = sorted(embedding_data.keys(), key=int)
    X = np.array([embedding_data[node] for node in nodes])
    
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    
    # Color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#bcbd22']
    
    for i, node in enumerate(nodes):
        graph_data["nodes"][node]["cluster"] = int(labels[i])
        graph_data["nodes"][node]["color"] = colors[labels[i] % len(colors)]
    
    return graph_data


@app.callback(
    Output("graph-viz", "figure"),
    Input("graph-store", "data")
)
def update_graph_viz(graph_data):
    nodes = graph_data.get("nodes", {})
    edges = graph_data.get("edges", [])
    
    if not nodes:
        return go.Figure()
    
    # Edge trace
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
    
    # Node trace
    node_x = [nodes[n]["x"] for n in nodes]
    node_y = [nodes[n]["y"] for n in nodes]
    node_text = [f"Node {n}<br>Degree: computed via groggy" for n in nodes]
    node_colors = [nodes[n].get("color", "lightblue") for n in nodes]
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        text=node_text,
        hoverinfo='text',
        marker=dict(
            size=12,
            color=node_colors,
            line=dict(width=1, color='white')
        )
    )
    
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white'
    )
    
    return fig


@app.callback(
    Output("embedding-viz", "figure"),
    Input("embedding-store", "data"),
    Input("graph-store", "data")
)
def update_embedding_viz(embedding_data, graph_data):
    if not embedding_data:
        return go.Figure()
    
    nodes = sorted(embedding_data.keys(), key=int)
    embeddings = np.array([embedding_data[n] for n in nodes])
    
    if embeddings.shape[1] < 2:
        return go.Figure()
    
    # Get colors from clusters if available
    node_colors = [graph_data["nodes"][n].get("color", "lightblue") for n in nodes]
    node_text = [f"Node {n}" for n in nodes]
    
    if embeddings.shape[1] >= 3:
        # 3D plot
        fig = go.Figure(data=[go.Scatter3d(
            x=embeddings[:, 0],
            y=embeddings[:, 1],
            z=embeddings[:, 2],
            mode='markers',
            text=node_text,
            hoverinfo='text',
            marker=dict(
                size=8,
                color=node_colors,
                line=dict(width=1, color='white')
            )
        )])
        fig.update_layout(
            scene=dict(
                xaxis_title='Dim 1',
                yaxis_title='Dim 2',
                zaxis_title='Dim 3',
                bgcolor='white'
            ),
            margin=dict(l=0, r=0, b=0, t=0)
        )
    else:
        # 2D plot
        fig = go.Figure(data=[go.Scatter(
            x=embeddings[:, 0],
            y=embeddings[:, 1],
            mode='markers',
            text=node_text,
            hoverinfo='text',
            marker=dict(
                size=12,
                color=node_colors,
                line=dict(width=1, color='white')
            )
        )])
        fig.update_layout(
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            showlegend=False,
            plot_bgcolor='white',
            margin=dict(l=40, r=20, b=40, t=20)
        )
    
    return fig


if __name__ == '__main__':
    print("Starting Groggy Similarity Embedding Explorer...")
    print("Make sure groggy is installed: maturin develop --release")
    app.run(debug=True, port=8052)
