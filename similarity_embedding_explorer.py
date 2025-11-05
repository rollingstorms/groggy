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

def compute_similarity_matrix(G, nodes, method="adjacency"):
    """
    Compute similarity matrix S from graph.
    
    Methods:
    - adjacency: Binary adjacency (1 if connected, 0 otherwise)
    - weighted: Use edge weights if available
    - gaussian: Gaussian similarity based on node positions
    - cosine: Cosine similarity of node features
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
            "extra": G.nodes[node].get('extra', 0)
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
            html.Button("Generate Graph", id="generate-btn", n_clicks=0,
                       style={"width": "100%", "padding": "10px", "fontSize": "16px"}),
            
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
                       style={"width": "100%", "padding": "10px", "fontSize": "16px"}),
            
            html.Hr(),
            
            html.H3("Clustering"),
            html.Label("Number of Clusters:"),
            dcc.Slider(id="num-clusters", min=2, max=8, step=1, value=3,
                      marks={i: str(i) for i in range(2, 9)}),
            html.Br(),
            html.Button("Cluster Embeddings", id="cluster-btn", n_clicks=0,
                       style={"width": "100%", "padding": "10px", "fontSize": "16px"}),
            
            html.Hr(),
            html.Div(id="stats-output", style={"fontSize": "12px", "marginTop": "20px"})
            
        ], style={"width": "25%", "padding": "20px", "overflowY": "auto", "height": "90vh"}),
        
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
    G = generate_graph(family, num_nodes)
    return graph_to_data(G)


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
    
    # Build NetworkX graph
    G = nx.Graph()
    for node_id, props in graph_data["nodes"].items():
        G.add_node(node_id, **props)
    for u, v in graph_data["edges"]:
        G.add_edge(u, v)
    
    nodes = sorted(G.nodes())
    
    # Compute similarity matrix
    S = compute_similarity_matrix(G, nodes, method=sim_method)
    
    # Compute embeddings
    if embed_method == "eigen":
        embeddings, eigenvalues = compute_normalized_laplacian_embeddings(S, k)
    else:
        embeddings = power_iteration_embeddings(S, k)
        eigenvalues = np.array([])
    
    # Store embeddings
    embedding_data = {
        node: embeddings[i].tolist() for i, node in enumerate(nodes)
    }
    
    # Stats
    stats = html.Div([
        html.P(f"Nodes: {len(nodes)}"),
        html.P(f"Edges: {len(G.edges())}"),
        html.P(f"Similarity: {sim_method}"),
        html.P(f"Method: {embed_method}"),
        html.P(f"Dimensions: {k}"),
        html.P(f"Avg Similarity: {S.mean():.3f}"),
        html.P(f"Sparsity: {(S > 0).mean():.3f}"),
    ])
    
    if len(eigenvalues) > 0:
        top_eigs = eigenvalues[:min(5, len(eigenvalues))]
        eig_str = ", ".join([f"{e:.3f}" for e in top_eigs])
        stats.children.append(html.P(f"Top eigenvalues: {eig_str}"))
    
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
    
    nodes = sorted(embedding_data.keys())
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
    node_text = [f"Node {n}" for n in nodes]
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
    
    nodes = sorted(embedding_data.keys())
    embeddings = np.array([embedding_data[n] for n in nodes])
    
    if embeddings.shape[1] < 2:
        return go.Figure()
    
    # Create node index mapping
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # Get colors from clusters if available
    node_colors = [graph_data["nodes"][n].get("color", "lightblue") for n in nodes]
    node_text = [f"Node {n}" for n in nodes]
    
    # Get edges from graph data
    edges = graph_data.get("edges", [])
    
    if embeddings.shape[1] >= 3:
        # 3D plot with edges
        edge_x, edge_y, edge_z = [], [], []
        for u, v in edges:
            if u in node_to_idx and v in node_to_idx:
                u_idx = node_to_idx[u]
                v_idx = node_to_idx[v]
                edge_x.extend([embeddings[u_idx, 0], embeddings[v_idx, 0], None])
                edge_y.extend([embeddings[u_idx, 1], embeddings[v_idx, 1], None])
                edge_z.extend([embeddings[u_idx, 2], embeddings[v_idx, 2], None])
        
        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        node_trace = go.Scatter3d(
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
        )
        
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            scene=dict(
                xaxis_title='Dim 1',
                yaxis_title='Dim 2',
                zaxis_title='Dim 3',
                bgcolor='white'
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            showlegend=False
        )
    else:
        # 2D plot with edges
        edge_x, edge_y = [], []
        for u, v in edges:
            if u in node_to_idx and v in node_to_idx:
                u_idx = node_to_idx[u]
                v_idx = node_to_idx[v]
                edge_x.extend([embeddings[u_idx, 0], embeddings[v_idx, 0], None])
                edge_y.extend([embeddings[u_idx, 1], embeddings[v_idx, 1], None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        node_trace = go.Scatter(
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
        )
        
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            showlegend=False,
            plot_bgcolor='white',
            margin=dict(l=40, r=20, b=40, t=20)
        )
    
    return fig


if __name__ == '__main__':
    app.run(debug=True, port=8051)
