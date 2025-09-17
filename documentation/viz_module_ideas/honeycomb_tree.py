# honeycomb_graph.py
# pip install numpy matplotlib networkx imageio torch

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import imageio.v2 as imageio
import networkx as nx

# Try to import PyTorch for advanced energy optimization
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not found. Falling back to NumPy-only optimization.")

# ------------------------- config knobs -------------------------
SEED         = 9
DEFAULT_N_NODES = 20        # default for generated graphs
HEX_R        = 0.10         # hex cell radius (smaller => more cells)
DEFAULT_GIF_FRAMES = 12     # default frame count
DEFAULT_GIF_DUR_S  = 0.18   # default per-frame duration seconds
SHOW_EDGES_IN_GIF = True    # True draws edges, but makes GIF slower/heavier

# Available rotation types
ROTATION_TYPES = ["rotate", "oscillate", "spiral", "zoom", "breathe", "wobble", "nd_rotate"]

# n-D embedding options
EMBEDDING_TYPES = ["energy_2d", "spectral", "random_nd"]

# Optimization method: "numpy" or "torch_energy" (requires PyTorch)
OPTIMIZER    = "torch_energy" if HAS_TORCH else "numpy"

# Graph generation options
GRAPH_TYPES = {
    "random_tree": lambda n, seed: nx.random_tree(n=n, seed=seed),
    "barabasi_albert": lambda n, seed: nx.barabasi_albert_graph(n=n, m=2, seed=seed),
    "watts_strogatz": lambda n, seed: nx.watts_strogatz_graph(n=n, k=4, p=0.3, seed=seed),
    "erdos_renyi": lambda n, seed: nx.erdos_renyi_graph(n=n, p=0.1, seed=seed),
    "karate_club": lambda n, seed: nx.karate_club_graph(),
    "les_miserables": lambda n, seed: nx.les_miserables_graph(),
    "florentine_families": lambda n, seed: nx.florentine_families_graph(),
    "petersen": lambda n, seed: nx.petersen_graph(),
    "complete": lambda n, seed: nx.complete_graph(n),
    "cycle": lambda n, seed: nx.cycle_graph(n),
    "grid_2d": lambda n, seed: nx.grid_2d_graph(int(np.sqrt(n)), int(np.sqrt(n))),
}

# ------------------------- utils -------------------------
def hex_grid_in_circle(radius=1.0, hex_radius=0.1, margin=0.02):
    """Centers of a pointy-top hex grid clipped to a circle."""
    R = hex_radius
    w = math.sqrt(3) * R     # x spacing
    h = 1.5 * R              # y spacing
    centers = []
    y = -radius + R
    while y <= radius - R:
        row = int(round((y + radius - R) / h))
        x_offset = 0.0 if (row % 2 == 0) else (w / 2)
        x = -radius + R + x_offset
        while x <= radius - R:
            if (x*x + y*y) <= (radius - margin) ** 2:
                centers.append([x, y])
            x += w
        y += h
    return np.array(centers)

def assign_unique_cells(points2, centers2):
    """Greedy, globally-nearest unique assignment of points -> hex centers."""
    N, M = len(points2), len(centers2)
    D = np.sqrt(((points2[:, None, :] - centers2[None, :, :]) ** 2).sum(axis=2))
    assigned = np.full(N, -1, dtype=int)
    used = set(); taken = set()
    flat = np.argsort(D, axis=None)  # all (i,j) sorted by distance
    for fi in flat:
        i, j = divmod(fi, M)
        if i in taken or j in used: 
            continue
        assigned[i] = j
        taken.add(i); used.add(j)
        if len(taken) == N:
            break
    return assigned

def optimize_flat_embedding(N, edges, steps=200, lr=0.15,
                            lam_rep=0.25, p=1.4, lam_var=0.15,
                            radius_clip=0.90, seed=0):
    """
    Learn 2D coords Y that separate nodes (repulsion + spread) while
    keeping adjacent nodes reasonable (Laplacian smoothness).
    All NumPy, O(N^2) repulsion (fine for a few hundred nodes).
    """
    rng = np.random.default_rng(seed)
    Y = rng.normal(size=(N, 2)) * 0.15

    # Graph Laplacian for edge cohesion
    deg = np.zeros(N)
    for (u, v) in edges:
        deg[u] += 1; deg[v] += 1
    L = np.diag(deg)
    for (u, v) in edges:
        L[u, v] -= 1; L[v, u] -= 1

    for t in range(steps):
        # 1) edge cohesion grad: 2 L Y
        grad = 2.0 * (L @ Y)

        # 2) repulsion: -p * sum_j ((Yi - Yj) / ||Yi-Yj||^(p+2))
        diff = Y[:, None, :] - Y[None, :, :]           # [N,N,2]
        d2 = (diff ** 2).sum(-1) + 1e-9                # [N,N]
        np.fill_diagonal(d2, 1.0)                      # avoid inf on diag
        inv = d2 ** (-(p/2) - 1.0)                     # 1/r^(p+2)
        np.fill_diagonal(inv, 0.0)
        rep_grad = -p * (inv[:, :, None] * diff).sum(axis=1)
        grad += lam_rep * rep_grad

        # 3) spread: maximize variance => -lam_var * d/dY trace(Cov)
        mu = Y.mean(axis=0, keepdims=True)
        grad += -lam_var * (2.0 * (Y - mu) / max(1, N - 1))

        # step
        Y -= lr * grad

        # clip to disk to keep it inside the circle
        r = np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12
        Y /= np.maximum(1.0, r / radius_clip)

        # a touch of decay keeps it stable
        if (t + 1) % 80 == 0:
            lr *= 0.6
    return Y

def optimize_flat_embedding_torch(N, edges, steps=800, lr=3e-2,
                                 lam_stress=0.0, lam_rep=0.1, p=1.5, lam_spread=0.05,
                                 radius_clip=0.98, seed=0):
    """
    PyTorch-based energy optimization that actively diverges similar nodes.
    
    Energy function:
    L(Y) = edge_cohesion + λ_stress * tree_stress + λ_rep * repulsion - λ_spread * log_det(Cov)
    
    This forces orthogonality between similar branches while respecting tree structure.
    """
    if not HAS_TORCH:
        print("PyTorch not available, falling back to NumPy method")
        return optimize_flat_embedding(N, edges, seed=seed)
    
    torch.manual_seed(seed)
    
    # Initialize positions
    Y = torch.randn(N, 2, requires_grad=True)
    
    # Convert edges to tensor
    edges_tensor = torch.tensor(edges, dtype=torch.long)
    
    # Optimizer
    opt = torch.optim.Adam([Y], lr=lr)
    
    for t in range(steps):
        opt.zero_grad()
        
        # Project to disk (radial clip to unit circle)
        with torch.no_grad():
            r = Y.norm(dim=1, keepdim=True) + 1e-9
            Y.data = Y.data / torch.clamp(r / radius_clip, min=1.0)
        
        # Compute energy
        loss = flat_energy_torch(Y, edges_tensor, 
                               lam_stress=lam_stress, lam_rep=lam_rep, 
                               p=p, lam_spread=lam_spread)
        
        loss.backward()
        opt.step()
        
        # Optional: print progress
        if t % 200 == 0:
            print(f"Step {t}: Loss = {loss.item():.6f}")
    
    return Y.detach().numpy()

def flat_energy_torch(Y, edges, D_pairs=None, lam_stress=0.0, lam_rep=0.1, p=1.5, lam_spread=0.05):
    """
    Energy function for diverging flat embeddings.
    
    Y: [N,2] node positions
    edges: [E,2] edge list
    """
    
    # 1) Edge cohesion: keeps tree neighbors reasonably close
    e_u, e_v = edges[:, 0], edges[:, 1]
    L_local = ((Y[e_u] - Y[e_v]) ** 2).sum(dim=1).mean()
    
    # 2) Optional tree stress (not used by default)
    L_stress = torch.tensor(0.0, device=Y.device)
    if D_pairs is not None and lam_stress > 0:
        (i, j), Dij = D_pairs["idx"], D_pairs["dist"].to(Y.device)
        d = (Y[i] - Y[j]).norm(dim=1) + 1e-6
        L_stress = ((d - Dij) ** 2).mean()
    
    # 3) Repulsion: pushes all nodes apart (key for divergence!)
    # O(N^2) - fine for moderate N, use neighbor sampling for large graphs
    diff = Y.unsqueeze(1) - Y.unsqueeze(0)  # [N,N,2]
    d2 = (diff ** 2).sum(-1) + 1e-6          # [N,N] 
    L_rep = (d2.pow(-p/2)).mean()            # 1/||y_i - y_j||^p
    
    # 4) Spread: maximize covariance determinant (encourages orthogonal axes)
    C = torch.cov(Y.T) + 1e-3 * torch.eye(2, device=Y.device)
    L_spread = -torch.logdet(C)
    
    return L_local + lam_stress * L_stress + lam_rep * L_rep + lam_spread * L_spread

def get_node_embeddings(G, embedding_type="energy_2d", n_dims=10):
    """Generate high-dimensional node embeddings."""
    N = G.number_of_nodes()
    
    if embedding_type == "energy_2d":
        # Use our existing energy optimization (returns 2D)
        edges = np.array(G.edges(), dtype=int)
        if OPTIMIZER == "torch_energy" and HAS_TORCH:
            Y = optimize_flat_embedding_torch(N, edges, seed=SEED)
        else:
            Y = optimize_flat_embedding(N, edges, seed=SEED)
        return Y, 2
        
    elif embedding_type == "spectral":
        # Use Laplacian eigenvectors as high-D node features
        try:
            L = nx.laplacian_matrix(G).astype(float)
            evals, evecs = np.linalg.eigh(L.toarray())
            # Use the first n_dims eigenvectors (skip the constant eigenvector)
            # Note: eigenvalues are sorted ascending, so we take indices 1:n_dims+1
            X = evecs[:, 1:min(n_dims+1, N)]
            actual_dims = X.shape[1]
            print(f"Using {actual_dims} spectral dimensions")
            return X, actual_dims
        except Exception as e:
            print(f"Spectral embedding failed: {e}, falling back to energy_2d")
            return get_node_embeddings(G, "energy_2d", n_dims)
            
    elif embedding_type == "random_nd":
        # Random high-dimensional embeddings (for testing)
        rng = np.random.default_rng(SEED)
        X = rng.normal(size=(N, n_dims))
        # Normalize
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        return X, n_dims
        
    else:
        return get_node_embeddings(G, "energy_2d", n_dims)

# ------------------------- main -------------------------
def create_graph(graph_type="random_tree", n_nodes=None, seed=None, graph_file=None):
    """Create or load a NetworkX graph."""
    if graph_file:
        # Load from file (supports GraphML, GML, etc.)
        if graph_file.endswith('.graphml'):
            G = nx.read_graphml(graph_file)
        elif graph_file.endswith('.gml'):
            G = nx.read_gml(graph_file)
        elif graph_file.endswith('.edgelist'):
            G = nx.read_edgelist(graph_file)
        else:
            raise ValueError(f"Unsupported file format: {graph_file}")
        
        # Convert to integers if needed
        G = nx.convert_node_labels_to_integers(G)
        return G
    
    # Generate graph
    if graph_type not in GRAPH_TYPES:
        raise ValueError(f"Unknown graph type: {graph_type}. Available: {list(GRAPH_TYPES.keys())}")
    
    n_nodes = n_nodes or DEFAULT_N_NODES
    seed = seed or SEED
    
    G = GRAPH_TYPES[graph_type](n_nodes, seed)
    
    # Ensure it's connected (for better visualization)
    if not nx.is_connected(G):
        # Take the largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        print(f"Using largest connected component with {G.number_of_nodes()} nodes")
    
    return G

def get_node_colors(G, color_by="depth", root_node=None):
    """Generate node colors based on various graph properties."""
    N = G.number_of_nodes()
    cmap = plt.cm.viridis
    
    if color_by == "depth":
        # Color by distance from root
        if root_node is None:
            # Choose node with highest centrality as root
            centrality = nx.betweenness_centrality(G)
            root_node = max(centrality, key=centrality.get)
        
        depths = nx.single_source_shortest_path_length(G, root_node)
        maxd = max(depths.values()) if depths else 1
        return np.array([cmap(depths.get(i, 0) / maxd) for i in range(N)])
    
    elif color_by == "degree":
        degrees = [G.degree(i) for i in range(N)]
        max_deg = max(degrees) if degrees else 1
        return np.array([cmap(deg / max_deg) for deg in degrees])
    
    elif color_by == "betweenness":
        centrality = nx.betweenness_centrality(G)
        max_cent = max(centrality.values()) if centrality else 1
        return np.array([cmap(centrality.get(i, 0) / max_cent) for i in range(N)])
    
    elif color_by == "clustering":
        clustering = nx.clustering(G)
        max_clust = max(clustering.values()) if clustering else 1
        return np.array([cmap(clustering.get(i, 0) / max_clust) for i in range(N)])
    
    else:
        # Default: uniform coloring
        return np.array([cmap(0.5) for _ in range(N)])

def main(graph_type="random_tree", n_nodes=None, seed=None, graph_file=None, 
         color_by="depth", output_prefix=None, rotation_type="rotate", rotation_speed=1.0,
         embedding_type="energy_2d", n_dims=10, hex_radius=None, gif_frames=None, gif_duration=None):
    
    # Create graph
    G = create_graph(graph_type, n_nodes, seed, graph_file)
    edges = np.array(G.edges(), dtype=int)
    N = G.number_of_nodes()
    
    print(f"Graph: {N} nodes, {G.number_of_edges()} edges")
    
    # Generate output filenames
    if not output_prefix:
        if graph_file:
            base_name = graph_file.split('/')[-1].split('.')[0]
        else:
            base_name = f"{graph_type}_{N}"
        output_prefix = f"honeycomb_{base_name}"
    
    png_path = f"{output_prefix}_{OPTIMIZER}.png"
    gif_path = f"{output_prefix}_{OPTIMIZER}_rotate.gif"

    # Get node embeddings (2D or n-D)
    print(f"Using {embedding_type} embeddings...")
    X, actual_dims = get_node_embeddings(G, embedding_type, n_dims)
    
    if embedding_type == "energy_2d":
        Y = X  # Already 2D from energy optimization
        print(f"Using {OPTIMIZER} optimization method...")
    else:
        # For n-D embeddings, we'll project them to 2D for the static image
        if actual_dims > 2:
            # Simple PCA projection for static image
            from sklearn.decomposition import PCA
            try:
                pca = PCA(n_components=2)
                Y = pca.fit_transform(X)
                # Normalize to unit circle
                r = np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12
                Y = Y / np.maximum(1.0, r / 0.95)
            except ImportError:
                # Fallback without sklearn
                U, s, Vt = np.linalg.svd(X, full_matrices=False)
                Y = U[:, :2] @ np.diag(s[:2])
                r = np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12
                Y = Y / np.maximum(1.0, r / 0.95)
        else:
            Y = X

    # Build honeycomb centers (ensure honeycomb >> nodes)
    hex_r = hex_radius or HEX_R
    centers = hex_grid_in_circle(hex_radius=hex_r, margin=0.02)
    
    print(f"Initial honeycomb has {len(centers)} cells for {N} nodes")
    
    # Auto-adjust hex size if needed
    if len(centers) < N:
        print(f"Not enough hex cells ({len(centers)} < {N}), making grid denser...")
        hex_r = hex_r * 0.85
        centers = hex_grid_in_circle(hex_radius=hex_r, margin=0.02)
        if len(centers) < N:
            hex_r = hex_r * 0.82  # 0.85 * 0.82 ≈ 0.70
            centers = hex_grid_in_circle(hex_radius=hex_r, margin=0.02)
        print(f"Adjusted to {len(centers)} hex cells (radius={hex_r:.3f})")
    
    # Update the effective hex radius for rendering
    effective_hex_r = hex_r

    # Unique assign: each node -> one hex cell
    idx = assign_unique_cells(Y, centers)
    snapped = centers[idx]

    # Colors by graph properties
    node_colors = get_node_colors(G, color_by=color_by)

    # --------- static PNG (with edges) ---------
    fig = plt.figure(figsize=(4.5, 4.5), dpi=110)
    ax = plt.gca(); ax.set_aspect('equal'); ax.axis('off')
    
    graph_name = graph_type if not graph_file else graph_file.split('/')[-1].split('.')[0]
    ax.set_title(f"{N}-node {graph_name} — {OPTIMIZER} → honeycomb", pad=8)

    for (u, v) in edges:
        ax.plot([snapped[u, 0], snapped[v, 0]],
                [snapped[u, 1], snapped[v, 1]],
                linewidth=1.1, alpha=0.35, color="gray")

    for (x, y), col in zip(snapped, node_colors):
        ax.add_patch(RegularPolygon((x, y), 6, radius=effective_hex_r*0.85,
                                    orientation=np.pi/6,
                                    facecolor=col, edgecolor="black", linewidth=0.4))

    t = np.linspace(0, 2*np.pi, 360)
    ax.plot(np.cos(t), np.sin(t), linewidth=1.0, alpha=0.6)
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")

    # --------- animated GIF with various rotation types ---------
    frames = gif_frames or DEFAULT_GIF_FRAMES
    duration = gif_duration or DEFAULT_GIF_DUR_S
    
    if rotation_type == "nd_rotate" and actual_dims > 2:
        create_nd_animation_gif(X, centers, edges, node_colors, gif_path, rotation_speed, 
                              effective_hex_r, frames, duration)
    else:
        create_animation_gif(Y, centers, edges, node_colors, gif_path, rotation_type, rotation_speed,
                           effective_hex_r, frames, duration)

def create_animation_gif(Y, centers, edges, node_colors, gif_path, rotation_type="rotate", speed=1.0,
                       hex_r=None, gif_frames=None, gif_duration=None):
    """Create animated GIF with different rotation/transformation types."""
    
    hex_r = hex_r or HEX_R
    gif_frames = gif_frames or DEFAULT_GIF_FRAMES
    gif_duration = gif_duration or DEFAULT_GIF_DUR_S
    
    def apply_transformation(pts, frame_idx, total_frames, transform_type, rotation_speed=1.0):
        """Apply different transformation types to points."""
        t = frame_idx / total_frames  # normalized time [0,1]
        t_speed = t * rotation_speed  # apply speed multiplier
        
        if transform_type == "rotate":
            # Simple continuous rotation
            angle = 2 * math.pi * t_speed
            c, s = math.cos(angle), math.sin(angle)
            R = np.array([[c, -s], [s, c]])
            return pts @ R.T
            
        elif transform_type == "oscillate":
            # Back-and-forth rotation (pendulum)
            angle = math.pi * 0.8 * math.sin(2 * math.pi * t_speed)  # ±0.8π swing
            c, s = math.cos(angle), math.sin(angle)
            R = np.array([[c, -s], [s, c]])
            return pts @ R.T
            
        elif transform_type == "spiral":
            # Rotate + slight zoom in/out
            angle = 2 * math.pi * t_speed
            scale = 0.85 + 0.3 * math.sin(2 * math.pi * t_speed)  # zoom 0.85-1.15x
            c, s = math.cos(angle), math.sin(angle)
            R = np.array([[c, -s], [s, c]]) * scale
            return pts @ R.T
            
        elif transform_type == "zoom":
            # Pulsing zoom (no rotation)
            scale = 0.7 + 0.4 * (0.5 + 0.5 * math.cos(2 * math.pi * t_speed))  # 0.7-1.1x
            return pts * scale
            
        elif transform_type == "breathe":
            # Gentle radial breathing
            scale = 0.9 + 0.2 * math.sin(2 * math.pi * t_speed)  # 0.9-1.1x smooth
            return pts * scale
            
        elif transform_type == "wobble":
            # Rotation + slight x/y shear for "wobble" effect
            angle = 2 * math.pi * t_speed
            shear_x = 0.1 * math.sin(4 * math.pi * t_speed)
            shear_y = 0.1 * math.cos(4 * math.pi * t_speed) 
            c, s = math.cos(angle), math.sin(angle)
            R = np.array([[c + shear_x, -s + shear_y], 
                         [s + shear_y, c + shear_x]])
            return pts @ R.T
            
        elif transform_type == "nd_rotate":
            # This is handled separately in create_nd_animation_gif
            return pts
            
        else:
            # Default: simple rotation
            angle = 2 * math.pi * t_speed
            c, s = math.cos(angle), math.sin(angle)
            R = np.array([[c, -s], [s, c]])
            return pts @ R.T

    frames = []
    for f in range(gif_frames):
        # Apply transformation
        Yf = apply_transformation(Y, f, gif_frames, rotation_type, speed)
        idxf = assign_unique_cells(Yf, centers)
        snapf = centers[idxf]

        fig = plt.figure(figsize=(3.6, 3.6), dpi=80)
        ax = plt.gca(); ax.set_aspect('equal'); ax.axis('off')

        if SHOW_EDGES_IN_GIF:
            for (u, v) in edges:
                ax.plot([snapf[u, 0], snapf[v, 0]],
                        [snapf[u, 1], snapf[v, 1]],
                        linewidth=1.0, alpha=0.28, color="gray")

        # node hexes
        for (x, y), col in zip(snapf, node_colors):
            ax.add_patch(RegularPolygon((x, y), 6, radius=hex_r*0.85,
                                        orientation=np.pi/6,
                                        facecolor=col, edgecolor="black", linewidth=0.35))
        # circle boundary
        tt = np.linspace(0, 2*np.pi, 240)
        ax.plot(np.cos(tt), np.sin(tt), linewidth=1.0, alpha=0.45)

        fig.savefig('temp_frame.png', bbox_inches='tight', dpi=80)
        frame = imageio.imread('temp_frame.png')[:, :, :3]
        plt.close(fig)
        frames.append(frame)

    imageio.mimsave(gif_path, frames, duration=gif_duration)
    print(f"Saved: {gif_path} ({gif_frames} frames, {gif_duration*1000:.0f}ms/frame)")
    
    # Cleanup temp file
    import os
    if os.path.exists('temp_frame.png'):
        os.remove('temp_frame.png')

def create_nd_animation_gif(X, centers, edges, node_colors, gif_path, speed=1.0, 
                          hex_r=None, gif_frames=None, gif_duration=None, rotation_axes=None):
    """Create n-dimensional rotation animation by rotating the 2D view subspace through n-D space."""
    
    hex_r = hex_r or HEX_R
    gif_frames = gif_frames or DEFAULT_GIF_FRAMES
    gif_duration = gif_duration or DEFAULT_GIF_DUR_S
    
    N, d = X.shape
    print(f"Creating n-D rotation through {d}-dimensional space")
    
    # Initialize orthogonal 2D view basis V ∈ R^{d×2}
    rng = np.random.default_rng(SEED)
    V = rng.normal(size=(d, 2))
    V, _ = np.linalg.qr(V)  # Make orthogonal
    
    # Default rotation axes: cycle through all possible 2-planes
    if rotation_axes is None:
        # Create rotation sequence through different 2-planes in n-D space
        rotation_axes = []
        for i in range(min(d, 8)):  # Limit to avoid too many rotations
            for j in range(i+1, min(d, i+4)):  # Each axis with a few others
                rotation_axes.append((i, j))
    
    frames = []
    total_rotations = len(rotation_axes)
    
    for f in range(gif_frames):
        # Choose which 2-plane to rotate in
        rot_idx = int((f / gif_frames) * total_rotations * speed) % total_rotations
        axis_i, axis_j = rotation_axes[rot_idx]
        
        # Create Givens rotation in the selected 2-plane
        angle = 2 * math.pi * (f / gif_frames) * speed
        c, s = math.cos(angle), math.sin(angle)
        
        # Apply Givens rotation G(i,j,θ) to the view basis V
        V_rot = V.copy()
        # Givens rotation: rotate coordinates i and j
        V_rot[axis_i, :] = c * V[axis_i, :] - s * V[axis_j, :]
        V_rot[axis_j, :] = s * V[axis_i, :] + c * V[axis_j, :]
        
        # Re-orthogonalize to prevent drift
        V_rot, _ = np.linalg.qr(V_rot)
        
        # Project n-D embeddings to current 2D view
        Y_2d = X @ V_rot
        
        # Normalize to unit circle
        r = np.linalg.norm(Y_2d, axis=1, keepdims=True) + 1e-12
        Y_2d = Y_2d / np.maximum(1.0, r / 0.95)  # Keep within circle
        
        # Assign to honeycomb
        idxf = assign_unique_cells(Y_2d, centers)
        snapf = centers[idxf]
        
        # Render frame
        fig = plt.figure(figsize=(3.6, 3.6), dpi=80)
        ax = plt.gca(); ax.set_aspect('equal'); ax.axis('off')

        if SHOW_EDGES_IN_GIF:
            for (u, v) in edges:
                ax.plot([snapf[u, 0], snapf[v, 0]],
                        [snapf[u, 1], snapf[v, 1]],
                        linewidth=1.0, alpha=0.28, color="gray")

        # node hexes
        for (x, y), col in zip(snapf, node_colors):
            ax.add_patch(RegularPolygon((x, y), 6, radius=hex_r*0.85,
                                        orientation=np.pi/6,
                                        facecolor=col, edgecolor="black", linewidth=0.35))
        # circle boundary + show which axes we're rotating
        tt = np.linspace(0, 2*np.pi, 240)
        ax.plot(np.cos(tt), np.sin(tt), linewidth=1.0, alpha=0.45)
        ax.text(0, -1.15, f"Rotating axes {axis_i}↔{axis_j}", 
                ha='center', va='center', fontsize=8, alpha=0.7)

        fig.savefig('temp_frame.png', bbox_inches='tight', dpi=80)
        frame = imageio.imread('temp_frame.png')[:, :, :3]
        plt.close(fig)
        frames.append(frame)

    imageio.mimsave(gif_path, frames, duration=gif_duration)
    print(f"Saved: {gif_path} ({gif_frames} frames, {gif_duration*1000:.0f}ms/frame)")
    
    # Cleanup temp file
    import os
    if os.path.exists('temp_frame.png'):
        os.remove('temp_frame.png')

def compare_methods(graph_type="barabasi_albert", n_nodes=50):
    """Generate outputs with both optimization methods for comparison."""
    global OPTIMIZER
    
    original_optimizer = OPTIMIZER
    
    # Test NumPy method
    OPTIMIZER = "numpy"
    print("\n" + "="*50)
    print("GENERATING NUMPY VERSION")
    print("="*50)
    main(graph_type=graph_type, n_nodes=n_nodes, output_prefix=f"compare_{graph_type}_{n_nodes}")
    
    # Test PyTorch method (if available)
    if HAS_TORCH:
        OPTIMIZER = "torch_energy"
        print("\n" + "="*50)
        print("GENERATING TORCH ENERGY VERSION")
        print("="*50)
        main(graph_type=graph_type, n_nodes=n_nodes, output_prefix=f"compare_{graph_type}_{n_nodes}")
    
    # Restore original
    OPTIMIZER = original_optimizer
    print(f"\n✅ Comparison complete! Check the generated files.")
    print(f"NumPy method: compare_{graph_type}_{n_nodes}_numpy.png")
    if HAS_TORCH:
        print(f"PyTorch energy method: compare_{graph_type}_{n_nodes}_torch_energy.png")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate honeycomb visualizations of NetworkX graphs")
    parser.add_argument("--graph-type", "-g", default="barabasi_albert", 
                        choices=list(GRAPH_TYPES.keys()),
                        help="Type of graph to generate")
    parser.add_argument("--nodes", "-n", type=int, default=50,
                        help="Number of nodes (for generated graphs)")
    parser.add_argument("--file", "-f", type=str, 
                        help="Load graph from file (.graphml, .gml, .edgelist)")
    parser.add_argument("--color-by", "-c", default="betweenness",
                        choices=["depth", "degree", "betweenness", "clustering"],
                        help="Node coloring scheme")
    parser.add_argument("--seed", "-s", type=int, default=SEED,
                        help="Random seed")
    parser.add_argument("--optimizer", "-o", default=OPTIMIZER,
                        choices=["numpy", "torch_energy"],
                        help="Optimization method")
    parser.add_argument("--rotation-type", "-r", default="rotate",
                        choices=ROTATION_TYPES,
                        help="Animation rotation/transformation type")
    parser.add_argument("--rotation-speed", "-rs", type=float, default=1.0,
                        help="Rotation speed multiplier (0.5=slow, 2.0=fast)")
    parser.add_argument("--embedding-type", "-e", default="energy_2d",
                        choices=EMBEDDING_TYPES,
                        help="Type of node embeddings to use")
    parser.add_argument("--n-dims", "-nd", type=int, default=10,
                        help="Number of dimensions for high-D embeddings")
    parser.add_argument("--hex-radius", "-hr", type=float, default=HEX_R,
                        help="Honeycomb cell radius (smaller=more cells)")
    parser.add_argument("--gif-frames", "-gf", type=int, default=DEFAULT_GIF_FRAMES,
                        help="Number of frames in animation")
    parser.add_argument("--gif-duration", "-gd", type=float, default=DEFAULT_GIF_DUR_S,
                        help="Duration per frame in seconds")
    parser.add_argument("--compare", action="store_true",
                        help="Generate both numpy and torch versions for comparison")
    
    args = parser.parse_args()
    
    # Update global settings
    OPTIMIZER = args.optimizer
    
    if args.compare:
        compare_methods(args.graph_type, args.nodes)
    else:
        main(graph_type=args.graph_type, 
             n_nodes=args.nodes,
             seed=args.seed,
             graph_file=args.file,
             color_by=args.color_by,
             rotation_type=args.rotation_type,
             rotation_speed=args.rotation_speed,
             embedding_type=args.embedding_type,
             n_dims=args.n_dims,
             hex_radius=args.hex_radius,
             gif_frames=args.gif_frames,
             gif_duration=args.gif_duration)