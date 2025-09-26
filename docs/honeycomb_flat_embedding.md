# Honeycomb Flat Embedding TODOs

This page tracks the outstanding work required to support the "flat, divergent embedding" pathway described in the recent design conversation.

## Core Tasks

- [ ] Implement the flat energy solver in `src/viz/layouts/flat_embedding.rs`
  - Support direct optimisation of `Y \in R^{N x 2}`
  - Add optional linear projector variant `W \in R^{d x 2}` with Stiefel constraint
  - Include configurable weights for cohesion, repulsion, spread, and (optional) stress
  - Provide a CPU-friendly minibatch option for large graphs
- [ ] Integrate the solver with `RealTimeVizEngine::apply_layout_algorithm`
  - Honour new layout params (`layout.flat_embed`, `layout.repulsion_power`, etc.)
  - Cache the resulting 2D positions alongside the existing embedding cache
- [ ] Extend the honeycomb quantisation to use a unique assignment (Hungarian/Sinkhorn)
  - Build hex-centre lookup given a target radius and granularity
  - Offer soft assignment during interaction and harden on release
- [ ] Expose configuration in the UI drawer (layout tab)
  - Toggle for flat embedding, sliders for repulsion & spread weights
  - Optional projector mode dropdown
- [ ] Update documentation/examples once the flow is stabilised

## Nice-to-haves

- Depth-aware repulsion scaling (branch separation)
- Community centroid repulsion (push whole subtrees apart)
- Live steering: retain projector `W` and allow small rotations + micro-optimisation rounds

Refer back to `sleek.css` for visual tweaks to keep the controls consistent.
