//! Honeycomb coordinate system and grid mapping utilities

use super::{HoneycombConfig, HoneycombLayoutStrategy};
use crate::api::graph::Graph;
use crate::errors::{GraphError, GraphResult};
use crate::storage::matrix::GraphMatrix;
use crate::viz::embeddings::flat_embedding::{compute_flat_embedding, FlatEmbedConfig};
use crate::viz::streaming::data_source::Position;
use std::collections::HashMap;

/// Hexagonal coordinates using axial coordinate system (q, r)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HexCoord {
    pub q: i32,
    pub r: i32,
}

impl HexCoord {
    pub fn new(q: i32, r: i32) -> Self {
        Self { q, r }
    }

    /// Convert to cube coordinates (x, y, z) where x + y + z = 0
    pub fn to_cube(&self) -> (i32, i32, i32) {
        let x = self.q;
        let z = self.r;
        let y = -x - z;
        (x, y, z)
    }

    /// Create from cube coordinates
    pub fn from_cube(x: i32, y: i32, z: i32) -> Result<Self, GraphError> {
        if x + y + z != 0 {
            return Err(GraphError::InvalidInput(
                "Invalid cube coordinates: x + y + z must equal 0".to_string(),
            ));
        }
        Ok(Self { q: x, r: z })
    }

    /// Distance between two hex coordinates
    pub fn distance(&self, other: &HexCoord) -> i32 {
        let (x1, y1, z1) = self.to_cube();
        let (x2, y2, z2) = other.to_cube();
        ((x1 - x2).abs() + (y1 - y2).abs() + (z1 - z2).abs()) / 2
    }

    /// Get the 6 neighboring hex coordinates
    pub fn neighbors(&self) -> Vec<HexCoord> {
        let directions = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)];

        directions
            .iter()
            .map(|(dq, dr)| HexCoord::new(self.q + dq, self.r + dr))
            .collect()
    }

    /// Linear interpolation between two hex coordinates
    pub fn lerp(&self, other: &HexCoord, t: f64) -> HexCoord {
        let q = self.q as f64 + t * (other.q - self.q) as f64;
        let r = self.r as f64 + t * (other.r - self.r) as f64;
        HexCoord::new(q.round() as i32, r.round() as i32)
    }
}

/// Honeycomb grid manager for mapping positions to hexagonal coordinates
#[derive(Debug)]
pub struct HoneycombGrid {
    config: HoneycombConfig,
    occupied_cells: HashMap<HexCoord, usize>, // Maps hex coord to node index
    node_positions: HashMap<usize, HexCoord>, // Maps node index to hex coord
    pixel_positions: HashMap<HexCoord, Position>, // Cache of pixel positions
}

impl HoneycombGrid {
    /// Create a new honeycomb grid
    pub fn new(config: HoneycombConfig) -> Self {
        Self {
            config,
            occupied_cells: HashMap::new(),
            node_positions: HashMap::new(),
            pixel_positions: HashMap::new(),
        }
    }

    /// Convert hex coordinates to pixel position
    pub fn hex_to_pixel(&self, hex: &HexCoord) -> Position {
        if let Some(&cached_pos) = self.pixel_positions.get(hex) {
            return cached_pos;
        }

        let size = self.config.cell_size;
        let x = size * (3.0f64.sqrt() * hex.q as f64 + 3.0f64.sqrt() / 2.0 * hex.r as f64);
        let y = size * (3.0 / 2.0 * hex.r as f64);

        Position { x, y }
    }

    /// Convert pixel position to hex coordinates
    pub fn pixel_to_hex(&self, pos: &Position) -> HexCoord {
        let size = self.config.cell_size;
        let q = (3.0f64.sqrt() / 3.0 * pos.x - 1.0 / 3.0 * pos.y) / size;
        let r = (2.0 / 3.0 * pos.y) / size;

        // Round to nearest hex coordinates using cube coordinate rounding
        let x = q;
        let z = r;
        let y = -x - z;

        let rx = x.round();
        let ry = y.round();
        let rz = z.round();

        let x_diff = (rx - x).abs();
        let y_diff = (ry - y).abs();
        let z_diff = (rz - z).abs();

        let (final_x, final_z) = if x_diff > y_diff && x_diff > z_diff {
            (-ry - rz, rz)
        } else if y_diff > z_diff {
            (rx, -rx - ry)
        } else {
            (rx, rz)
        };

        HexCoord::new(final_x as i32, final_z as i32)
    }

    /// Map 2D positions to honeycomb grid using the configured strategy
    pub fn map_positions_to_grid(&mut self, positions: &[Position]) -> GraphResult<Vec<Position>> {
        match self.config.layout_strategy {
            HoneycombLayoutStrategy::Spiral => self.map_spiral(positions),
            HoneycombLayoutStrategy::DensityBased => self.map_density_based(positions),
            HoneycombLayoutStrategy::DistancePreserving => self.map_distance_preserving(positions),
            HoneycombLayoutStrategy::EnergyBased => self.map_energy_based(positions),
            HoneycombLayoutStrategy::Custom { ref ordering_fn } => {
                let ordering_fn_copy = ordering_fn.clone();
                self.map_custom(positions, &ordering_fn_copy)
            }
        }
    }

    /// Map positions using spiral layout
    fn map_spiral(&mut self, positions: &[Position]) -> GraphResult<Vec<Position>> {
        let n = positions.len();
        let hex_coords = self.generate_spiral_coordinates(n);

        let mut result_positions = Vec::with_capacity(n);

        for (i, &hex_coord) in hex_coords.iter().enumerate() {
            self.occupied_cells.insert(hex_coord, i);
            self.node_positions.insert(i, hex_coord);

            let pixel_pos = if self.config.snap_to_centers {
                self.hex_to_pixel(&hex_coord)
            } else {
                // Blend between original position and hex center
                let hex_pixel = self.hex_to_pixel(&hex_coord);
                Position {
                    x: 0.8 * hex_pixel.x + 0.2 * positions[i].x,
                    y: 0.8 * hex_pixel.y + 0.2 * positions[i].y,
                }
            };

            result_positions.push(pixel_pos);
        }

        Ok(result_positions)
    }

    /// Map positions preserving density patterns
    fn map_density_based(&mut self, positions: &[Position]) -> GraphResult<Vec<Position>> {
        // Compute density map of original positions
        let density_map = self.compute_density_map(positions)?;

        // Sort positions by density (highest first)
        let mut position_indices: Vec<usize> = (0..positions.len()).collect();
        position_indices.sort_by(|&a, &b| density_map[b].partial_cmp(&density_map[a]).unwrap());

        // Generate hex coordinates in spiral order
        let hex_coords = self.generate_spiral_coordinates(positions.len());

        let mut result_positions = vec![Position { x: 0.0, y: 0.0 }; positions.len()];

        for (spiral_idx, &pos_idx) in position_indices.iter().enumerate() {
            let hex_coord = hex_coords[spiral_idx];
            self.occupied_cells.insert(hex_coord, pos_idx);
            self.node_positions.insert(pos_idx, hex_coord);

            let pixel_pos = if self.config.snap_to_centers {
                self.hex_to_pixel(&hex_coord)
            } else {
                let hex_pixel = self.hex_to_pixel(&hex_coord);
                Position {
                    x: 0.9 * hex_pixel.x + 0.1 * positions[pos_idx].x,
                    y: 0.9 * hex_pixel.y + 0.1 * positions[pos_idx].y,
                }
            };

            result_positions[pos_idx] = pixel_pos;
        }

        Ok(result_positions)
    }

    /// Map positions preserving relative distances
    fn map_distance_preserving(&mut self, positions: &[Position]) -> GraphResult<Vec<Position>> {
        if positions.is_empty() {
            return Ok(Vec::new());
        }

        // Start with the central point
        let center_idx = self.find_center_point(positions);
        let center_hex = HexCoord::new(0, 0);

        self.occupied_cells.insert(center_hex, center_idx);
        self.node_positions.insert(center_idx, center_hex);

        let mut result_positions = vec![Position { x: 0.0, y: 0.0 }; positions.len()];
        result_positions[center_idx] = self.hex_to_pixel(&center_hex);

        let mut placed = vec![false; positions.len()];
        placed[center_idx] = true;
        let mut placement_queue = vec![center_idx];

        // Place nodes iteratively, trying to preserve distances
        while placement_queue.len() < positions.len() {
            let mut best_placement = None;
            let mut best_error = f64::INFINITY;

            // For each unplaced node
            for i in 0..positions.len() {
                if placed[i] {
                    continue;
                }

                // Try each available hex position near placed nodes
                for &placed_idx in &placement_queue {
                    let placed_hex = self.node_positions[&placed_idx];

                    for neighbor_hex in placed_hex.neighbors() {
                        if self.occupied_cells.contains_key(&neighbor_hex) {
                            continue;
                        }

                        // Compute placement error
                        let error =
                            self.compute_placement_error(i, &neighbor_hex, positions, &placed);

                        if error < best_error {
                            best_error = error;
                            best_placement = Some((i, neighbor_hex));
                        }
                    }
                }
            }

            // Place the best candidate
            if let Some((node_idx, hex_coord)) = best_placement {
                self.occupied_cells.insert(hex_coord, node_idx);
                self.node_positions.insert(node_idx, hex_coord);
                result_positions[node_idx] = self.hex_to_pixel(&hex_coord);
                placed[node_idx] = true;
                placement_queue.push(node_idx);
            } else {
                // Fallback: place remaining nodes in spiral order
                let remaining: Vec<usize> = (0..positions.len()).filter(|&i| !placed[i]).collect();

                let spiral_coords = self.generate_spiral_coordinates(remaining.len());
                let mut spiral_idx = 0;

                for &node_idx in &remaining {
                    // Find next available spiral position
                    while spiral_idx < spiral_coords.len()
                        && self.occupied_cells.contains_key(&spiral_coords[spiral_idx])
                    {
                        spiral_idx += 1;
                    }

                    if spiral_idx < spiral_coords.len() {
                        let hex_coord = spiral_coords[spiral_idx];
                        self.occupied_cells.insert(hex_coord, node_idx);
                        self.node_positions.insert(node_idx, hex_coord);
                        result_positions[node_idx] = self.hex_to_pixel(&hex_coord);
                        spiral_idx += 1;
                    }
                }
                break;
            }
        }

        Ok(result_positions)
    }

    /// Map positions using custom ordering function
    fn map_custom(
        &mut self,
        positions: &[Position],
        _ordering_fn: &str,
    ) -> GraphResult<Vec<Position>> {
        // For now, fallback to spiral layout
        // In a full implementation, this would parse and execute the custom function
        self.map_spiral(positions)
    }

    /// Map positions using flat embedding energy optimization + honeycomb assignment
    /// This solves quantization issues by using gradient descent for continuous optimization
    /// followed by optimal assignment to hex centers.
    pub fn map_with_flat_embedding(
        &mut self,
        embedding: &GraphMatrix,
        graph: &Graph,
    ) -> GraphResult<Vec<Position>> {
        // Use flat embedding to get optimized 2D positions
        let flat_config = FlatEmbedConfig::default();
        let optimized_positions = compute_flat_embedding(embedding, graph, &flat_config)?;

        // Assign optimized positions to honeycomb grid
        self.assign_to_honeycomb_grid(&optimized_positions)
    }

    /// Map positions using energy-based unique assignment to hex centers
    /// This is the correct honeycomb approach: positions come from energy optimization,
    /// we just assign each position to the closest available hex center.
    fn map_energy_based(&mut self, positions: &[Position]) -> GraphResult<Vec<Position>> {
        if positions.is_empty() {
            return Ok(Vec::new());
        }

        // Generate all hex centers in a circular region
        let hex_centers = self.generate_hex_centers_in_circle(positions.len() * 2);

        // Scale the hex centers from unit circle to pixel space (multiply by cell_size * 100)
        let scale_factor = self.config.cell_size * 100.0;
        let scaled_hex_centers: Vec<Position> = hex_centers
            .iter()
            .map(|pos| Position {
                x: pos.x * scale_factor,
                y: pos.y * scale_factor,
            })
            .collect();

        // Use unique assignment algorithm (greedy global nearest neighbor)
        let assignments = self.assign_unique_cells(positions, &scaled_hex_centers)?;

        let mut result_positions = Vec::with_capacity(positions.len());

        for (node_idx, &hex_center_idx) in assignments.iter().enumerate() {
            if hex_center_idx >= scaled_hex_centers.len() {
                return Err(crate::errors::GraphError::InvalidInput(format!(
                    "Invalid hex center assignment: {} >= {}",
                    hex_center_idx,
                    scaled_hex_centers.len()
                )));
            }

            let hex_center_pos = scaled_hex_centers[hex_center_idx];
            let hex_coord = self.pixel_to_hex(&hex_center_pos);

            // Track the assignment
            self.occupied_cells.insert(hex_coord, node_idx);
            self.node_positions.insert(node_idx, hex_coord);

            let final_pos = if self.config.snap_to_centers {
                hex_center_pos
            } else {
                // Blend between original position and hex center (80% hex, 20% original)
                Position {
                    x: 0.8 * hex_center_pos.x + 0.2 * positions[node_idx].x,
                    y: 0.8 * hex_center_pos.y + 0.2 * positions[node_idx].y,
                }
            };

            result_positions.push(final_pos);
        }

        //
        // Debug message
        // Debug parameters
        // Debug parameters

        Ok(result_positions)
    }

    /// Generate spiral hex coordinates
    fn generate_spiral_coordinates(&self, count: usize) -> Vec<HexCoord> {
        let mut coords = Vec::with_capacity(count);

        if count == 0 {
            return coords;
        }

        // Start with center
        coords.push(HexCoord::new(0, 0));
        if count == 1 {
            return coords;
        }

        // Generate spiral outward
        let mut ring = 1;
        while coords.len() < count {
            for side in 0..6 {
                for i in 0..ring {
                    if coords.len() >= count {
                        break;
                    }

                    let angle = side as f64 * std::f64::consts::PI / 3.0;
                    let q = (ring as f64 * angle.cos()
                        + i as f64 * (angle + std::f64::consts::PI / 3.0).cos())
                    .round() as i32;
                    let r = (ring as f64 * angle.sin()
                        + i as f64 * (angle + std::f64::consts::PI / 3.0).sin())
                    .round() as i32;

                    coords.push(HexCoord::new(q, r));
                }
            }
            ring += 1;
        }

        coords.truncate(count);
        coords
    }

    /// Compute density map for positions
    fn compute_density_map(&self, positions: &[Position]) -> GraphResult<Vec<f64>> {
        let n = positions.len();
        let mut densities = vec![0.0; n];

        let bandwidth = 50.0; // Density estimation bandwidth

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let dx = positions[i].x - positions[j].x;
                    let dy = positions[i].y - positions[j].y;
                    let dist_sq = dx * dx + dy * dy;
                    let weight = (-dist_sq / (2.0 * bandwidth * bandwidth)).exp();
                    densities[i] += weight;
                }
            }
        }

        Ok(densities)
    }

    /// Find the most central point in the position set
    fn find_center_point(&self, positions: &[Position]) -> usize {
        let mut min_total_dist = f64::INFINITY;
        let mut center_idx = 0;

        for i in 0..positions.len() {
            let mut total_dist = 0.0;
            for j in 0..positions.len() {
                if i != j {
                    let dx = positions[i].x - positions[j].x;
                    let dy = positions[i].y - positions[j].y;
                    total_dist += (dx * dx + dy * dy).sqrt();
                }
            }

            if total_dist < min_total_dist {
                min_total_dist = total_dist;
                center_idx = i;
            }
        }

        center_idx
    }

    /// Compute placement error for a node at a given hex position
    fn compute_placement_error(
        &self,
        node_idx: usize,
        hex_coord: &HexCoord,
        positions: &[Position],
        placed: &[bool],
    ) -> f64 {
        let mut error = 0.0;
        let target_pixel = self.hex_to_pixel(hex_coord);

        for (other_idx, &is_placed) in placed.iter().enumerate() {
            if !is_placed || other_idx == node_idx {
                continue;
            }

            let other_hex = self.node_positions[&other_idx];
            let other_pixel = self.hex_to_pixel(&other_hex);

            // Original distance
            let dx_orig = positions[node_idx].x - positions[other_idx].x;
            let dy_orig = positions[node_idx].y - positions[other_idx].y;
            let orig_dist = (dx_orig * dx_orig + dy_orig * dy_orig).sqrt();

            // New distance
            let dx_new = target_pixel.x - other_pixel.x;
            let dy_new = target_pixel.y - other_pixel.y;
            let new_dist = (dx_new * dx_new + dy_new * dy_new).sqrt();

            // Distance preservation error
            let dist_error = (orig_dist - new_dist).abs() / (orig_dist + 1e-12);
            error += dist_error * dist_error;
        }

        error
    }

    /// Get hex coordinate for a node
    pub fn get_hex_coord(&self, node_idx: usize) -> Option<HexCoord> {
        self.node_positions.get(&node_idx).copied()
    }

    /// Get node at hex coordinate
    pub fn get_node_at_hex(&self, hex_coord: &HexCoord) -> Option<usize> {
        self.occupied_cells.get(hex_coord).copied()
    }

    /// Check if a hex coordinate is occupied
    pub fn is_occupied(&self, hex_coord: &HexCoord) -> bool {
        self.occupied_cells.contains_key(hex_coord)
    }

    /// Get all occupied hex coordinates
    pub fn get_occupied_coords(&self) -> Vec<HexCoord> {
        self.occupied_cells.keys().copied().collect()
    }

    /// Get grid bounding box
    pub fn get_bounding_box(&self) -> Option<(HexCoord, HexCoord)> {
        if self.occupied_cells.is_empty() {
            return None;
        }

        let coords: Vec<HexCoord> = self.occupied_cells.keys().copied().collect();
        let min_q = coords.iter().map(|c| c.q).min().unwrap();
        let max_q = coords.iter().map(|c| c.q).max().unwrap();
        let min_r = coords.iter().map(|c| c.r).min().unwrap();
        let max_r = coords.iter().map(|c| c.r).max().unwrap();

        Some((HexCoord::new(min_q, min_r), HexCoord::new(max_q, max_r)))
    }

    /// Generate hex centers in a circular pattern (like the Python script)
    /// Returns pixel positions of hex centers clipped to a unit circle
    fn generate_hex_centers_in_circle(&self, min_count: usize) -> Vec<Position> {
        let radius = 1.0; // Unit circle radius
        let hex_radius = self.config.cell_size / 100.0; // Normalize cell size to reasonable range
        let margin = 0.02;

        // Hexagonal grid spacing
        let w = (3_f64).sqrt() * hex_radius; // x spacing
        let h = 1.5 * hex_radius; // y spacing

        let mut centers = Vec::new();
        let mut y = -radius + hex_radius;

        while y <= radius - hex_radius {
            let row = ((y + radius - hex_radius) / h).round() as i32;
            let x_offset = if row % 2 == 0 { 0.0 } else { w / 2.0 };
            let mut x = -radius + hex_radius + x_offset;

            while x <= radius - hex_radius {
                // Check if point is within circle with margin
                if x * x + y * y <= (radius - margin).powi(2) {
                    centers.push(Position { x, y });
                }
                x += w;
            }
            y += h;
        }

        // If we don't have enough centers, try with smaller hex radius
        if centers.len() < min_count {
            //
            // Debug message
            // Debug parameters
            // Debug parameters
            // Debug parameters
            let smaller_hex_radius = hex_radius * 0.8;
            let w = (3_f64).sqrt() * smaller_hex_radius;
            let h = 1.5 * smaller_hex_radius;

            centers.clear();
            let mut y = -radius + smaller_hex_radius;

            while y <= radius - smaller_hex_radius {
                let row = ((y + radius - smaller_hex_radius) / h).round() as i32;
                let x_offset = if row % 2 == 0 { 0.0 } else { w / 2.0 };
                let mut x = -radius + smaller_hex_radius + x_offset;

                while x <= radius - smaller_hex_radius {
                    if x * x + y * y <= (radius - margin).powi(2) {
                        centers.push(Position { x, y });
                    }
                    x += w;
                }
                y += h;
            }
        }

        //
        // Debug message
        // Debug parameters
        // Debug parameters
        centers
    }

    /// Greedy global nearest-neighbor assignment of positions to hex centers
    /// Based on the assign_unique_cells function from the Python script
    fn assign_unique_cells(
        &self,
        positions: &[Position],
        centers: &[Position],
    ) -> GraphResult<Vec<usize>> {
        if positions.is_empty() || centers.is_empty() {
            return Ok(Vec::new());
        }

        let n_positions = positions.len();
        let n_centers = centers.len();

        if n_centers < n_positions {
            return Err(crate::errors::GraphError::InvalidInput(format!(
                "Not enough hex centers ({}) for positions ({})",
                n_centers, n_positions
            )));
        }

        // Compute distance matrix [positions × centers]
        let mut distances = Vec::with_capacity(n_positions * n_centers);
        for pos in positions {
            for center in centers {
                let dx = pos.x - center.x;
                let dy = pos.y - center.y;
                let dist = (dx * dx + dy * dy).sqrt();
                distances.push(dist);
            }
        }

        // Greedy assignment: for each (position, center) pair in distance order,
        // assign if both are still available
        let mut assigned = vec![None; n_positions]; // position -> center index
        let mut used_centers = std::collections::HashSet::new();
        let mut assigned_positions = std::collections::HashSet::new();

        // Create sorted list of (distance, position_idx, center_idx)
        let mut distance_tuples = Vec::with_capacity(n_positions * n_centers);
        for pos_idx in 0..n_positions {
            for center_idx in 0..n_centers {
                let dist = distances[pos_idx * n_centers + center_idx];
                distance_tuples.push((dist, pos_idx, center_idx));
            }
        }
        distance_tuples.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Greedy assignment
        for (_dist, pos_idx, center_idx) in distance_tuples {
            if assigned_positions.contains(&pos_idx) || used_centers.contains(&center_idx) {
                continue;
            }

            assigned[pos_idx] = Some(center_idx);
            assigned_positions.insert(pos_idx);
            used_centers.insert(center_idx);

            if assigned_positions.len() == n_positions {
                break;
            }
        }

        // Convert to result format
        let mut result = Vec::with_capacity(n_positions);
        for assignment in assigned {
            match assignment {
                Some(center_idx) => result.push(center_idx),
                None => {
                    return Err(crate::errors::GraphError::InvalidInput(
                        "Failed to assign all positions to hex centers".to_string(),
                    ));
                }
            }
        }

        //
        // Debug message
        // Debug parameters
        // Debug parameters
        // Debug parameters

        Ok(result)
    }

    /// Assign flat embedding positions to honeycomb grid cells
    fn assign_to_honeycomb_grid(&mut self, positions: &[Position]) -> GraphResult<Vec<Position>> {
        if positions.is_empty() {
            return Ok(Vec::new());
        }

        // Generate hex centers covering the area of the optimized positions
        let hex_centers = self.generate_adaptive_hex_centers(positions)?;

        // Use the existing unique assignment algorithm
        let assignments = self.assign_unique_cells(positions, &hex_centers)?;

        let mut result_positions = Vec::with_capacity(positions.len());

        for (node_idx, &hex_center_idx) in assignments.iter().enumerate() {
            if hex_center_idx >= hex_centers.len() {
                return Err(GraphError::InvalidInput(format!(
                    "Invalid hex center assignment: {} >= {}",
                    hex_center_idx,
                    hex_centers.len()
                )));
            }

            let hex_center_pos = hex_centers[hex_center_idx];
            let hex_coord = self.pixel_to_hex(&hex_center_pos);

            // Track the assignment
            self.occupied_cells.insert(hex_coord, node_idx);
            self.node_positions.insert(node_idx, hex_coord);

            let final_pos = if self.config.snap_to_centers {
                hex_center_pos
            } else {
                // Blend between optimized position and hex center
                Position {
                    x: 0.7 * hex_center_pos.x + 0.3 * positions[node_idx].x,
                    y: 0.7 * hex_center_pos.y + 0.3 * positions[node_idx].y,
                }
            };

            result_positions.push(final_pos);
        }

        //
        // Debug message
        // Debug parameters
        // Debug parameters

        Ok(result_positions)
    }

    /// Generate hex centers adapted to the bounding box of positions
    fn generate_adaptive_hex_centers(&self, positions: &[Position]) -> GraphResult<Vec<Position>> {
        if positions.is_empty() {
            return Ok(Vec::new());
        }

        // Find bounding box of positions
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for pos in positions {
            min_x = min_x.min(pos.x);
            max_x = max_x.max(pos.x);
            min_y = min_y.min(pos.y);
            max_y = max_y.max(pos.y);
        }

        // Compute adaptive parameters
        let width = max_x - min_x;
        let height = max_y - min_y;
        let radius = ((width * width + height * height).sqrt() / 2.0).max(1.0);
        let center_x = (min_x + max_x) / 2.0;
        let center_y = (min_y + max_y) / 2.0;

        // Adaptive cell size based on density
        let area = width * height;
        let density = positions.len() as f64 / area.max(1.0);
        let adaptive_cell_size = (self.config.cell_size / 100.0) / (density.sqrt().max(0.1));

        // Generate hex grid within the bounding circle
        let margin = 0.1; // Larger margin for better coverage
        let w = (3_f64).sqrt() * adaptive_cell_size; // x spacing
        let h = 1.5 * adaptive_cell_size; // y spacing

        let mut centers = Vec::new();
        let mut y = center_y - radius;

        while y <= center_y + radius {
            let row = ((y - center_y + radius) / h).round() as i32;
            let x_offset = if row % 2 == 0 { 0.0 } else { w / 2.0 };
            let mut x = center_x - radius + x_offset;

            while x <= center_x + radius {
                let dx = x - center_x;
                let dy = y - center_y;

                // Check if point is within expanded bounding region
                if dx * dx + dy * dy <= (radius + margin).powi(2) {
                    centers.push(Position { x, y });
                }
                x += w;
            }
            y += h;
        }

        // Ensure we have enough centers
        if centers.len() < positions.len() {
            //
            // Debug message
            // Debug parameters
            // Debug parameters
            // Debug parameters

            // Fallback: use the original circle generation with smaller cell size
            let fallback_radius = radius * 1.5;
            let smaller_cell_size = adaptive_cell_size * 0.7;
            let fallback_centers = self.generate_hex_centers_in_circle_with_params(
                fallback_radius,
                smaller_cell_size,
                positions.len() * 2,
            );

            if fallback_centers.len() > centers.len() {
                return Ok(fallback_centers);
            }
        }

        Ok(centers)
    }

    /// Generate hex centers in circle with custom parameters
    fn generate_hex_centers_in_circle_with_params(
        &self,
        radius: f64,
        hex_radius: f64,
        min_count: usize,
    ) -> Vec<Position> {
        let margin = 0.02;
        let w = (3_f64).sqrt() * hex_radius; // x spacing
        let h = 1.5 * hex_radius; // y spacing

        let mut centers = Vec::new();
        let mut y = -radius + hex_radius;

        while y <= radius - hex_radius {
            let row = ((y + radius - hex_radius) / h).round() as i32;
            let x_offset = if row % 2 == 0 { 0.0 } else { w / 2.0 };
            let mut x = -radius + hex_radius + x_offset;

            while x <= radius - hex_radius {
                if x * x + y * y <= (radius - margin).powi(2) {
                    centers.push(Position { x, y });
                }
                x += w;
            }
            y += h;
        }

        // Try smaller radius if we don't have enough
        if centers.len() < min_count && hex_radius > 0.01 {
            return self.generate_hex_centers_in_circle_with_params(
                radius,
                hex_radius * 0.8,
                min_count,
            );
        }

        centers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hex_coord_distance() {
        let hex1 = HexCoord::new(0, 0);
        let hex2 = HexCoord::new(3, -2);
        assert_eq!(hex1.distance(&hex2), 3);

        let hex3 = HexCoord::new(1, -1);
        assert_eq!(hex1.distance(&hex3), 1);
    }

    #[test]
    fn test_hex_coord_neighbors() {
        let hex = HexCoord::new(0, 0);
        let neighbors = hex.neighbors();
        assert_eq!(neighbors.len(), 6);

        // Check that all neighbors are distance 1 away
        for neighbor in neighbors {
            assert_eq!(hex.distance(&neighbor), 1);
        }
    }

    #[test]
    fn test_hex_to_pixel_conversion() {
        let config = HoneycombConfig::default();
        let grid = HoneycombGrid::new(config);

        let hex = HexCoord::new(1, 0);
        let pixel = grid.hex_to_pixel(&hex);
        let back_to_hex = grid.pixel_to_hex(&pixel);

        assert_eq!(hex, back_to_hex);
    }

    #[test]
    fn test_spiral_coordinates() {
        let config = HoneycombConfig::default();
        let grid = HoneycombGrid::new(config);

        let coords = grid.generate_spiral_coordinates(7);
        assert_eq!(coords.len(), 7);

        // Center should be first
        assert_eq!(coords[0], HexCoord::new(0, 0));

        // All coordinates should be unique
        let mut unique_coords = coords.clone();
        unique_coords.sort_by_key(|c| (c.q, c.r));
        unique_coords.dedup();
        assert_eq!(unique_coords.len(), coords.len());
    }

    #[test]
    fn test_grid_mapping() {
        let config = HoneycombConfig::default();
        let mut grid = HoneycombGrid::new(config);

        let positions = vec![
            Position { x: 0.0, y: 0.0 },
            Position { x: 10.0, y: 0.0 },
            Position { x: -5.0, y: 8.0 },
        ];

        let mapped = grid.map_positions_to_grid(&positions);
        assert!(mapped.is_ok());

        let mapped_positions = mapped.unwrap();
        assert_eq!(mapped_positions.len(), 3);

        // Check that grid tracking is working
        assert_eq!(grid.occupied_cells.len(), 3);
        assert_eq!(grid.node_positions.len(), 3);
    }

    #[test]
    fn test_cube_coordinates() {
        let hex = HexCoord::new(2, -1);
        let (x, y, z) = hex.to_cube();
        assert_eq!(x + y + z, 0);

        let back_to_hex = HexCoord::from_cube(x, y, z).unwrap();
        assert_eq!(hex, back_to_hex);
    }
}
