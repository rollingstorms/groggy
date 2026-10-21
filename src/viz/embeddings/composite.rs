//! Checked row-aligned composition of existing embedding engines/matrices.
use super::{
    row_node_ids, CombinationStrategy, EmbeddingConfig, EmbeddingEngine, EmbeddingEngineFactory,
    EmbeddingMethod,
};
use crate::{Graph, GraphError, GraphMatrix, GraphResult};
use std::collections::HashMap;
fn invalid(message: impl Into<String>) -> GraphError {
    GraphError::InvalidInput(message.into())
}

impl CombinationStrategy {
    /// Parse the names accepted by thin language adapters; computation validates controls.
    pub fn from_name(name: &str, chunk_size: usize) -> GraphResult<Self> {
        match name {
            "concatenate" => Ok(Self::Concatenate),
            "weighted_average" => Ok(Self::WeightedAverage),
            "interleave" => Ok(Self::Interleave { chunk_size }),
            _ => Err(invalid(
                "Embedding strategy must be concatenate, weighted_average or interleave",
            )),
        }
    }
}
fn coefficients(
    count: usize,
    weights: Option<&[f64]>,
    strategy: &CombinationStrategy,
) -> GraphResult<Vec<f64>> {
    if count == 0 {
        return Err(invalid("At least one embedding is required"));
    }
    if matches!(strategy, CombinationStrategy::Interleave { chunk_size: 0 }) {
        return Err(invalid("Interleave chunk_size must be positive"));
    }
    let mut values = weights
        .map(<[f64]>::to_vec)
        .unwrap_or_else(|| vec![1.0; count]);
    if values.len() != count {
        return Err(invalid(
            "Embedding weights must match the number of matrices",
        ));
    }
    if values.iter().any(|w| !w.is_finite() || *w < 0.0) {
        return Err(invalid("Embedding weights must be finite and nonnegative"));
    }
    if matches!(strategy, CombinationStrategy::WeightedAverage) {
        let maximum = values.iter().copied().fold(0.0_f64, f64::max);
        if maximum == 0.0 {
            return Err(invalid("Weighted average requires a positive total weight"));
        }
        for value in &mut values {
            *value /= maximum;
        }
        let total: f64 = values.iter().sum();
        for value in &mut values {
            *value /= total;
        }
    }
    Ok(values)
}

/// Combine owned-value embedding matrices after validating and aligning their rows.
/// Unlabeled rows use ascending active IDs; labeled rows may be permuted.
/// Concatenation/interleaving preserve every column; averaging requires equal widths.
pub fn combine_embedding_matrices(
    graph: &Graph,
    matrices: &[GraphMatrix],
    weights: Option<&[f64]>,
    strategy: &CombinationStrategy,
) -> GraphResult<GraphMatrix> {
    let coefficients = coefficients(matrices.len(), weights, strategy)?;
    let mut nodes = graph.space().node_ids();
    nodes.sort_unstable();
    if nodes.is_empty() {
        return Err(invalid("Cannot combine embeddings for an empty graph"));
    }
    let mut row_maps = Vec::with_capacity(matrices.len());
    let mut total_columns = 0usize;
    for matrix in matrices {
        if matrix.shape().1 == 0 {
            return Err(invalid("Embedding matrices must have at least one column"));
        }
        let order = row_node_ids(matrix, graph)?;
        let index: HashMap<_, _> = order
            .iter()
            .enumerate()
            .map(|(row, &node)| (node, row))
            .collect();
        row_maps.push(
            nodes
                .iter()
                .map(|node| {
                    index
                        .get(node)
                        .copied()
                        .ok_or_else(|| invalid("Missing composite row identity"))
                })
                .collect::<GraphResult<Vec<_>>>()?,
        );
        total_columns = total_columns
            .checked_add(matrix.shape().1)
            .ok_or_else(|| invalid("Composite column count overflow"))?;
        for row in 0..matrix.shape().0 {
            for col in 0..matrix.shape().1 {
                if !matrix.get_checked(row, col)?.is_finite() {
                    return Err(invalid("Composite embeddings require finite values"));
                }
            }
        }
    }
    let average = matches!(strategy, CombinationStrategy::WeightedAverage);
    let first_width = matrices[0].shape().1;
    if average && matrices.iter().any(|m| m.shape().1 != first_width) {
        return Err(invalid("Weighted average requires equal embedding widths"));
    }
    let output_columns = if average { first_width } else { total_columns };
    nodes
        .len()
        .checked_mul(output_columns)
        .and_then(|count| count.checked_mul(std::mem::size_of::<f64>()))
        .ok_or_else(|| invalid("Composite output size overflow"))?;
    let mut columns = Vec::with_capacity(output_columns);
    match strategy {
        CombinationStrategy::Concatenate => {
            for (child, matrix) in matrices.iter().enumerate() {
                columns.extend((0..matrix.shape().1).map(|col| (child, col)));
            }
        }
        CombinationStrategy::Interleave { chunk_size } => {
            let maximum = matrices.iter().map(|m| m.shape().1).max().unwrap_or(0);
            let mut start = 0;
            while start < maximum {
                for (child, matrix) in matrices.iter().enumerate() {
                    let end = start.saturating_add(*chunk_size).min(matrix.shape().1);
                    if start < end {
                        columns.extend((start..end).map(|col| (child, col)));
                    }
                }
                start = start.saturating_add(*chunk_size);
            }
        }
        CombinationStrategy::WeightedAverage => {
            columns.extend((0..first_width).map(|col| (0, col)))
        }
    }
    let mut result = GraphMatrix::zeros(nodes.len(), output_columns);
    for row in 0..nodes.len() {
        for (out, &(child, col)) in columns.iter().enumerate() {
            let value = if average {
                let mut scale = 0.0_f64;
                for (matrix, rows) in matrices.iter().zip(&row_maps) {
                    scale = scale.max(matrix.get_checked(rows[row], col)?.abs());
                }
                if scale == 0.0 {
                    0.0
                } else {
                    let (mut sum, mut compensation, mut low, mut high) =
                        (0.0, 0.0, f64::INFINITY, f64::NEG_INFINITY);
                    for ((matrix, rows), &weight) in
                        matrices.iter().zip(&row_maps).zip(&coefficients)
                    {
                        let normalized = matrix.get_checked(rows[row], col)? / scale;
                        low = low.min(normalized);
                        high = high.max(normalized);
                        let term = normalized * weight - compensation;
                        let next = sum + term;
                        compensation = (next - sum) - term;
                        sum = next;
                    }
                    // A convex mean is within its input range; clip only roundoff.
                    sum.clamp(low, high) * scale
                }
            } else {
                matrices[child].get_checked(row_maps[child][row], col)?
            };
            if !value.is_finite() {
                return Err(invalid("Composite result exceeds finite f64 range"));
            }
            result.set(row, out, value)?;
        }
    }
    result.set_row_labels(Some(
        nodes.iter().map(|node| format!("node_{node}")).collect(),
    ));
    let names = if average {
        matrices[0].column_names().to_vec()
    } else {
        columns
            .iter()
            .map(|&(child, col)| {
                format!(
                    "embedding_{child}.{}",
                    matrices[child]
                        .column_names()
                        .get(col)
                        .cloned()
                        .unwrap_or_else(|| format!("col_{col}"))
                )
            })
            .collect()
    };
    result.set_column_names(names);
    Ok(result)
}

#[derive(Debug)]
pub(super) struct CompositeEmbedding {
    methods: Vec<(EmbeddingMethod, f64)>,
    combination_strategy: CombinationStrategy,
}
impl CompositeEmbedding {
    pub(super) fn new(methods: Vec<(EmbeddingMethod, f64)>, strategy: CombinationStrategy) -> Self {
        Self {
            methods,
            combination_strategy: strategy,
        }
    }
}
impl EmbeddingEngine for CompositeEmbedding {
    fn compute_embedding(&self, graph: &Graph, dimensions: usize) -> GraphResult<GraphMatrix> {
        self.validate_graph(graph)?;
        if dimensions == 0 {
            return Err(invalid("Composite child dimensions must be positive"));
        }
        let weights: Vec<_> = self.methods.iter().map(|(_, weight)| *weight).collect();
        coefficients(
            self.methods.len(),
            Some(&weights),
            &self.combination_strategy,
        )?;
        let mut matrices = Vec::with_capacity(self.methods.len());
        for (method, _) in &self.methods {
            let config = EmbeddingConfig {
                method: method.clone(),
                dimensions,
                ..Default::default()
            };
            matrices.push(
                EmbeddingEngineFactory::create_engine(&config)?
                    .compute_embedding(graph, dimensions)?,
            );
        }
        combine_embedding_matrices(graph, &matrices, Some(&weights), &self.combination_strategy)
    }
    fn supports_incremental(&self) -> bool {
        false
    }
    fn supports_streaming(&self) -> bool {
        false
    }
    fn name(&self) -> &str {
        "composite"
    }
    fn default_dimensions(&self) -> usize {
        10
    }
}
