//! Normalization step primitives.

use std::collections::HashMap;

use anyhow::{anyhow, Result};

use super::super::{AlgorithmParamValue, Context, CostHint};
use super::core::{Step, StepMetadata, StepScope};

/// Normalization methods.
#[derive(Clone, Copy, Debug)]
pub enum NormalizeMethod {
    Sum,
    Max,
    MinMax,
}

fn value_as_f64(value: &AlgorithmParamValue, context: &str) -> Result<f64> {
    match value {
        AlgorithmParamValue::Float(v) => Ok(*v),
        AlgorithmParamValue::Int(v) => Ok(*v as f64),
        other => Err(anyhow!(
            "expected numeric value for {context}, found {:?}",
            other
        )),
    }
}

/// Generic normalize values step that works on both node and edge maps.
pub struct NormalizeValuesStep {
    source: String,
    target: String,
    epsilon: f64,
    method: NormalizeMethod,
}

impl NormalizeValuesStep {
    pub fn new(
        source: impl Into<String>,
        target: impl Into<String>,
        method: NormalizeMethod,
        epsilon: f64,
    ) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            epsilon,
            method,
        }
    }
}

impl Step for NormalizeValuesStep {
    fn id(&self) -> &'static str {
        "core.normalize_values"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Normalize values using sum/max/minmax".to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        if ctx.is_cancelled() {
            return Err(anyhow!("normalize_values cancelled"));
        }

        // Try node map first, then edge map
        if let Ok(map) = scope.variables().node_map(&self.source) {
            let normalized = normalize_map(map, self.method, self.epsilon, &self.source)?;
            scope
                .variables_mut()
                .set_node_map(self.target.clone(), normalized);
            Ok(())
        } else if let Ok(map) = scope.variables().edge_map(&self.source) {
            let normalized = normalize_map(map, self.method, self.epsilon, &self.source)?;
            scope
                .variables_mut()
                .set_edge_map(self.target.clone(), normalized);
            Ok(())
        } else {
            Err(anyhow!("variable '{}' not found or not a map", self.source))
        }
    }
}

fn normalize_map<K>(
    map: &HashMap<K, AlgorithmParamValue>,
    method: NormalizeMethod,
    epsilon: f64,
    source_name: &str,
) -> Result<HashMap<K, AlgorithmParamValue>>
where
    K: Copy + std::cmp::Eq + std::hash::Hash + Ord,
{
    let mut normalized = HashMap::with_capacity(map.len());
    let mut keys: Vec<K> = map.keys().copied().collect();
    keys.sort_unstable();

    match method {
        NormalizeMethod::Sum => {
            let mut sum = 0.0;
            for key in &keys {
                let value = map.get(key).expect("key must exist");
                sum += value_as_f64(value, source_name)?;
            }

            if sum.abs() <= epsilon {
                return Err(anyhow!(
                    "cannot normalize '{}': total magnitude below epsilon ({})",
                    source_name,
                    epsilon
                ));
            }

            for key in keys {
                let value = map.get(&key).expect("key must exist");
                let raw = value_as_f64(value, source_name)?;
                normalized.insert(key, AlgorithmParamValue::Float(raw / sum));
            }
        }
        NormalizeMethod::Max => {
            let mut max_value = f64::NEG_INFINITY;
            for key in &keys {
                let value = map.get(key).expect("key must exist");
                let raw = value_as_f64(value, source_name)?;
                if raw > max_value {
                    max_value = raw;
                }
            }

            if max_value.abs() <= epsilon {
                return Err(anyhow!(
                    "cannot normalize '{}': max magnitude below epsilon ({})",
                    source_name,
                    epsilon
                ));
            }

            for key in keys {
                let value = map.get(&key).expect("key must exist");
                let raw = value_as_f64(value, source_name)?;
                normalized.insert(key, AlgorithmParamValue::Float(raw / max_value));
            }
        }
        NormalizeMethod::MinMax => {
            let mut min_value = f64::INFINITY;
            let mut max_value = f64::NEG_INFINITY;
            for key in &keys {
                let value = map.get(key).expect("key must exist");
                let raw = value_as_f64(value, source_name)?;
                if raw < min_value {
                    min_value = raw;
                }
                if raw > max_value {
                    max_value = raw;
                }
            }

            let range = max_value - min_value;
            if range.abs() <= epsilon {
                return Err(anyhow!(
                    "cannot normalize '{}': range below epsilon ({})",
                    source_name,
                    epsilon
                ));
            }

            for key in keys {
                let value = map.get(&key).expect("key must exist");
                let raw = value_as_f64(value, source_name)?;
                let normalized_value = (raw - min_value) / range;
                normalized.insert(key, AlgorithmParamValue::Float(normalized_value));
            }
        }
    }

    Ok(normalized)
}

/// Backward-compatible alias for normalize values (node-specific name).
pub struct NormalizeNodeValuesStep {
    inner: NormalizeValuesStep,
}

impl NormalizeNodeValuesStep {
    pub fn new(
        source: impl Into<String>,
        target: impl Into<String>,
        method: NormalizeMethod,
        epsilon: f64,
    ) -> Self {
        Self {
            inner: NormalizeValuesStep::new(source, target, method, epsilon),
        }
    }
}

impl Step for NormalizeNodeValuesStep {
    fn id(&self) -> &'static str {
        "core.normalize_node_values"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Normalize node values using sum/max/minmax".to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        self.inner.apply(ctx, scope)
    }
}

/// Standardize values using Z-score normalization (mean=0, std=1).
pub struct StandardizeStep {
    source: String,
    target: String,
    epsilon: f64,
}

impl StandardizeStep {
    pub fn new(source: impl Into<String>, target: impl Into<String>, epsilon: f64) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            epsilon,
        }
    }
}

impl Step for StandardizeStep {
    fn id(&self) -> &'static str {
        "core.standardize"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Z-score standardization (mean=0, std=1)".to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        if ctx.is_cancelled() {
            return Err(anyhow!("standardize cancelled"));
        }

        // Try node map first, then edge map
        if let Ok(map) = scope.variables().node_map(&self.source) {
            let standardized = standardize_map(map, self.epsilon, &self.source)?;
            scope
                .variables_mut()
                .set_node_map(self.target.clone(), standardized);
            Ok(())
        } else if let Ok(map) = scope.variables().edge_map(&self.source) {
            let standardized = standardize_map(map, self.epsilon, &self.source)?;
            scope
                .variables_mut()
                .set_edge_map(self.target.clone(), standardized);
            Ok(())
        } else {
            Err(anyhow!("variable '{}' not found or not a map", self.source))
        }
    }
}

fn standardize_map<K>(
    map: &HashMap<K, AlgorithmParamValue>,
    epsilon: f64,
    source_name: &str,
) -> Result<HashMap<K, AlgorithmParamValue>>
where
    K: Copy + std::cmp::Eq + std::hash::Hash,
{
    if map.is_empty() {
        return Ok(HashMap::new());
    }

    // Compute mean
    let mut sum = 0.0;
    let mut count = 0;
    for value in map.values() {
        sum += value_as_f64(value, source_name)?;
        count += 1;
    }
    let mean = sum / count as f64;

    // Compute standard deviation
    let mut variance_sum = 0.0;
    for value in map.values() {
        let val = value_as_f64(value, source_name)?;
        let diff = val - mean;
        variance_sum += diff * diff;
    }
    let std_dev = (variance_sum / count as f64).sqrt();

    if std_dev <= epsilon {
        return Err(anyhow!(
            "cannot standardize '{}': standard deviation below epsilon ({})",
            source_name,
            epsilon
        ));
    }

    // Standardize
    let mut standardized = HashMap::with_capacity(map.len());
    for (&key, value) in map.iter() {
        let val = value_as_f64(value, source_name)?;
        let z_score = (val - mean) / std_dev;
        standardized.insert(key, AlgorithmParamValue::Float(z_score));
    }

    Ok(standardized)
}

/// Clip values to a specified range [min, max].
pub struct ClipValuesStep {
    source: String,
    target: String,
    min: f64,
    max: f64,
}

impl ClipValuesStep {
    pub fn new(source: impl Into<String>, target: impl Into<String>, min: f64, max: f64) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            min,
            max,
        }
    }
}

impl Step for ClipValuesStep {
    fn id(&self) -> &'static str {
        "core.clip"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: format!("Clip values to range [{}, {}]", self.min, self.max),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        if ctx.is_cancelled() {
            return Err(anyhow!("clip cancelled"));
        }

        if self.min > self.max {
            return Err(anyhow!(
                "invalid clip range: min ({}) > max ({})",
                self.min,
                self.max
            ));
        }

        // Try node map first, then edge map
        if let Ok(map) = scope.variables().node_map(&self.source) {
            let clipped = clip_map(map, self.min, self.max, &self.source)?;
            scope
                .variables_mut()
                .set_node_map(self.target.clone(), clipped);
            Ok(())
        } else if let Ok(map) = scope.variables().edge_map(&self.source) {
            let clipped = clip_map(map, self.min, self.max, &self.source)?;
            scope
                .variables_mut()
                .set_edge_map(self.target.clone(), clipped);
            Ok(())
        } else {
            Err(anyhow!("variable '{}' not found or not a map", self.source))
        }
    }
}

fn clip_map<K>(
    map: &HashMap<K, AlgorithmParamValue>,
    min: f64,
    max: f64,
    source_name: &str,
) -> Result<HashMap<K, AlgorithmParamValue>>
where
    K: Copy + std::cmp::Eq + std::hash::Hash,
{
    let mut clipped = HashMap::with_capacity(map.len());

    for (&key, value) in map.iter() {
        let val = value_as_f64(value, source_name)?;
        let clipped_val = val.clamp(min, max);
        clipped.insert(key, AlgorithmParamValue::Float(clipped_val));
    }

    Ok(clipped)
}
