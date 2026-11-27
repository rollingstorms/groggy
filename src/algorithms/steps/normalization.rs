//! Normalization step primitives.

use super::read_scope::{sealed, ReadStep, ReadStepScope};
use std::collections::HashMap;

use anyhow::{anyhow, Result};

use super::super::{AlgorithmParamValue, Context, CostHint};
use super::core::{AtomicStepEffect, Step, StepMetadata, StepScope};

/// Normalization methods.
#[derive(Clone, Copy, Debug)]
pub enum NormalizeMethod {
    Sum,
    Mean,
    Min,
    Max,
    MinMax,
}

fn finite(value: f64, context: &str) -> Result<f64> {
    if !value.is_finite() {
        return Err(anyhow!(
            "{context} requires finite numeric values and intermediates"
        ));
    }
    Ok(value)
}
pub(super) fn validate_epsilon(epsilon: f64) -> Result<()> {
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err(anyhow!("epsilon must be finite and non-negative"));
    }
    Ok(())
}
pub(super) fn validate_clip(min: f64, max: f64) -> Result<()> {
    if min.is_nan() || max.is_nan() || min > max {
        return Err(anyhow!(
            "invalid clip range: bounds must be ordered and not NaN"
        ));
    }
    Ok(())
}
fn value_as_f64(value: &AlgorithmParamValue, context: &str) -> Result<f64> {
    match value {
        AlgorithmParamValue::Float(v) => finite(*v, context),
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
    fn atomic_effect(&self) -> AtomicStepEffect {
        AtomicStepEffect::ImmutableInputs
    }

    fn supports_atomic_properties(&self) -> bool {
        true
    }

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

    fn read_kernel(&self) -> Option<&dyn ReadStep> {
        Some(self)
    }
    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        self.apply_read(ctx, &mut scope.read_scope())
    }
}

impl sealed::Sealed for NormalizeValuesStep {}
impl ReadStep for NormalizeValuesStep {
    fn apply_read(&self, ctx: &mut Context, scope: &mut ReadStepScope<'_>) -> Result<()> {
        if ctx.is_cancelled() {
            return Err(anyhow!("normalize_values cancelled"));
        }

        validate_epsilon(self.epsilon)?;

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
    validate_epsilon(epsilon)?;
    // Preserve existing empty policies without evaluating infinite sentinels.
    if map.is_empty() && !matches!(method, NormalizeMethod::Sum) {
        return Ok(HashMap::new());
    }
    let mut normalized = HashMap::with_capacity(map.len());
    let mut keys: Vec<K> = map.keys().copied().collect();
    keys.sort_unstable();

    match method {
        NormalizeMethod::Sum | NormalizeMethod::Mean => {
            let mut sum = 0.0;
            for key in &keys {
                let value = map.get(key).expect("key must exist");
                sum += value_as_f64(value, source_name)?;
            }

            finite(sum, source_name)?;
            if matches!(method, NormalizeMethod::Mean) {
                sum /= keys.len() as f64;
            }
            if sum.abs() <= epsilon {
                return Err(anyhow!(
                    "cannot normalize '{}': sum/mean magnitude below epsilon ({})",
                    source_name,
                    epsilon
                ));
            }

            for key in keys {
                let value = map.get(&key).expect("key must exist");
                let raw = value_as_f64(value, source_name)?;
                normalized.insert(
                    key,
                    AlgorithmParamValue::Float(finite(raw / sum, source_name)?),
                );
            }
        }
        NormalizeMethod::Min | NormalizeMethod::Max => {
            let is_min = matches!(method, NormalizeMethod::Min);
            let mut max_value = if is_min {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            };
            for key in &keys {
                let value = map.get(key).expect("key must exist");
                let raw = value_as_f64(value, source_name)?;
                if (is_min && raw < max_value) || (!is_min && raw > max_value) {
                    max_value = raw;
                }
            }

            finite(max_value, source_name)?;
            if max_value.abs() <= epsilon {
                return Err(anyhow!(
                    "cannot normalize '{}': min/max magnitude below epsilon ({})",
                    source_name,
                    epsilon
                ));
            }

            for key in keys {
                let value = map.get(&key).expect("key must exist");
                let raw = value_as_f64(value, source_name)?;
                normalized.insert(
                    key,
                    AlgorithmParamValue::Float(finite(raw / max_value, source_name)?),
                );
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

            let range = finite(max_value - min_value, source_name)?;
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
                let normalized_value = finite((raw - min_value) / range, source_name)?;
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
    fn atomic_effect(&self) -> AtomicStepEffect {
        AtomicStepEffect::ImmutableInputs
    }

    fn supports_atomic_properties(&self) -> bool {
        true
    }

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

    fn read_kernel(&self) -> Option<&dyn ReadStep> {
        Some(self)
    }
    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        self.apply_read(ctx, &mut scope.read_scope())
    }
}
impl sealed::Sealed for NormalizeNodeValuesStep {}
impl ReadStep for NormalizeNodeValuesStep {
    fn apply_read(&self, ctx: &mut Context, scope: &mut ReadStepScope<'_>) -> Result<()> {
        self.inner.apply_read(ctx, scope)
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
    fn atomic_effect(&self) -> AtomicStepEffect {
        AtomicStepEffect::ImmutableInputs
    }

    fn supports_atomic_properties(&self) -> bool {
        true
    }

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

    fn read_kernel(&self) -> Option<&dyn ReadStep> {
        Some(self)
    }
    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        self.apply_read(ctx, &mut scope.read_scope())
    }
}

impl sealed::Sealed for StandardizeStep {}
impl ReadStep for StandardizeStep {
    fn apply_read(&self, ctx: &mut Context, scope: &mut ReadStepScope<'_>) -> Result<()> {
        if ctx.is_cancelled() {
            return Err(anyhow!("standardize cancelled"));
        }

        validate_epsilon(self.epsilon)?;

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
    validate_epsilon(epsilon)?;
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
    let mean = finite(sum / count as f64, source_name)?;

    // Compute standard deviation
    let mut variance_sum = 0.0;
    for value in map.values() {
        let val = value_as_f64(value, source_name)?;
        let diff = val - mean;
        variance_sum += diff * diff;
    }
    let std_dev = finite((variance_sum / count as f64).sqrt(), source_name)?;

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
        let z_score = finite((val - mean) / std_dev, source_name)?;
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
    fn atomic_effect(&self) -> AtomicStepEffect {
        AtomicStepEffect::ImmutableInputs
    }

    fn supports_atomic_properties(&self) -> bool {
        true
    }

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

    fn read_kernel(&self) -> Option<&dyn ReadStep> {
        Some(self)
    }
    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        self.apply_read(ctx, &mut scope.read_scope())
    }
}

impl sealed::Sealed for ClipValuesStep {}
impl ReadStep for ClipValuesStep {
    fn apply_read(&self, ctx: &mut Context, scope: &mut ReadStepScope<'_>) -> Result<()> {
        if ctx.is_cancelled() {
            return Err(anyhow!("clip cancelled"));
        }

        validate_clip(self.min, self.max)?;

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
    validate_clip(min, max)?;
    let mut clipped = HashMap::with_capacity(map.len());

    for (&key, value) in map.iter() {
        let val = value_as_f64(value, source_name)?;
        let clipped_val = val.clamp(min, max);
        clipped.insert(key, AlgorithmParamValue::Float(clipped_val));
    }

    Ok(clipped)
}

#[cfg(test)]
mod safety_tests {
    use super::*;
    fn values(xs: &[f64]) -> HashMap<usize, AlgorithmParamValue> {
        xs.iter()
            .enumerate()
            .map(|(i, &x)| (i, AlgorithmParamValue::Float(x)))
            .collect()
    }
    #[test]
    fn invalid_epsilon_rejects_even_for_empty_maps() {
        for epsilon in [-1.0, f64::NAN, f64::INFINITY] {
            for data in [values(&[]), values(&[0.0])] {
                for method in [
                    NormalizeMethod::Sum,
                    NormalizeMethod::Mean,
                    NormalizeMethod::Min,
                    NormalizeMethod::Max,
                    NormalizeMethod::MinMax,
                ] {
                    assert!(normalize_map(&data, method, epsilon, "test").is_err());
                }
                assert!(standardize_map(&data, epsilon, "test").is_err());
            }
        }
    }
    #[test]
    fn invalid_clip_controls_return_errors_without_panicking() {
        for (min, max) in [(f64::NAN, 1.0), (0.0, f64::NAN), (2.0, 1.0)] {
            for data in [values(&[]), values(&[0.5])] {
                let result = std::panic::catch_unwind(|| clip_map(&data, min, max, "test"));
                assert!(result.is_ok(), "clip controls must not panic");
                assert!(result.unwrap().is_err());
            }
        }
    }
    #[test]
    fn nonfinite_inputs_and_unrepresentable_intermediates_never_install_numeric_results() {
        for x in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let data = values(&[1.0, x]);
            for method in [
                NormalizeMethod::Sum,
                NormalizeMethod::Mean,
                NormalizeMethod::Min,
                NormalizeMethod::Max,
                NormalizeMethod::MinMax,
            ] {
                assert!(normalize_map(&data, method, 0.0, "test").is_err());
            }
            assert!(standardize_map(&data, 0.0, "test").is_err());
            assert!(clip_map(&data, 0.0, 1.0, "test").is_err());
        }
        assert!(normalize_map(
            &values(&[f64::MAX, f64::MAX]),
            NormalizeMethod::Sum,
            0.0,
            "test"
        )
        .is_err());
        assert!(normalize_map(
            &values(&[-f64::MAX, f64::MAX]),
            NormalizeMethod::MinMax,
            0.0,
            "test"
        )
        .is_err());
        assert!(standardize_map(&values(&[-f64::MAX, f64::MAX]), 0.0, "test").is_err());
    }
    #[test]
    fn valid_normalization_keeps_population_variance_signed_max_and_empty_rules() {
        let data = values(&[-2.0, -4.0]);
        let out = normalize_map(&data, NormalizeMethod::Max, 0.0, "test").unwrap();
        assert_eq!(out[&0], AlgorithmParamValue::Float(1.0));
        assert_eq!(out[&1], AlgorithmParamValue::Float(2.0));
        let standardized = standardize_map(&values(&[2.0, 4.0]), 0.0, "test").unwrap();
        assert_eq!(standardized[&0], AlgorithmParamValue::Float(-1.0));
        assert_eq!(standardized[&1], AlgorithmParamValue::Float(1.0));
        assert!(normalize_map(&values(&[]), NormalizeMethod::Sum, 0.0, "test").is_err());
        assert!(
            normalize_map(&values(&[]), NormalizeMethod::Max, 0.0, "test")
                .unwrap()
                .is_empty()
        );
        assert!(standardize_map(&values(&[]), 0.0, "test")
            .unwrap()
            .is_empty());
        assert!(clip_map(&values(&[]), 0.0, 1.0, "test").unwrap().is_empty());
        for (min, max) in [
            (f64::NEG_INFINITY, 1.0),
            (0.0, f64::INFINITY),
            (f64::NEG_INFINITY, f64::INFINITY),
        ] {
            assert!(clip_map(&values(&[0.5]), min, max, "test").is_ok());
        }
    }
    #[test]
    fn minimum_and_mean_use_signed_divisors_and_reject_zero_or_overflow() {
        for sign in [-1.0, 1.0] {
            let input = values(&[sign * 2.0, sign * 4.0]);
            let mean = normalize_map(&input, NormalizeMethod::Mean, 0.0, "test").unwrap();
            assert_eq!(mean[&0], AlgorithmParamValue::Float(2.0 / 3.0));
            assert_eq!(mean[&1], AlgorithmParamValue::Float(4.0 / 3.0));
            let min = normalize_map(&input, NormalizeMethod::Min, 0.0, "test").unwrap();
            let expected = if sign > 0.0 { [1.0, 2.0] } else { [0.5, 1.0] };
            for (i, x) in expected.into_iter().enumerate() {
                assert_eq!(min[&i], AlgorithmParamValue::Float(x));
            }
        }
        for method in [NormalizeMethod::Min, NormalizeMethod::Mean] {
            assert!(normalize_map(&values(&[]), method, 0.0, "test")
                .unwrap()
                .is_empty());
            assert!(normalize_map(&values(&[0.0]), method, 0.0, "test").is_err());
            assert!(normalize_map(&values(&[1e-30]), method, 1e-20, "test").is_err());
            assert!(normalize_map(&values(&[1e-30]), method, 0.0, "test").is_ok());
        }
        assert!(normalize_map(&values(&[-1.0, 1.0]), NormalizeMethod::Mean, 0.0, "test").is_err());
        assert!(normalize_map(
            &values(&[f64::MAX, f64::MAX]),
            NormalizeMethod::Mean,
            0.0,
            "test"
        )
        .is_err());
        assert!(normalize_map(
            &values(&[f64::MIN_POSITIVE, f64::MAX]),
            NormalizeMethod::Min,
            0.0,
            "test"
        )
        .is_err());
    }
    #[test]
    fn normalization_errors_preserve_node_and_edge_outputs() {
        use super::super::core::StepVariables;
        use crate::{subgraphs::Subgraph, Graph};
        let mut graph = Graph::new_directed_manual();
        graph.add_nodes(2).unwrap();
        graph.add_edge(0, 1).unwrap();
        graph.add_edge(1, 1).unwrap();
        let selected = Subgraph::from_nodes(
            graph.into_shared(),
            [0, 1].into_iter().collect(),
            "test".into(),
        )
        .unwrap();
        for edges in [false, true] {
            let steps: Vec<Box<dyn Step>> = vec![
                Box::new(NormalizeValuesStep::new(
                    "input",
                    "out",
                    NormalizeMethod::Sum,
                    -1.0,
                )),
                Box::new(NormalizeNodeValuesStep::new(
                    "input",
                    "out",
                    NormalizeMethod::Max,
                    0.0,
                )),
                Box::new(StandardizeStep::new("input", "out", 0.0)),
                Box::new(ClipValuesStep::new("input", "out", f64::NAN, 1.0)),
            ];
            for step in steps {
                let mut vars = StepVariables::default();
                let source = values(&[1.0, f64::INFINITY]);
                let old = values(&[42.0]);
                if edges {
                    vars.set_edge_map("input", source);
                    vars.set_edge_map("out", old.clone());
                } else {
                    vars.set_node_map("input", source);
                    vars.set_node_map("out", old.clone());
                }
                let mut scope = StepScope::new(&selected, &mut vars);
                assert!(step.apply(&mut Context::new(), &mut scope).is_err());
                assert_eq!(
                    if edges {
                        vars.edge_map("out").unwrap()
                    } else {
                        vars.node_map("out").unwrap()
                    },
                    &old
                );
            }
        }
    }
}
