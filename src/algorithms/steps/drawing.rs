use anyhow::{anyhow, Result};
use std::collections::HashMap;

use crate::algorithms::Context;
use crate::traits::SubgraphOperations;
use crate::types::{AttrName, AttrValue, NodeId};

use super::{Step, StepMetadata, StepScope};

/// Smooth stroke point attributes (e.g., x/y) grouped by stroke id and ordered by time.
pub struct SmoothStrokeAttrsStep {
    attr_x: AttrName,
    attr_y: AttrName,
    attr_t: AttrName,
    attr_stroke: AttrName,
    target_x: AttrName,
    target_y: AttrName,
    window: usize,
    iterations: usize,
    target_stroke: Option<i64>,
}

impl SmoothStrokeAttrsStep {
    pub fn new(
        attr_x: AttrName,
        attr_y: AttrName,
        attr_t: AttrName,
        attr_stroke: AttrName,
        target_x: AttrName,
        target_y: AttrName,
        window: usize,
        iterations: usize,
        target_stroke: Option<i64>,
    ) -> Self {
        Self {
            attr_x,
            attr_y,
            attr_t,
            attr_stroke,
            target_x,
            target_y,
            window: window.max(1),
            iterations: iterations.max(1),
            target_stroke,
        }
    }
}

impl Step for SmoothStrokeAttrsStep {
    fn id(&self) -> &'static str {
        "drot.smooth_strokes"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Smooth stroke attributes grouped by stroke id".to_string(),
            cost_hint: crate::algorithms::CostHint::Linear,
        }
    }

    fn apply(&self, _ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        let mut by_stroke: HashMap<i64, Vec<(f64, NodeId, f64, f64)>> = HashMap::new();

        for &node in scope.subgraph().nodes().iter() {
            let stroke_val = scope
                .subgraph()
                .get_node_attribute(node, &self.attr_stroke)
                .map_err(|err| anyhow!(err))?;
            let stroke_id = match stroke_val.and_then(attr_to_i64) {
                Some(value) => value,
                None => continue,
            };
            if let Some(target) = self.target_stroke {
                if target != stroke_id {
                    continue;
                }
            }

            let t = scope
                .subgraph()
                .get_node_attribute(node, &self.attr_t)
                .map_err(|err| anyhow!(err))?
                .and_then(attr_to_f64)
                .unwrap_or(0.0);
            let x = match scope
                .subgraph()
                .get_node_attribute(node, &self.attr_x)
                .map_err(|err| anyhow!(err))?
                .and_then(attr_to_f64)
            {
                Some(value) => value,
                None => continue,
            };
            let y = match scope
                .subgraph()
                .get_node_attribute(node, &self.attr_y)
                .map_err(|err| anyhow!(err))?
                .and_then(attr_to_f64)
            {
                Some(value) => value,
                None => continue,
            };

            by_stroke
                .entry(stroke_id)
                .or_default()
                .push((t, node, x, y));
        }

        if by_stroke.is_empty() {
            return Ok(());
        }

        let mut attrs_by_name: HashMap<AttrName, Vec<(NodeId, AttrValue)>> = HashMap::new();
        let target_x = self.target_x.clone();
        let target_y = self.target_y.clone();

        for samples in by_stroke.values_mut() {
            samples.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let mut points: Vec<(f64, f64)> = samples.iter().map(|(_, _, x, y)| (*x, *y)).collect();
            for _ in 0..self.iterations {
                points = smooth_points(&points, self.window);
            }

            for ((_, node, _, _), (sx, sy)) in samples.iter().zip(points.iter()) {
                attrs_by_name
                    .entry(target_x.clone())
                    .or_default()
                    .push((*node, AttrValue::Float(*sx as f32)));
                attrs_by_name
                    .entry(target_y.clone())
                    .or_default()
                    .push((*node, AttrValue::Float(*sy as f32)));
            }
        }

        scope
            .subgraph()
            .set_node_attrs(attrs_by_name)
            .map_err(|err| anyhow!(err))?;

        Ok(())
    }
}

fn attr_to_f64(value: AttrValue) -> Option<f64> {
    match value {
        AttrValue::Float(v) => Some(v as f64),
        AttrValue::Int(v) => Some(v as f64),
        AttrValue::SmallInt(v) => Some(v as f64),
        AttrValue::Bool(v) => Some(if v { 1.0 } else { 0.0 }),
        _ => None,
    }
}

fn attr_to_i64(value: AttrValue) -> Option<i64> {
    match value {
        AttrValue::Int(v) => Some(v),
        AttrValue::SmallInt(v) => Some(v as i64),
        AttrValue::Float(v) => Some(v as i64),
        AttrValue::Text(text) => text.parse::<i64>().ok(),
        _ => None,
    }
}

fn smooth_points(points: &[(f64, f64)], window: usize) -> Vec<(f64, f64)> {
    if points.len() <= 2 {
        return points.to_vec();
    }
    let window = window.max(1);
    let half = window / 2;
    let mut result = Vec::with_capacity(points.len());
    for i in 0..points.len() {
        let start = i.saturating_sub(half);
        let end = (i + half).min(points.len().saturating_sub(1));
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut count = 0.0;
        for j in start..=end {
            sum_x += points[j].0;
            sum_y += points[j].1;
            count += 1.0;
        }
        result.push((sum_x / count, sum_y / count));
    }
    result
}
