//! Algorithm execution core and shared types.
//!
//! Phase 1 of the algorithm roadmap establishes the foundational traits,
//! metadata, telemetry context, and typed parameter plumbing that higher-level
//! pipeline and builder layers will compose.

use std::collections::HashMap;
use std::ops::Deref;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Once};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

use crate::subgraphs::Subgraph;
use crate::types::AttrValue;

pub mod builder;
pub mod centrality;
pub mod community;
pub mod pathfinding;
pub mod pipeline;
pub mod registry;
pub mod steps;

pub use pipeline::{AlgorithmSpec, Pipeline, PipelineBuilder, PipelineSpec};
pub use registry::{global_registry, AlgorithmFactory, Registry};
pub use steps::{
    ensure_core_steps_registered, global_step_registry, register_core_steps, Step, StepMetadata,
    StepRegistry, StepScope, StepSpec, StepValue,
};

static ALGORITHMS_INIT: Once = Once::new();

/// Ensure that all built-in algorithms and step primitives are registered.
pub fn ensure_algorithms_registered() {
    ensure_core_steps_registered();
    ALGORITHMS_INIT.call_once(|| {
        if let Err(err) = community::register_algorithms(global_registry()) {
            panic!("failed to register core algorithms: {err}");
        }
        if let Err(err) = centrality::register_algorithms(global_registry()) {
            panic!("failed to register centrality algorithms: {err}");
        }
        if let Err(err) = pathfinding::register_algorithms(global_registry()) {
            panic!("failed to register pathfinding algorithms: {err}");
        }
        if let Err(err) = builder::register_algorithms(global_registry()) {
            panic!("failed to register builder algorithms: {err}");
        }
    });
}

/// Performance classification for algorithms and primitive steps.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "info")]
pub enum CostHint {
    /// Constant time regardless of graph size.
    Constant,
    /// Logarithmic in the number of active entities.
    Logarithmic,
    /// Linear in the number of active entities.
    Linear,
    /// Linearithmic (n log n) complexity.
    Linearithmic,
    /// Quadratic complexity.
    Quadratic,
    /// Cubic complexity.
    Cubic,
    /// Higher than cubic or unspecified exponential behaviour.
    Exponential,
    /// Unknown or data-dependent complexity.
    Unknown,
    /// Custom textual hint for nuanced cases.
    Custom(String),
}

impl Default for CostHint {
    fn default() -> Self {
        Self::Unknown
    }
}

/// Rich metadata describing an algorithm implementation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AlgorithmMetadata {
    /// Stable identifier (mirrors the registry key).
    pub id: String,
    /// Optional human-friendly display name.
    pub name: String,
    /// Short description suitable for discovery UIs.
    pub description: String,
    /// Semantic version or release identifier.
    pub version: String,
    /// Complexity hint for planning and UX.
    pub cost_hint: CostHint,
    /// Whether the algorithm supports cooperative cancellation.
    pub supports_cancellation: bool,
    /// Schema for accepted parameters.
    pub parameters: Vec<ParameterMetadata>,
}

impl Default for AlgorithmMetadata {
    fn default() -> Self {
        Self {
            id: String::new(),
            name: String::new(),
            description: String::new(),
            version: "0.0.0".to_string(),
            cost_hint: CostHint::Unknown,
            supports_cancellation: false,
            parameters: Vec::new(),
        }
    }
}

/// Metadata for a single algorithm or step parameter.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParameterMetadata {
    pub name: String,
    pub description: String,
    pub value_type: ParameterType,
    pub required: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_value: Option<AlgorithmParamValue>,
}

/// Lightweight type information so builders can validate inputs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ParameterType {
    Int,
    Float,
    Bool,
    Text,
    IntList,
    FloatList,
    BoolList,
    TextList,
    Json,
}

/// Typed parameter storage shared by algorithms and step primitives.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AlgorithmParams {
    values: HashMap<String, AlgorithmParamValue>,
}

impl AlgorithmParams {
    /// Create an empty parameter map.
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }

    /// Insert a raw parameter value.
    pub fn insert(&mut self, key: impl Into<String>, value: AlgorithmParamValue) {
        self.values.insert(key.into(), value);
    }

    /// Convenience setter for integers.
    pub fn set_int(&mut self, key: impl Into<String>, value: i64) {
        self.insert(key, AlgorithmParamValue::Int(value));
    }

    /// Convenience setter for floating point values.
    pub fn set_float(&mut self, key: impl Into<String>, value: f64) {
        self.insert(key, AlgorithmParamValue::Float(value));
    }

    /// Convenience setter for booleans.
    pub fn set_bool(&mut self, key: impl Into<String>, value: bool) {
        self.insert(key, AlgorithmParamValue::Bool(value));
    }

    /// Convenience setter for textual values.
    pub fn set_text(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.insert(key, AlgorithmParamValue::Text(value.into()));
    }

    /// Fetch a parameter by name.
    pub fn get(&self, key: &str) -> Option<&AlgorithmParamValue> {
        self.values.get(key)
    }

    /// Require an integer parameter.
    pub fn expect_int(&self, key: &str) -> Result<i64> {
        self.get_int(key)
            .ok_or_else(|| anyhow!("missing required integer parameter '{key}'"))
    }

    /// Fetch an integer parameter if present.
    pub fn get_int(&self, key: &str) -> Option<i64> {
        match self.get(key) {
            Some(AlgorithmParamValue::Int(v)) => Some(*v),
            Some(AlgorithmParamValue::Float(v)) => Some(*v as i64),
            _ => None,
        }
    }

    /// Require a float parameter.
    pub fn expect_float(&self, key: &str) -> Result<f64> {
        self.get_float(key)
            .ok_or_else(|| anyhow!("missing required float parameter '{key}'"))
    }

    /// Fetch a float parameter if present.
    pub fn get_float(&self, key: &str) -> Option<f64> {
        match self.get(key) {
            Some(AlgorithmParamValue::Float(v)) => Some(*v),
            Some(AlgorithmParamValue::Int(v)) => Some(*v as f64),
            _ => None,
        }
    }

    /// Require a boolean parameter.
    pub fn expect_bool(&self, key: &str) -> Result<bool> {
        self.get_bool(key)
            .ok_or_else(|| anyhow!("missing required boolean parameter '{key}'"))
    }

    /// Fetch a boolean parameter if present.
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        match self.get(key) {
            Some(AlgorithmParamValue::Bool(v)) => Some(*v),
            _ => None,
        }
    }

    /// Require a text parameter.
    pub fn expect_text(&self, key: &str) -> Result<&str> {
        self.get_text(key)
            .ok_or_else(|| anyhow!("missing required text parameter '{key}'"))
    }

    /// Fetch a text parameter if present.
    pub fn get_text(&self, key: &str) -> Option<&str> {
        match self.get(key) {
            Some(AlgorithmParamValue::Text(v)) => Some(v.as_str()),
            _ => None,
        }
    }

    /// Iterate over stored parameters.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &AlgorithmParamValue)> {
        self.values.iter()
    }
}

impl Deref for AlgorithmParams {
    type Target = HashMap<String, AlgorithmParamValue>;

    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

/// Typed value representation for algorithm specs, parameters, and step state.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum AlgorithmParamValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    Text(String),
    IntList(Vec<i64>),
    FloatList(Vec<f64>),
    BoolList(Vec<bool>),
    TextList(Vec<String>),
    Json(serde_json::Value),
    None,
}

impl AlgorithmParamValue {
    /// Attempt to convert a value into a graph attribute.
    pub fn as_attr_value(&self) -> Option<crate::types::AttrValue> {
        use crate::types::AttrValue;
        match self {
            AlgorithmParamValue::Int(v) => Some(AttrValue::Int(*v)),
            AlgorithmParamValue::Float(v) => Some(AttrValue::Float(*v as f32)),
            AlgorithmParamValue::Bool(v) => Some(AttrValue::Bool(*v)),
            AlgorithmParamValue::Text(v) => Some(AttrValue::Text(v.clone())),
            AlgorithmParamValue::IntList(v) => Some(AttrValue::IntVec(v.clone())),
            AlgorithmParamValue::FloatList(v) => {
                Some(AttrValue::FloatVec(v.iter().map(|f| *f as f32).collect()))
            }
            AlgorithmParamValue::BoolList(v) => Some(AttrValue::BoolVec(v.clone())),
            AlgorithmParamValue::TextList(v) => Some(AttrValue::TextVec(v.clone())),
            AlgorithmParamValue::Json(v) => Some(AttrValue::Json(v.to_string())),
            AlgorithmParamValue::None => Some(AttrValue::Null),
        }
    }

    /// Attempt to construct a parameter value from a graph attribute.
    pub fn from_attr_value(value: AttrValue) -> Option<Self> {
        match value {
            AttrValue::Float(v) => Some(AlgorithmParamValue::Float(f64::from(v))),
            AttrValue::Int(v) => Some(AlgorithmParamValue::Int(v)),
            AttrValue::Bool(v) => Some(AlgorithmParamValue::Bool(v)),
            AttrValue::Text(v) => Some(AlgorithmParamValue::Text(v)),
            AttrValue::FloatVec(v) => Some(AlgorithmParamValue::FloatList(
                v.into_iter().map(f64::from).collect(),
            )),
            AttrValue::IntVec(v) => Some(AlgorithmParamValue::IntList(v)),
            AttrValue::TextVec(v) => Some(AlgorithmParamValue::TextList(v)),
            AttrValue::BoolVec(v) => Some(AlgorithmParamValue::BoolList(v)),
            AttrValue::CompactText(v) => Some(AlgorithmParamValue::Text(v.as_str().to_string())),
            AttrValue::SmallInt(v) => Some(AlgorithmParamValue::Int(v as i64)),
            AttrValue::Json(v) => serde_json::from_str::<serde_json::Value>(&v)
                .ok()
                .map(AlgorithmParamValue::Json),
            AttrValue::CompressedText(data) => {
                data.decompress_text().ok().map(AlgorithmParamValue::Text)
            }
            AttrValue::CompressedFloatVec(data) => data.decompress_float_vec().ok().map(|vals| {
                AlgorithmParamValue::FloatList(vals.into_iter().map(f64::from).collect())
            }),
            AttrValue::Null => Some(AlgorithmParamValue::None),
            AttrValue::Bytes(_)
            | AttrValue::SubgraphRef(_)
            | AttrValue::NodeArray(_)
            | AttrValue::EdgeArray(_) => None,
        }
    }
}

impl From<i64> for AlgorithmParamValue {
    fn from(value: i64) -> Self {
        Self::Int(value)
    }
}

impl From<f64> for AlgorithmParamValue {
    fn from(value: f64) -> Self {
        Self::Float(value)
    }
}

impl From<bool> for AlgorithmParamValue {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

impl From<&str> for AlgorithmParamValue {
    fn from(value: &str) -> Self {
        Self::Text(value.to_string())
    }
}

impl From<String> for AlgorithmParamValue {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<Vec<i64>> for AlgorithmParamValue {
    fn from(value: Vec<i64>) -> Self {
        Self::IntList(value)
    }
}

impl From<Vec<f64>> for AlgorithmParamValue {
    fn from(value: Vec<f64>) -> Self {
        Self::FloatList(value)
    }
}

impl From<Vec<bool>> for AlgorithmParamValue {
    fn from(value: Vec<bool>) -> Self {
        Self::BoolList(value)
    }
}

impl From<Vec<String>> for AlgorithmParamValue {
    fn from(value: Vec<String>) -> Self {
        Self::TextList(value)
    }
}

impl From<serde_json::Value> for AlgorithmParamValue {
    fn from(value: serde_json::Value) -> Self {
        Self::Json(value)
    }
}

/// Execution telemetry surfaced to algorithms and steps.
#[derive(Default, Debug)]
pub struct Context {
    timers: HashMap<String, Duration>,
    iteration_events: Vec<IterationEvent>,
    active_step: Option<ActiveStep>,
    cancel_token: Option<Arc<AtomicBool>>,
}

impl Context {
    /// Create a context without cancellation capabilities.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a context wired to an external cancellation flag.
    pub fn with_cancel_token(token: Arc<AtomicBool>) -> Self {
        Self {
            cancel_token: Some(token),
            ..Self::default()
        }
    }

    /// Record a named duration, aggregating if called repeatedly.
    pub fn record_duration(&mut self, name: impl Into<String>, duration: Duration) {
        let entry = self
            .timers
            .entry(name.into())
            .or_insert_with(Duration::default);
        *entry += duration;
    }

    /// Execute a closure while timing it.
    pub fn with_scoped_timer<F, R>(&mut self, name: impl Into<String>, func: F) -> R
    where
        F: FnOnce() -> R,
    {
        let name = name.into();
        let start = Instant::now();
        let output = func();
        let elapsed = start.elapsed();
        self.record_duration(name, elapsed);
        output
    }

    /// Begin tracking a pipeline step.
    pub fn begin_step(&mut self, index: usize, _algorithm_id: impl Into<String>) {
        self.active_step = Some(ActiveStep {
            index,
            started_at: Instant::now(),
        });
    }

    /// Finish the active step, recording total duration under its identifier.
    pub fn finish_step(&mut self) {
        if let Some(active) = self.active_step.take() {
            let elapsed = active.started_at.elapsed();
            let metric = format!("pipeline.step.{}", active.index);
            self.record_duration(metric, elapsed);
        }
    }

    /// Emit iteration telemetry for iterative algorithms.
    pub fn emit_iteration(&mut self, iteration: usize, updates: usize) {
        self.iteration_events
            .push(IterationEvent { iteration, updates });
    }

    /// Drain accumulated iteration events (for tests/telemetry upload).
    pub fn take_iteration_events(&mut self) -> Vec<IterationEvent> {
        std::mem::take(&mut self.iteration_events)
    }

    /// Access aggregated timer metrics.
    pub fn timers(&self) -> &HashMap<String, Duration> {
        &self.timers
    }

    /// Whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.cancel_token
            .as_ref()
            .map(|token| token.load(Ordering::Relaxed))
            .unwrap_or(false)
    }

    /// Provide external visibility into the cancellation token.
    pub fn cancel_token(&self) -> Option<Arc<AtomicBool>> {
        self.cancel_token.as_ref().map(Arc::clone)
    }
}

#[derive(Debug)]
struct ActiveStep {
    index: usize,
    started_at: Instant,
}

/// Iteration telemetry emitted by algorithms.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IterationEvent {
    pub iteration: usize,
    pub updates: usize,
}

/// Trait implemented by all core algorithms.
pub trait Algorithm: Send + Sync {
    /// Stable identifier used for registry lookup.
    fn id(&self) -> &'static str;

    /// Optional rich metadata. Default implementation provides minimal info.
    fn metadata(&self) -> AlgorithmMetadata {
        AlgorithmMetadata {
            id: self.id().to_string(),
            name: self.id().to_string(),
            ..AlgorithmMetadata::default()
        }
    }

    /// Execute the algorithm, producing a new subgraph or mutating in place.
    fn execute(&self, ctx: &mut Context, subgraph: Subgraph) -> Result<Subgraph>;
}

/// Helper trait for algorithms that expose default parameter sets.
pub trait ConfigurableAlgorithm: Algorithm {
    /// Advertise default parameters to simplify builder usage.
    fn default_params(&self) -> AlgorithmParams {
        AlgorithmParams::new()
    }
}

/// Convenient alias for algorithm results.
pub type AlgorithmResult<T> = Result<T>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_tracks_timers() {
        let mut ctx = Context::new();
        ctx.with_scoped_timer("unit", || std::thread::sleep(Duration::from_millis(1)));
        assert!(ctx.timers().get("unit").is_some());
    }

    #[test]
    fn algorithm_params_round_trip() {
        let mut params = AlgorithmParams::new();
        params.set_int("iters", 10);
        params.set_float("alpha", 0.25);
        params.set_bool("normalize", true);
        params.set_text("strategy", "lpa");

        assert_eq!(params.get_int("iters"), Some(10));
        assert_eq!(params.get_float("alpha"), Some(0.25));
        assert_eq!(params.get_bool("normalize"), Some(true));
        assert_eq!(params.get_text("strategy"), Some("lpa"));
    }

    #[test]
    fn algorithm_param_value_to_attr() {
        use crate::types::AttrValue;
        let val = AlgorithmParamValue::Text("hello".to_string());
        assert_eq!(
            val.as_attr_value(),
            Some(AttrValue::Text("hello".to_string()))
        );
    }

    #[test]
    fn expect_helpers_validate_presence() {
        let mut params = AlgorithmParams::new();
        params.set_int("iters", 5);
        params.set_float("alpha", 0.3);
        params.set_bool("flag", true);
        params.set_text("mode", "fast");

        assert_eq!(params.expect_int("iters").unwrap(), 5);
        assert!((params.expect_float("alpha").unwrap() - 0.3).abs() < f64::EPSILON);
        assert!(params.expect_bool("flag").unwrap());
        assert_eq!(params.expect_text("mode").unwrap(), "fast");
        assert!(params.expect_int("missing").is_err());
    }
}
