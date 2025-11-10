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
use crate::types::{AttrValue, NodeId};

pub mod builder;
pub mod centrality;
pub mod community;
pub mod execution;
pub mod pathfinding;
pub mod pipeline;
pub mod registry;
pub mod steps;
pub mod temporal;

pub use pipeline::{AlgorithmSpec, Pipeline, PipelineBuilder, PipelineSpec};
pub use registry::{global_registry, AlgorithmFactory, Registry};
pub use steps::{
    ensure_core_steps_registered, global_step_registry, register_core_steps, Step, StepMetadata,
    StepRegistry, StepScope, StepSpec, StepValue,
};
pub use temporal::{
    ChangeType, ChangedEntities, EdgeAttrChange, NodeAttrChange, TemporalDelta, TemporalMetadata,
    TemporalScope,
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

    /// Create AlgorithmParams from a JSON map, converting values to AlgorithmParamValue.
    pub fn from_json_map(map: &serde_json::Map<String, serde_json::Value>) -> Result<Self> {
        let mut params = Self::new();
        for (key, value) in map.iter() {
            let param_value = match value {
                serde_json::Value::Number(n) => {
                    if let Some(i) = n.as_i64() {
                        AlgorithmParamValue::Int(i)
                    } else if let Some(f) = n.as_f64() {
                        AlgorithmParamValue::Float(f)
                    } else {
                        return Err(anyhow!("Invalid number: {}", n));
                    }
                }
                serde_json::Value::String(s) => AlgorithmParamValue::Text(s.clone()),
                serde_json::Value::Bool(b) => AlgorithmParamValue::Bool(*b),
                serde_json::Value::Array(arr) => {
                    // Try to convert to IntList
                    let ints: Result<Vec<i64>> = arr
                        .iter()
                        .map(|v| {
                            v.as_i64()
                                .ok_or_else(|| anyhow!("Array element is not an integer"))
                        })
                        .collect();
                    if let Ok(int_list) = ints {
                        AlgorithmParamValue::IntList(int_list)
                    } else {
                        // Fall back to JSON
                        AlgorithmParamValue::Json(value.clone())
                    }
                }
                _ => AlgorithmParamValue::Json(value.clone()),
            };
            params.insert(key.clone(), param_value);
        }
        Ok(params)
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

/// Call counter for profiling - tracks invocation counts
#[derive(Debug, Clone, Default)]
pub struct CallCounter {
    count: usize,
    total_duration: Duration,
}

impl CallCounter {
    pub fn increment(&mut self, duration: Duration) {
        self.count += 1;
        self.total_duration += duration;
    }

    pub fn count(&self) -> usize {
        self.count
    }

    pub fn total_duration(&self) -> Duration {
        self.total_duration
    }

    pub fn avg_duration(&self) -> Option<Duration> {
        if self.count > 0 {
            Some(self.total_duration / self.count as u32)
        } else {
            None
        }
    }
}

/// Execution telemetry surfaced to algorithms and steps.
#[derive(Debug)]
pub struct Context {
    timers: HashMap<String, Duration>,
    call_counters: HashMap<String, CallCounter>,
    stats: HashMap<String, f64>,
    iteration_events: Vec<IterationEvent>,
    active_step: Option<ActiveStep>,
    cancel_token: Option<Arc<AtomicBool>>,
    temporal_scope: Option<temporal::TemporalScope>,
    persist_results: bool,
    outputs: HashMap<String, AlgorithmOutput>,
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

    /// Create a context with a temporal scope.
    pub fn with_temporal_scope(scope: temporal::TemporalScope) -> Self {
        Self {
            temporal_scope: Some(scope),
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
    pub fn begin_step(&mut self, index: usize, algorithm_id: impl Into<String>) {
        self.active_step = Some(ActiveStep {
            index,
            label: algorithm_id.into(),
            started_at: Instant::now(),
        });
    }

    /// Finish the active step, recording total duration under its identifier.
    pub fn finish_step(&mut self) {
        if let Some(active) = self.active_step.take() {
            let elapsed = active.started_at.elapsed();
            let metric = if active.label.is_empty() {
                format!("pipeline.step.{}", active.index)
            } else {
                format!("pipeline.step.{}.{}", active.index, active.label)
            };
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

    /// Snapshot the timers as an owned map for external reporting.
    pub fn timer_snapshot(&self) -> HashMap<String, Duration> {
        self.timers.clone()
    }

    // === CALL COUNTER PROFILING ===

    /// Record a function call with its duration for profiling.
    /// This tracks both call counts and cumulative duration.
    pub fn record_call(&mut self, name: impl Into<String>, duration: Duration) {
        let entry = self
            .call_counters
            .entry(name.into())
            .or_insert_with(CallCounter::default);
        entry.increment(duration);
    }

    /// Execute a closure while timing it and recording the call count.
    pub fn with_counted_timer<F, R>(&mut self, name: impl Into<String>, func: F) -> R
    where
        F: FnOnce() -> R,
    {
        let name = name.into();
        let start = Instant::now();
        let output = func();
        let elapsed = start.elapsed();
        self.record_call(name, elapsed);
        output
    }

    /// Get call counters for detailed profiling analysis.
    pub fn call_counters(&self) -> &HashMap<String, CallCounter> {
        &self.call_counters
    }

    /// Take a snapshot of call counters for reporting.
    pub fn call_counter_snapshot(&self) -> HashMap<String, CallCounter> {
        self.call_counters.clone()
    }

    /// Record a scalar statistic (e.g., counts) for reporting.
    pub fn record_stat(&mut self, name: impl Into<String>, value: f64) {
        self.stats.insert(name.into(), value);
    }

    /// Snapshot recorded scalar statistics.
    pub fn stat_snapshot(&self) -> HashMap<String, f64> {
        self.stats.clone()
    }

    /// Print a detailed profiling report to stdout.
    pub fn print_profiling_report(&self, algorithm_name: &str) {
        println!("\n{}", "=".repeat(80));
        println!("Profiling Report: {}", algorithm_name);
        println!("{}", "=".repeat(80));

        // Sort by total duration (descending)
        let mut counters: Vec<_> = self.call_counters.iter().collect();
        counters.sort_by(|a, b| b.1.total_duration().cmp(&a.1.total_duration()));

        println!(
            "\n{:<50} {:>10} {:>12} {:>12}",
            "Phase", "Calls", "Total (ms)", "Avg (μs)"
        );
        println!("{}", "-".repeat(84));

        for (name, counter) in counters.iter() {
            let total_ms = counter.total_duration().as_secs_f64() * 1000.0;
            let avg_us = counter
                .avg_duration()
                .map(|d| d.as_secs_f64() * 1_000_000.0)
                .unwrap_or(0.0);

            println!(
                "{:<50} {:>10} {:>12.3} {:>12.3}",
                name,
                counter.count(),
                total_ms,
                avg_us
            );
        }

        if !self.stats.is_empty() {
            println!("\nRecorded Statistics");
            println!("{}", "-".repeat(80));
            let mut stats: Vec<_> = self.stats.iter().collect();
            stats.sort_by(|a, b| a.0.cmp(b.0));
            for (name, value) in stats {
                println!("{:<50} {:>20.3}", name, value);
            }
        }

        println!("{}", "=".repeat(80));
        println!();
    }

    /// Returns whether algorithms should persist their results into the graph.
    pub fn persist_results(&self) -> bool {
        self.persist_results
    }

    /// Configure whether algorithms should persist their results into the graph.
    pub fn set_persist_results(&mut self, persist: bool) {
        self.persist_results = persist;
    }

    /// Record an output payload produced by an algorithm.
    pub fn add_output(&mut self, key: impl Into<String>, output: AlgorithmOutput) {
        self.outputs.insert(key.into(), output);
    }

    /// Drain collected algorithm outputs.
    pub fn take_outputs(&mut self) -> HashMap<String, AlgorithmOutput> {
        std::mem::take(&mut self.outputs)
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

    // === TEMPORAL EXTENSIONS ===

    /// Get the current temporal scope, if set.
    pub fn temporal_scope(&self) -> Option<&temporal::TemporalScope> {
        self.temporal_scope.as_ref()
    }

    /// Set or update the temporal scope for this context.
    pub fn set_temporal_scope(&mut self, scope: temporal::TemporalScope) {
        self.temporal_scope = Some(scope);
    }

    /// Clear the temporal scope.
    pub fn clear_temporal_scope(&mut self) {
        self.temporal_scope = None;
    }

    /// Compute the delta between two temporal snapshots.
    ///
    /// This is a convenience method that wraps TemporalDelta::compute,
    /// allowing algorithms to easily compare snapshots during execution.
    pub fn delta(
        &self,
        prev: &crate::temporal::TemporalSnapshot,
        cur: &crate::temporal::TemporalSnapshot,
    ) -> Result<temporal::TemporalDelta> {
        temporal::TemporalDelta::compute(prev, cur)
            .map_err(|e| anyhow!("delta computation failed: {}", e))
    }

    /// Get entities that changed within the current temporal scope's window.
    ///
    /// This uses the TemporalIndex to identify all nodes and edges that
    /// were created, deleted, or modified within the time window defined
    /// by the temporal scope.
    ///
    /// Returns an error if no temporal scope with a window is set, or if
    /// the index is not provided.
    pub fn changed_entities(
        &self,
        index: &crate::temporal::TemporalIndex,
    ) -> Result<temporal::ChangedEntities> {
        let scope = self
            .temporal_scope
            .as_ref()
            .ok_or_else(|| anyhow!("no temporal scope set"))?;

        let (start, end) = scope
            .window
            .ok_or_else(|| anyhow!("temporal scope has no window defined"))?;

        let mut entities = temporal::ChangedEntities::empty();

        // Get all commits in the window
        for commit_id in start..=end {
            let changed_nodes = index.nodes_changed_in_commit(commit_id);
            let changed_edges = index.edges_changed_in_commit(commit_id);

            // For each changed node/edge, determine the change type
            for node_id in changed_nodes {
                // Check if created or deleted at this commit
                if index.node_exists_at(node_id, commit_id)
                    && !index.node_exists_at(node_id, commit_id.saturating_sub(1))
                {
                    entities.add_node(node_id, temporal::ChangeType::Created);
                } else if !index.node_exists_at(node_id, commit_id)
                    && (commit_id == 0
                        || index.node_exists_at(node_id, commit_id.saturating_sub(1)))
                {
                    entities.add_node(node_id, temporal::ChangeType::Deleted);
                } else {
                    entities.add_node(node_id, temporal::ChangeType::AttributeModified);
                }
            }

            for edge_id in changed_edges {
                // Similar logic for edges
                if index.edge_exists_at(edge_id, commit_id)
                    && !index.edge_exists_at(edge_id, commit_id.saturating_sub(1))
                {
                    entities.add_edge(edge_id, temporal::ChangeType::Created);
                } else if !index.edge_exists_at(edge_id, commit_id)
                    && (commit_id == 0
                        || index.edge_exists_at(edge_id, commit_id.saturating_sub(1)))
                {
                    entities.add_edge(edge_id, temporal::ChangeType::Deleted);
                } else {
                    entities.add_edge(edge_id, temporal::ChangeType::AttributeModified);
                }
            }
        }

        Ok(entities)
    }
}

impl Default for Context {
    fn default() -> Self {
        Self {
            timers: HashMap::new(),
            call_counters: HashMap::new(),
            stats: HashMap::new(),
            iteration_events: Vec::new(),
            active_step: None,
            cancel_token: None,
            temporal_scope: None,
            persist_results: true,
            outputs: HashMap::new(),
        }
    }
}

#[derive(Debug)]
struct ActiveStep {
    index: usize,
    label: String,
    started_at: Instant,
}

/// Iteration telemetry emitted by algorithms.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IterationEvent {
    pub iteration: usize,
    pub updates: usize,
}

/// Structured payload returned by algorithm execution when persistence is disabled.
#[derive(Clone, Debug)]
pub enum AlgorithmOutput {
    Components(Vec<Vec<NodeId>>),
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
