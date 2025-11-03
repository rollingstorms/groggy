//! Step composition helpers and templates for common patterns.

use std::collections::HashMap;

use anyhow::Result;

use super::core::StepSpec;
use crate::algorithms::{AlgorithmParamValue, AlgorithmParams};

/// Template for composing multiple steps into a reusable pattern.
pub trait StepTemplate {
    /// Generate step specifications from template parameters.
    fn generate(&self, params: &HashMap<String, AlgorithmParamValue>) -> Result<Vec<StepSpec>>;

    /// Template identifier.
    fn id(&self) -> &str;

    /// Human-readable description.
    fn description(&self) -> &str;
}

/// Builder for creating step sequences with common patterns.
pub struct StepComposer {
    steps: Vec<StepSpec>,
    var_counter: usize,
}

impl StepComposer {
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            var_counter: 0,
        }
    }

    /// Generate a unique variable name.
    pub fn auto_var(&mut self, prefix: &str) -> String {
        let name = format!("{}_{}", prefix, self.var_counter);
        self.var_counter += 1;
        name
    }

    /// Add a step to the sequence.
    pub fn add_step(&mut self, spec: StepSpec) {
        self.steps.push(spec);
    }

    /// Initialize nodes with a value.
    pub fn init_nodes(&mut self, var_name: &str, value: AlgorithmParamValue) {
        let mut params = AlgorithmParams::new();
        params.insert("target", AlgorithmParamValue::Text(var_name.to_string()));
        params.insert("value", value);

        self.steps.push(StepSpec {
            id: "core.init_nodes".to_string(),
            params,
            inputs: vec![],
            outputs: vec![],
        });
    }

    /// Load node attribute into variable.
    pub fn load_node_attr(&mut self, attr: &str, var_name: &str, default: AlgorithmParamValue) {
        let mut params = AlgorithmParams::new();
        params.insert("target", AlgorithmParamValue::Text(var_name.to_string()));
        params.insert("attr", AlgorithmParamValue::Text(attr.to_string()));
        params.insert("default", default);

        self.steps.push(StepSpec {
            id: "core.load_node_attr".to_string(),
            params,
            inputs: vec![],
            outputs: vec![],
        });
    }

    /// Attach variable as node attribute.
    pub fn attach_node_attr(&mut self, var_name: &str, attr: &str) {
        let mut params = AlgorithmParams::new();
        params.insert("source", AlgorithmParamValue::Text(var_name.to_string()));
        params.insert("attr", AlgorithmParamValue::Text(attr.to_string()));

        self.steps.push(StepSpec {
            id: "core.attach_node_attr".to_string(),
            params,
            inputs: vec![],
            outputs: vec![],
        });
    }

    /// Binary arithmetic operation.
    pub fn binary_op(&mut self, op: &str, left: &str, right: &str, target: &str) {
        let mut params = AlgorithmParams::new();
        params.insert("left", AlgorithmParamValue::Text(left.to_string()));
        params.insert("right", AlgorithmParamValue::Text(right.to_string()));
        params.insert("target", AlgorithmParamValue::Text(target.to_string()));

        let step_id = format!("core.{}", op);
        self.steps.push(StepSpec {
            id: step_id,
            params,
            inputs: vec![],
            outputs: vec![],
        });
    }

    /// Add two variables.
    pub fn add(&mut self, left: &str, right: &str, target: &str) {
        self.binary_op("add", left, right, target);
    }

    /// Subtract two variables.
    pub fn sub(&mut self, left: &str, right: &str, target: &str) {
        self.binary_op("sub", left, right, target);
    }

    /// Multiply two variables.
    pub fn mul(&mut self, left: &str, right: &str, target: &str) {
        self.binary_op("mul", left, right, target);
    }

    /// Divide two variables.
    pub fn div(&mut self, left: &str, right: &str, target: &str) {
        self.binary_op("div", left, right, target);
    }

    /// Normalize values.
    pub fn normalize(&mut self, source: &str, target: &str, method: &str) {
        let mut params = AlgorithmParams::new();
        params.insert("source", AlgorithmParamValue::Text(source.to_string()));
        params.insert("target", AlgorithmParamValue::Text(target.to_string()));
        params.insert("method", AlgorithmParamValue::Text(method.to_string()));
        params.insert("epsilon", AlgorithmParamValue::Float(1e-9));

        self.steps.push(StepSpec {
            id: "core.normalize_values".to_string(),
            params,
            inputs: vec![],
            outputs: vec![],
        });
    }

    /// Compute node degrees.
    pub fn node_degree(&mut self, target: &str) {
        let mut params = AlgorithmParams::new();
        params.insert("target", AlgorithmParamValue::Text(target.to_string()));

        self.steps.push(StepSpec {
            id: "core.node_degree".to_string(),
            params,
            inputs: vec![],
            outputs: vec![],
        });
    }

    /// Reduce node values to scalar.
    pub fn reduce_nodes(&mut self, source: &str, target: &str, reducer: &str) {
        let mut params = AlgorithmParams::new();
        params.insert("source", AlgorithmParamValue::Text(source.to_string()));
        params.insert("target", AlgorithmParamValue::Text(target.to_string()));
        params.insert("reducer", AlgorithmParamValue::Text(reducer.to_string()));

        self.steps.push(StepSpec {
            id: "core.reduce_nodes".to_string(),
            params,
            inputs: vec![],
            outputs: vec![],
        });
    }

    /// Build and return the composed steps.
    pub fn build(self) -> Vec<StepSpec> {
        self.steps
    }
}

impl Default for StepComposer {
    fn default() -> Self {
        Self::new()
    }
}

/// Macro for quickly composing step sequences.
#[macro_export]
macro_rules! compose_steps {
    ($composer:expr, init_nodes $var:expr => $value:expr) => {
        $composer.init_nodes($var, $value)
    };
    ($composer:expr, load_attr $attr:expr => $var:expr, default = $default:expr) => {
        $composer.load_node_attr($attr, $var, $default)
    };
    ($composer:expr, attach_attr $var:expr => $attr:expr) => {
        $composer.attach_node_attr($var, $attr)
    };
    ($composer:expr, add $left:expr, $right:expr => $target:expr) => {
        $composer.add($left, $right, $target)
    };
    ($composer:expr, sub $left:expr, $right:expr => $target:expr) => {
        $composer.sub($left, $right, $target)
    };
    ($composer:expr, mul $left:expr, $right:expr => $target:expr) => {
        $composer.mul($left, $right, $target)
    };
    ($composer:expr, div $left:expr, $right:expr => $target:expr) => {
        $composer.div($left, $right, $target)
    };
    ($composer:expr, normalize $source:expr => $target:expr, method = $method:expr) => {
        $composer.normalize($source, $target, $method)
    };
    ($composer:expr, node_degree => $target:expr) => {
        $composer.node_degree($target)
    };
    ($composer:expr, reduce $source:expr => $target:expr, $reducer:expr) => {
        $composer.reduce_nodes($source, $target, $reducer)
    };
}

/// Common algorithm patterns as templates.
pub mod templates {
    use super::*;

    /// Degree centrality template: compute and normalize node degrees.
    pub struct DegreeCentrality;

    impl StepTemplate for DegreeCentrality {
        fn id(&self) -> &str {
            "template.degree_centrality"
        }

        fn description(&self) -> &str {
            "Compute and normalize node degree centrality"
        }

        fn generate(&self, params: &HashMap<String, AlgorithmParamValue>) -> Result<Vec<StepSpec>> {
            let output_attr = params
                .get("output_attr")
                .and_then(|v| {
                    if let AlgorithmParamValue::Text(s) = v {
                        Some(s.as_str())
                    } else {
                        None
                    }
                })
                .unwrap_or("degree_centrality");

            let mut composer = StepComposer::new();
            let degrees = composer.auto_var("degrees");
            let normalized = composer.auto_var("normalized");

            composer.node_degree(&degrees);
            composer.normalize(&degrees, &normalized, "sum");
            composer.attach_node_attr(&normalized, output_attr);

            Ok(composer.build())
        }
    }

    /// Weighted average template: combine multiple attributes with weights.
    pub struct WeightedAverage;

    impl StepTemplate for WeightedAverage {
        fn id(&self) -> &str {
            "template.weighted_average"
        }

        fn description(&self) -> &str {
            "Compute weighted average of node attributes"
        }

        fn generate(&self, params: &HashMap<String, AlgorithmParamValue>) -> Result<Vec<StepSpec>> {
            let output_attr = params
                .get("output_attr")
                .and_then(|v| {
                    if let AlgorithmParamValue::Text(s) = v {
                        Some(s.as_str())
                    } else {
                        None
                    }
                })
                .unwrap_or("weighted_avg");

            let mut composer = StepComposer::new();

            let attr1 = composer.auto_var("attr1");
            let attr2 = composer.auto_var("attr2");
            let weighted1 = composer.auto_var("weighted1");
            let weighted2 = composer.auto_var("weighted2");
            let sum = composer.auto_var("sum");

            composer.load_node_attr("attr1", &attr1, AlgorithmParamValue::Float(0.0));
            composer.load_node_attr("attr2", &attr2, AlgorithmParamValue::Float(0.0));

            composer.mul(&attr1, "weight1", &weighted1);
            composer.mul(&attr2, "weight2", &weighted2);
            composer.add(&weighted1, &weighted2, &sum);

            composer.attach_node_attr(&sum, output_attr);

            Ok(composer.build())
        }
    }

    /// Z-score normalization template: standardize attribute values.
    pub struct ZScoreNormalization;

    impl StepTemplate for ZScoreNormalization {
        fn id(&self) -> &str {
            "template.zscore"
        }

        fn description(&self) -> &str {
            "Standardize node attribute using z-score normalization"
        }

        fn generate(&self, params: &HashMap<String, AlgorithmParamValue>) -> Result<Vec<StepSpec>> {
            let input_attr = params
                .get("input_attr")
                .and_then(|v| {
                    if let AlgorithmParamValue::Text(s) = v {
                        Some(s.as_str())
                    } else {
                        None
                    }
                })
                .unwrap_or("value");

            let output_attr = params
                .get("output_attr")
                .and_then(|v| {
                    if let AlgorithmParamValue::Text(s) = v {
                        Some(s.as_str())
                    } else {
                        None
                    }
                })
                .unwrap_or("zscore");

            let mut composer = StepComposer::new();
            let values = composer.auto_var("values");
            let normalized = composer.auto_var("normalized");

            composer.load_node_attr(input_attr, &values, AlgorithmParamValue::Float(0.0));

            let mut params = AlgorithmParams::new();
            params.insert("source", AlgorithmParamValue::Text(values));
            params.insert("target", AlgorithmParamValue::Text(normalized.clone()));
            params.insert("epsilon", AlgorithmParamValue::Float(1e-9));

            composer.add_step(StepSpec {
                id: "core.standardize".to_string(),
                params,
                inputs: vec![],
                outputs: vec![],
            });

            composer.attach_node_attr(&normalized, output_attr);

            Ok(composer.build())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn composer_generates_steps() {
        let mut composer = StepComposer::new();
        composer.init_nodes("values", AlgorithmParamValue::Float(1.0));
        composer.node_degree("degrees");
        composer.add("values", "degrees", "result");

        let steps = composer.build();
        assert_eq!(steps.len(), 3);
        assert_eq!(steps[0].id, "core.init_nodes");
        assert_eq!(steps[1].id, "core.node_degree");
        assert_eq!(steps[2].id, "core.add");
    }

    #[test]
    fn auto_var_generates_unique_names() {
        let mut composer = StepComposer::new();
        let var1 = composer.auto_var("temp");
        let var2 = composer.auto_var("temp");

        assert_ne!(var1, var2);
        assert!(var1.starts_with("temp_"));
        assert!(var2.starts_with("temp_"));
    }

    #[test]
    fn degree_centrality_template() {
        let template = templates::DegreeCentrality;
        let mut params = HashMap::new();
        params.insert(
            "output_attr".to_string(),
            AlgorithmParamValue::Text("centrality".to_string()),
        );

        let steps = template.generate(&params).unwrap();
        assert_eq!(steps.len(), 3);
        assert_eq!(steps[0].id, "core.node_degree");
        assert_eq!(steps[1].id, "core.normalize_values");
        assert_eq!(steps[2].id, "core.attach_node_attr");
    }
}
