//! Pipeline validation framework with data-flow analysis and error reporting.

use std::collections::{HashMap, HashSet};
use std::fmt;

use anyhow::{anyhow, Result};

use super::core::StepSpec;
use super::schema::{Constraint, ParameterType, SchemaRegistry, StepSchema};
use crate::algorithms::AlgorithmParamValue;

/// Result of pipeline validation.
#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<ValidationWarning>,
}

impl ValidationReport {
    pub fn new() -> Self {
        Self {
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }

    pub fn add_error(&mut self, error: ValidationError) {
        self.errors.push(error);
    }

    pub fn add_warning(&mut self, warning: ValidationWarning) {
        self.warnings.push(warning);
    }

    /// Format report as human-readable text.
    pub fn format(&self) -> String {
        let mut output = String::new();

        if !self.errors.is_empty() {
            output.push_str(&format!(
                "❌ {} validation error(s):\n\n",
                self.errors.len()
            ));
            for (i, err) in self.errors.iter().enumerate() {
                output.push_str(&format!("{}. {}\n", i + 1, err));
            }
        }

        if !self.warnings.is_empty() {
            if !output.is_empty() {
                output.push('\n');
            }
            output.push_str(&format!(
                "⚠️  {} validation warning(s):\n\n",
                self.warnings.len()
            ));
            for (i, warn) in self.warnings.iter().enumerate() {
                output.push_str(&format!("{}. {}\n", i + 1, warn));
            }
        }

        if self.is_valid() && self.warnings.is_empty() {
            output.push_str("✅ Pipeline validation passed\n");
        }

        output
    }
}

impl Default for ValidationReport {
    fn default() -> Self {
        Self::new()
    }
}

/// Structured validation error with context.
#[derive(Debug, Clone)]
pub struct ValidationError {
    pub step_index: Option<usize>,
    pub step_id: String,
    pub category: ErrorCategory,
    pub message: String,
    pub suggestion: Option<String>,
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(idx) = self.step_index {
            write!(f, "Step {} ({}): ", idx, self.step_id)?;
        } else {
            write!(f, "Step ({}): ", self.step_id)?;
        }
        write!(f, "{}", self.message)?;
        if let Some(suggestion) = &self.suggestion {
            write!(f, "\n   Suggestion: {}", suggestion)?;
        }
        Ok(())
    }
}

/// Validation warning (non-fatal issues).
#[derive(Debug, Clone)]
pub struct ValidationWarning {
    pub step_index: Option<usize>,
    pub step_id: String,
    pub message: String,
}

impl fmt::Display for ValidationWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(idx) = self.step_index {
            write!(f, "Step {} ({}): {}", idx, self.step_id, self.message)
        } else {
            write!(f, "Step ({}): {}", self.step_id, self.message)
        }
    }
}

/// Categories of validation errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    /// Missing required parameter or input.
    MissingRequired,
    /// Type mismatch between producer and consumer.
    TypeMismatch,
    /// Variable referenced but never defined.
    UndefinedVariable,
    /// Output variable name conflicts with existing variable.
    VariableConflict,
    /// Parameter value violates constraint.
    ConstraintViolation,
    /// Step ID not found in registry.
    UnknownStep,
    /// Data-flow cycle detected.
    CyclicDependency,
    /// Other validation error.
    Other,
}

/// Pipeline validator with data-flow analysis.
pub struct PipelineValidator<'a> {
    schema_registry: &'a SchemaRegistry,
}

impl<'a> PipelineValidator<'a> {
    pub fn new(schema_registry: &'a SchemaRegistry) -> Self {
        Self { schema_registry }
    }

    /// Validate a complete pipeline.
    pub fn validate(&self, steps: &[StepSpec]) -> ValidationReport {
        let mut report = ValidationReport::new();
        let mut variable_types: HashMap<String, ParameterType> = HashMap::new();
        let mut defined_variables: HashSet<String> = HashSet::new();

        for (index, step) in steps.iter().enumerate() {
            self.validate_step(
                index,
                step,
                &mut report,
                &mut variable_types,
                &mut defined_variables,
            );
        }

        self.check_unused_variables(&variable_types, &defined_variables, &mut report);

        report
    }

    fn validate_step(
        &self,
        index: usize,
        step: &StepSpec,
        report: &mut ValidationReport,
        variable_types: &mut HashMap<String, ParameterType>,
        defined_variables: &mut HashSet<String>,
    ) {
        let schema = match self.schema_registry.get(&step.id) {
            Some(s) => s,
            None => {
                report.add_error(ValidationError {
                    step_index: Some(index),
                    step_id: step.id.clone(),
                    category: ErrorCategory::UnknownStep,
                    message: format!("Step '{}' is not registered", step.id),
                    suggestion: Some("Check step ID spelling or ensure step is registered".into()),
                });
                return;
            }
        };

        self.validate_inputs(index, step, schema, report, variable_types);
        self.validate_parameters(index, step, schema, report);
        self.validate_outputs(
            index,
            step,
            schema,
            report,
            variable_types,
            defined_variables,
        );
    }

    fn validate_inputs(
        &self,
        index: usize,
        step: &StepSpec,
        schema: &StepSchema,
        report: &mut ValidationReport,
        variable_types: &HashMap<String, ParameterType>,
    ) {
        for input_schema in &schema.inputs {
            let provided = step.inputs.iter().any(|i| i == &input_schema.name)
                || step
                    .params
                    .get_text(&input_schema.name)
                    .or_else(|| step.params.get_text("source"))
                    .is_some();

            if input_schema.required && !provided {
                report.add_error(ValidationError {
                    step_index: Some(index),
                    step_id: step.id.clone(),
                    category: ErrorCategory::MissingRequired,
                    message: format!("Missing required input '{}'", input_schema.name),
                    suggestion: Some(format!("Add '{}' to inputs or params", input_schema.name)),
                });
            }

            if let Some(var_name) = step
                .params
                .get_text(&input_schema.name)
                .or_else(|| step.params.get_text("source"))
            {
                if let Some(actual_type) = variable_types.get(var_name) {
                    if !input_schema.param_type.is_compatible(actual_type) {
                        report.add_error(ValidationError {
                            step_index: Some(index),
                            step_id: step.id.clone(),
                            category: ErrorCategory::TypeMismatch,
                            message: format!(
                                "Input '{}' expects type {}, but variable '{}' has type {}",
                                input_schema.name, input_schema.param_type, var_name, actual_type
                            ),
                            suggestion: None,
                        });
                    }
                } else {
                    report.add_error(ValidationError {
                        step_index: Some(index),
                        step_id: step.id.clone(),
                        category: ErrorCategory::UndefinedVariable,
                        message: format!(
                            "Input '{}' references undefined variable '{}'",
                            input_schema.name, var_name
                        ),
                        suggestion: Some(format!(
                            "Ensure '{}' is defined by a previous step",
                            var_name
                        )),
                    });
                }
            }
        }
    }

    fn validate_parameters(
        &self,
        index: usize,
        step: &StepSpec,
        schema: &StepSchema,
        report: &mut ValidationReport,
    ) {
        for param_schema in &schema.params {
            if let Some(value) = step.params.get(&param_schema.name) {
                self.validate_parameter_value(
                    index,
                    &step.id,
                    &param_schema.name,
                    value,
                    &param_schema.param_type,
                    &param_schema.constraints,
                    report,
                );
            } else if param_schema.required && param_schema.default.is_none() {
                report.add_error(ValidationError {
                    step_index: Some(index),
                    step_id: step.id.clone(),
                    category: ErrorCategory::MissingRequired,
                    message: format!("Missing required parameter '{}'", param_schema.name),
                    suggestion: Some(format!(
                        "Add '{}' parameter of type {}",
                        param_schema.name, param_schema.param_type
                    )),
                });
            }
        }
    }

    fn validate_parameter_value(
        &self,
        index: usize,
        step_id: &str,
        param_name: &str,
        value: &AlgorithmParamValue,
        expected_type: &ParameterType,
        constraints: &[Constraint],
        report: &mut ValidationReport,
    ) {
        let actual_type = self.infer_param_type(value);
        if !expected_type.is_compatible(&actual_type) {
            report.add_error(ValidationError {
                step_index: Some(index),
                step_id: step_id.to_string(),
                category: ErrorCategory::TypeMismatch,
                message: format!(
                    "Parameter '{}' expects type {}, but got {}",
                    param_name, expected_type, actual_type
                ),
                suggestion: None,
            });
        }

        for constraint in constraints {
            if let Err(msg) = self.check_constraint(value, constraint) {
                report.add_error(ValidationError {
                    step_index: Some(index),
                    step_id: step_id.to_string(),
                    category: ErrorCategory::ConstraintViolation,
                    message: format!("Parameter '{}': {}", param_name, msg),
                    suggestion: None,
                });
            }
        }
    }

    fn validate_outputs(
        &self,
        index: usize,
        step: &StepSpec,
        schema: &StepSchema,
        report: &mut ValidationReport,
        variable_types: &mut HashMap<String, ParameterType>,
        defined_variables: &mut HashSet<String>,
    ) {
        for output_schema in &schema.outputs {
            if let Some(var_name) = step
                .params
                .get_text(&output_schema.name)
                .or_else(|| step.params.get_text("target"))
                .or_else(|| step.params.get_text("output"))
            {
                if defined_variables.contains(var_name) {
                    report.add_warning(ValidationWarning {
                        step_index: Some(index),
                        step_id: step.id.clone(),
                        message: format!(
                            "Output variable '{}' overwrites existing variable",
                            var_name
                        ),
                    });
                }

                variable_types.insert(var_name.to_string(), output_schema.param_type.clone());
                defined_variables.insert(var_name.to_string());
            } else if output_schema.required {
                report.add_error(ValidationError {
                    step_index: Some(index),
                    step_id: step.id.clone(),
                    category: ErrorCategory::MissingRequired,
                    message: format!(
                        "Missing required output variable name for '{}'",
                        output_schema.name
                    ),
                    suggestion: Some(format!(
                        "Add '{}', 'target', or 'output' parameter",
                        output_schema.name
                    )),
                });
            }
        }
    }

    fn check_unused_variables(
        &self,
        variable_types: &HashMap<String, ParameterType>,
        defined_variables: &HashSet<String>,
        report: &mut ValidationReport,
    ) {
        for var in variable_types.keys() {
            if !defined_variables.contains(var) {
                report.add_warning(ValidationWarning {
                    step_index: None,
                    step_id: "pipeline".to_string(),
                    message: format!("Variable '{}' is referenced but never produced", var),
                });
            }
        }
    }

    fn infer_param_type(&self, value: &AlgorithmParamValue) -> ParameterType {
        match value {
            AlgorithmParamValue::Int(_) => ParameterType::Int,
            AlgorithmParamValue::Float(_) => ParameterType::Float,
            AlgorithmParamValue::Text(_) => ParameterType::Text,
            AlgorithmParamValue::Bool(_) => ParameterType::Bool,
            _ => ParameterType::Any,
        }
    }

    fn check_constraint(&self, value: &AlgorithmParamValue, constraint: &Constraint) -> Result<()> {
        match constraint {
            Constraint::Range { min, max } => {
                let num = match value {
                    AlgorithmParamValue::Int(i) => *i as f64,
                    AlgorithmParamValue::Float(f) => *f,
                    _ => return Ok(()),
                };
                if num < *min || num > *max {
                    return Err(anyhow!(
                        "value {} is outside valid range [{}, {}]",
                        num,
                        min,
                        max
                    ));
                }
            }
            Constraint::Positive => {
                let num = match value {
                    AlgorithmParamValue::Int(i) => *i as f64,
                    AlgorithmParamValue::Float(f) => *f,
                    _ => return Ok(()),
                };
                if num <= 0.0 {
                    return Err(anyhow!("value must be positive, got {}", num));
                }
            }
            Constraint::NonNegative => {
                let num = match value {
                    AlgorithmParamValue::Int(i) => *i as f64,
                    AlgorithmParamValue::Float(f) => *f,
                    _ => return Ok(()),
                };
                if num < 0.0 {
                    return Err(anyhow!("value must be non-negative, got {}", num));
                }
            }
            Constraint::Enum { options } => {
                if let AlgorithmParamValue::Text(s) = value {
                    if !options.contains(s) {
                        return Err(anyhow!(
                            "value '{}' is not one of: {}",
                            s,
                            options.join(", ")
                        ));
                    }
                }
            }
            Constraint::Pattern { pattern } => {
                if let AlgorithmParamValue::Text(s) = value {
                    if let Ok(re) = regex::Regex::new(pattern) {
                        if !re.is_match(s) {
                            return Err(anyhow!(
                                "value '{}' does not match pattern '{}'",
                                s,
                                pattern
                            ));
                        }
                    }
                }
            }
            Constraint::Custom { message } => {
                return Err(anyhow!("{}", message));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::steps::schema::{ParameterSchemaBuilder, StepSchemaBuilder};
    use crate::algorithms::AlgorithmParams;

    #[test]
    fn validate_missing_required_input() {
        let mut registry = SchemaRegistry::new();
        registry.register(
            StepSchemaBuilder::new("test.step")
                .input(
                    ParameterSchemaBuilder::new("source", ParameterType::Float)
                        .required()
                        .build(),
                )
                .build(),
        );

        let validator = PipelineValidator::new(&registry);
        let spec = StepSpec {
            id: "test.step".to_string(),
            params: AlgorithmParams::new(),
            inputs: vec![],
            outputs: vec![],
        };

        let report = validator.validate(&[spec]);
        assert!(!report.is_valid());
        assert_eq!(report.errors.len(), 1);
        assert_eq!(report.errors[0].category, ErrorCategory::MissingRequired);
    }

    #[test]
    fn validate_type_mismatch() {
        let mut registry = SchemaRegistry::new();

        registry.register(
            StepSchemaBuilder::new("producer")
                .output(
                    ParameterSchemaBuilder::new("result", ParameterType::Int)
                        .required()
                        .build(),
                )
                .param(
                    ParameterSchemaBuilder::new("target", ParameterType::Text)
                        .required()
                        .build(),
                )
                .build(),
        );

        registry.register(
            StepSchemaBuilder::new("consumer")
                .input(
                    ParameterSchemaBuilder::new("source", ParameterType::Text)
                        .required()
                        .build(),
                )
                .param(
                    ParameterSchemaBuilder::new("source", ParameterType::Text)
                        .required()
                        .build(),
                )
                .build(),
        );

        let validator = PipelineValidator::new(&registry);

        let mut params1 = AlgorithmParams::new();
        params1.insert("target", AlgorithmParamValue::Text("var1".into()));

        let mut params2 = AlgorithmParams::new();
        params2.insert("source", AlgorithmParamValue::Text("var1".into()));

        let steps = vec![
            StepSpec {
                id: "producer".to_string(),
                params: params1,
                inputs: vec![],
                outputs: vec![],
            },
            StepSpec {
                id: "consumer".to_string(),
                params: params2,
                inputs: vec![],
                outputs: vec![],
            },
        ];

        let report = validator.validate(&steps);
        assert!(!report.is_valid());
    }

    #[test]
    fn validate_constraint_violation() {
        let mut registry = SchemaRegistry::new();
        registry.register(
            StepSchemaBuilder::new("test.step")
                .param(
                    ParameterSchemaBuilder::new("value", ParameterType::Int)
                        .required()
                        .constraint(Constraint::Range {
                            min: 0.0,
                            max: 10.0,
                        })
                        .build(),
                )
                .build(),
        );

        let validator = PipelineValidator::new(&registry);

        let mut params = AlgorithmParams::new();
        params.insert("value", AlgorithmParamValue::Int(100));

        let spec = StepSpec {
            id: "test.step".to_string(),
            params,
            inputs: vec![],
            outputs: vec![],
        };

        let report = validator.validate(&[spec]);
        assert!(!report.is_valid());
        assert_eq!(
            report.errors[0].category,
            ErrorCategory::ConstraintViolation
        );
    }
}
