//! Integration tests for builder validation, schema registry, and composition helpers.

use groggy::algorithms::steps::composition::templates;
use groggy::algorithms::steps::{
    Constraint, ErrorCategory, ParameterSchemaBuilder, ParameterType, PipelineValidator,
    SchemaRegistry, StepComposer, StepSchemaBuilder, StepSpec, StepTemplate,
};
use groggy::algorithms::{AlgorithmParamValue, AlgorithmParams, CostHint};
use std::collections::HashMap;

#[test]
fn test_schema_builder_fluent_api() {
    let schema = StepSchemaBuilder::new("test.add")
        .description("Add two values together")
        .cost_hint(CostHint::Linear)
        .input(
            ParameterSchemaBuilder::new("left", ParameterType::Float)
                .description("Left operand")
                .required()
                .build(),
        )
        .input(
            ParameterSchemaBuilder::new("right", ParameterType::Float)
                .description("Right operand")
                .required()
                .build(),
        )
        .output(
            ParameterSchemaBuilder::new("target", ParameterType::Float)
                .description("Result variable")
                .required()
                .build(),
        )
        .param(
            ParameterSchemaBuilder::new("precision", ParameterType::Int)
                .description("Decimal precision")
                .optional()
                .default(serde_json::json!(2))
                .constraint(Constraint::Range {
                    min: 0.0,
                    max: 10.0,
                })
                .build(),
        )
        .tag("arithmetic")
        .tag("binary")
        .build();

    assert_eq!(schema.id, "test.add");
    assert_eq!(schema.inputs.len(), 2);
    assert_eq!(schema.outputs.len(), 1);
    assert_eq!(schema.params.len(), 1);
    assert_eq!(schema.tags, vec!["arithmetic", "binary"]);
}

#[test]
fn test_schema_registry_operations() {
    let mut registry = SchemaRegistry::new();

    let schema1 = StepSchemaBuilder::new("test.step1")
        .tag("category_a")
        .build();
    let schema2 = StepSchemaBuilder::new("test.step2")
        .tag("category_a")
        .tag("category_b")
        .build();

    registry.register(schema1);
    registry.register(schema2);

    assert!(registry.contains("test.step1"));
    assert!(registry.contains("test.step2"));
    assert!(!registry.contains("test.nonexistent"));

    let by_tag = registry.find_by_tag("category_a");
    assert_eq!(by_tag.len(), 2);

    let step_ids = registry.step_ids();
    assert_eq!(step_ids.len(), 2);
}

#[test]
fn test_validation_missing_required_input() {
    let mut registry = SchemaRegistry::new();
    registry.register(
        StepSchemaBuilder::new("test.step")
            .input(
                ParameterSchemaBuilder::new("source", ParameterType::Float)
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
    let spec = StepSpec {
        id: "test.step".to_string(),
        params: AlgorithmParams::new(),
        inputs: vec![],
        outputs: vec![],
    };

    let report = validator.validate(&[spec]);

    assert!(!report.is_valid());
    assert_eq!(report.errors.len(), 2); // Missing both input and param
    assert!(report
        .errors
        .iter()
        .all(|e| e.category == ErrorCategory::MissingRequired));
    assert!(report.errors.iter().any(|e| e.message.contains("source")));
}

#[test]
fn test_validation_constraint_violation() {
    let mut registry = SchemaRegistry::new();
    registry.register(
        StepSchemaBuilder::new("test.step")
            .param(
                ParameterSchemaBuilder::new("count", ParameterType::Int)
                    .required()
                    .constraint(Constraint::Range {
                        min: 1.0,
                        max: 100.0,
                    })
                    .build(),
            )
            .build(),
    );

    let validator = PipelineValidator::new(&registry);

    let mut params = AlgorithmParams::new();
    params.insert("count", AlgorithmParamValue::Int(500));

    let spec = StepSpec {
        id: "test.step".to_string(),
        params,
        inputs: vec![],
        outputs: vec![],
    };

    let report = validator.validate(&[spec]);

    assert!(!report.is_valid());
    assert!(report
        .errors
        .iter()
        .any(|e| e.category == ErrorCategory::ConstraintViolation));
}

#[test]
fn test_validation_type_compatibility() {
    let mut registry = SchemaRegistry::new();

    // Producer creates an Int output
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

    // Consumer expects Float input (compatible with Int)
    registry.register(
        StepSchemaBuilder::new("consumer")
            .input(
                ParameterSchemaBuilder::new("source", ParameterType::Float)
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

    // Should pass because Float accepts Int
    assert!(report.is_valid(), "Validation report: {}", report.format());
}

#[test]
fn test_validation_report_formatting() {
    let mut registry = SchemaRegistry::new();
    registry.register(
        StepSchemaBuilder::new("test.step")
            .param(
                ParameterSchemaBuilder::new("value", ParameterType::Int)
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
    let formatted = report.format();

    assert!(formatted.contains("validation error"));
    assert!(formatted.contains("test.step"));
    assert!(formatted.contains("value"));
}

#[test]
fn test_step_composer_basic_operations() {
    let mut composer = StepComposer::new();

    composer.init_nodes("values", AlgorithmParamValue::Float(1.0));
    composer.node_degree("degrees");
    composer.add("values", "degrees", "result");
    composer.normalize("result", "normalized", "sum");
    composer.attach_node_attr("normalized", "output");

    let steps = composer.build();

    assert_eq!(steps.len(), 5);
    assert_eq!(steps[0].id, "core.init_nodes");
    assert_eq!(steps[1].id, "core.node_degree");
    assert_eq!(steps[2].id, "core.add");
    assert_eq!(steps[3].id, "core.normalize_values");
    assert_eq!(steps[4].id, "core.attach_node_attr");
}

#[test]
fn test_step_composer_auto_variables() {
    let mut composer = StepComposer::new();

    let var1 = composer.auto_var("temp");
    let var2 = composer.auto_var("temp");
    let var3 = composer.auto_var("result");

    assert_ne!(var1, var2);
    assert_ne!(var2, var3);
    assert!(var1.starts_with("temp_"));
    assert!(var2.starts_with("temp_"));
    assert!(var3.starts_with("result_"));
}

#[test]
fn test_degree_centrality_template() {
    let template = templates::DegreeCentrality;

    assert_eq!(template.id(), "template.degree_centrality");

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

    // Verify the output attribute is set correctly
    let attach_step = &steps[2];
    assert_eq!(attach_step.params.get_text("attr"), Some("centrality"));
}

#[test]
fn test_zscore_template() {
    let template = templates::ZScoreNormalization;

    let mut params = HashMap::new();
    params.insert(
        "input_attr".to_string(),
        AlgorithmParamValue::Text("score".to_string()),
    );
    params.insert(
        "output_attr".to_string(),
        AlgorithmParamValue::Text("zscore".to_string()),
    );

    let steps = template.generate(&params).unwrap();

    assert_eq!(steps.len(), 3);
    assert_eq!(steps[0].id, "core.load_node_attr");
    assert_eq!(steps[1].id, "core.standardize");
    assert_eq!(steps[2].id, "core.attach_node_attr");
}

#[test]
fn test_parameter_type_compatibility_rules() {
    // Int compatible with Float
    assert!(ParameterType::Float.is_compatible(&ParameterType::Int));

    // But not vice versa
    assert!(!ParameterType::Int.is_compatible(&ParameterType::Float));

    // Any is compatible with everything
    assert!(ParameterType::Any.is_compatible(&ParameterType::Int));
    assert!(ParameterType::Any.is_compatible(&ParameterType::Text));
    assert!(ParameterType::Int.is_compatible(&ParameterType::Any));

    // Same types are compatible
    assert!(ParameterType::Text.is_compatible(&ParameterType::Text));

    // Different types are incompatible
    assert!(!ParameterType::Text.is_compatible(&ParameterType::Int));
}

#[test]
fn test_constraint_display() {
    let range = Constraint::Range {
        min: 0.0,
        max: 10.0,
    };
    assert_eq!(range.to_string(), "range [0, 10]");

    let positive = Constraint::Positive;
    assert_eq!(positive.to_string(), "must be positive");

    let enum_constraint = Constraint::Enum {
        options: vec!["a".to_string(), "b".to_string()],
    };
    assert_eq!(enum_constraint.to_string(), "must be one of: a, b");
}

#[test]
fn test_complete_pipeline_validation_workflow() {
    // Create a schema registry with multiple steps
    let mut registry = SchemaRegistry::new();

    // Register init_nodes step
    registry.register(
        StepSchemaBuilder::new("core.init_nodes")
            .description("Initialize node values")
            .cost_hint(CostHint::Linear)
            .output(
                ParameterSchemaBuilder::new(
                    "target",
                    ParameterType::NodeColumn {
                        value_type: Box::new(ParameterType::Float),
                    },
                )
                .required()
                .build(),
            )
            .param(
                ParameterSchemaBuilder::new("target", ParameterType::Text)
                    .required()
                    .build(),
            )
            .param(
                ParameterSchemaBuilder::new("value", ParameterType::Float)
                    .optional()
                    .default(serde_json::json!(0.0))
                    .build(),
            )
            .tag("initialization")
            .build(),
    );

    // Register normalize step
    registry.register(
        StepSchemaBuilder::new("core.normalize_values")
            .description("Normalize values")
            .cost_hint(CostHint::Linear)
            .input(
                ParameterSchemaBuilder::new(
                    "source",
                    ParameterType::NodeColumn {
                        value_type: Box::new(ParameterType::Float),
                    },
                )
                .required()
                .build(),
            )
            .output(
                ParameterSchemaBuilder::new(
                    "target",
                    ParameterType::NodeColumn {
                        value_type: Box::new(ParameterType::Float),
                    },
                )
                .required()
                .build(),
            )
            .param(
                ParameterSchemaBuilder::new("source", ParameterType::Text)
                    .required()
                    .build(),
            )
            .param(
                ParameterSchemaBuilder::new("target", ParameterType::Text)
                    .required()
                    .build(),
            )
            .param(
                ParameterSchemaBuilder::new("method", ParameterType::Text)
                    .optional()
                    .constraint(Constraint::Enum {
                        options: vec!["sum".into(), "max".into(), "minmax".into()],
                    })
                    .build(),
            )
            .tag("normalization")
            .build(),
    );

    // Build a valid pipeline
    let mut params1 = AlgorithmParams::new();
    params1.insert("target", AlgorithmParamValue::Text("values".into()));
    params1.insert("value", AlgorithmParamValue::Float(1.0));

    let mut params2 = AlgorithmParams::new();
    params2.insert("source", AlgorithmParamValue::Text("values".into()));
    params2.insert("target", AlgorithmParamValue::Text("normalized".into()));
    params2.insert("method", AlgorithmParamValue::Text("sum".into()));

    let steps = vec![
        StepSpec {
            id: "core.init_nodes".to_string(),
            params: params1,
            inputs: vec![],
            outputs: vec![],
        },
        StepSpec {
            id: "core.normalize_values".to_string(),
            params: params2,
            inputs: vec![],
            outputs: vec![],
        },
    ];

    let validator = PipelineValidator::new(&registry);
    let report = validator.validate(&steps);

    assert!(report.is_valid(), "Validation failed: {}", report.format());
    assert_eq!(report.errors.len(), 0);
}
