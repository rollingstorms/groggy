//! End-to-end demonstration of the complete builder validation system.
//!
//! This test shows the full workflow:
//! 1. Define step schemas
//! 2. Register them in a schema registry
//! 3. Build a pipeline using the step composer
//! 4. Validate the pipeline
//! 5. Execute it (simulated)

use groggy::algorithms::steps::{
    Constraint, ParameterSchemaBuilder, ParameterType, PipelineValidator, SchemaRegistry,
    StepComposer, StepSchemaBuilder,
};
use groggy::algorithms::{AlgorithmParamValue, CostHint};

#[test]
fn test_complete_builder_workflow() {
    // ========================================
    // Step 1: Define Step Schemas
    // ========================================

    let mut schema_registry = SchemaRegistry::new();

    // Schema for init_nodes step
    schema_registry.register(
        StepSchemaBuilder::new("core.init_nodes")
            .description("Initialize all node values with a constant")
            .cost_hint(CostHint::Linear)
            .output(
                ParameterSchemaBuilder::new(
                    "target",
                    ParameterType::NodeColumn {
                        value_type: Box::new(ParameterType::Float),
                    },
                )
                .description("Variable name for the initialized values")
                .required()
                .build(),
            )
            .param(
                ParameterSchemaBuilder::new("target", ParameterType::Text)
                    .description("Output variable name")
                    .required()
                    .build(),
            )
            .param(
                ParameterSchemaBuilder::new("value", ParameterType::Float)
                    .description("Initial value for all nodes")
                    .optional()
                    .default(serde_json::json!(0.0))
                    .constraint(Constraint::NonNegative)
                    .build(),
            )
            .tag("initialization")
            .build(),
    );

    // Schema for node_degree step
    schema_registry.register(
        StepSchemaBuilder::new("core.node_degree")
            .description("Compute the degree of each node")
            .cost_hint(CostHint::Linear)
            .output(
                ParameterSchemaBuilder::new(
                    "target",
                    ParameterType::NodeColumn {
                        value_type: Box::new(ParameterType::Int),
                    },
                )
                .description("Variable name for the computed degrees")
                .required()
                .build(),
            )
            .param(
                ParameterSchemaBuilder::new("target", ParameterType::Text)
                    .description("Output variable name")
                    .required()
                    .build(),
            )
            .tag("structural")
            .tag("degree")
            .build(),
    );

    // Schema for add step
    schema_registry.register(
        StepSchemaBuilder::new("core.add")
            .description("Add two node value maps element-wise")
            .cost_hint(CostHint::Linear)
            .input(
                ParameterSchemaBuilder::new(
                    "left",
                    ParameterType::NodeColumn {
                        value_type: Box::new(ParameterType::Float),
                    },
                )
                .description("Left operand variable")
                .required()
                .build(),
            )
            .input(
                ParameterSchemaBuilder::new(
                    "right",
                    ParameterType::NodeColumn {
                        value_type: Box::new(ParameterType::Float),
                    },
                )
                .description("Right operand variable")
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
                .description("Result variable")
                .required()
                .build(),
            )
            .param(
                ParameterSchemaBuilder::new("left", ParameterType::Text)
                    .required()
                    .build(),
            )
            .param(
                ParameterSchemaBuilder::new("right", ParameterType::Text)
                    .required()
                    .build(),
            )
            .param(
                ParameterSchemaBuilder::new("target", ParameterType::Text)
                    .required()
                    .build(),
            )
            .tag("arithmetic")
            .tag("binary")
            .build(),
    );

    // Schema for normalize_values step
    schema_registry.register(
        StepSchemaBuilder::new("core.normalize_values")
            .description("Normalize node values using specified method")
            .cost_hint(CostHint::Linear)
            .input(
                ParameterSchemaBuilder::new(
                    "source",
                    ParameterType::NodeColumn {
                        value_type: Box::new(ParameterType::Float),
                    },
                )
                .description("Source values to normalize")
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
                .description("Normalized output values")
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
                    .description("Normalization method")
                    .optional()
                    .default(serde_json::json!("sum"))
                    .constraint(Constraint::Enum {
                        options: vec!["sum".into(), "max".into(), "minmax".into()],
                    })
                    .build(),
            )
            .param(
                ParameterSchemaBuilder::new("epsilon", ParameterType::Float)
                    .description("Small value to prevent division by zero")
                    .optional()
                    .default(serde_json::json!(1e-9))
                    .constraint(Constraint::Positive)
                    .build(),
            )
            .tag("normalization")
            .tag("preprocessing")
            .build(),
    );

    // Schema for attach_node_attr step
    schema_registry.register(
        StepSchemaBuilder::new("core.attach_node_attr")
            .description("Persist a node value map as a node attribute")
            .cost_hint(CostHint::Linear)
            .input(
                ParameterSchemaBuilder::new(
                    "source",
                    ParameterType::NodeColumn {
                        value_type: Box::new(ParameterType::Any),
                    },
                )
                .description("Values to attach")
                .required()
                .build(),
            )
            .param(
                ParameterSchemaBuilder::new("source", ParameterType::Text)
                    .required()
                    .build(),
            )
            .param(
                ParameterSchemaBuilder::new("attr", ParameterType::Text)
                    .description("Attribute name to attach as")
                    .required()
                    .build(),
            )
            .tag("output")
            .tag("attributes")
            .build(),
    );

    // ========================================
    // Step 2: Build Pipeline Using Composer
    // ========================================

    let mut composer = StepComposer::new();

    // Initialize base values
    let base_values = composer.auto_var("base");
    composer.init_nodes(&base_values, AlgorithmParamValue::Float(1.0));

    // Compute node degrees
    let degrees = composer.auto_var("degrees");
    composer.node_degree(&degrees);

    // Combine base values with degrees
    let combined = composer.auto_var("combined");
    composer.add(&base_values, &degrees, &combined);

    // Normalize the combined values
    let normalized = composer.auto_var("normalized");
    composer.normalize(&combined, &normalized, "sum");

    // Attach as output attribute
    composer.attach_node_attr(&normalized, "weighted_degree");

    let pipeline_steps = composer.build();

    // ========================================
    // Step 3: Validate Pipeline
    // ========================================

    let validator = PipelineValidator::new(&schema_registry);
    let validation_report = validator.validate(&pipeline_steps);

    // Print validation report
    println!("\n{}", validation_report.format());

    // ========================================
    // Step 4: Assert Validation Success
    // ========================================

    assert!(
        validation_report.is_valid(),
        "Pipeline validation failed:\n{}",
        validation_report.format()
    );

    // ========================================
    // Step 5: Verify Pipeline Structure
    // ========================================

    assert_eq!(pipeline_steps.len(), 5);
    assert_eq!(pipeline_steps[0].id, "core.init_nodes");
    assert_eq!(pipeline_steps[1].id, "core.node_degree");
    assert_eq!(pipeline_steps[2].id, "core.add");
    assert_eq!(pipeline_steps[3].id, "core.normalize_values");
    assert_eq!(pipeline_steps[4].id, "core.attach_node_attr");

    // Verify final output attribute name
    assert_eq!(
        pipeline_steps[4].params.get_text("attr"),
        Some("weighted_degree")
    );
}

#[test]
fn test_validation_catches_errors() {
    // Create a schema registry
    let mut schema_registry = SchemaRegistry::new();

    // Register a simple schema
    schema_registry.register(
        StepSchemaBuilder::new("test.requires_input")
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

    // Build an invalid pipeline (missing required input)
    // This step has no parameters, so it will fail validation
    let steps = vec![groggy::algorithms::steps::StepSpec {
        id: "test.requires_input".to_string(),
        params: groggy::algorithms::AlgorithmParams::new(),
        inputs: vec![],
        outputs: vec![],
    }];

    // Validate
    let validator = PipelineValidator::new(&schema_registry);
    let report = validator.validate(&steps);

    // Should fail validation
    assert!(!report.is_valid());
    assert!(!report.errors.is_empty());

    // Print the helpful error message
    println!("\nExpected validation error:\n{}", report.format());
}

#[test]
fn test_schema_discovery_by_tags() {
    let mut registry = SchemaRegistry::new();

    // Register schemas with tags
    registry.register(
        StepSchemaBuilder::new("math.add")
            .tag("arithmetic")
            .tag("binary")
            .build(),
    );

    registry.register(
        StepSchemaBuilder::new("math.multiply")
            .tag("arithmetic")
            .tag("binary")
            .build(),
    );

    registry.register(
        StepSchemaBuilder::new("math.sqrt")
            .tag("arithmetic")
            .tag("unary")
            .build(),
    );

    registry.register(
        StepSchemaBuilder::new("graph.degree")
            .tag("structural")
            .build(),
    );

    // Find arithmetic operations
    let arithmetic = registry.find_by_tag("arithmetic");
    assert_eq!(arithmetic.len(), 3);

    // Find binary operations
    let binary = registry.find_by_tag("binary");
    assert_eq!(binary.len(), 2);

    // Find structural operations
    let structural = registry.find_by_tag("structural");
    assert_eq!(structural.len(), 1);
}
