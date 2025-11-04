# Pipeline Builder Validation Guide

## Quick Start

The pipeline builder now includes comprehensive validation, schema definitions, and composition helpers to catch errors early and simplify pipeline construction.

## Basic Validation

### Validate a Pipeline

```rust
use groggy::algorithms::builder::validate_pipeline;
use groggy::algorithms::steps::{StepSpec, AlgorithmParams, AlgorithmParamValue};

// Build your pipeline
let steps = vec![
    StepSpec {
        id: "core.init_nodes".to_string(),
        params: params1,
        inputs: vec![],
        outputs: vec![],
    },
    // ... more steps
];

// Validate before executing
let report = validate_pipeline(&steps);

if !report.is_valid() {
    eprintln!("Validation failed:\n{}", report.format());
    return;
}
```

### Understanding Validation Reports

```rust
let report = validator.validate(&steps);

// Check if valid
if report.is_valid() {
    println!("✅ Pipeline is valid");
}

// Print formatted report
println!("{}", report.format());

// Programmatic access to errors
for error in &report.errors {
    println!("Error at step {}: {}", 
        error.step_index.unwrap_or(0), 
        error.message);
    
    if let Some(suggestion) = &error.suggestion {
        println!("  Suggestion: {}", suggestion);
    }
}
```

## Defining Step Schemas

### Basic Schema

```rust
use groggy::algorithms::steps::{
    StepSchemaBuilder, ParameterSchemaBuilder, ParameterType, CostHint
};

let schema = StepSchemaBuilder::new("my.custom_step")
    .description("Does something useful")
    .cost_hint(CostHint::Linear)
    .input(
        ParameterSchemaBuilder::new("source", ParameterType::NodeColumn {
            value_type: Box::new(ParameterType::Float)
        })
        .description("Input values")
        .required()
        .build()
    )
    .output(
        ParameterSchemaBuilder::new("target", ParameterType::NodeColumn {
            value_type: Box::new(ParameterType::Float)
        })
        .description("Output values")
        .required()
        .build()
    )
    .param(
        ParameterSchemaBuilder::new("threshold", ParameterType::Float)
            .description("Filtering threshold")
            .optional()
            .default(serde_json::json!(0.5))
            .constraint(Constraint::Range { min: 0.0, max: 1.0 })
            .build()
    )
    .tag("filtering")
    .tag("threshold")
    .build();
```

### Parameter Types

Available types:
- `ParameterType::Int`: Integer values
- `ParameterType::Float`: Floating-point values
- `ParameterType::Text`: String values
- `ParameterType::Bool`: Boolean values
- `ParameterType::NodeMap`: Map from NodeId to value
- `ParameterType::EdgeMap`: Map from EdgeId to value
- `ParameterType::NodeColumn`: Columnar node storage (most common)
- `ParameterType::Scalar`: Single scalar value
- `ParameterType::Snapshot`: Temporal snapshot
- `ParameterType::TemporalIndex`: Temporal index
- `ParameterType::Union`: Union of multiple types
- `ParameterType::Any`: Accepts any type

### Constraints

Add validation rules to parameters:

```rust
use groggy::algorithms::steps::Constraint;

// Range constraint
.constraint(Constraint::Range { min: 0.0, max: 100.0 })

// Must be positive
.constraint(Constraint::Positive)

// Must be non-negative
.constraint(Constraint::NonNegative)

// Must match regex pattern
.constraint(Constraint::Pattern { 
    pattern: r"^[a-z_]+$".to_string() 
})

// Must be one of enumerated values
.constraint(Constraint::Enum {
    options: vec!["sum".into(), "max".into(), "mean".into()]
})

// Custom constraint with message
.constraint(Constraint::Custom {
    message: "Value must be even".to_string()
})
```

## Using the Step Composer

### Basic Composition

```rust
use groggy::algorithms::steps::StepComposer;

let mut composer = StepComposer::new();

// Initialize node values
composer.init_nodes("initial", AlgorithmParamValue::Float(0.0));

// Load an attribute
composer.load_node_attr("score", "scores", AlgorithmParamValue::Float(0.0));

// Compute node degrees
composer.node_degree("degrees");

// Add two variables
composer.add("scores", "degrees", "combined");

// Normalize
composer.normalize("combined", "normalized", "sum");

// Attach as attribute
composer.attach_node_attr("normalized", "result");

// Build the pipeline
let steps = composer.build();
```

### Auto-Generated Variables

```rust
let mut composer = StepComposer::new();

// Generate unique variable names automatically
let temp1 = composer.auto_var("temp");  // "temp_0"
let temp2 = composer.auto_var("temp");  // "temp_1"
let result = composer.auto_var("result");  // "result_2"

composer.init_nodes(&temp1, AlgorithmParamValue::Float(1.0));
composer.node_degree(&temp2);
composer.add(&temp1, &temp2, &result);
```

### Helper Methods

```rust
// Arithmetic operations
composer.add("a", "b", "sum");
composer.sub("a", "b", "diff");
composer.mul("a", "b", "product");
composer.div("a", "b", "quotient");

// Aggregation
composer.reduce_nodes("values", "total", "sum");

// Normalization
composer.normalize("values", "normalized", "minmax");
```

## Using Templates

### Built-in Templates

#### Degree Centrality

```rust
use groggy::algorithms::steps::composition::templates::DegreeCentrality;
use groggy::algorithms::steps::StepTemplate;

let template = DegreeCentrality;
let mut params = HashMap::new();
params.insert(
    "output_attr".to_string(),
    AlgorithmParamValue::Text("centrality".to_string())
);

let steps = template.generate(&params)?;
```

#### Z-Score Normalization

```rust
use groggy::algorithms::steps::composition::templates::ZScoreNormalization;

let template = ZScoreNormalization;
let mut params = HashMap::new();
params.insert("input_attr", AlgorithmParamValue::Text("score".into()));
params.insert("output_attr", AlgorithmParamValue::Text("zscore".into()));

let steps = template.generate(&params)?;
```

### Creating Custom Templates

```rust
use groggy::algorithms::steps::{StepTemplate, StepComposer};

pub struct MyCustomTemplate;

impl StepTemplate for MyCustomTemplate {
    fn id(&self) -> &str {
        "template.my_custom"
    }
    
    fn description(&self) -> &str {
        "Custom multi-step pattern"
    }
    
    fn generate(&self, params: &HashMap<String, AlgorithmParamValue>) 
        -> Result<Vec<StepSpec>> 
    {
        let mut composer = StepComposer::new();
        
        // Extract parameters
        let input = params.get("input")
            .and_then(|v| match v {
                AlgorithmParamValue::Text(s) => Some(s.as_str()),
                _ => None,
            })
            .unwrap_or("input");
        
        // Build pattern
        let temp = composer.auto_var("temp");
        composer.load_node_attr(input, &temp, AlgorithmParamValue::Float(0.0));
        composer.normalize(&temp, "result", "sum");
        
        Ok(composer.build())
    }
}
```

## Common Validation Errors

### Missing Required Parameter

```
❌ Step 0 (core.normalize_values): Missing required parameter 'source'
   Suggestion: Add 'source' parameter of type Text
```

**Fix**: Add the missing parameter to the step spec.

### Type Mismatch

```
❌ Step 1 (core.add): Input 'left' expects type Float, but variable 'count' has type Int
```

**Fix**: Either change the producer to output Float, or use a conversion step.

### Undefined Variable

```
❌ Step 2 (core.attach_node_attr): Input 'source' references undefined variable 'result'
   Suggestion: Ensure 'result' is defined by a previous step
```

**Fix**: Make sure the variable is produced by an earlier step in the pipeline.

### Constraint Violation

```
❌ Step 0 (core.top_k): Parameter 'k': value -5 is outside valid range [0, 1000]
```

**Fix**: Adjust the parameter value to meet the constraint.

## Schema Registry

### Creating a Registry

```rust
use groggy::algorithms::steps::SchemaRegistry;

let mut registry = SchemaRegistry::new();

// Register schemas
registry.register(my_schema1);
registry.register(my_schema2);
```

### Querying Schemas

```rust
// Check if step has schema
if registry.contains("core.add") {
    // Get schema
    if let Some(schema) = registry.get("core.add") {
        println!("Description: {}", schema.description);
        println!("Cost: {:?}", schema.cost_hint);
    }
}

// Find by tag
let arithmetic_steps = registry.find_by_tag("arithmetic");
for schema in arithmetic_steps {
    println!("- {}: {}", schema.id, schema.description);
}

// Export for documentation
let json = registry.export_json()?;
std::fs::write("step_schemas.json", json)?;
```

## Best Practices

### 1. Validate Early

Always validate pipelines during construction, not at execution time:

```rust
let report = validate_pipeline(&steps);
if !report.is_valid() {
    return Err(anyhow!("Invalid pipeline: {}", report.format()));
}
```

### 2. Use Auto-Variables for Intermediate Results

```rust
let mut composer = StepComposer::new();
let temp1 = composer.auto_var("temp");
let temp2 = composer.auto_var("temp");
// Names are unique, avoiding conflicts
```

### 3. Add Descriptive Constraints

```rust
ParameterSchemaBuilder::new("iterations", ParameterType::Int)
    .description("Number of iterations (higher = more accurate but slower)")
    .constraint(Constraint::Range { min: 1.0, max: 1000.0 })
    .default(serde_json::json!(10))
    .build()
```

### 4. Use Templates for Common Patterns

Instead of rebuilding the same multi-step patterns, create and reuse templates.

### 5. Tag Your Schemas

Tags enable discovery and categorization:

```rust
.tag("arithmetic")
.tag("normalization")
.tag("vectorization")
```

## Integration with Existing Code

The validation system is **opt-in** and **backward compatible**:

- Existing pipelines work without modification
- Validation only runs if schemas are registered
- No performance impact when schemas are not used
- Gradual migration is supported

## Performance Notes

- Schema lookup: O(1)
- Pipeline validation: O(n) steps
- Validation overhead: ~1-5ms for typical pipelines
- No runtime cost after validation

## Troubleshooting

### "Step not registered" Error

Make sure step schemas are registered before validation:
```rust
registry.register(my_schema);
```

### Type Compatibility Issues

Check the type compatibility rules:
- `Float` accepts `Int` (implicit conversion)
- `Any` accepts all types
- `Union` types accept any of their constituent types
- Other type combinations require exact match

### Validation Disabled

If validation isn't running:
1. Check that `get_schema_registry()` returns a registry
2. Ensure schemas are registered for your steps
3. Verify the validator is created and called

## Next Steps

- Read `BUILDER_VALIDATION_COMPLETE.md` for implementation details
- See `tests/builder_validation_integration.rs` for complete examples
- Explore `src/algorithms/steps/composition.rs` for more templates
- Check `src/algorithms/steps/schema.rs` for all available types and constraints
