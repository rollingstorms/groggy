# Expression System for map_nodes

## Design Goals

1. **Serializable** - Can be expressed in JSON/TOML
2. **Type-safe** - Evaluated at runtime with error handling
3. **Accessible** - Easy syntax for common operations
4. **Extensible** - Can add new operations
5. **Fast** - Minimal overhead for simple operations

## Proposed Syntax

### Simple Operations
```json
{
  "id": "core.map_nodes",
  "params": {
    "source": "values",
    "target": "doubled",
    "expr": {
      "op": "mul",
      "left": {"var": "value"},
      "right": {"const": 2.0}
    }
  }
}
```

### Accessing Attributes
```json
{
  "expr": {
    "op": "add",
    "left": {"attr": "degree"},
    "right": {"var": "score"}
  }
}
```

### Function Calls
```json
{
  "expr": {
    "fn": "sqrt",
    "arg": {"var": "value"}
  }
}
```

### Conditionals
```json
{
  "expr": {
    "if": {"op": "gt", "left": {"var": "x"}, "right": {"const": 0}},
    "then": {"var": "x"},
    "else": {"const": 0}
  }
}
```

## Implementation Plan

### Phase 1: Core Expression Type
```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Expr {
    // Literals
    Const(AlgorithmParamValue),
    
    // Variable references
    Var(String),              // From step variables
    Attr(String),             // From node attributes
    
    // Binary operations
    BinaryOp {
        op: BinaryOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    
    // Unary operations
    UnaryOp {
        op: UnaryOp,
        arg: Box<Expr>,
    },
    
    // Function calls
    Call {
        func: String,
        args: Vec<Expr>,
    },
    
    // Conditional
    If {
        condition: Box<Expr>,
        then_expr: Box<Expr>,
        else_expr: Box<Expr>,
    },
}
```

### Phase 2: Evaluation Context
```rust
pub struct ExprContext<'a> {
    node: NodeId,
    input: &'a StepInput<'a>,
    current_value: Option<&'a AlgorithmParamValue>,
}

impl Expr {
    pub fn eval(&self, ctx: &ExprContext) -> Result<AlgorithmParamValue> {
        match self {
            Expr::Const(v) => Ok(v.clone()),
            Expr::Var(name) => ctx.get_var(name),
            Expr::Attr(name) => ctx.get_attr(name),
            // ... implement operations
        }
    }
}
```

### Phase 3: Simplified JSON Syntax

For common cases, support simplified string syntax:
```json
{
  "expr": "value * 2.0"
}
```

Parse into Expr tree. This requires a mini parser but covers 80% of use cases.

## Operations to Support

### Binary Ops
- Arithmetic: `add`, `sub`, `mul`, `div`, `mod`, `pow`
- Comparison: `eq`, `ne`, `lt`, `le`, `gt`, `ge`
- Logical: `and`, `or`

### Unary Ops
- Arithmetic: `neg`, `abs`
- Logical: `not`
- Math: `sqrt`, `log`, `exp`, `floor`, `ceil`, `round`

### Functions
- Math: `min(a, b)`, `max(a, b)`, `clamp(x, min, max)`
- Aggregation: `sum(list)`, `mean(list)`, `count(list)`
- String: `concat(a, b)`, `format(...)`

### Special
- `neighbor_values(attr)` - Get attribute from all neighbors
- `neighbor_count()` - Degree
- `has_attr(name)` - Check if attribute exists

## Example Use Cases

### Double all values
```json
{"expr": "value * 2"}
```

### Compute weighted score
```json
{
  "expr": {
    "op": "mul",
    "left": {"attr": "importance"},
    "right": {"var": "centrality"}
  }
}
```

### Normalize by degree
```json
{
  "expr": {
    "op": "div",
    "left": {"var": "score"},
    "right": {"fn": "max", "args": [{"call": "neighbor_count"}, {"const": 1}]}
  }
}
```

### Conditional scoring
```json
{
  "expr": {
    "if": {"op": "gt", "left": {"attr": "degree"}, "right": {"const": 10}},
    "then": {"op": "mul", "left": {"var": "value"}, "right": {"const": 2.0}},
    "else": {"var": "value"}
  }
}
```

## Implementation Files

1. `src/algorithms/steps/expression.rs` - Core Expr type and evaluation
2. `src/algorithms/steps/expression/parser.rs` - Parse string syntax
3. Update `transformations.rs` - Add MapNodesExprStep
4. Update `registry.rs` - Register with Expr parameter

## Migration Path

1. Keep existing `MapNodesStep` for Rust API
2. Add new `MapNodesExprStep` with expression system
3. Register `core.map_nodes` to use expressions
4. Update docs with examples

## Performance Considerations

- Cache parsed expressions
- Inline simple operations (const folding)
- JIT compilation for hot paths (future optimization)
- Benchmark target: <100ns overhead per node

## Security

- No arbitrary code execution
- Limited operation set
- No file/network access
- Stack depth limits for recursion
