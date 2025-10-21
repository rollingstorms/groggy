# {OBJECT_NAME} API Reference

## Conceptual Overview

{Brief explanation of what this object represents in Groggy - 150-200 words}

**Key characteristics:**
- {Main feature 1}
- {Main feature 2}
- {Main feature 3}

**When to use {OBJECT_NAME}:**
- {Use case 1}
- {Use case 2}
- {Use case 3}

---

## Architecture & Design

### In Groggy's Three-Tier Architecture

```
Python {OBJECT_NAME} Object (this layer)
         ↓
    FFI Bridge (PyO3)
         ↓
   Rust Core Implementation
    ├─ {Core component 1}
    ├─ {Core component 2}
    └─ {Core component 3}
```

### Design Philosophy

**1. {Design Principle 1}**

{Explanation with code example if relevant}

**2. {Design Principle 2}**

{Explanation}

**3. {Design Principle 3}**

{Explanation}

### Why {OBJECT_NAME} Exists

{Explain the problem this object solves - 200-300 words}

---

## Object Transformations

{OBJECT_NAME} can transform into:

### {OBJECT_NAME} → {TargetType1}
{How and when to use this transformation}

```python
# Example
{code}
```

### {OBJECT_NAME} → {TargetType2}
{How and when to use this transformation}

```python
# Example
{code}
```

### {OBJECT_NAME} → {TargetType3}
{How and when to use this transformation}

```python
# Example
{code}
```

### Reverse Transformations

Other objects can create {OBJECT_NAME}:

```python
# From {SourceType1}
{code}

# From {SourceType2}
{code}
```

---

## Common Patterns

### Pattern 1: {Pattern Name}

{Description of pattern and when to use it}

```python
# Example
{code}
```

**Use case:** {When to use this pattern}

### Pattern 2: {Pattern Name}

{Description}

```python
# Example
{code}
```

**Use case:** {When to use this pattern}

### Pattern 3: {Pattern Name}

{Description}

```python
# Example
{code}
```

**Use case:** {When to use this pattern}

---

## Performance Notes

### Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| {operation1} | O({complexity}) | {explanation} |
| {operation2} | O({complexity}) | {explanation} |
| {operation3} | O({complexity}) | {explanation} |

### Memory

- **Storage**: {memory characteristics}
- **Views**: {view cost}
- **Materialization**: {when data is copied}

### Optimization Tips

1. **{Tip 1 title}**: {explanation}
2. **{Tip 2 title}**: {explanation}
3. **{Tip 3 title}**: {explanation}

---

## Complete Method Reference

::: groggy.{OBJECT_NAME}
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3
      members:
        - {method1}
        - {method2}
        - {method3}
        # ... add all methods here

---

## Next Steps

- **[User Guide: {Related Guide}](../guide/{guide_file}.md)**: Comprehensive tutorial
- **[{Related API 1}](related1.md)**: Related object documentation
- **[{Related API 2}](related2.md)**: Related object documentation
