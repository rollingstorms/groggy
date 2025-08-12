# Groggy Query Engine - Quick Reference Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [API Cheat Sheet](#api-cheat-sheet)
3. [Common Patterns](#common-patterns)
4. [Performance Tips](#performance-tips)
5. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Python Setup
```python
import groggy

# Create graph and add data
graph = groggy.Graph()
graph.add_node(0)
graph.set_node_attribute(0, "name", groggy.AttrValue("Alice"))
graph.set_node_attribute(0, "age", groggy.AttrValue(30))
```

### Basic Querying
```python
# Filter nodes by attribute
name_filter = groggy.NodeFilter.attribute_equals("name", groggy.AttrValue("Alice"))
matching_nodes = graph.filter_nodes(name_filter)

# Basic statistics
avg_age = graph.aggregate_node_attribute("age", "average")
print(f"Average age: {avg_age.as_float()}")
```

---

## API Cheat Sheet

### Node Filtering
```python
# Has attribute
groggy.NodeFilter.has_attribute("name")

# Attribute equals value
groggy.NodeFilter.attribute_equals("age", groggy.AttrValue(25))

# Logical combinations
groggy.NodeFilter.and_filters([filter1, filter2])
groggy.NodeFilter.or_filters([filter1, filter2])
groggy.NodeFilter.not_filter(filter1)
```

### Advanced Attribute Filtering
```python
# Create attribute filters
age_filter = groggy.AttributeFilter.between(
    groggy.AttrValue(18), 
    groggy.AttrValue(65)
)

# Use in node filter
node_filter = groggy.NodeFilter.attribute_filter("age", age_filter)
```

### Graph Traversal
```python
# BFS traversal
result = graph.traverse_bfs(
    start_node=0,
    max_depth=3,
    node_filter=None,  # Optional filtering
    edge_filter=None   # Optional filtering
)
print(f"Found {len(result.nodes)} nodes via {result.algorithm}")

# DFS traversal
result = graph.traverse_dfs(start_node=0, max_depth=2)

# Connected components
components = graph.connected_components()
```

### Aggregation Operations
```python
# Basic statistics
count = graph.aggregate_node_attribute("any_attr", "count")
sum_val = graph.aggregate_node_attribute("numeric_attr", "sum") 
avg = graph.aggregate_node_attribute("numeric_attr", "average")
min_val = graph.aggregate_node_attribute("numeric_attr", "min")
max_val = graph.aggregate_node_attribute("numeric_attr", "max")

# Advanced statistics
stddev = graph.aggregate_node_attribute("numeric_attr", "stddev")
median = graph.aggregate_node_attribute("numeric_attr", "median")
p95 = graph.aggregate_node_attribute("numeric_attr", "percentile_95")

# Unique count
unique_count = graph.aggregate_node_attribute("category_attr", "unique_count")

# Grouping
grouped = graph.group_nodes_by_attribute(
    group_by_attr="department",
    aggregate_attr="salary",
    operation="average"
)
```

### AttrValue Creation
```python
# Different value types
groggy.AttrValue(42)                    # Integer
groggy.AttrValue(3.14)                  # Float
groggy.AttrValue("text")                # String
groggy.AttrValue(True)                  # Boolean
groggy.AttrValue([1.0, 2.0, 3.0])       # Float vector
groggy.AttrValue(b"binary_data")        # Bytes
```

---

## Common Patterns

### 1. Multi-Criteria Filtering
```python
# Find adults in specific location
adult_filter = groggy.AttributeFilter.greater_than(groggy.AttrValue(18))
location_filter = groggy.AttributeFilter.equals(groggy.AttrValue("NYC"))

complex_filter = groggy.NodeFilter.and_filters([
    groggy.NodeFilter.attribute_filter("age", adult_filter),
    groggy.NodeFilter.attribute_filter("location", location_filter)
])

adults_in_nyc = graph.filter_nodes(complex_filter)
```

### 2. Range Queries
```python
# Find nodes with numeric values in range
range_filter = groggy.AttributeFilter.between(
    groggy.AttrValue(1000),  # min
    groggy.AttrValue(5000)   # max
)
node_filter = groggy.NodeFilter.attribute_filter("salary", range_filter)
result = graph.filter_nodes(node_filter)
```

### 3. Text Searching
```python
# Find nodes with text containing substring  
contains_filter = groggy.AttributeFilter.contains("engineer")
job_filter = groggy.NodeFilter.attribute_filter("job_title", contains_filter)
engineers = graph.filter_nodes(job_filter)
```

### 4. Statistical Analysis
```python
# Get comprehensive statistics for an attribute
def analyze_attribute(graph, attr_name):
    results = {
        'count': graph.aggregate_node_attribute(attr_name, "count").as_int(),
        'average': graph.aggregate_node_attribute(attr_name, "average").as_float(),
        'min': graph.aggregate_node_attribute(attr_name, "min").as_float(),
        'max': graph.aggregate_node_attribute(attr_name, "max").as_float(),
        'stddev': graph.aggregate_node_attribute(attr_name, "stddev").as_float(),
        'median': graph.aggregate_node_attribute(attr_name, "median").as_float(),
        'p25': graph.aggregate_node_attribute(attr_name, "percentile_25").as_float(),
        'p75': graph.aggregate_node_attribute(attr_name, "percentile_75").as_float(),
        'p95': graph.aggregate_node_attribute(attr_name, "percentile_95").as_float(),
    }
    return results

age_stats = analyze_attribute(graph, "age")
print(f"Age statistics: {age_stats}")
```

### 5. Neighborhood Analysis
```python
# Find all neighbors of specific nodes and analyze them
def analyze_neighborhood(graph, center_node, max_depth=2):
    # Get neighborhood via BFS
    traversal = graph.traverse_bfs(
        start_node=center_node,
        max_depth=max_depth
    )
    
    print(f"Neighborhood of node {center_node}:")
    print(f"  - {len(traversal.nodes)} nodes within {max_depth} steps")
    print(f"  - {len(traversal.edges)} edges traversed")
    
    # Could add more analysis of the neighborhood nodes
    return traversal.nodes

neighborhood = analyze_neighborhood(graph, 0)
```

### 6. Category Grouping
```python
# Group by category and get statistics for each group
def group_analysis(graph, group_attr, value_attr, operation="average"):
    grouped = graph.group_nodes_by_attribute(
        group_by_attr=group_attr,
        aggregate_attr=value_attr,
        operation=operation
    )
    
    print(f"{operation.title()} of {value_attr} by {group_attr}:")
    for category, result in grouped.items():
        value = result.as_float() if operation in ["average", "sum", "min", "max"] else result.as_int()
        print(f"  {category}: {value}")
    
    return grouped

# Example: Average salary by department
salary_by_dept = group_analysis(graph, "department", "salary", "average")
```

### 7. Connected Component Analysis
```python
# Analyze graph connectivity
def connectivity_analysis(graph):
    components = graph.connected_components()
    
    print(f"Graph connectivity analysis:")
    print(f"  - {len(components)} connected components")
    
    component_sizes = [len(component) for component in components]
    print(f"  - Largest component: {max(component_sizes)} nodes")
    print(f"  - Smallest component: {min(component_sizes)} nodes")
    print(f"  - Average component size: {sum(component_sizes) / len(component_sizes):.1f}")
    
    return components

components = connectivity_analysis(graph)
```

---

## Performance Tips

### 1. Filter Ordering
```python
# Put most selective filters first in AND combinations
# GOOD: Specific equality first, then range
fast_filter = groggy.NodeFilter.and_filters([
    groggy.NodeFilter.attribute_equals("type", groggy.AttrValue("person")),  # Very selective
    groggy.NodeFilter.attribute_filter("age", age_range_filter)              # Less selective
])

# AVOID: Range first, then equality  
slow_filter = groggy.NodeFilter.and_filters([
    groggy.NodeFilter.attribute_filter("age", age_range_filter),             # Less selective first
    groggy.NodeFilter.attribute_equals("type", groggy.AttrValue("person"))   # Very selective second
])
```

### 2. Use Specific Operations
```python
# GOOD: Use has_attribute for existence checks
has_email = groggy.NodeFilter.has_attribute("email")

# AVOID: Using not_equals with null for existence (not supported)
```

### 3. Limit Traversal Depth
```python
# GOOD: Limit depth for large graphs
result = graph.traverse_bfs(start_node=0, max_depth=3)  # Reasonable limit

# AVOID: Unlimited depth on large graphs
# result = graph.traverse_bfs(start_node=0, max_depth=100)  # Too deep
```

### 4. Batch Operations
```python
# GOOD: Use complex filters instead of multiple simple ones
complex_filter = groggy.NodeFilter.and_filters([filter1, filter2, filter3])
result = graph.filter_nodes(complex_filter)

# AVOID: Multiple separate filter calls
# result1 = graph.filter_nodes(filter1)
# result2 = graph.filter_nodes(filter2)  
# result3 = graph.filter_nodes(filter3)
```

### 5. Use Appropriate Data Types
```python
# GOOD: Use integers for IDs and counts
groggy.AttrValue(42)        # Integer for ID

# GOOD: Use floats for measurements
groggy.AttrValue(3.14159)   # Float for precise values

# GOOD: Use compact representations when possible
groggy.AttrValue("short")   # Will use CompactText internally if ≤ 22 chars
```

---

## Troubleshooting

### Common Errors

#### 1. Node/Edge Not Found
```python
# Error: Node 999 not found during filtering operation
# Solution: Check if node exists before filtering
if graph.has_node(999):
    result = graph.traverse_bfs(start_node=999, max_depth=2)
```

#### 2. Type Mismatch in Filtering
```python
# Error: Cannot compare text attribute with numeric filter
# Problem: Using numeric comparison on text attribute
age_filter = groggy.AttributeFilter.greater_than(groggy.AttrValue(18))
name_filter = groggy.NodeFilter.attribute_filter("name", age_filter)  # Wrong!

# Solution: Use appropriate filter for data type
text_filter = groggy.AttributeFilter.equals(groggy.AttrValue("Alice"))
name_filter = groggy.NodeFilter.attribute_filter("name", text_filter)  # Correct
```

#### 3. Empty Results
```python
# Issue: Query returns empty results unexpectedly
# Debug: Check if attributes exist
has_attr = groggy.NodeFilter.has_attribute("problematic_attr")
nodes_with_attr = graph.filter_nodes(has_attr)
print(f"Nodes with attribute: {len(nodes_with_attr)}")

# Debug: Check attribute values
if nodes_with_attr:
    # Inspect first node's attributes manually
    pass
```

#### 4. Performance Issues
```python
# Issue: Query takes too long
# Solution 1: Add depth limits to traversal
result = graph.traverse_bfs(start_node=0, max_depth=3)  # Limit depth

# Solution 2: Use more selective filters first
selective_filter = groggy.NodeFilter.attribute_equals("rare_attr", groggy.AttrValue("value"))
broad_filter = groggy.NodeFilter.has_attribute("common_attr") 
combined = groggy.NodeFilter.and_filters([selective_filter, broad_filter])
```

### Debugging Strategies

#### 1. Check Data First
```python
# Get basic graph statistics
def graph_stats(graph):
    # This would need to be implemented to count nodes/edges
    print("Graph overview:")
    # print(f"  Nodes: {graph.node_count()}")
    # print(f"  Edges: {graph.edge_count()}")
```

#### 2. Test Filters Incrementally
```python
# Start with simple filters, then combine
filter1 = groggy.NodeFilter.has_attribute("type")
result1 = graph.filter_nodes(filter1)
print(f"Filter 1 results: {len(result1)}")

filter2 = groggy.NodeFilter.attribute_equals("type", groggy.AttrValue("person"))
result2 = graph.filter_nodes(filter2)
print(f"Filter 2 results: {len(result2)}")

combined = groggy.NodeFilter.and_filters([filter1, filter2])
result_combined = graph.filter_nodes(combined)
print(f"Combined results: {len(result_combined)}")
```

#### 3. Validate Aggregation Results
```python
# Check if aggregation makes sense
count = graph.aggregate_node_attribute("age", "count").as_int()
avg = graph.aggregate_node_attribute("age", "average").as_float()
min_val = graph.aggregate_node_attribute("age", "min").as_float()
max_val = graph.aggregate_node_attribute("age", "max").as_float()

print(f"Age validation:")
print(f"  Count: {count}")
print(f"  Range: {min_val} to {max_val}")
print(f"  Average: {avg}")
print(f"  Range check: {'✓' if min_val <= avg <= max_val else '✗'}")
```

### Best Practices

1. **Always validate inputs** before complex operations
2. **Use has_attribute** to check existence before value filtering
3. **Start with simple queries** and build complexity gradually
4. **Check intermediate results** when debugging complex filters
5. **Use appropriate data types** for AttrValue creation
6. **Limit traversal depth** for performance on large graphs
7. **Test edge cases** like empty graphs or missing attributes

---

## Examples by Use Case

### Social Network Analysis
```python
# Find influential users (high degree)
# Note: This requires implementing degree calculation
def find_influential_users(graph, min_connections=10):
    # This would need degree calculation support
    pass

# Find communities via connected components
communities = graph.connected_components()
large_communities = [c for c in communities if len(c) >= 5]
```

### Data Science Workflow
```python
# Outlier detection using percentiles
def find_outliers(graph, attr_name, threshold=95):
    p95 = graph.aggregate_node_attribute(attr_name, "percentile_95").as_float()
    outlier_filter = groggy.AttributeFilter.greater_than(groggy.AttrValue(p95))
    node_filter = groggy.NodeFilter.attribute_filter(attr_name, outlier_filter)
    return graph.filter_nodes(node_filter)

outliers = find_outliers(graph, "income", threshold=90)
```

### Business Intelligence
```python
# Customer segmentation
def customer_segments(graph):
    segments = graph.group_nodes_by_attribute(
        group_by_attr="customer_tier",
        aggregate_attr="total_purchases",
        operation="sum"
    )
    
    for tier, total in segments.items():
        print(f"{tier}: ${total.as_float():,.2f} total purchases")
    
    return segments
```

This quick reference provides practical, ready-to-use code patterns for the most common querying scenarios!
