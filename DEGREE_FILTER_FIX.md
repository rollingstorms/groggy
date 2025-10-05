# Degree Filter Hiding High-Degree Nodes Fix

## Problem

Nodes like "BaseTable" appeared in the table view but were invisible in the graph view, even though they existed with valid positions.

## Root Cause

The degree filter slider had a **default maximum of 50**, which filtered out any nodes with degree > 50.

**BaseTable has degree 142** (142 edges connected to it, including 20 self-loops), so it was being hidden by the filter!

### How the Filter Works

From `web/app.js` lines 1262-1269:
```javascript
this.nodes.forEach(node => {
    const degree = degrees[node.id] ?? 0;
    if (degree < this.filters.degreeMin) {
        return; // Node filtered out
    }
    if (degree > this.filters.degreeMax) {
        return; // Node filtered out - THIS WAS HAPPENING TO BASETABLE
    }
    visible.push(node.id);
});
```

### Why BaseTable Was Hidden

1. **BaseTable degree**: 142 edges
2. **Default degree-max slider**: value="50", max="50"
3. **Initial filter**: `this.filters.degreeMax = 50`
4. **Result**: 142 > 50 → filtered out!

## The Fix

**File:** `web/index.html` lines 132-137

Increased the degree filter range from 0-50 to 0-500:

```html
<!-- BEFORE: Max degree of 50 -->
<input type="range" id="degree-max" min="0" max="50" value="50" class="slider">

<!-- AFTER: Max degree of 500 -->
<input type="range" id="degree-max" min="0" max="500" value="500" class="slider">
```

## Why This Makes Sense

Real-world graphs often have high-degree hub nodes:

- **Meta-graph**: BaseTable with 142 edges (many methods)
- **Social networks**: Celebrities with thousands of connections  
- **Citation graphs**: Seminal papers cited hundreds of times
- **Web graphs**: Popular pages with many links

A max degree of 50 was too restrictive for real datasets.

## Benefits

1. **All nodes visible by default** - No surprising hidden nodes
2. **Filter still useful** - Can still reduce clutter by filtering high-degree nodes
3. **Handles hub nodes** - Graphs with hubs (like BaseTable) work correctly
4. **Better UX** - Users expect to see all nodes unless they explicitly filter

## Testing

To verify the fix:

```python
import groggy as gr

gt = gr.from_csv(
    nodes_filepath='comprehensive_test_objects.csv',
    edges_filepath='comprehensive_test_methods.csv',
    node_id_column='object_name',
    source_id_column='object_name',
    target_id_column='result_type'
)
g = gt.to_graph()
g.viz.show(layout='circular')
```

**Expected behavior:**
- ✅ BaseTable visible in graph view (degree 142)
- ✅ All 54 nodes visible by default
- ✅ Degree filter slider goes from 0 to 500
- ✅ Can still filter to lower degrees if needed

## Alternative Solutions Considered

1. **Remove filter entirely** - But users might want to filter high-degree nodes
2. **Dynamic max based on graph** - More complex, requires computing max degree
3. **Start with no upper limit** - Could be confusing when filtering
4. **Set to 500** - Simple, covers most graphs ✅ (chosen)

## Console Debug Commands

To check node degrees in browser console:
```javascript
// Show node degrees
app.nodes.forEach(n => {
    const degree = app.nodeDegrees[n.id] || 0;
    if (degree > 50) {
        console.log(`${n.label} (${n.id}): degree ${degree}`);
    }
});

// Check current filter
console.log('Degree filter:', app.filters.degreeMin, '-', app.filters.degreeMax);

// Check visible nodes
console.log('Visible nodes:', app.visibleNodes.length, 'of', app.nodes.length);
```

## Related Issues

This also fixes:
- Hub nodes disappearing mysteriously
- "Node count mismatch" between table and graph
- Autofit not centering properly (missing high-degree nodes)
- Search finding nodes that don't appear on canvas
