# Dynamic Degree Filter Implementation

## Overview

Changed the degree filter to **dynamically adapt to the actual graph**, ensuring ALL nodes are visible by default regardless of their degree.

## The Problem

Previously, the degree filter had a hardcoded maximum (50 or 500), which meant:
- Hub nodes with high degree were filtered out by default
- Different graphs needed different filter ranges
- No way to know if nodes were missing without checking manually

## The Solution

**Auto-set the degree filter maximum to match the graph's actual maximum degree.**

### Implementation

**File:** `web/app.js` - `computeNodeDegrees()` method

```javascript
computeNodeDegrees() {
    // ... compute degrees for each node ...
    
    // Calculate max degree in the graph
    const maxDegree = Math.max(0, ...Object.values(degrees));
    
    // Update slider to match graph's max degree
    const degreeMaxInput = document.getElementById('degree-max');
    if (degreeMaxInput) {
        const sliderMax = Math.max(10, maxDegree);  // Minimum 10 for usability
        degreeMaxInput.max = sliderMax;
        degreeMaxInput.value = sliderMax;
        
        // Update filter to include all nodes
        this.filters.degreeMax = sliderMax;
    }
}
```

**File:** `web/app.js` - `handleSnapshot()` method (lines 490-520)

Removed the code that read degreeMax from slider before computeNodeDegrees:
```javascript
// OLD: Read slider value before computing degrees
if (degreeMaxInput) {
    const max = parseInt(degreeMaxInput.value, 10);
    this.filters.degreeMax = max;  // Uses old hardcoded value!
}

// NEW: Let computeNodeDegrees set it based on actual graph
// computeNodeDegrees() will set the slider and filter automatically
```

**File:** `web/index.html` - Initial slider values (lines 132-137)

Set reasonable defaults that will be overridden:
```html
<!-- Initial placeholder values -->
<input type="range" id="degree-max" min="0" max="100" value="100" class="slider">
```

These are just placeholders - they'll be replaced when the graph loads.

## How It Works

1. **Graph loads** → `handleSnapshot()` called
2. **Compute degrees** → `computeNodeDegrees()` calculates each node's degree
3. **Find max** → `Math.max(...Object.values(degrees))` finds highest degree
4. **Update slider** → Sets `slider.max` and `slider.value` to max degree
5. **Update filter** → Sets `this.filters.degreeMax` to max degree
6. **Result** → All nodes visible by default!

## Benefits

### 1. **No Hidden Nodes by Default**
Every node in the graph is visible when you first load it. No surprises.

### 2. **Graph-Specific Ranges**
- Small graph with max degree 10 → slider 0-10
- Medium graph with max degree 50 → slider 0-50  
- Large hub graph with max degree 500 → slider 0-500

### 3. **Still Useful for Filtering**
Users can still reduce the slider to focus on low-degree nodes:
- Set max to 10 → only show nodes with ≤10 connections
- Set max to 5 → only show peripheral nodes

### 4. **Performance Safety**
The filter is still available if someone loads a massive graph and wants to hide high-degree hubs for performance.

## Examples

### Example 1: Meta-Graph
```
Nodes: 54
Max degree: 142 (BaseTable with many edges)

Slider auto-sets to: 0-142
Result: All 54 nodes visible ✅
```

### Example 2: Small Test Graph
```
Nodes: 10  
Max degree: 5

Slider auto-sets to: 0-10 (minimum 10 for usability)
Result: All 10 nodes visible ✅
```

### Example 3: Social Network
```
Nodes: 1000
Max degree: 2500 (celebrity hub)

Slider auto-sets to: 0-2500
Result: All 1000 nodes visible ✅
User can reduce to 50 to hide celebrities
```

## Testing

To verify the dynamic behavior:

```python
import groggy as gr

# Load graph
gt = gr.from_csv(...)
g = gt.to_graph()
g.viz.show()
```

**Check in browser:**
1. Open Controls → Filter tab
2. Look at Degree Filter slider
3. Max value should equal the highest degree in your graph
4. All nodes should be visible

**Console check:**
```javascript
// Check max degree
console.log('Max degree:', Math.max(...Object.values(app.nodeDegrees)));

// Check slider
const slider = document.getElementById('degree-max');
console.log('Slider max:', slider.max, 'value:', slider.value);

// Check filter
console.log('Filter degreeMax:', app.filters.degreeMax);

// All should match!
```

## Edge Cases Handled

1. **Empty graph** → Max degree 0, slider 0-10 (minimum for usability)
2. **Single node** → Max degree 0, slider 0-10
3. **Disconnected graph** → Max degree based on largest component
4. **Graph updates** → computeNodeDegrees() called on updates, slider adjusts
5. **User manually changes slider** → Works as expected, overrides default

## Philosophy

This change aligns with the principle that **a general-purpose graph library should show the complete graph by default**. Filtering should be opt-in, not required to see your data.

### Before:
- ❌ Some nodes hidden by default
- ❌ Need to know about degree filter
- ❌ Need to manually adjust range
- ❌ Different graphs need different settings

### After:
- ✅ All nodes visible by default
- ✅ No configuration required
- ✅ Auto-adapts to each graph
- ✅ Filter still available when needed
