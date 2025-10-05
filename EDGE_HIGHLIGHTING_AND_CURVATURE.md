# Edge Highlighting and Curvature Multiplier

## Overview

Implemented two visualization enhancements:
1. **Highlight edges connected to selected node** - Makes it easy to see a node's connections
2. **Curvature multiplier** - Global control to scale all edge curvatures uniformly

---

## Feature 1: Edge Highlighting on Node Selection

### Behavior

When a node is selected:
- ✨ **Connected edges** → Highlighted with slightly thicker lines and blue color
- 🌫️ **Non-connected edges** → Dimmed to 20% opacity
- 🔴 **Selected edge** → Red (unchanged)
- 🟠 **Hovered edge** → Orange (unchanged)

### Implementation

**File:** `web/app.js` lines 1487-1518

```javascript
// Check if edge is connected to selected node
const isConnectedToSelected = this.selectedNode && 
    (edge.source === this.selectedNode.id || edge.target === this.selectedNode.id);

if (isConnectedToSelected) {
    // Highlight connected edges
    ctx.strokeStyle = edge.color || '#4a90e2'; // Blue default
    ctx.lineWidth = (edge.width || 1) * 1.5 / this.camera.zoom; // 50% thicker
    ctx.globalAlpha = this.opacity * 0.9; // Nearly full opacity
} else if (this.selectedNode) {
    // Dim non-connected edges
    ctx.strokeStyle = edge.color || '#cccccc';
    ctx.lineWidth = (edge.width || 1) / this.camera.zoom;
    ctx.globalAlpha = this.opacity * 0.2; // Very dim
}
```

### User Experience

**Before:**
- Select node → Only node highlighted
- Hard to see which edges connect to it
- All edges same prominence

**After:**
- Select node → Node highlighted + connected edges highlighted
- Non-connected edges fade into background
- Easy to trace node's connections

### Example

For BaseTable with 142 edges:
1. Click BaseTable node
2. 142 connected edges become bright blue and slightly thicker
3. All other edges dim to 20% opacity
4. Clear visual of BaseTable's connectivity!

---

## Feature 2: Curvature Multiplier

### Problem

Previously, the curvature was **additive**:
```javascript
// OLD: Addition
curvature = curvatureMultiplier + edgeCurvature
```

This meant:
- Slider at 0.25 → adds 0.25 to all edges
- Edge with curvature 1.0 → becomes 1.25
- Edge with curvature 0.5 → becomes 0.75
- Doesn't scale uniformly!

### Solution

Changed to **multiplicative**:
```javascript
// NEW: Multiplication
curvature = edgeCurvature * curvatureMultiplier
```

Now:
- Slider at 1.0 → no change (identity)
- Slider at 2.0 → doubles all curvatures
- Slider at 0.5 → halves all curvatures
- Slider at 0.0 → straight lines
- Slider at 3.0 → triple curvature

### Implementation

**File:** `web/app.js` lines 1534-1537 and 1050-1053

Changed from additive to multiplicative:
```javascript
// Before
const curvature = this.curvatureMultiplier + edgeCurvature;

// After
const curvature = edgeCurvature * this.curvatureMultiplier;
```

**File:** `web/app.js` line 32

Changed default from 0.25 to 1.0:
```javascript
this.curvatureMultiplier = 1.0;  // Default multiplier (1.0 = no change)
```

**File:** `web/index.html` - Slider control

Updated range and default:
```html
<!-- Before -->
<label>Edge Curvature:</label>
<input type="range" id="curvature-slider" min="-2" max="2" step="0.1" value="0.25">

<!-- After -->
<label>Edge Curvature Multiplier:</label>
<input type="range" id="curvature-slider" min="0" max="3" step="0.1" value="1.0">
```

### Behavior

| Slider Value | Effect |
|--------------|--------|
| 0.0 | All edges straight (no curvature) |
| 0.5 | Half curvature (gentler curves) |
| 1.0 | **Default** - original curvature |
| 1.5 | 50% more curvature |
| 2.0 | Double curvature (stronger curves) |
| 3.0 | Triple curvature (very curved) |

### Benefits

1. **Intuitive control** - 1.0 means "as designed", scale up/down from there
2. **Preserves relationships** - Relative curvatures stay proportional
3. **No negative values** - 0-3 range is more intuitive than -2 to +2
4. **Better for multi-edges** - Scales all parallel edges uniformly

### Use Cases

**Reduce clutter:**
- Set to 0.0 → All edges straight, easiest to see
- Set to 0.3 → Slight curves, minimal clutter

**Emphasize structure:**
- Set to 1.5 → More pronounced curves
- Set to 2.0+ → Very curved, good for seeing multi-edges

**Dense graphs with many parallel edges:**
- Set to 2-3 → Curves help distinguish parallel edges

---

## Combined Effect

When both features work together:

1. **Select a hub node** (e.g., BaseTable)
2. **Connected edges highlighted** → See all 142 connections
3. **Adjust curvature slider** → Scale curves to reduce visual clutter
4. **Result** → Clear view of node's connectivity with controllable edge curvature

---

## Testing

### Test Edge Highlighting

```python
import groggy as gr

gt = gr.from_csv(...)
g = gt.to_graph()
g.viz.show()
```

In browser:
1. Click any node
2. Observe: Connected edges bright, others dimmed
3. Click another node
4. Observe: Highlighting updates to new node
5. Click empty space
6. Observe: All edges return to normal

### Test Curvature Multiplier

1. Open Controls → Style tab
2. Find "Edge Curvature Multiplier" slider
3. Set to 0.0 → All edges straight
4. Set to 1.0 → Original curvatures
5. Set to 2.0 → Double curvature
6. Observe: All edges scale proportionally

### Test Combined

1. Select BaseTable node (high degree)
2. Observe: Many edges highlighted
3. Adjust curvature to 0.5
4. Observe: Easier to see with gentler curves
5. Adjust curvature to 2.0
6. Observe: More pronounced separation between edges

---

## Technical Details

### Edge States Priority

1. **Selected edge** → Red, thick (highest priority)
2. **Hovered edge** → Orange, medium
3. **Connected to selected node** → Blue/highlighted, slightly thick
4. **Selected node exists** → Dimmed (non-connected edges)
5. **Default** → Normal color and width

### Performance

- No significant performance impact
- Highlighting is pure CSS-style logic (no extra calculations)
- Curvature multiplication is same cost as addition

### Accessibility

- Clear visual hierarchy
- High contrast between highlighted and dimmed
- Works with all graph sizes
- Scales with zoom level

---

## Future Enhancements

Possible improvements:
1. **Directional highlighting** - Different colors for incoming vs outgoing edges
2. **Hop distance** - Highlight 2-hop or 3-hop neighbors
3. **Edge bundling** - Bundle similar edges together
4. **Animated selection** - Smooth transition when selecting nodes
5. **Curvature presets** - Quick buttons for 0, 0.5, 1.0, 2.0
