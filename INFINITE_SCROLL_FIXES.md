# Infinite Scroll Bug Fixes

## Issues Fixed

### Issue 1: Nodes button not clearing edges data
**Problem:** After loading edges and clicking the "Nodes" button, the old edge data remained visible in the table.

**Root Cause:** The `switchToNodes()` and `switchToEdges()` methods weren't clearing the existing table body and header before requesting new data.

**Fix:** Added table clearing code to both methods:
```javascript
// Clear table body and header for fresh load
const tableBody = document.getElementById('table-body');
const tableHeader = document.getElementById('table-header').querySelector('tr');
if (tableBody) tableBody.innerHTML = '';
if (tableHeader) tableHeader.innerHTML = '';
```

**File:** `web/app.js` lines 1910-1945

### Issue 2: Graph view missing after switching back from table
**Problem:** When switching back to graph view from table view, the graph appeared blank and autofit couldn't find it.

**Root Cause:** Two issues:
1. The canvas view was shown AFTER resuming, but needed to be visible first
2. Canvas wasn't being redrawn after the display change
3. Canvas might need resizing after being hidden

**Fix:** Reordered operations and added forced redraw:
```javascript
// 1. Show graph view FIRST
const canvasView = document.getElementById('canvas-view');
if (canvasView) canvasView.style.display = 'block';

// 2. Hide table view
const tableView = document.getElementById('table-view');
if (tableView) tableView.style.display = 'none';

// 3. Resume animation
this.app.resume();

// 4. Force canvas resize and redraw
this.app.resizeCanvas();
this.app.render();
```

**File:** `web/app.js` lines 1812-1836

## Why These Fixes Work

### Fix 1: Clear table on switch
- `innerHTML = ''` removes all existing rows and headers
- Prevents old data from lingering when switching between nodes/edges
- Combined with `allRows.clear()` ensures clean state

### Fix 2: Proper view switching order
- **Show canvas first** - Canvas needs to be visible before rendering
- **Resize canvas** - Ensures canvas dimensions are correct after being hidden
- **Force render** - Triggers immediate redraw, doesn't wait for next animation frame
- **Resume last** - Ensures `isPaused` flag is correct

## Testing

To verify fixes:

1. **Test nodes/edges switching:**
   - Load table view
   - Click "Edges" button → edges load
   - Click "Nodes" button → should clear and show nodes (not edges)
   - Repeat several times → should work consistently

2. **Test graph view return:**
   - Start in graph view
   - Switch to table view
   - Switch back to graph view → graph should be visible
   - Click autofit button → should center graph correctly
   - Nodes and edges should be rendered properly

## Additional Notes

The `renderTableData()` method still uses incremental rendering for infinite scroll:
- Only updates header if empty (first load)
- Appends new rows without clearing existing
- This is correct for scrolling within same view

But `switchToNodes()` and `switchToEdges()` now explicitly clear everything because they're switching to a DIFFERENT dataset, not scrolling within the same one.

## Files Modified

1. `web/app.js`:
   - Lines 1910-1945: `switchToNodes()` and `switchToEdges()` with table clearing
   - Lines 1812-1836: `switchToGraph()` with proper ordering and forced redraw
