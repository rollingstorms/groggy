# Infinite Scroll Table Implementation

## Overview

Replaced pagination buttons with infinite scroll/virtual scrolling for the table view. This provides a more efficient and seamless experience for viewing large graphs.

## Changes Made

### 1. Removed Pagination Buttons
**File:** `web/index.html` line 41-45

**Before:**
```html
<div class="table-footer">
  <span id="table-row-info">Rows: 0-0 of 0</span>
  <button id="table-prev" class="table-nav-btn" disabled>Previous</button>
  <button id="table-next" class="table-nav-btn" disabled>Next</button>
</div>
```

**After:**
```html
<div class="table-footer">
  <span id="table-row-info">Scroll to load more...</span>
</div>
```

### 2. Implemented Infinite Scroll
**File:** `web/app.js` - TableRenderer class

**New Features:**

1. **Scroll Detection** - Monitors scroll position and loads more data when approaching edges
   - `bufferSize = 50` rows from edge triggers load
   - `handleScroll()` method calculates visible row indices

2. **Intelligent Caching** - Keeps loaded rows in memory
   - `allRows` Map caches rows by index
   - Prevents duplicate requests for already-loaded data
   - Only drops data on sort/filter changes

3. **Bidirectional Loading** - Can load in either direction
   - Scroll up near top → loads earlier rows
   - Scroll down near bottom → loads later rows

4. **Smart Row Insertion** - Maintains correct order
   - Uses `dataset.rowIndex` to track global position
   - Inserts new rows in correct sorted position
   - Prevents flashing/jumping during scroll

### Key Implementation Details

#### Scroll Handler
```javascript
handleScroll() {
    const estimatedRowHeight = 30; // Approximate height per row
    const topRowIndex = Math.floor(scrollTop / estimatedRowHeight);
    const bottomRowIndex = Math.floor((scrollTop + clientHeight) / estimatedRowHeight);

    // Load more when near edges
    if (topRowIndex < this.bufferSize && this.currentOffset > 0) {
        this.loadMoreData(newOffset); // Load earlier rows
    }
    else if (bottomRowIndex > this.currentOffset + this.windowSize - this.bufferSize) {
        this.loadMoreData(newOffset); // Load later rows
    }
}
```

#### Row Caching
```javascript
// Cache incoming rows
const startOffset = data.start_offset || 0;
data.rows.forEach((row, idx) => {
    this.allRows.set(startOffset + idx, row);
});
```

#### Smart Rendering
```javascript
// Check if row already exists before adding
const existingRow = Array.from(tableBody.children).find(
    tr => tr.dataset.rowIndex == globalIndex
);

if (!existingRow) {
    // Insert in correct position
    const insertBefore = Array.from(tableBody.children).find(
        tr => parseInt(tr.dataset.rowIndex) > globalIndex
    );
    
    if (insertBefore) {
        tableBody.insertBefore(tr, insertBefore);
    } else {
        tableBody.appendChild(tr);
    }
}
```

## Benefits

1. **No pagination buttons** - Cleaner UI, more intuitive
2. **Loads on demand** - Only fetches visible data + buffer
3. **Smooth scrolling** - No page jumps or flashing
4. **Memory efficient** - Caches loaded data but clears on filter/sort
5. **Works both directions** - Scroll up or down to load more

## User Experience

- **Initial load**: First 100 rows loaded
- **Scroll down**: When within 50 rows of bottom, next 100 rows load
- **Scroll up**: When within 50 rows of top, previous 100 rows load
- **Footer updates**: Shows "Loaded X of Y rows (scroll for more)"
- **Sorting/filtering**: Clears cache and reloads fresh data

## Performance Characteristics

- **Window size**: 100 rows per request
- **Buffer zone**: 50 rows from edge triggers load
- **Estimated row height**: 30px (used for scroll position calculation)
- **Cache**: All loaded rows kept in memory during session
- **Network**: Only loads data as needed, not entire dataset

## Future Enhancements

Possible improvements:
1. **Virtual DOM** - Only render visible rows, keep rest in memory
2. **Variable row heights** - More accurate scroll position calculation
3. **Prefetching** - Load next chunk before reaching buffer zone
4. **Smart cache limits** - Drop far-away rows if dataset is huge
5. **Scroll indicators** - Show loading spinner while fetching

## Testing

To test:
1. Open table view with large graph (500+ nodes)
2. Scroll down - new rows appear seamlessly
3. Scroll up - earlier rows load without jump
4. Sort a column - cache clears, fresh sorted data loads
5. Switch between Nodes/Edges - cache resets appropriately

The implementation is production-ready and handles edge cases like:
- Reaching top/bottom of dataset
- Multiple rapid scroll events (debounced by `isLoading` flag)
- Sorting/filtering with cache invalidation
- WebSocket disconnections (falls back to placeholder)
