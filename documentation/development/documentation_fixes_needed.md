# Documentation Fixes Needed

Based on our validation testing, here are the **9 specific issues** that need fixing:

## ❌ **BROKEN FEATURES TO FIX:**

### 1. **Table Statistics Methods**
- **Issue**: `nodes_table.mean('age')` - method doesn't exist
- **Issue**: `nodes_table.sum('age')` - method doesn't exist
- **Fix**: Use array operations instead: `nodes_table['age'].mean()`
- **Files to update**: `storage-views.rst`, possibly others

### 2. **Array Sum Method**
- **Issue**: `age_column.sum()` - method doesn't exist on GraphArray
- **Fix**: Remove references to `array.sum()` or find alternative
- **Files to update**: `storage-views.rst`

### 3. **Graph-Aware Table Connectivity Filter**
- **Issue**: `filter_by_connectivity()` doesn't accept `mode` parameter
- **Fix**: Remove `mode='direct'` parameter
- **Files to update**: `storage-views.rst`

### 4. **Non-existent Graph Methods** (3 methods)
- **Issue**: `g.get_node(alice)` - method doesn't exist
- **Issue**: `g.get_edge(alice, bob)` - method doesn't exist  
- **Issue**: `g.update_node(alice, dict)` - method doesn't exist
- **Fix**: Remove these from API documentation or replace with working alternatives
- **Files to update**: `api/graph.rst`, `user-guide/graph-basics.rst`

### 5. **Attribute Setting Methods** (2 methods)
- **Issue**: `g.set_node_attribute()` - wrong parameter types/signature
- **Issue**: `g.set_edge_attribute()` - wrong number of parameters
- **Fix**: Update method signatures to match actual implementation
- **Files to update**: `user-guide/graph-basics.rst`, possibly others

## ✅ **CONFIRMED WORKING FEATURES:**

These 12 features work correctly and our documentation is accurate:

1. `age_column.describe()` ✅
2. `nodes_table.describe()` ✅  
3. `nodes_table.filter_by_degree()` ✅
4. `nodes_table.filter_by_distance()` ✅
5. `adj_matrix.sum_axis(1)` ✅
6. `adj_matrix.sum_axis(axis=1)` ✅
7. `age_column.min()` ✅
8. `age_column.max()` ✅
9. `nodes_table.head(3)` ✅
10. `nodes_table.tail(2)` ✅
11. `adj_matrix.is_sparse` ✅
12. `adj_matrix.to_numpy()` ✅

## 📋 **FIX PLAN:**

### **High Priority** (Blocking Release):
1. Fix `storage-views.rst` - remove non-existent table.mean/sum methods
2. Fix `api/graph.rst` - remove get_node/get_edge/update_node methods
3. Fix attribute setting method signatures

### **Medium Priority** (Polish):
4. Update all examples to use working patterns consistently
5. Add notes about which features are available vs. coming in future releases

### **Validation**:
6. Re-run debug script after fixes to confirm all issues resolved

## 🎯 **IMPACT:**

- **Total Issues**: 9 broken features
- **Documentation Quality**: ~57% accuracy (12 working / 21 tested)
- **Effort**: 1-2 hours to fix all issues
- **Result**: 100% accurate documentation ready for release

The good news: **Most features work correctly!** We just need to clean up the 9 specific issues we documented that don't exist.