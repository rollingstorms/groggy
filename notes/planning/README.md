# Planning Documentation

## 📋 **Active Implementation Plan**

**Primary Document**: [UNIFIED_IMPLEMENTATION_PLAN.md](./UNIFIED_IMPLEMENTATION_PLAN.md)

This is the **single source of truth** for SubgraphOperations & HierarchicalOperations implementation.

### **What's in the Unified Plan:**
- **Complete method audit**: 70 methods across 2 traits
- **Implementation strategy**: Core Rust + FFI traits approach  
- **File-by-file breakdown**: Exact locations and method signatures
- **6-week timeline**: Detailed week-by-week deliverables
- **Success metrics**: Clear completion criteria

---

## 📁 **Archived Planning Documents**

The following documents have been **consolidated into the unified plan**:

- `archived/integrated_traits_hierarchy_plan.md` - Early traits + hierarchy planning
- `archived/methods_audit.md` - Detailed method availability analysis  
- `archived/shared_traits_migration_plan.md` - Initial trait migration approach
- `archived/week2_enhancement_plan.md` - Week 2 enhancement analysis

**These are kept for historical reference but should NOT be used for implementation.**

---

## 🎯 **Implementation Status**

**Current Phase**: Week 1 - Core SubgraphOperations Implementation
**Next Steps**: Implement missing structural metrics in `/src/core/subgraph.rs`

**Key Principle**: 
- ✅ **Preserve existing optimizations** (connected_components, bfs, dfs - 4.5x faster than NetworkX)
- ✅ **Only implement missing functionality** (~35 new methods)
- ✅ **Use thin FFI wrappers** that delegate to core implementations

Refer to the [Unified Implementation Plan](./UNIFIED_IMPLEMENTATION_PLAN.md) for all implementation details.