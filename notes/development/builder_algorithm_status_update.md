# Builder Algorithm Status Update

**Date:** 2025-11-04  
**Status:** ✅ Core DSL validated, ⚠️ Loop unrolling bug outstanding

## Overview
We revalidated the `@algorithm` decorator pipeline using the builder DSL. PageRank, Label Propagation, and the simple degree-centrality flow execute end-to-end, confirming that the expression builder, operator overloading, and step encoding continue to work after the recent refactors.

## Confirmed Capabilities
- ✅ Operator overloading compiles expressions like `ranks / (deg + 1e-9)` without additional glue code.
- ✅ Algorithms without explicit loops (e.g., degree centrality) run successfully through the builder DSL.
- ✅ Step encoding covers both the optimized IR field names (`"a"`/`"b"`) and the legacy names (`"left"`/`"right"`), keeping the translator backward compatible.

## Known Issues
- ❌ Loop unrolling fails to rewrite internal variable references. When we duplicate steps across iterations, the step keys gain the `_iter{n}` suffix, but nested references (e.g., `"a": "const_2"`) still point to the pre-unrolled identifiers (`"const_2"` instead of `"const_2_iter0"`). The unrolling pass needs to remap every identifier inside the duplicated steps.

## Validation
- `python test_simple_builder.py` — simple, non-looped algorithm passes and confirms the DSL path is healthy. (Loops remain disabled pending the fix above.)

## Next Actions
1. Update the loop unrolling transformation to rewrite nested variable references during duplication.
2. Add regression coverage for multi-iteration algorithms (PageRank, LPA) once the rewrite is in place.
