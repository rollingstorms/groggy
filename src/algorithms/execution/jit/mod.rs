//! JIT compilation for batch plans (Tier 2)
//!
//! This module provides Just-In-Time compilation of BatchPlans to native machine code
//! using Cranelift. Hot loops are automatically compiled for maximum performance.
//!
//! Architecture:
//! - `compiler`: BatchPlan → native code compilation
//! - `context`: JIT execution context and memory management
//! - `templates`: Pre-optimized kernels for common algorithms

pub mod compiler;
pub mod context;

use crate::algorithms::execution::batch_plan::BatchPlan;
use anyhow::Result;

/// Compiled native function signature
/// For now, just takes basic parameters (we'll add StepScope later)
pub type CompiledFunction = unsafe extern "C" fn(
    node_count: usize,
    iterations: usize,
    scope_ptr: *mut u8, // Opaque pointer for now
) -> i32;

/// JIT compilation manager
pub struct JitManager {
    context: context::JitContext,
}

impl JitManager {
    /// Create a new JIT manager
    pub fn new() -> Result<Self> {
        Ok(Self {
            context: context::JitContext::new()?,
        })
    }

    /// Compile a batch plan to native code
    pub fn compile(&mut self, plan: &BatchPlan) -> Result<CompiledFunction> {
        compiler::compile_batch_plan(&mut self.context, plan)
    }

    /// Check if a plan is JIT-compatible
    pub fn is_compatible(plan: &BatchPlan) -> bool {
        // For now, all batch plans are JIT-compatible
        // Later: check for unsupported instructions
        !plan.instructions.is_empty()
    }
}

impl Default for JitManager {
    fn default() -> Self {
        Self::new().expect("Failed to create JIT manager")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jit_manager_creation() {
        let manager = JitManager::new();
        assert!(manager.is_ok());
    }
}
