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

use crate::algorithms::execution::batch_plan::{BatchInstruction, BatchPlan};
use anyhow::Result;

/// Compiled native function signature
///
/// Parameters:
/// - node_count: Number of nodes to process
/// - iterations: Number of loop iterations
/// - slot_ptrs: Pointer to array of slot pointers (f64 arrays)
///
/// Returns: 0 on success, non-zero on error
pub type CompiledFunction =
    unsafe extern "C" fn(node_count: usize, iterations: usize, slot_ptrs: *const *mut f64) -> i32;

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

    /// Check if JIT compilation is supported on this platform
    pub fn is_platform_supported() -> bool {
        // ARM64 support requires Cranelift 0.107+
        cfg!(not(target_arch = "aarch64"))
    }

    /// Check if a plan is JIT-compatible
    pub fn is_compatible(plan: &BatchPlan) -> bool {
        // Platform check
        if !Self::is_platform_supported() {
            return false;
        }

        // Check for supported instructions
        // Currently: arithmetic and scalar ops only
        // TODO: Add StepScope-dependent operations (Load/StoreNodeProp, NeighborAggregate)
        for instr in &plan.instructions {
            match instr {
                BatchInstruction::LoadScalar { .. }
                | BatchInstruction::Add { .. }
                | BatchInstruction::Sub { .. }
                | BatchInstruction::Mul { .. }
                | BatchInstruction::Div { .. }
                | BatchInstruction::FusedMADD { .. }
                | BatchInstruction::FusedAXPY { .. } => {}
                _ => return false, // Unsupported instruction
            }
        }

        !plan.instructions.is_empty()
    }
}

impl Default for JitManager {
    fn default() -> Self {
        Self::new().expect("Failed to create JIT manager")
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;

    #[test]
    fn test_jit_manager_creation() {
        let manager = JitManager::new();
        assert!(manager.is_ok());
    }
}
