//! Batch execution infrastructure for efficient loop interpretation.
//!
//! This module provides a register-based execution model that eliminates
//! per-step FFI overhead by executing entire loop bodies in native Rust code.
//!
//! ## Architecture
//!
//! 1. **Python side**: Analyzes loop bodies and generates BatchPlans
//!    - Slot allocation (register allocation)
//!    - IR → BatchInstruction lowering
//!    - Lifetime analysis and optimization
//!
//! 2. **Rust side**: Interprets BatchPlans efficiently
//!    - Slot-based execution (no variable lookups)
//!    - Tight interpreter loop
//!    - Integration with existing fused kernels
//!
//! ## Performance Target
//!
//! - Current: LoopStep executes body steps one-by-one (~10-50 FFI calls/iteration)
//! - Target: Single FFI call to enter batch executor
//! - Expected speedup: 10-100× (PageRank: 113-196× → <10× slower than native)

pub mod batch_executor;
pub mod batch_plan;
// JIT compilation disabled temporarily due to Cranelift ARM64 compatibility issues
// pub mod jit; // Tier 2: JIT compilation

pub use batch_executor::BatchExecutor;
pub use batch_plan::{BatchInstruction, BatchPlan, SlotId};
// pub use jit::JitManager;
