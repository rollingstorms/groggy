//! Batch execution infrastructure for efficient loop interpretation.
//!
//! This module provides a register-based execution model that eliminates
//! per-step FFI overhead by executing entire loop bodies in native Rust code.
//!
//! Numeric plans lower to the same checked step kernels used by ordinary native
//! execution. The private register frame retains typed values and shares staging,
//! cancellation and error boundaries. Unported instruction families retain their
//! non-atomic compatibility interpreter and are not certified for durable runs.
//!
//! Both routes execute within Rust. A serialized register plan is not itself a
//! promise of acceleration; see the checked batch benchmark for measured costs.

pub mod batch_executor;
pub mod batch_plan;
// JIT compilation disabled temporarily due to Cranelift ARM64 compatibility issues
// pub mod jit; // Tier 2: JIT compilation

pub use batch_executor::BatchExecutor;
pub use batch_plan::{BatchInstruction, BatchPlan, SlotId};
// pub use jit::JitManager;
