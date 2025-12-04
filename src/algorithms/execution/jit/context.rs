//! JIT execution context and memory management

use anyhow::Result;
use cranelift::prelude::*;
use cranelift_jit::JITModule;

#[cfg(not(target_arch = "aarch64"))]
use cranelift_jit::JITBuilder;
#[cfg(not(target_arch = "aarch64"))]
use cranelift_module::Module;

/// JIT execution context
pub struct JitContext {
    /// Cranelift module for code generation
    pub module: JITModule,

    /// Code generator context
    pub ctx: codegen::Context,
}

impl JitContext {
    /// Create a new JIT context
    ///
    /// # Platform Support
    /// - ✅ x86_64: Full support
    /// - ⚠️  aarch64 (ARM64): Limited support in Cranelift 0.102
    ///   - Known issue: PLT generation not implemented for ARM64
    ///   - Workaround: Requires Cranelift 0.107+ or custom patching
    ///
    /// For ARM64 systems, this will currently fail with:
    /// "PLT is currently only supported on x86_64"
    pub fn new() -> Result<Self> {
        #[cfg(not(target_arch = "aarch64"))]
        {
            // x86_64 and other architectures - standard path
            let builder = JITBuilder::new(cranelift_module::default_libcall_names())?;
            let module = JITModule::new(builder);
            let ctx = module.make_context();
            Ok(Self { module, ctx })
        }

        #[cfg(target_arch = "aarch64")]
        {
            // ARM64: Cranelift 0.102 has incomplete ARM64 JIT support
            // This is a known limitation - PLT stubs not implemented
            Err(anyhow::anyhow!(
                "JIT compilation is not yet supported on aarch64 (ARM64) with Cranelift 0.102.\n\
                 This is a known Cranelift limitation. Workarounds:\n\
                 1. Use x86_64 for JIT development\n\
                 2. Upgrade to Cranelift 0.107+ (requires updating dependencies)\n\
                 3. Use BatchExecutor (Tier 1) which works on all platforms"
            ))
        }
    }

    /// Clear context for next function
    pub fn clear(&mut self) {
        self.ctx.clear();
    }
}
