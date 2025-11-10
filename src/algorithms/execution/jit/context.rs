//! JIT execution context and memory management

use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use anyhow::Result;

/// JIT execution context
pub struct JitContext {
    /// Cranelift module for code generation
    pub module: JITModule,
    
    /// Code generator context
    pub ctx: codegen::Context,
}

impl JitContext {
    /// Create a new JIT context
    pub fn new() -> Result<Self> {
        // Create JIT builder with native target
        let builder = JITBuilder::new(cranelift_module::default_libcall_names())?;
        
        // Set up module
        let mut module = JITModule::new(builder);
        let ctx = module.make_context();
        
        Ok(Self {
            module,
            ctx,
        })
    }
    
    /// Clear context for next function
    pub fn clear(&mut self) {
        self.ctx.clear();
    }
}
