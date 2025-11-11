//! BatchPlan → native code compiler

use super::context::JitContext;
use super::CompiledFunction;
use crate::algorithms::execution::batch_plan::{BatchInstruction, BatchPlan};
use anyhow::{anyhow, Result};
use cranelift::prelude::*;
use cranelift_module::{Linkage, Module};

/// Compile a batch plan to native machine code
pub fn compile_batch_plan(jit: &mut JitContext, plan: &BatchPlan) -> Result<CompiledFunction> {
    // Clear previous function
    jit.clear();

    // Define function signature: (node_count: usize, iterations: usize, scope: *mut StepScope) -> i32
    let pointer_type = jit.module.target_config().pointer_type();
    let mut sig = jit.module.make_signature();
    sig.params.push(AbiParam::new(pointer_type)); // node_count
    sig.params.push(AbiParam::new(pointer_type)); // iterations
    sig.params.push(AbiParam::new(pointer_type)); // scope_ptr
    sig.returns.push(AbiParam::new(types::I32)); // return code (0 = success)

    // Create function
    let func_id = jit
        .module
        .declare_function("batch_execute", Linkage::Export, &sig)?;

    // Build function body
    jit.ctx.func.signature = sig;
    let mut builder_context = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut jit.ctx.func, &mut builder_context);

    // Entry block
    let entry_block = builder.create_block();
    builder.append_block_params_for_function_params(entry_block);
    builder.switch_to_block(entry_block);
    builder.seal_block(entry_block);

    // Get function parameters
    let node_count = builder.block_params(entry_block)[0];
    let iterations = builder.block_params(entry_block)[1];
    let scope_ptr = builder.block_params(entry_block)[2];

    // TODO: For now, just return 0 (success)
    // Next step: actually compile the instructions!
    let zero = builder.ins().iconst(types::I32, 0);
    builder.ins().return_(&[zero]);

    // Finalize function
    builder.finalize();

    // Compile to machine code
    jit.module.define_function(func_id, &mut jit.ctx)?;
    jit.module.clear_context(&mut jit.ctx);
    jit.module.finalize_definitions()?;

    // Get function pointer
    let code_ptr = jit.module.get_finalized_function(func_id);

    Ok(unsafe { std::mem::transmute(code_ptr) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::execution::batch_plan::BatchPlan;

    #[test]
    fn test_compile_empty_plan() {
        let mut jit = JitContext::new().unwrap();
        let plan = BatchPlan {
            name: "test_plan".to_string(),
            instructions: vec![],
            slot_count: 0,
            carried_slots: vec![],
        };

        let result = compile_batch_plan(&mut jit, &plan);
        assert!(result.is_ok());

        // Try calling the compiled function (should just return 0)
        let func = result.unwrap();
        let ret = unsafe { func(100, 10, std::ptr::null_mut()) };
        assert_eq!(ret, 0);
    }
}
