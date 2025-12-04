//! BatchPlan → native code compiler

#![allow(unused_mut, dead_code)]

use super::context::JitContext;
use super::CompiledFunction;
use crate::algorithms::execution::batch_plan::{BatchInstruction, BatchPlan};
use anyhow::{anyhow, Result};
use cranelift::prelude::*;
use cranelift_module::{Linkage, Module};

/// Compilation context for building a single function
struct CompilerContext<'a> {
    builder: FunctionBuilder<'a>,
    pointer_type: Type,
    node_count: Value,
    /// Pointers to slot arrays (Vec<Vec<f64>>)
    slot_ptrs: Vec<Value>,
}

impl<'a> CompilerContext<'a> {
    /// Compile arithmetic operation: dst[i] = lhs[i] op rhs[i]
    fn compile_arithmetic_op(
        &mut self,
        dst: usize,
        lhs: usize,
        rhs: usize,
        node_idx: Value,
        op: impl Fn(&mut FunctionBuilder, Value, Value) -> Value,
    ) {
        // Load lhs[node_idx]
        let lhs_ptr = self.slot_ptrs[lhs];
        let node_offset = self.builder.ins().imul_imm(node_idx, 8); // f64 = 8 bytes
        let lhs_elem_addr = self.builder.ins().iadd(lhs_ptr, node_offset);
        let lhs_val = self
            .builder
            .ins()
            .load(types::F64, MemFlags::new(), lhs_elem_addr, 0);

        // Load rhs[node_idx]
        let rhs_ptr = self.slot_ptrs[rhs];
        let rhs_elem_addr = self.builder.ins().iadd(rhs_ptr, node_offset);
        let rhs_val = self
            .builder
            .ins()
            .load(types::F64, MemFlags::new(), rhs_elem_addr, 0);

        // Perform operation
        let result = op(&mut self.builder, lhs_val, rhs_val);

        // Store to dst[node_idx]
        let dst_ptr = self.slot_ptrs[dst];
        let dst_elem_addr = self.builder.ins().iadd(dst_ptr, node_offset);
        self.builder
            .ins()
            .store(MemFlags::new(), result, dst_elem_addr, 0);
    }

    /// Compile scalar broadcast: dst[i] = value
    fn compile_load_scalar(&mut self, dst: usize, value: f64, node_idx: Value) {
        let val = self.builder.ins().f64const(value);
        let dst_ptr = self.slot_ptrs[dst];
        let node_offset = self.builder.ins().imul_imm(node_idx, 8);
        let dst_elem_addr = self.builder.ins().iadd(dst_ptr, node_offset);
        self.builder
            .ins()
            .store(MemFlags::new(), val, dst_elem_addr, 0);
    }

    /// Compile fused multiply-add: dst[i] = a[i] * b[i] + c[i]
    fn compile_fused_madd(&mut self, dst: usize, a: usize, b: usize, c: usize, node_idx: Value) {
        let node_offset = self.builder.ins().imul_imm(node_idx, 8);

        // Load a[node_idx]
        let a_ptr = self.slot_ptrs[a];
        let a_elem_addr = self.builder.ins().iadd(a_ptr, node_offset);
        let a_val = self
            .builder
            .ins()
            .load(types::F64, MemFlags::new(), a_elem_addr, 0);

        // Load b[node_idx]
        let b_ptr = self.slot_ptrs[b];
        let b_elem_addr = self.builder.ins().iadd(b_ptr, node_offset);
        let b_val = self
            .builder
            .ins()
            .load(types::F64, MemFlags::new(), b_elem_addr, 0);

        // Load c[node_idx]
        let c_ptr = self.slot_ptrs[c];
        let c_elem_addr = self.builder.ins().iadd(c_ptr, node_offset);
        let c_val = self
            .builder
            .ins()
            .load(types::F64, MemFlags::new(), c_elem_addr, 0);

        // Compute a * b + c (use fma if available, otherwise separate ops)
        let result = self.builder.ins().fma(a_val, b_val, c_val);

        // Store to dst[node_idx]
        let dst_ptr = self.slot_ptrs[dst];
        let dst_elem_addr = self.builder.ins().iadd(dst_ptr, node_offset);
        self.builder
            .ins()
            .store(MemFlags::new(), result, dst_elem_addr, 0);
    }

    /// Copy one slot to another: to[i] = from[i]
    fn compile_copy_slot(&mut self, from: usize, to: usize, node_idx: Value) {
        let node_offset = self.builder.ins().imul_imm(node_idx, 8);

        let from_ptr = self.slot_ptrs[from];
        let from_elem_addr = self.builder.ins().iadd(from_ptr, node_offset);
        let val = self
            .builder
            .ins()
            .load(types::F64, MemFlags::new(), from_elem_addr, 0);

        let to_ptr = self.slot_ptrs[to];
        let to_elem_addr = self.builder.ins().iadd(to_ptr, node_offset);
        self.builder
            .ins()
            .store(MemFlags::new(), val, to_elem_addr, 0);
    }
}

/// Compile a batch plan to native machine code
pub fn compile_batch_plan(jit: &mut JitContext, plan: &BatchPlan) -> Result<CompiledFunction> {
    // Validate plan first
    plan.validate()?;

    // Clear previous function
    jit.clear();

    // Define function signature: (node_count: usize, iterations: usize, slot_ptrs: *const *mut f64) -> i32
    let pointer_type = jit.module.target_config().pointer_type();
    let mut sig = jit.module.make_signature();
    sig.params.push(AbiParam::new(pointer_type)); // node_count
    sig.params.push(AbiParam::new(pointer_type)); // iterations
    sig.params.push(AbiParam::new(pointer_type)); // slot_ptrs (array of pointers)
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
    let slot_ptrs_base = builder.block_params(entry_block)[2];

    // Load all slot pointers upfront
    let mut slot_ptrs = Vec::new();
    for slot_idx in 0..plan.slot_count {
        let offset = (slot_idx * std::mem::size_of::<usize>()) as i64;
        let slot_ptr_addr = builder.ins().iadd_imm(slot_ptrs_base, offset);
        let slot_ptr = builder
            .ins()
            .load(pointer_type, MemFlags::new(), slot_ptr_addr, 0);
        slot_ptrs.push(slot_ptr);
    }

    // Create iteration loop
    let iter_block = builder.create_block();
    let iter_body_block = builder.create_block();
    let iter_done_block = builder.create_block();

    // Add phi variable for iteration counter
    builder.append_block_param(iter_block, pointer_type);

    // Jump to iteration loop
    let zero = builder.ins().iconst(pointer_type, 0);
    builder.ins().jump(iter_block, &[zero]);

    // Iteration loop header
    builder.switch_to_block(iter_block);
    let iter_var = builder.block_params(iter_block)[0];
    let cond = builder
        .ins()
        .icmp(IntCC::UnsignedLessThan, iter_var, iterations);
    builder
        .ins()
        .brif(cond, iter_body_block, &[], iter_done_block, &[]);

    // Iteration body
    builder.switch_to_block(iter_body_block);

    // Create node loop
    let node_block = builder.create_block();
    let node_body_block = builder.create_block();
    let node_done_block = builder.create_block();

    builder.append_block_param(node_block, pointer_type);
    builder.ins().jump(node_block, &[zero]);

    // Node loop header
    builder.switch_to_block(node_block);
    let node_var = builder.block_params(node_block)[0];
    let node_cond = builder
        .ins()
        .icmp(IntCC::UnsignedLessThan, node_var, node_count);
    builder
        .ins()
        .brif(node_cond, node_body_block, &[], node_done_block, &[]);

    // Node body - execute all instructions for this node
    builder.switch_to_block(node_body_block);

    let mut ctx = CompilerContext {
        builder,
        pointer_type,
        node_count,
        slot_ptrs,
    };

    // Compile each instruction
    for instr in &plan.instructions {
        compile_instruction(&mut ctx, instr, node_var)?;
    }

    // Increment node counter and loop
    let one = ctx.builder.ins().iconst(pointer_type, 1);
    let next_node = ctx.builder.ins().iadd(node_var, one);
    ctx.builder.ins().jump(node_block, &[next_node]);

    // After all nodes processed - handle carried slots
    ctx.builder.switch_to_block(node_done_block);

    // Copy carried slots (only if not last iteration)
    if !plan.carried_slots.is_empty() {
        let is_last_iter = ctx.builder.ins().iadd_imm(iter_var, 1);
        let is_last = ctx
            .builder
            .ins()
            .icmp(IntCC::Equal, is_last_iter, iterations);

        let copy_block = ctx.builder.create_block();
        let skip_copy_block = ctx.builder.create_block();

        ctx.builder
            .ins()
            .brif(is_last, skip_copy_block, &[], copy_block, &[]);

        // Copy block - copy all carried slots
        ctx.builder.switch_to_block(copy_block);

        // Create copy loop for each carried slot
        for (from_slot, to_slot) in &plan.carried_slots {
            let copy_node_block = ctx.builder.create_block();
            let copy_node_body_block = ctx.builder.create_block();
            let copy_node_done_block = ctx.builder.create_block();

            ctx.builder
                .append_block_param(copy_node_block, pointer_type);
            ctx.builder.ins().jump(copy_node_block, &[zero]);

            ctx.builder.switch_to_block(copy_node_block);
            let copy_node_var = ctx.builder.block_params(copy_node_block)[0];
            let copy_cond =
                ctx.builder
                    .ins()
                    .icmp(IntCC::UnsignedLessThan, copy_node_var, node_count);
            ctx.builder.ins().brif(
                copy_cond,
                copy_node_body_block,
                &[],
                copy_node_done_block,
                &[],
            );

            ctx.builder.switch_to_block(copy_node_body_block);
            ctx.compile_copy_slot(*from_slot, *to_slot, copy_node_var);
            let next_copy_node = ctx.builder.ins().iadd(copy_node_var, one);
            ctx.builder.ins().jump(copy_node_block, &[next_copy_node]);

            ctx.builder.switch_to_block(copy_node_done_block);
        }

        ctx.builder.ins().jump(skip_copy_block, &[]);
        ctx.builder.switch_to_block(skip_copy_block);
    }

    // Increment iteration counter and loop back
    let next_iter = ctx.builder.ins().iadd(iter_var, one);
    ctx.builder.ins().jump(iter_block, &[next_iter]);

    // Seal blocks
    ctx.builder.seal_block(iter_block);
    ctx.builder.seal_block(iter_body_block);
    ctx.builder.seal_block(node_block);
    ctx.builder.seal_block(node_body_block);
    ctx.builder.seal_block(node_done_block);

    // Done - return success
    ctx.builder.switch_to_block(iter_done_block);
    let success = ctx.builder.ins().iconst(types::I32, 0);
    ctx.builder.ins().return_(&[success]);

    ctx.builder.seal_block(iter_done_block);

    // Finalize function
    ctx.builder.finalize();

    // Compile to machine code
    jit.module.define_function(func_id, &mut jit.ctx)?;
    jit.module.clear_context(&mut jit.ctx);
    jit.module.finalize_definitions()?;

    // Get function pointer
    let code_ptr = jit.module.get_finalized_function(func_id);

    #[allow(clippy::missing_transmute_annotations)]
    Ok(unsafe { std::mem::transmute(code_ptr) })
}

/// Compile a single instruction
fn compile_instruction(
    ctx: &mut CompilerContext,
    instr: &BatchInstruction,
    node_idx: Value,
) -> Result<()> {
    match instr {
        BatchInstruction::LoadScalar { dst, value } => {
            ctx.compile_load_scalar(*dst, *value, node_idx);
        }
        BatchInstruction::Add { dst, lhs, rhs } => {
            ctx.compile_arithmetic_op(*dst, *lhs, *rhs, node_idx, |builder, a, b| {
                builder.ins().fadd(a, b)
            });
        }
        BatchInstruction::Sub { dst, lhs, rhs } => {
            ctx.compile_arithmetic_op(*dst, *lhs, *rhs, node_idx, |builder, a, b| {
                builder.ins().fsub(a, b)
            });
        }
        BatchInstruction::Mul { dst, lhs, rhs } => {
            ctx.compile_arithmetic_op(*dst, *lhs, *rhs, node_idx, |builder, a, b| {
                builder.ins().fmul(a, b)
            });
        }
        BatchInstruction::Div { dst, lhs, rhs } => {
            ctx.compile_arithmetic_op(*dst, *lhs, *rhs, node_idx, |builder, a, b| {
                builder.ins().fdiv(a, b)
            });
        }
        BatchInstruction::FusedMADD { dst, a, b, c } => {
            ctx.compile_fused_madd(*dst, *a, *b, *c, node_idx);
        }
        BatchInstruction::FusedAXPY { dst, alpha, x, y } => {
            // AXPY: dst = alpha * x + y
            ctx.compile_fused_madd(*dst, *alpha, *x, *y, node_idx);
        }
        // TODO: These need StepScope access - will be handled in Phase 2
        BatchInstruction::LoadNodeProp { .. }
        | BatchInstruction::StoreNodeProp { .. }
        | BatchInstruction::NeighborAggregate { .. }
        | BatchInstruction::NeighborMode { .. }
        | BatchInstruction::FusedNeighborMulAgg { .. } => {
            return Err(anyhow!(
                "Instruction {:?} requires StepScope access (not yet supported in JIT)",
                instr
            ));
        }
    }
    Ok(())
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use crate::algorithms::execution::batch_plan::{BatchInstruction, BatchPlan};

    /// Helper to create slot arrays for testing
    fn create_slots(count: usize, node_count: usize) -> (Vec<Vec<f64>>, Vec<*mut f64>) {
        let mut slots = vec![vec![0.0; node_count]; count];
        let mut slot_ptrs: Vec<*mut f64> = slots.iter_mut().map(|v| v.as_mut_ptr()).collect();
        (slots, slot_ptrs)
    }

    #[test]
    fn test_compile_empty_plan() {
        let mut jit = JitContext::new().unwrap();
        let plan = BatchPlan::new(vec![], 0, vec![]);

        let result = compile_batch_plan(&mut jit, &plan);
        assert!(result.is_ok());

        // Call with no slots
        let func = result.unwrap();
        let ret = unsafe { func(0, 1, std::ptr::null()) };
        assert_eq!(ret, 0);
    }

    #[test]
    fn test_compile_load_scalar() {
        let mut jit = JitContext::new().unwrap();
        let plan = BatchPlan::new(
            vec![BatchInstruction::LoadScalar {
                dst: 0,
                value: 42.0,
            }],
            1, // 1 slot
            vec![],
        );

        let func = compile_batch_plan(&mut jit, &plan).unwrap();

        // Create slots
        let node_count = 10;
        let (mut slots, slot_ptrs) = create_slots(1, node_count);

        // Execute
        let ret = unsafe { func(node_count, 1, slot_ptrs.as_ptr()) };
        assert_eq!(ret, 0);

        // Check all values are 42.0
        for &val in &slots[0] {
            assert_eq!(val, 42.0);
        }
    }

    #[test]
    fn test_compile_add() {
        let mut jit = JitContext::new().unwrap();
        let plan = BatchPlan::new(
            vec![
                BatchInstruction::LoadScalar {
                    dst: 0,
                    value: 10.0,
                },
                BatchInstruction::LoadScalar {
                    dst: 1,
                    value: 32.0,
                },
                BatchInstruction::Add {
                    dst: 2,
                    lhs: 0,
                    rhs: 1,
                },
            ],
            3, // 3 slots
            vec![],
        );

        let func = compile_batch_plan(&mut jit, &plan).unwrap();

        // Create slots
        let node_count = 5;
        let (mut slots, slot_ptrs) = create_slots(3, node_count);

        // Execute
        let ret = unsafe { func(node_count, 1, slot_ptrs.as_ptr()) };
        assert_eq!(ret, 0);

        // Check result: slot[2] should be 10 + 32 = 42
        for &val in &slots[2] {
            assert_eq!(val, 42.0);
        }
    }

    #[test]
    fn test_compile_arithmetic_ops() {
        let mut jit = JitContext::new().unwrap();
        let plan = BatchPlan::new(
            vec![
                BatchInstruction::LoadScalar {
                    dst: 0,
                    value: 20.0,
                },
                BatchInstruction::LoadScalar { dst: 1, value: 5.0 },
                BatchInstruction::Sub {
                    dst: 2,
                    lhs: 0,
                    rhs: 1,
                }, // 20 - 5 = 15
                BatchInstruction::Mul {
                    dst: 3,
                    lhs: 2,
                    rhs: 1,
                }, // 15 * 5 = 75
                BatchInstruction::Div {
                    dst: 4,
                    lhs: 3,
                    rhs: 0,
                }, // 75 / 20 = 3.75
            ],
            5,
            vec![],
        );

        let func = compile_batch_plan(&mut jit, &plan).unwrap();

        let node_count = 3;
        let (mut slots, slot_ptrs) = create_slots(5, node_count);

        let ret = unsafe { func(node_count, 1, slot_ptrs.as_ptr()) };
        assert_eq!(ret, 0);

        // Check results
        for &val in &slots[2] {
            assert_eq!(val, 15.0);
        }
        for &val in &slots[3] {
            assert_eq!(val, 75.0);
        }
        for &val in &slots[4] {
            assert_eq!(val, 3.75);
        }
    }

    #[test]
    fn test_compile_fused_madd() {
        let mut jit = JitContext::new().unwrap();
        let plan = BatchPlan::new(
            vec![
                BatchInstruction::LoadScalar { dst: 0, value: 2.0 },
                BatchInstruction::LoadScalar { dst: 1, value: 3.0 },
                BatchInstruction::LoadScalar { dst: 2, value: 5.0 },
                BatchInstruction::FusedMADD {
                    dst: 3,
                    a: 0,
                    b: 1,
                    c: 2,
                }, // 2 * 3 + 5 = 11
            ],
            4,
            vec![],
        );

        let func = compile_batch_plan(&mut jit, &plan).unwrap();

        let node_count = 4;
        let (mut slots, slot_ptrs) = create_slots(4, node_count);

        let ret = unsafe { func(node_count, 1, slot_ptrs.as_ptr()) };
        assert_eq!(ret, 0);

        for &val in &slots[3] {
            assert_eq!(val, 11.0);
        }
    }

    #[test]
    fn test_compile_multiple_iterations() {
        let mut jit = JitContext::new().unwrap();
        // Plan: increment by 1.0 each iteration
        let plan = BatchPlan::new(
            vec![
                BatchInstruction::LoadScalar { dst: 0, value: 0.0 },
                BatchInstruction::LoadScalar { dst: 1, value: 1.0 },
                BatchInstruction::Add {
                    dst: 0,
                    lhs: 0,
                    rhs: 1,
                },
            ],
            2,
            vec![(0, 0)], // carry slot 0 to itself
        );

        let func = compile_batch_plan(&mut jit, &plan).unwrap();

        let node_count = 3;
        let (mut slots, slot_ptrs) = create_slots(2, node_count);

        // Run 10 iterations
        let ret = unsafe { func(node_count, 10, slot_ptrs.as_ptr()) };
        assert_eq!(ret, 0);

        // After 10 iterations, slot[0] should be 10.0
        for &val in &slots[0] {
            assert_eq!(val, 10.0);
        }
    }

    #[test]
    fn test_compile_carried_slots() {
        let mut jit = JitContext::new().unwrap();
        // PageRank-like pattern: multiply by damping factor
        let plan = BatchPlan::new(
            vec![
                BatchInstruction::LoadScalar {
                    dst: 1,
                    value: 0.85,
                },
                BatchInstruction::Mul {
                    dst: 2,
                    lhs: 0,
                    rhs: 1,
                },
            ],
            3,
            vec![(2, 0)], // carry result back
        );

        let func = compile_batch_plan(&mut jit, &plan).unwrap();

        let node_count = 2;
        let (mut slots, slot_ptrs) = create_slots(3, node_count);

        // Initialize slot 0 with 1.0
        slots[0].fill(1.0);

        // Run 5 iterations
        let ret = unsafe { func(node_count, 5, slot_ptrs.as_ptr()) };
        assert_eq!(ret, 0);

        // After 5 iterations: 1.0 * 0.85^5 ≈ 0.4437
        let expected = 0.85_f64.powi(5);
        for &val in &slots[0] {
            assert!((val - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_compile_validation_fails_invalid_slot() {
        let mut jit = JitContext::new().unwrap();
        let plan = BatchPlan::new(
            vec![BatchInstruction::LoadScalar {
                dst: 5, // out of bounds
                value: 1.0,
            }],
            3, // only 3 slots
            vec![],
        );

        let result = compile_batch_plan(&mut jit, &plan);
        assert!(result.is_err());
    }
}
