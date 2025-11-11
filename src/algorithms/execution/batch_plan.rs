//! Batch execution plan representation.
//!
//! A BatchPlan is a compiled representation of a loop body that can be
//! executed efficiently without per-step overhead.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

/// Slot identifier for register-based execution.
///
/// Slots are temporary storage locations that hold intermediate values
/// during batch execution. They're analogous to CPU registers.
pub type SlotId = usize;

/// A single instruction in a batch execution plan.
///
/// Instructions operate on slots rather than named variables, eliminating
/// the need for variable lookups during execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BatchInstruction {
    /// Load a node property into a slot
    LoadNodeProp { dst: SlotId, var_name: String },

    /// Store a slot value back to a node property
    StoreNodeProp { src: SlotId, var_name: String },

    /// Load a scalar value into a slot (broadcast to all nodes)
    LoadScalar { dst: SlotId, value: f64 },

    /// Element-wise addition: dst = lhs + rhs
    Add {
        dst: SlotId,
        lhs: SlotId,
        rhs: SlotId,
    },

    /// Element-wise subtraction: dst = lhs - rhs
    Sub {
        dst: SlotId,
        lhs: SlotId,
        rhs: SlotId,
    },

    /// Element-wise multiplication: dst = lhs * rhs
    Mul {
        dst: SlotId,
        lhs: SlotId,
        rhs: SlotId,
    },

    /// Element-wise division: dst = lhs / rhs
    Div {
        dst: SlotId,
        lhs: SlotId,
        rhs: SlotId,
    },

    /// Aggregate neighbor values: dst = sum/mean/max/min(neighbors[src])
    NeighborAggregate {
        dst: SlotId,
        src: SlotId,
        operation: AggregateOp,
        direction: Direction,
    },

    /// Compute mode of neighbor values: dst = mode(neighbors[src])
    NeighborMode {
        dst: SlotId,
        src: SlotId,
        tie_break: TieBreak,
        direction: Direction,
    },

    /// Fused neighbor multiply-aggregate: dst = agg(neighbors[src] * multiplier)
    FusedNeighborMulAgg {
        dst: SlotId,
        src: SlotId,
        multiplier: SlotId,
        operation: AggregateOp,
        direction: Direction,
    },

    /// Fused multiply-add: dst = a * b + c
    FusedMADD {
        dst: SlotId,
        a: SlotId,
        b: SlotId,
        c: SlotId,
    },

    /// Fused axpy: dst = alpha * x + y (where alpha is scalar)
    FusedAXPY {
        dst: SlotId,
        alpha: SlotId,
        x: SlotId,
        y: SlotId,
    },
}

/// Aggregation operation for neighbor values
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AggregateOp {
    Sum,
    Mean,
    Min,
    Max,
}

/// Edge direction for neighbor operations
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Direction {
    In,
    Out,
    Undirected,
}

/// Tie-breaking strategy for mode operations
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TieBreak {
    Lowest,
    Highest,
    First,
}

/// A compiled batch execution plan.
///
/// Contains all information needed to execute a loop body efficiently:
/// - Instruction sequence (topologically sorted)
/// - Number of slots needed (register count)
/// - Loop-carried variables (phi nodes)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchPlan {
    /// Ordered sequence of instructions to execute
    pub instructions: Vec<BatchInstruction>,

    /// Number of slots (registers) needed
    pub slot_count: usize,

    /// Variables that are carried across loop iterations
    /// Format: (from_slot, to_slot) - copy from_slot to to_slot at iteration end
    pub carried_slots: Vec<(SlotId, SlotId)>,

    /// Optional: name for debugging/telemetry
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl BatchPlan {
    /// Create a new batch plan
    pub fn new(
        instructions: Vec<BatchInstruction>,
        slot_count: usize,
        carried_slots: Vec<(SlotId, SlotId)>,
    ) -> Self {
        Self {
            instructions,
            slot_count,
            carried_slots,
            name: None,
        }
    }

    /// Create a batch plan with a name (for debugging)
    pub fn with_name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }

    /// Validate the plan (check slot indices, detect common issues)
    pub fn validate(&self) -> Result<()> {
        // Check all slot references are within bounds
        for (i, instr) in self.instructions.iter().enumerate() {
            self.validate_instruction(i, instr)?;
        }

        // Check carried slots
        for (from, to) in &self.carried_slots {
            if *from >= self.slot_count {
                return Err(anyhow!(
                    "Carried slot from={} exceeds slot_count={}",
                    from,
                    self.slot_count
                ));
            }
            if *to >= self.slot_count {
                return Err(anyhow!(
                    "Carried slot to={} exceeds slot_count={}",
                    to,
                    self.slot_count
                ));
            }
        }

        Ok(())
    }

    fn validate_instruction(&self, index: usize, instr: &BatchInstruction) -> Result<()> {
        let check_slot = |slot: SlotId, name: &str| -> Result<()> {
            if slot >= self.slot_count {
                Err(anyhow!(
                    "Instruction {}: {} slot {} exceeds slot_count {}",
                    index,
                    name,
                    slot,
                    self.slot_count
                ))
            } else {
                Ok(())
            }
        };

        match instr {
            BatchInstruction::LoadNodeProp { dst, .. } => check_slot(*dst, "dst"),
            BatchInstruction::StoreNodeProp { src, .. } => check_slot(*src, "src"),
            BatchInstruction::LoadScalar { dst, .. } => check_slot(*dst, "dst"),
            BatchInstruction::Add { dst, lhs, rhs }
            | BatchInstruction::Sub { dst, lhs, rhs }
            | BatchInstruction::Mul { dst, lhs, rhs }
            | BatchInstruction::Div { dst, lhs, rhs } => {
                check_slot(*dst, "dst")?;
                check_slot(*lhs, "lhs")?;
                check_slot(*rhs, "rhs")
            }
            BatchInstruction::NeighborAggregate { dst, src, .. }
            | BatchInstruction::NeighborMode { dst, src, .. } => {
                check_slot(*dst, "dst")?;
                check_slot(*src, "src")
            }
            BatchInstruction::FusedNeighborMulAgg {
                dst,
                src,
                multiplier,
                ..
            } => {
                check_slot(*dst, "dst")?;
                check_slot(*src, "src")?;
                check_slot(*multiplier, "multiplier")
            }
            BatchInstruction::FusedMADD { dst, a, b, c } => {
                check_slot(*dst, "dst")?;
                check_slot(*a, "a")?;
                check_slot(*b, "b")?;
                check_slot(*c, "c")
            }
            BatchInstruction::FusedAXPY { dst, alpha, x, y } => {
                check_slot(*dst, "dst")?;
                check_slot(*alpha, "alpha")?;
                check_slot(*x, "x")?;
                check_slot(*y, "y")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_plan_validation() {
        // Valid plan
        let plan = BatchPlan::new(
            vec![
                BatchInstruction::LoadNodeProp {
                    dst: 0,
                    var_name: "ranks".to_string(),
                },
                BatchInstruction::LoadScalar {
                    dst: 1,
                    value: 0.85,
                },
                BatchInstruction::Mul {
                    dst: 2,
                    lhs: 0,
                    rhs: 1,
                },
                BatchInstruction::StoreNodeProp {
                    src: 2,
                    var_name: "ranks".to_string(),
                },
            ],
            3,            // 3 slots needed
            vec![(2, 0)], // carry slot 2 to slot 0
        );

        assert!(plan.validate().is_ok());
    }

    #[test]
    fn test_batch_plan_invalid_slot() {
        // Invalid: references slot 3 but only 3 slots (0-2) allocated
        let plan = BatchPlan::new(
            vec![BatchInstruction::LoadNodeProp {
                dst: 3,
                var_name: "ranks".to_string(),
            }],
            3,
            vec![],
        );

        assert!(plan.validate().is_err());
    }

    #[test]
    fn test_serialization() {
        let plan = BatchPlan::new(
            vec![
                BatchInstruction::LoadNodeProp {
                    dst: 0,
                    var_name: "ranks".to_string(),
                },
                BatchInstruction::Add {
                    dst: 1,
                    lhs: 0,
                    rhs: 0,
                },
            ],
            2,
            vec![],
        )
        .with_name("test_plan".to_string());

        let json = serde_json::to_string(&plan).unwrap();
        let deserialized: BatchPlan = serde_json::from_str(&json).unwrap();

        assert_eq!(plan.instructions.len(), deserialized.instructions.len());
        assert_eq!(plan.slot_count, deserialized.slot_count);
        assert_eq!(plan.name, deserialized.name);
    }
}
