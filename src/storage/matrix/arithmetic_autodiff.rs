//! Composition of owned numerical results with shared saved-input tapes.
use super::*;
impl<T: NumericType> GraphMatrix<T> {
    pub(super) fn finite_tape_values(&self) -> GraphResult<()> {
        let invalid = |message: String| GraphError::InvalidInput(message);
        let finite = if let Some(sparse) = self
            .storage
            .checked_sparse()
            .map_err(|e| invalid(e.to_string()))?
        {
            sparse.values.iter().all(|value| value.is_finite())
        } else {
            self.storage
                .to_vec()
                .map_err(|e| invalid(e.to_string()))?
                .iter()
                .all(|value| value.is_finite())
        };
        if finite {
            Ok(())
        } else {
            Err(invalid(
                "Automatic differentiation requires finite values".into(),
            ))
        }
    }
    pub(super) fn bound_tape(&self) -> GraphResult<(ComputationGraph<T>, NodeId)> {
        if !T::DTYPE.is_float() {
            return Err(GraphError::InvalidInput(
                "Automatic differentiation requires a floating dtype".into(),
            ));
        }
        self.finite_tape_values()?;
        let invalid = |message: &str| GraphError::InvalidInput(message.into());
        let graph = self
            .computation_graph
            .clone()
            .ok_or_else(|| invalid("Missing Gradient input tape"))?;
        let input_id = self
            .current_node_id
            .ok_or_else(|| invalid("Missing Gradient input node"))?;
        let input = graph
            .get_node(input_id)
            .ok_or_else(|| invalid("Missing Gradient input node"))?;
        {
            let input = input
                .lock()
                .map_err(|_| invalid("Gradient input tape lock poisoned"))?;
            let saved = input
                .value
                .as_ref()
                .ok_or_else(|| invalid("Missing Gradient input snapshot"))?;
            let saved = Self::from_storage(saved.clone());
            let same_values = if let (Some(a), Some(b)) = (
                saved
                    .storage
                    .checked_sparse()
                    .map_err(|e| invalid(&e.to_string()))?,
                self.storage
                    .checked_sparse()
                    .map_err(|e| invalid(&e.to_string()))?,
            ) {
                a.shape == b.shape
                    && a.values == b.values
                    && a.col_indices == b.col_indices
                    && a.row_pointers == b.row_pointers
            } else {
                saved
                    .storage
                    .to_vec()
                    .map_err(|e| invalid(&e.to_string()))?
                    == self.storage.to_vec().map_err(|e| invalid(&e.to_string()))?
            };
            if saved.shape() != self.shape() || !same_values {
                return Err(invalid("Gradient input changed since gradient binding; rebind with requires_grad(false).requires_grad(true)"));
            }
        }
        Ok((graph, input_id))
    }
    pub(super) fn track_binary(
        &self,
        other: &Self,
        mut result: Self,
        operation: Operation,
    ) -> GraphResult<Self> {
        if !self.requires_grad && !other.requires_grad {
            return Ok(result);
        }
        let invalid = |message: String| GraphError::InvalidInput(message);
        result.finite_tape_values()?;
        let mut graph = ComputationGraph::new();
        let mut inputs = Vec::with_capacity(2);
        for input in [self, other] {
            input.finite_tape_values()?;
            if input.requires_grad {
                let (tape, id) = input.bound_tape()?;
                graph.merge(&tape).map_err(|e| invalid(e.to_string()))?;
                inputs.push(id);
            } else {
                inputs.push(graph.create_leaf(input.storage.clone(), false));
            }
        }
        let id = graph
            .create_operation(operation, inputs, result.storage.shape(), true)
            .map_err(|e| invalid(e.to_string()))?;
        graph
            .get_node(id)
            .ok_or_else(|| invalid("Missing arithmetic output node".into()))?
            .lock()
            .map_err(|_| invalid("Arithmetic tape lock poisoned".into()))?
            .value = Some(result.storage.clone());
        result.requires_grad = true;
        result.current_node_id = Some(id);
        result.computation_graph = Some(graph);
        Ok(result)
    }
}
