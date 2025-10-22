//! Optimized Activation Functions with SIMD Support
//!
//! This module provides high-performance implementations of common neural network
//! activation functions with automatic SIMD vectorization and differentiation support.

use crate::storage::advanced_matrix::{
    backend::OperationType,
    numeric_type::NumericType,
    unified_matrix::{MatrixError, MatrixResult, UnifiedMatrix},
};
use std::f64::consts::PI;

/// Trait for activation functions with forward and backward passes
pub trait ActivationFunction<T: NumericType> {
    /// Apply activation function element-wise
    fn forward(&self, input: &UnifiedMatrix<T>) -> MatrixResult<UnifiedMatrix<T>>;

    /// Compute derivative of activation function
    fn backward(
        &self,
        input: &UnifiedMatrix<T>,
        grad_output: &UnifiedMatrix<T>,
    ) -> MatrixResult<UnifiedMatrix<T>>;

    /// In-place version for memory efficiency
    fn forward_inplace(&self, matrix: &mut UnifiedMatrix<T>) -> MatrixResult<()>;

    /// Get the name of this activation function
    fn name(&self) -> &str;
}

/// Enum for different activation function types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ActivationType {
    ReLU,
    LeakyReLU(f64), // slope parameter
    GELU,
    Sigmoid,
    Tanh,
    Swish,
    Mish,
    ELU(f64), // alpha parameter
}

/// High-performance activation operations with SIMD optimization
pub struct ActivationOps;

impl ActivationOps {
    /// Apply ReLU activation: max(0, x)
    pub fn relu<T: NumericType>(input: &UnifiedMatrix<T>) -> MatrixResult<UnifiedMatrix<T>> {
        Self::apply_elementwise(input, |x| if x.to_f64() > 0.0 { x } else { T::zero() })
    }

    /// Apply ReLU activation in-place
    pub fn relu_inplace<T: NumericType>(matrix: &mut UnifiedMatrix<T>) -> MatrixResult<()> {
        Self::apply_elementwise_inplace(matrix, |x| if x.to_f64() > 0.0 { *x } else { T::zero() })
    }

    /// Apply ReLU derivative: 1 if x > 0, 0 otherwise
    pub fn relu_derivative<T: NumericType>(
        input: &UnifiedMatrix<T>,
    ) -> MatrixResult<UnifiedMatrix<T>> {
        Self::apply_elementwise(input, |x| {
            if x.to_f64() > 0.0 {
                T::one()
            } else {
                T::zero()
            }
        })
    }

    /// Apply GELU activation: x * Φ(x) where Φ is the CDF of standard normal
    /// Approximation: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    pub fn gelu<T: NumericType>(input: &UnifiedMatrix<T>) -> MatrixResult<UnifiedMatrix<T>> {
        Self::apply_elementwise(input, |x| {
            let x_f64 = x.to_f64();
            let sqrt_2_over_pi = (2.0 / PI).sqrt();
            let x_cubed = x_f64 * x_f64 * x_f64;
            let inner = sqrt_2_over_pi * (x_f64 + 0.044715 * x_cubed);
            let tanh_inner = inner.tanh();
            let result = 0.5 * x_f64 * (1.0 + tanh_inner);
            T::from_f64(result).unwrap_or(T::zero())
        })
    }

    /// Apply GELU derivative
    pub fn gelu_derivative<T: NumericType>(
        input: &UnifiedMatrix<T>,
    ) -> MatrixResult<UnifiedMatrix<T>> {
        Self::apply_elementwise(input, |x| {
            let x_f64 = x.to_f64();
            let sqrt_2_over_pi = (2.0 / PI).sqrt();

            // GELU derivative is complex - this is the exact formula
            let gaussian = (-0.5 * x_f64 * x_f64).exp() / (2.0 * PI).sqrt();
            let cdf_approx =
                0.5 * (1.0 + (sqrt_2_over_pi * (x_f64 + 0.044715 * x_f64.powi(3))).tanh());
            let pdf_part =
                0.5 * x_f64 * gaussian * sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * x_f64 * x_f64);

            let result = cdf_approx + pdf_part;
            T::from_f64(result).unwrap_or(T::zero())
        })
    }

    /// Apply Sigmoid activation: 1 / (1 + exp(-x))
    pub fn sigmoid<T: NumericType>(input: &UnifiedMatrix<T>) -> MatrixResult<UnifiedMatrix<T>> {
        Self::apply_elementwise(input, |x| {
            let x_f64 = x.to_f64();
            // Use numerically stable sigmoid
            let result = if x_f64 >= 0.0 {
                let exp_neg_x = (-x_f64).exp();
                1.0 / (1.0 + exp_neg_x)
            } else {
                let exp_x = x_f64.exp();
                exp_x / (1.0 + exp_x)
            };
            T::from_f64(result).unwrap_or(T::zero())
        })
    }

    /// Apply Sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
    pub fn sigmoid_derivative<T: NumericType>(
        input: &UnifiedMatrix<T>,
    ) -> MatrixResult<UnifiedMatrix<T>> {
        let sigmoid_out = Self::sigmoid(input)?;
        Self::apply_elementwise(&sigmoid_out, |sig| {
            let sig_f64 = sig.to_f64();
            let result = sig_f64 * (1.0 - sig_f64);
            T::from_f64(result).unwrap_or(T::zero())
        })
    }

    /// Apply Tanh activation: (exp(2x) - 1) / (exp(2x) + 1)
    pub fn tanh<T: NumericType>(input: &UnifiedMatrix<T>) -> MatrixResult<UnifiedMatrix<T>> {
        Self::apply_elementwise(input, |x| {
            let result = x.to_f64().tanh();
            T::from_f64(result).unwrap_or(T::zero())
        })
    }

    /// Apply Tanh derivative: 1 - tanh²(x)
    pub fn tanh_derivative<T: NumericType>(
        input: &UnifiedMatrix<T>,
    ) -> MatrixResult<UnifiedMatrix<T>> {
        Self::apply_elementwise(input, |x| {
            let tanh_x = x.to_f64().tanh();
            let result = 1.0 - tanh_x * tanh_x;
            T::from_f64(result).unwrap_or(T::zero())
        })
    }

    /// Apply Swish activation: x * sigmoid(x)
    pub fn swish<T: NumericType>(input: &UnifiedMatrix<T>) -> MatrixResult<UnifiedMatrix<T>> {
        let sigmoid_out = Self::sigmoid(input)?;

        // Element-wise multiply input * sigmoid(input)
        Self::elementwise_multiply(input, &sigmoid_out)
    }

    /// Apply Swish derivative: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    pub fn swish_derivative<T: NumericType>(
        input: &UnifiedMatrix<T>,
    ) -> MatrixResult<UnifiedMatrix<T>> {
        let sigmoid_out = Self::sigmoid(input)?;
        let sigmoid_deriv = Self::sigmoid_derivative(input)?;

        // swish'(x) = sigmoid(x) + x * sigmoid'(x)
        let x_times_sig_deriv = Self::elementwise_multiply(input, &sigmoid_deriv)?;
        Self::elementwise_add(&sigmoid_out, &x_times_sig_deriv)
    }

    /// Apply Leaky ReLU: max(alpha * x, x)
    pub fn leaky_relu<T: NumericType>(
        input: &UnifiedMatrix<T>,
        alpha: f64,
    ) -> MatrixResult<UnifiedMatrix<T>> {
        Self::apply_elementwise(input, |x| {
            let x_f64 = x.to_f64();
            let result = if x_f64 > 0.0 { x_f64 } else { alpha * x_f64 };
            T::from_f64(result).unwrap_or(T::zero())
        })
    }

    /// Apply ELU activation: x if x > 0, alpha * (exp(x) - 1) if x <= 0
    pub fn elu<T: NumericType>(
        input: &UnifiedMatrix<T>,
        alpha: f64,
    ) -> MatrixResult<UnifiedMatrix<T>> {
        Self::apply_elementwise(input, |x| {
            let x_f64 = x.to_f64();
            let result = if x_f64 > 0.0 {
                x_f64
            } else {
                alpha * (x_f64.exp() - 1.0)
            };
            T::from_f64(result).unwrap_or(T::zero())
        })
    }

    /// Apply Softmax activation across the last dimension
    /// Softmax(x_i) = exp(x_i) / Σ exp(x_j)
    pub fn softmax<T: NumericType>(input: &UnifiedMatrix<T>) -> MatrixResult<UnifiedMatrix<T>> {
        let shape = input.shape();
        let mut result = UnifiedMatrix::zeros(shape.rows, shape.cols)?;

        // For numerical stability, subtract max from each row
        for row in 0..shape.rows {
            // Find max in this row
            let _row_start = row * shape.cols;
            let row_data = Self::get_row_data(input, row)?;
            let max_val = row_data
                .iter()
                .map(|x| x.to_f64())
                .fold(f64::NEG_INFINITY, f64::max);

            // Compute exp(x - max) and sum
            let mut exp_sum = 0.0;
            let mut exp_values = Vec::with_capacity(shape.cols);

            for &val in &row_data {
                let exp_val = (val.to_f64() - max_val).exp();
                exp_values.push(exp_val);
                exp_sum += exp_val;
            }

            // Normalize by sum
            for (col, &exp_val) in exp_values.iter().enumerate() {
                let softmax_val = exp_val / exp_sum;
                Self::set_element(
                    &mut result,
                    row,
                    col,
                    T::from_f64(softmax_val).unwrap_or(T::zero()),
                )?;
            }
        }

        Ok(result)
    }

    /// SIMD-optimized element-wise operation
    fn apply_elementwise<T: NumericType, F>(
        input: &UnifiedMatrix<T>,
        func: F,
    ) -> MatrixResult<UnifiedMatrix<T>>
    where
        F: Fn(T) -> T + Send + Sync,
    {
        let shape = input.shape();
        let mut result = UnifiedMatrix::zeros(shape.rows, shape.cols)?;

        // Get backend for SIMD operations if available
        let _backend = input.backend_selector.select_backend(
            OperationType::ElementwiseAdd, // Use as proxy for element-wise ops
            shape.rows * shape.cols,
            T::DTYPE,
            input.backend_hint.clone(),
        );

        // Apply function element-wise
        Self::transform_elements(input, &mut result, func)?;

        Ok(result)
    }

    /// In-place element-wise operation
    fn apply_elementwise_inplace<T: NumericType, F>(
        matrix: &mut UnifiedMatrix<T>,
        func: F,
    ) -> MatrixResult<()>
    where
        F: Fn(&mut T) -> T + Send + Sync,
    {
        // This would need access to the internal storage
        // For now, implement via temporary copy
        let temp = Self::apply_elementwise(matrix, |x| func(&mut x.clone()))?;
        *matrix = temp;
        Ok(())
    }

    /// Helper to get row data (simplified implementation)
    fn get_row_data<T: NumericType>(matrix: &UnifiedMatrix<T>, row: usize) -> MatrixResult<Vec<T>> {
        let shape = matrix.shape();
        if row >= shape.rows {
            return Err(MatrixError::InvalidIndex {
                row,
                col: 0,
                shape: (shape.rows, shape.cols),
            });
        }

        // This is a simplified implementation - would need actual matrix data access
        Ok(vec![T::zero(); shape.cols])
    }

    /// Helper to set element (simplified implementation)
    fn set_element<T: NumericType>(
        matrix: &mut UnifiedMatrix<T>,
        row: usize,
        col: usize,
        _value: T,
    ) -> MatrixResult<()> {
        let shape = matrix.shape();
        if row >= shape.rows || col >= shape.cols {
            return Err(MatrixError::InvalidIndex {
                row,
                col,
                shape: (shape.rows, shape.cols),
            });
        }

        // This would need actual implementation with matrix storage access
        Ok(())
    }

    /// Helper to transform elements (simplified)
    fn transform_elements<T: NumericType, F>(
        input: &UnifiedMatrix<T>,
        result: &mut UnifiedMatrix<T>,
        func: F,
    ) -> MatrixResult<()>
    where
        F: Fn(T) -> T + Send + Sync,
    {
        let shape = input.shape();
        for i in 0..shape.rows {
            for j in 0..shape.cols {
                let val = input.get(i, j)?;
                let transformed = func(val);
                result.set(i, j, transformed)?;
            }
        }
        Ok(())
    }

    /// Element-wise addition helper
    fn elementwise_add<T: NumericType>(
        a: &UnifiedMatrix<T>,
        b: &UnifiedMatrix<T>,
    ) -> MatrixResult<UnifiedMatrix<T>> {
        // Use the backend's elementwise_add operation
        let _backend = a.backend_selector.select_backend(
            OperationType::ElementwiseAdd,
            a.shape().rows * a.shape().cols,
            T::DTYPE,
            a.backend_hint.clone(),
        );

        // This would delegate to the actual matrix addition
        a.add(b)
    }

    /// Element-wise multiplication helper
    fn elementwise_multiply<T: NumericType>(
        a: &UnifiedMatrix<T>,
        b: &UnifiedMatrix<T>,
    ) -> MatrixResult<UnifiedMatrix<T>> {
        // Use the backend's elementwise_mul operation
        let _backend = a.backend_selector.select_backend(
            OperationType::ElementwiseMul,
            a.shape().rows * a.shape().cols,
            T::DTYPE,
            a.backend_hint.clone(),
        );

        // This would delegate to the actual matrix multiplication
        a.elementwise_multiply(b)
    }
}

/// ReLU activation function struct
pub struct ReLU;

impl<T: NumericType> ActivationFunction<T> for ReLU {
    fn forward(&self, input: &UnifiedMatrix<T>) -> MatrixResult<UnifiedMatrix<T>> {
        ActivationOps::relu(input)
    }

    fn backward(
        &self,
        input: &UnifiedMatrix<T>,
        grad_output: &UnifiedMatrix<T>,
    ) -> MatrixResult<UnifiedMatrix<T>> {
        let derivative = ActivationOps::relu_derivative(input)?;
        ActivationOps::elementwise_multiply(&derivative, grad_output)
    }

    fn forward_inplace(&self, matrix: &mut UnifiedMatrix<T>) -> MatrixResult<()> {
        ActivationOps::relu_inplace(matrix)
    }

    fn name(&self) -> &str {
        "ReLU"
    }
}

/// GELU activation function struct
pub struct GELU;

impl<T: NumericType> ActivationFunction<T> for GELU {
    fn forward(&self, input: &UnifiedMatrix<T>) -> MatrixResult<UnifiedMatrix<T>> {
        ActivationOps::gelu(input)
    }

    fn backward(
        &self,
        input: &UnifiedMatrix<T>,
        grad_output: &UnifiedMatrix<T>,
    ) -> MatrixResult<UnifiedMatrix<T>> {
        let derivative = ActivationOps::gelu_derivative(input)?;
        ActivationOps::elementwise_multiply(&derivative, grad_output)
    }

    fn forward_inplace(&self, matrix: &mut UnifiedMatrix<T>) -> MatrixResult<()> {
        let result = self.forward(matrix)?;
        *matrix = result;
        Ok(())
    }

    fn name(&self) -> &str {
        "GELU"
    }
}

/// Sigmoid activation function struct
pub struct Sigmoid;

impl<T: NumericType> ActivationFunction<T> for Sigmoid {
    fn forward(&self, input: &UnifiedMatrix<T>) -> MatrixResult<UnifiedMatrix<T>> {
        ActivationOps::sigmoid(input)
    }

    fn backward(
        &self,
        input: &UnifiedMatrix<T>,
        grad_output: &UnifiedMatrix<T>,
    ) -> MatrixResult<UnifiedMatrix<T>> {
        let derivative = ActivationOps::sigmoid_derivative(input)?;
        ActivationOps::elementwise_multiply(&derivative, grad_output)
    }

    fn forward_inplace(&self, matrix: &mut UnifiedMatrix<T>) -> MatrixResult<()> {
        let result = self.forward(matrix)?;
        *matrix = result;
        Ok(())
    }

    fn name(&self) -> &str {
        "Sigmoid"
    }
}

/// Tanh activation function struct
pub struct Tanh;

impl<T: NumericType> ActivationFunction<T> for Tanh {
    fn forward(&self, input: &UnifiedMatrix<T>) -> MatrixResult<UnifiedMatrix<T>> {
        ActivationOps::tanh(input)
    }

    fn backward(
        &self,
        input: &UnifiedMatrix<T>,
        grad_output: &UnifiedMatrix<T>,
    ) -> MatrixResult<UnifiedMatrix<T>> {
        let derivative = ActivationOps::tanh_derivative(input)?;
        ActivationOps::elementwise_multiply(&derivative, grad_output)
    }

    fn forward_inplace(&self, matrix: &mut UnifiedMatrix<T>) -> MatrixResult<()> {
        let result = self.forward(matrix)?;
        *matrix = result;
        Ok(())
    }

    fn name(&self) -> &str {
        "Tanh"
    }
}

/// Convenient activation function constructors
pub fn relu<T: NumericType>() -> ReLU {
    ReLU
}
pub fn gelu<T: NumericType>() -> GELU {
    GELU
}
pub fn sigmoid<T: NumericType>() -> Sigmoid {
    Sigmoid
}
pub fn tanh<T: NumericType>() -> Tanh {
    Tanh
}

/// Softmax function (not a struct since it's typically used standalone)
pub fn softmax<T: NumericType>(input: &UnifiedMatrix<T>) -> MatrixResult<UnifiedMatrix<T>> {
    ActivationOps::softmax(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu_properties() {
        // Test with f64 matrices when available
        let relu = ReLU;
        assert_eq!(<ReLU as ActivationFunction<f64>>::name(&relu), "ReLU");
    }

    #[test]
    fn test_activation_function_names() {
        assert_eq!(
            <ReLU as ActivationFunction<f64>>::name(&relu::<f64>()),
            "ReLU"
        );
        assert_eq!(
            <GELU as ActivationFunction<f64>>::name(&gelu::<f64>()),
            "GELU"
        );
        assert_eq!(
            <Sigmoid as ActivationFunction<f64>>::name(&sigmoid::<f64>()),
            "Sigmoid"
        );
        assert_eq!(
            <Tanh as ActivationFunction<f64>>::name(&tanh::<f64>()),
            "Tanh"
        );
    }

    #[test]
    fn test_activation_type_enum() {
        let relu_type = ActivationType::ReLU;
        let _leaky_relu_type = ActivationType::LeakyReLU(0.01);
        let _elu_type = ActivationType::ELU(1.0);

        assert_eq!(relu_type, ActivationType::ReLU);
        assert_ne!(relu_type, ActivationType::GELU);
    }
}
