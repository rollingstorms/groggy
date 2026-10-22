//! Checked native activation kernels and faithful derivatives.
//! Zero-preserving maps retain CSR storage; nonzero images use the shared expansion policy.

use crate::storage::matrix::{
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

/// Checked activation operations over logical matrix storage.
pub struct ActivationOps;

impl ActivationOps {
    fn invalid(message: &str) -> MatrixError {
        MatrixError::InvalidInput(message.into())
    }
    fn output<T: NumericType>(value: f64) -> MatrixResult<T> {
        use crate::storage::numeric::scalar::{convert, ConversionPolicy, NumericInput};
        if !value.is_finite() {
            return Err(Self::invalid("Activation result exceeds finite range"));
        }
        convert(
            NumericInput::Float(value),
            if T::DTYPE.is_float() {
                ConversionPolicy::Round
            } else {
                ConversionPolicy::Exact
            },
        )
        .map_err(|e| Self::invalid(&e.to_string()))
    }
    fn map<T: NumericType>(
        input: &UnifiedMatrix<T>,
        operation: impl Fn(T) -> MatrixResult<T>,
    ) -> MatrixResult<UnifiedMatrix<T>> {
        input.try_map_elementwise(|value| {
            if !value.is_finite() {
                return Err(Self::invalid("Activations require finite numeric input"));
            }
            operation(value)
        })
    }
    fn map_float<T: NumericType>(
        input: &UnifiedMatrix<T>,
        operation: impl Fn(f64) -> f64,
    ) -> MatrixResult<UnifiedMatrix<T>> {
        Self::map(input, |value| Self::output(operation(value.to_f64())))
    }
    fn sigmoid_value(x: f64) -> f64 {
        if x >= 0. {
            1. / (1. + (-x).exp())
        } else {
            let e = x.exp();
            e / (1. + e)
        }
    }
    fn gelu_value(x: f64) -> f64 {
        if x >= 12. {
            return x;
        }
        if x <= -12. {
            return 0.;
        }
        0.5 * x * (1. + ((2. / PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
    }
    fn gelu_slope(x: f64) -> f64 {
        if x >= 12. {
            return 1.;
        }
        if x <= -12. {
            return 0.;
        }
        let c = (2. / PI).sqrt();
        let t = (c * (x + 0.044715 * x * x * x)).tanh();
        0.5 * (1. + t) + 0.5 * x * (1. - t * t) * c * (1. + 3. * 0.044715 * x * x)
    }
    /// ReLU preserves exact integer values and zero-preserving sparse storage.
    pub fn relu<T: NumericType>(input: &UnifiedMatrix<T>) -> MatrixResult<UnifiedMatrix<T>> {
        Self::map(input, |x| Ok(if x.to_f64() > 0. { x } else { T::zero() }))
    }
    /// Atomic owned replacement: an error leaves the original unchanged.
    pub fn relu_inplace<T: NumericType>(matrix: &mut UnifiedMatrix<T>) -> MatrixResult<()> {
        *matrix = Self::relu(matrix)?;
        Ok(())
    }
    /// ReLU chooses derivative zero at the origin.
    pub fn relu_derivative<T: NumericType>(
        input: &UnifiedMatrix<T>,
    ) -> MatrixResult<UnifiedMatrix<T>> {
        Self::map(input, |x| {
            Ok(if x.to_f64() > 0. { T::one() } else { T::zero() })
        })
    }
    /// Tanh approximation to GELU; its derivative uses the same approximation.
    pub fn gelu<T: NumericType>(input: &UnifiedMatrix<T>) -> MatrixResult<UnifiedMatrix<T>> {
        Self::map_float(input, Self::gelu_value)
    }
    /// Faithful derivative of the tanh GELU forward formula.
    pub fn gelu_derivative<T: NumericType>(
        input: &UnifiedMatrix<T>,
    ) -> MatrixResult<UnifiedMatrix<T>> {
        Self::map_float(input, Self::gelu_slope)
    }
    /// Stable logistic activation.
    pub fn sigmoid<T: NumericType>(input: &UnifiedMatrix<T>) -> MatrixResult<UnifiedMatrix<T>> {
        Self::map_float(input, Self::sigmoid_value)
    }
    /// Logistic derivative.
    pub fn sigmoid_derivative<T: NumericType>(
        input: &UnifiedMatrix<T>,
    ) -> MatrixResult<UnifiedMatrix<T>> {
        Self::map_float(input, |x| {
            let s = Self::sigmoid_value(x);
            s * (1. - s)
        })
    }
    /// Hyperbolic tangent activation.
    pub fn tanh<T: NumericType>(input: &UnifiedMatrix<T>) -> MatrixResult<UnifiedMatrix<T>> {
        Self::map_float(input, f64::tanh)
    }
    /// Hyperbolic tangent derivative.
    pub fn tanh_derivative<T: NumericType>(
        input: &UnifiedMatrix<T>,
    ) -> MatrixResult<UnifiedMatrix<T>> {
        Self::map_float(input, |x| {
            let t = x.tanh();
            1. - t * t
        })
    }
    /// Swish activation x*sigmoid(x).
    pub fn swish<T: NumericType>(input: &UnifiedMatrix<T>) -> MatrixResult<UnifiedMatrix<T>> {
        Self::map_float(input, |x| x * Self::sigmoid_value(x))
    }
    /// Swish derivative.
    pub fn swish_derivative<T: NumericType>(
        input: &UnifiedMatrix<T>,
    ) -> MatrixResult<UnifiedMatrix<T>> {
        Self::map_float(input, |x| {
            let s = Self::sigmoid_value(x);
            s + x * s * (1. - s)
        })
    }
    /// Leaky ReLU with a finite negative-side slope.
    pub fn leaky_relu<T: NumericType>(
        input: &UnifiedMatrix<T>,
        alpha: f64,
    ) -> MatrixResult<UnifiedMatrix<T>> {
        if !alpha.is_finite() {
            return Err(Self::invalid("Leaky ReLU alpha must be finite"));
        }
        Self::map(input, |x| {
            if x.to_f64() > 0. {
                Ok(x)
            } else {
                Self::output(alpha * x.to_f64())
            }
        })
    }
    /// ELU with exp_m1 accuracy near zero and checked output conversion.
    pub fn elu<T: NumericType>(
        input: &UnifiedMatrix<T>,
        alpha: f64,
    ) -> MatrixResult<UnifiedMatrix<T>> {
        if !alpha.is_finite() {
            return Err(Self::invalid("ELU alpha must be finite"));
        }
        Self::map(input, |x| {
            if x.to_f64() > 0. {
                Ok(x)
            } else {
                Self::output(alpha * x.to_f64().exp_m1())
            }
        })
    }
    /// Stable row softmax; implicit sparse expansion uses the shared dense guard.
    pub fn softmax<T: NumericType>(input: &UnifiedMatrix<T>) -> MatrixResult<UnifiedMatrix<T>> {
        if !T::DTYPE.is_float() {
            return Err(Self::invalid("Softmax requires floating matrix storage"));
        }
        if input.checked_sparse()?.is_some() {
            super::super::resource_policy::implicit_dense(
                input.rows(),
                input.cols(),
                "Sparse softmax output",
            )?;
        }
        let mut output = UnifiedMatrix::zeros(input.rows(), input.cols())?;
        if input.cols() == 0 {
            return Ok(output);
        }
        for row in 0..input.rows() {
            let mut maximum = f64::NEG_INFINITY;
            for col in 0..input.cols() {
                let value = input.get(row, col)?.to_f64();
                if !value.is_finite() {
                    return Err(Self::invalid("Softmax requires finite input"));
                }
                maximum = maximum.max(value);
            }
            let mut total = 0.;
            for col in 0..input.cols() {
                total += (input.get(row, col)?.to_f64() - maximum).exp();
            }
            for col in 0..input.cols() {
                output.set(
                    row,
                    col,
                    Self::output((input.get(row, col)?.to_f64() - maximum).exp() / total)?,
                )?;
            }
        }
        Ok(output)
    }
    /// Pull back row softmax without constructing a dense Jacobian.
    pub fn softmax_vjp<T: NumericType>(
        input: &UnifiedMatrix<T>,
        gradient: &UnifiedMatrix<T>,
    ) -> MatrixResult<UnifiedMatrix<T>> {
        if input.shape() != gradient.shape() {
            return Err(Self::invalid("Softmax cotangent shape mismatch"));
        }
        let probabilities = Self::softmax(input)?;
        let mut output = UnifiedMatrix::zeros(input.rows(), input.cols())?;
        if input.cols() == 0 {
            return Ok(output);
        }
        for row in 0..input.rows() {
            let mut average = crate::storage::numeric::dot_product::Dot::new();
            for col in 0..input.cols() {
                let g = gradient.get(row, col)?.to_f64();
                if !g.is_finite() {
                    return Err(Self::invalid("Softmax cotangent must be finite"));
                }
                average.add(probabilities.get(row, col)?.to_f64(), g);
            }
            let average = average.total();
            for col in 0..input.cols() {
                let p = probabilities.get(row, col)?.to_f64();
                let mut product = crate::storage::numeric::dot_product::Dot::new();
                product.add(p, gradient.get(row, col)?.to_f64());
                product.add(-p, average);
                output.set(row, col, Self::output(product.total())?)?;
            }
        }
        Ok(output)
    }

    /// Inverted dropout; seeded masks depend on logical coordinates, not storage layout.
    /// Evaluation and p=0 are identity; p=1 training is all zero.
    pub fn dropout<T: NumericType>(
        input: &UnifiedMatrix<T>,
        p: f64,
        training: bool,
        seed: u64,
    ) -> MatrixResult<UnifiedMatrix<T>> {
        if !(0.0..=1.0).contains(&p) {
            return Err(Self::invalid(
                "Dropout probability must be finite and between zero and one",
            ));
        }
        if !training || p == 0. {
            return input.try_map_elementwise(Ok);
        }
        let map = |value: T, index: usize| -> MatrixResult<T> {
            if !value.is_finite() {
                return Err(Self::invalid("Dropout requires finite input"));
            }
            // SplitMix64 mixing gives reproducible coordinate decisions.
            let mut bits = seed.wrapping_add((index as u64).wrapping_mul(0x9e3779b97f4a7c15));
            bits = (bits ^ (bits >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            bits = (bits ^ (bits >> 27)).wrapping_mul(0x94d049bb133111eb);
            bits ^= bits >> 31;
            let uniform = (bits >> 11) as f64 / 9007199254740992.;
            if p == 1. || uniform < p {
                Ok(T::zero())
            } else {
                Self::output(value.to_f64() / (1. - p))
            }
        };
        if let Some(sparse) = input.checked_sparse()? {
            let mut values = Vec::new();
            let mut indices = Vec::new();
            let mut pointers = vec![0];
            input
                .rows()
                .checked_mul(input.cols())
                .ok_or_else(|| Self::invalid("Dropout shape overflow"))?;
            for row in 0..input.rows() {
                for entry in sparse.row_pointers[row]..sparse.row_pointers[row + 1] {
                    let col = sparse.col_indices[entry];
                    let value = map(sparse.values[entry], row * input.cols() + col)?;
                    if value != T::zero() {
                        values.push(value);
                        indices.push(col);
                    }
                }
                pointers.push(values.len());
            }
            return UnifiedMatrix::from_csr(input.rows(), input.cols(), values, indices, pointers);
        }
        let values = input
            .to_vec()?
            .into_iter()
            .enumerate()
            .map(|(index, value)| map(value, index))
            .collect::<MatrixResult<Vec<_>>>()?;
        UnifiedMatrix::from_data(values, input.rows(), input.cols())
    }
    fn elementwise_multiply<T: NumericType>(
        a: &UnifiedMatrix<T>,
        b: &UnifiedMatrix<T>,
    ) -> MatrixResult<UnifiedMatrix<T>> {
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
