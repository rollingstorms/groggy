//! NumericType trait - Foundation of the generic type system
//!
//! This trait provides the core abstraction for all numeric types supported
//! in the advanced matrix system, enabling type-safe and performant operations
//! across different numeric precisions.

use std::fmt::Debug;

/// Data type identifier for runtime type checking and backend selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    Bool,
    Int8,
    Int16,
    Int32,
    Int64,
    Float32,
    Float64,
    Complex64,
    Complex128,
}

impl DType {
    /// Get the size in bytes for this data type
    pub const fn byte_size(&self) -> usize {
        match self {
            DType::Bool => 1,
            DType::Int8 => 1,
            DType::Int16 => 2,
            DType::Int32 => 4,
            DType::Int64 => 8,
            DType::Float32 => 4,
            DType::Float64 => 8,
            DType::Complex64 => 8,
            DType::Complex128 => 16,
        }
    }

    /// Promote two data types to their common representation type
    pub fn promote(a: DType, b: DType) -> DType {
        use DType::*;
        match (a, b) {
            // Same types
            (x, y) if x == y => x,

            // Bool promotes to anything
            (Bool, other) | (other, Bool) => other,

            // Float types have priority
            (Float64, _) | (_, Float64) => Float64,
            (Float32, _) | (_, Float32) => Float32,
            (Complex128, _) | (_, Complex128) => Complex128,
            (Complex64, _) | (_, Complex64) => Complex64,

            // Integer promotions
            (Int64, _) | (_, Int64) => Int64,
            // Remaining patterns would be unreachable due to above patterns
            // All remaining cases default to highest precision integer
            _ => Int64, // Default fallback for any remaining integer combinations
        }
    }

    /// Check if this is a floating-point type
    pub const fn is_float(&self) -> bool {
        matches!(self, DType::Float32 | DType::Float64)
    }

    /// Check if this is a complex type
    pub const fn is_complex(&self) -> bool {
        matches!(self, DType::Complex64 | DType::Complex128)
    }

    /// Check if this is an integer type
    pub const fn is_integer(&self) -> bool {
        matches!(
            self,
            DType::Int8 | DType::Int16 | DType::Int32 | DType::Int64
        )
    }
}

/// Core trait for all numeric types in the advanced matrix system
///
/// This trait provides the fundamental operations and properties needed
/// for efficient matrix computations across different numeric precisions.
pub trait NumericType:
    Copy + Clone + Debug + PartialEq + PartialOrd + Send + Sync + 'static
{
    /// Accumulator type for reductions to prevent overflow/underflow
    type Accumulator: NumericType;

    /// Wide type for intermediate calculations
    type Wide: NumericType;

    /// The data type identifier
    const DTYPE: DType;

    /// Size in bytes for this type
    const BYTE_SIZE: usize;

    /// Zero value for this type
    fn zero() -> Self;

    /// One value for this type
    fn one() -> Self;

    /// Convert from f64 if possible
    fn from_f64(val: f64) -> Option<Self>;

    /// Convert to f64 for generic operations
    fn to_f64(self) -> f64;

    /// Check if this value is finite (not NaN or infinite)
    fn is_finite(self) -> bool;

    /// Get the absolute value
    fn abs(self) -> Self;

    /// Basic arithmetic operations
    fn add(self, other: Self) -> Self;
    fn sub(self, other: Self) -> Self;
    fn mul(self, other: Self) -> Self;
    fn div(self, other: Self) -> Self;

    /// Fused multiply-add operation (a * b + c)
    fn fma(a: Self, b: Self, c: Self) -> Self {
        Self::add(Self::mul(a, b), c)
    }

    /// SIMD operations - default implementations can be overridden for performance
    fn simd_add(a: &[Self], b: &[Self], result: &mut [Self]) {
        for ((a_val, b_val), result_val) in a.iter().zip(b.iter()).zip(result.iter_mut()) {
            *result_val = a_val.add(*b_val);
        }
    }

    fn simd_mul(a: &[Self], b: &[Self], result: &mut [Self]) {
        for ((a_val, b_val), result_val) in a.iter().zip(b.iter()).zip(result.iter_mut()) {
            *result_val = a_val.mul(*b_val);
        }
    }

    fn simd_fma(a: &[Self], b: &[Self], c: &[Self], result: &mut [Self]) {
        for (((a_val, b_val), c_val), result_val) in
            a.iter().zip(b.iter()).zip(c.iter()).zip(result.iter_mut())
        {
            *result_val = Self::fma(*a_val, *b_val, *c_val);
        }
    }

    fn simd_reduce_sum(values: &[Self]) -> Self::Accumulator;
    fn simd_reduce_max(values: &[Self]) -> Self;
    fn simd_reduce_min(values: &[Self]) -> Self;
}

// Implementations for standard types

impl NumericType for f64 {
    type Accumulator = f64;
    type Wide = f64;
    const DTYPE: DType = DType::Float64;
    const BYTE_SIZE: usize = 8;

    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }

    fn from_f64(val: f64) -> Option<Self> {
        Some(val)
    }
    fn to_f64(self) -> f64 {
        self
    }

    fn is_finite(self) -> bool {
        self.is_finite()
    }
    fn abs(self) -> Self {
        self.abs()
    }

    fn add(self, other: Self) -> Self {
        self + other
    }
    fn sub(self, other: Self) -> Self {
        self - other
    }
    fn mul(self, other: Self) -> Self {
        self * other
    }
    fn div(self, other: Self) -> Self {
        self / other
    }

    fn fma(a: Self, b: Self, c: Self) -> Self {
        a.mul_add(b, c) // Use hardware FMA if available
    }

    fn simd_reduce_sum(values: &[Self]) -> Self::Accumulator {
        values.iter().copied().sum()
    }

    fn simd_reduce_max(values: &[Self]) -> Self {
        values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    }

    fn simd_reduce_min(values: &[Self]) -> Self {
        values.iter().copied().fold(f64::INFINITY, f64::min)
    }
}

impl NumericType for f32 {
    type Accumulator = f64; // Use f64 for accumulation to prevent precision loss
    type Wide = f64; // Use f64 for intermediate calculations
    const DTYPE: DType = DType::Float32;
    const BYTE_SIZE: usize = 4;

    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }

    fn from_f64(val: f64) -> Option<Self> {
        if val.is_finite() && val >= f32::MIN as f64 && val <= f32::MAX as f64 {
            Some(val as f32)
        } else {
            None
        }
    }
    fn to_f64(self) -> f64 {
        self as f64
    }

    fn is_finite(self) -> bool {
        self.is_finite()
    }
    fn abs(self) -> Self {
        self.abs()
    }

    fn add(self, other: Self) -> Self {
        self + other
    }
    fn sub(self, other: Self) -> Self {
        self - other
    }
    fn mul(self, other: Self) -> Self {
        self * other
    }
    fn div(self, other: Self) -> Self {
        self / other
    }

    fn fma(a: Self, b: Self, c: Self) -> Self {
        a.mul_add(b, c) // Use hardware FMA if available
    }

    fn simd_reduce_sum(values: &[Self]) -> Self::Accumulator {
        values.iter().map(|&x| x as f64).sum()
    }

    fn simd_reduce_max(values: &[Self]) -> Self {
        values.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    }

    fn simd_reduce_min(values: &[Self]) -> Self {
        values.iter().copied().fold(f32::INFINITY, f32::min)
    }
}

impl NumericType for i64 {
    type Accumulator = i128; // Prevent overflow in reductions
    type Wide = f64; // Use f64 for division operations
    const DTYPE: DType = DType::Int64;
    const BYTE_SIZE: usize = 8;

    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }

    fn from_f64(val: f64) -> Option<Self> {
        if val.is_finite() && val >= i64::MIN as f64 && val <= i64::MAX as f64 {
            Some(val as i64)
        } else {
            None
        }
    }
    fn to_f64(self) -> f64 {
        self as f64
    }

    fn is_finite(self) -> bool {
        true
    } // Integers are always finite
    fn abs(self) -> Self {
        self.abs()
    }

    fn add(self, other: Self) -> Self {
        self.saturating_add(other)
    }
    fn sub(self, other: Self) -> Self {
        self.saturating_sub(other)
    }
    fn mul(self, other: Self) -> Self {
        self.saturating_mul(other)
    }
    fn div(self, other: Self) -> Self {
        if other != 0 {
            self / other
        } else {
            0
        }
    }

    fn simd_reduce_sum(values: &[Self]) -> Self::Accumulator {
        values.iter().map(|&x| x as i128).sum()
    }

    fn simd_reduce_max(values: &[Self]) -> Self {
        values.iter().copied().max().unwrap_or(i64::MIN)
    }

    fn simd_reduce_min(values: &[Self]) -> Self {
        values.iter().copied().min().unwrap_or(i64::MAX)
    }
}

impl NumericType for i32 {
    type Accumulator = i64; // Prevent overflow in reductions
    type Wide = f64; // Use f64 for division operations
    const DTYPE: DType = DType::Int32;
    const BYTE_SIZE: usize = 4;

    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }

    fn from_f64(val: f64) -> Option<Self> {
        if val.is_finite() && val >= i32::MIN as f64 && val <= i32::MAX as f64 {
            Some(val as i32)
        } else {
            None
        }
    }
    fn to_f64(self) -> f64 {
        self as f64
    }

    fn is_finite(self) -> bool {
        true
    } // Integers are always finite
    fn abs(self) -> Self {
        self.abs()
    }

    fn add(self, other: Self) -> Self {
        self.saturating_add(other)
    }
    fn sub(self, other: Self) -> Self {
        self.saturating_sub(other)
    }
    fn mul(self, other: Self) -> Self {
        self.saturating_mul(other)
    }
    fn div(self, other: Self) -> Self {
        if other != 0 {
            self / other
        } else {
            0
        }
    }

    fn simd_reduce_sum(values: &[Self]) -> Self::Accumulator {
        values.iter().map(|&x| x as i64).sum()
    }

    fn simd_reduce_max(values: &[Self]) -> Self {
        values.iter().copied().max().unwrap_or(i32::MIN)
    }

    fn simd_reduce_min(values: &[Self]) -> Self {
        values.iter().copied().min().unwrap_or(i32::MAX)
    }
}

impl NumericType for bool {
    type Accumulator = i32; // Count of true values
    type Wide = f64; // Use f64 for mathematical operations
    const DTYPE: DType = DType::Bool;
    const BYTE_SIZE: usize = 1;

    fn zero() -> Self {
        false
    }
    fn one() -> Self {
        true
    }

    fn from_f64(val: f64) -> Option<Self> {
        Some(val != 0.0)
    }
    fn to_f64(self) -> f64 {
        if self {
            1.0
        } else {
            0.0
        }
    }

    fn is_finite(self) -> bool {
        true
    }
    fn abs(self) -> Self {
        self
    } // abs(bool) = bool

    fn add(self, other: Self) -> Self {
        self | other
    } // Logical OR
    fn sub(self, other: Self) -> Self {
        self & !other
    } // Logical NAND
    fn mul(self, other: Self) -> Self {
        self & other
    } // Logical AND
    fn div(self, other: Self) -> Self {
        if other {
            self
        } else {
            false
        } // Division by false is false
    }

    fn simd_reduce_sum(values: &[Self]) -> Self::Accumulator {
        values.iter().map(|&x| if x { 1 } else { 0 }).sum()
    }

    fn simd_reduce_max(values: &[Self]) -> Self {
        values.iter().any(|&x| x) // True if any are true
    }

    fn simd_reduce_min(values: &[Self]) -> Self {
        values.iter().all(|&x| x) // True only if all are true
    }
}

impl NumericType for i128 {
    type Accumulator = i128; // i128 is large enough for its own accumulation
    type Wide = f64; // Use f64 for division operations
    const DTYPE: DType = DType::Int64; // Map to Int64 for compatibility
    const BYTE_SIZE: usize = 16;

    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }

    fn from_f64(val: f64) -> Option<Self> {
        // Check for extreme values that won't fit in i128
        if val.is_finite() && val >= i128::MIN as f64 && val <= i128::MAX as f64 {
            Some(val as i128)
        } else {
            None
        }
    }
    fn to_f64(self) -> f64 {
        self as f64
    }

    fn is_finite(self) -> bool {
        true
    } // Integers are always finite
    fn abs(self) -> Self {
        self.saturating_abs()
    } // Use saturating to avoid overflow on i128::MIN

    fn add(self, other: Self) -> Self {
        self.saturating_add(other)
    }
    fn sub(self, other: Self) -> Self {
        self.saturating_sub(other)
    }
    fn mul(self, other: Self) -> Self {
        self.saturating_mul(other)
    }
    fn div(self, other: Self) -> Self {
        if other != 0 {
            self / other
        } else {
            0
        }
    }

    fn simd_reduce_sum(values: &[Self]) -> Self::Accumulator {
        values.iter().fold(0i128, |acc, &x| acc.saturating_add(x))
    }

    fn simd_reduce_max(values: &[Self]) -> Self {
        values.iter().copied().max().unwrap_or(i128::MIN)
    }

    fn simd_reduce_min(values: &[Self]) -> Self {
        values.iter().copied().min().unwrap_or(i128::MAX)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_promotion() {
        assert_eq!(DType::promote(DType::Int32, DType::Float64), DType::Float64);
        assert_eq!(DType::promote(DType::Bool, DType::Int32), DType::Int32);
        assert_eq!(
            DType::promote(DType::Float32, DType::Float32),
            DType::Float32
        );
    }

    #[test]
    fn test_numeric_type_operations() {
        // Test f64 operations
        assert_eq!(f64::add(1.5, 2.5), 4.0);
        assert_eq!(f64::fma(2.0, 3.0, 1.0), 7.0);

        // Test i64 operations with saturation
        assert_eq!(i64::add(i64::MAX, 1), i64::MAX); // Saturating add

        // Test bool operations
        assert!(bool::add(true, false)); // Logical OR
        assert!(!bool::mul(true, false)); // Logical AND
    }

    #[test]
    fn test_simd_reductions() {
        let values = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(f64::simd_reduce_sum(&values), 10.0);
        assert_eq!(f64::simd_reduce_max(&values), 4.0);
        assert_eq!(f64::simd_reduce_min(&values), 1.0);
    }

    #[test]
    fn test_type_conversions() {
        assert_eq!(i32::from_f64(42.7), Some(42));
        assert_eq!(bool::from_f64(0.0), Some(false));
        assert_eq!(bool::from_f64(1.0), Some(true));

        // Test overflow handling
        assert_eq!(i32::from_f64(f64::INFINITY), None);
    }
}
