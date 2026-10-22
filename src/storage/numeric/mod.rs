//! Scalar numerical foundation shared by arrays, matrices and property bridges.
//! Owns dtypes, checked conversions/arithmetic, accurate accumulation and reductions.
//! It contains no graph/matrix/array ownership or execution policy.
pub mod attributes;
pub(crate) mod checked;
pub(crate) mod dot_product;
pub mod reductions;
pub mod scalar;
pub mod types;
pub use scalar::{ConversionPolicy, NumericComparison, NumericInput};
pub use types::{DType, NumericType};
