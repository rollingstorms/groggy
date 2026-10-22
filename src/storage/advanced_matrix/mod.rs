//! Compatibility imports for the former advanced matrix namespace.
//!
//! All implementations now live under [`crate::storage::matrix`]. Existing
//! module/type paths remain re-exports of those same types and traits.
pub use super::matrix::*;
