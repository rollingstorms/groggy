//! Datetime marshalling selected by the actual PyO3 build configuration.
//! The full CPython ABI retains direct C access; limited ABI builds use cached
//! standard-library types and their base descriptors, without float timestamps.
#[cfg(Py_LIMITED_API)]
mod limited;
#[cfg(not(Py_LIMITED_API))]
mod native;
#[cfg(Py_LIMITED_API)]
pub use limited::*;
#[cfg(not(Py_LIMITED_API))]
pub use native::*;
