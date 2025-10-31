//! Topology utilities shared across state/algorithm layers.
//!
//! This module hosts reusable data structures (such as CSR adjacency) that
//! support high-performance graph algorithms without duplicating boilerplate.

pub mod csr;

pub use csr::*;
