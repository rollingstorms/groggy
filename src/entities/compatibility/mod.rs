//! Historical unchecked entity constructors and specialized trait behavior.
//! Both adapters contain the canonical handle; storage owns no duplicate handles.
mod edge;
mod node;
pub use edge::EntityEdge;
pub use node::EntityNode;
