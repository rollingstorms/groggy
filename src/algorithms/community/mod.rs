mod louvain;
mod lpa;
pub mod modularity;

pub use louvain::Louvain;
pub use lpa::LabelPropagation;

use crate::algorithms::registry::Registry;

pub fn register_algorithms(registry: &Registry) -> anyhow::Result<()> {
    lpa::register(registry)?;
    louvain::register(registry)?;
    Ok(())
}
