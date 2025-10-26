mod components;
mod girvan_newman;
mod infomap;
mod leiden;
mod louvain;
mod lpa;
pub mod modularity;
pub mod utils;

pub use components::ConnectedComponents;
pub use girvan_newman::GirvanNewman;
pub use infomap::Infomap;
pub use leiden::Leiden;
pub use louvain::Louvain;
pub use lpa::LabelPropagation;

use crate::algorithms::registry::Registry;

pub fn register_algorithms(registry: &Registry) -> anyhow::Result<()> {
    lpa::register(registry)?;
    louvain::register(registry)?;
    leiden::register(registry)?;
    infomap::register(registry)?;
    components::register(registry)?;
    girvan_newman::register(registry)?;
    Ok(())
}
