mod components;
mod girvan_newman;
mod girvan_newman_paths;
mod infomap;
mod leiden;
mod leiden_levels;
mod louvain;
mod lpa;
mod map_equation;
pub mod modularity;
mod modularity_levels;
pub mod utils;

pub use components::{ComponentMode, ConnectedComponents};
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
