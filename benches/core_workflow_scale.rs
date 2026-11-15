//! Native vertical workflow; the allocation companion uses this same fixture.
#[path = "support/core_workflow.rs"]
mod workflow;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", serde_json::to_string_pretty(&workflow::run(|_| {})?)?);
    Ok(())
}
