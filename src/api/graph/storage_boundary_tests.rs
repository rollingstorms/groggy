//! Internal corruption/recovery regressions require trusted storage access.
#[path = "storage_boundary_tests/automatic_publication.rs"]
mod automatic_publication;
#[path = "storage_boundary_tests/current_property_reads.rs"]
mod current_property_reads;
#[path = "storage_boundary_tests/effective_time_completion.rs"]
mod effective_time_completion;
#[path = "storage_boundary_tests/endpoint_identity.rs"]
mod endpoint_identity;
#[path = "storage_boundary_tests/graph_copy.rs"]
mod graph_copy;
#[path = "storage_boundary_tests/owned_snapshots.rs"]
mod owned_snapshots;
#[path = "storage_boundary_tests/pinned_read_state.rs"]
mod pinned_read_state;
#[path = "storage_boundary_tests/pinned_view_writes.rs"]
mod pinned_view_writes;
#[path = "storage_boundary_tests/prepared_meta_creation.rs"]
mod prepared_meta_creation;
#[path = "storage_boundary_tests/retained_subgraphs.rs"]
mod retained_subgraphs;
#[path = "storage_boundary_tests/zero_bulk_allocation.rs"]
mod zero_bulk_allocation;
