//! Capture accepted first-parent state before detached effective-time replay.
use super::*;
use crate::state::effective_read::{EffectiveReadPlan, ReadTime};
use std::collections::BTreeMap;
impl Graph {
    /// Pin the selected published head. Pending manual edits reject.
    pub fn prepare_read_at(
        &self,
        effective_at: i64,
        recorded_by: Option<i64>,
    ) -> GraphResult<EffectiveReadPlan> {
        self.prepare_read_at_with_legacy(effective_at, recorded_by, BTreeMap::new())
    }
    /// Assign explicit clocks to legacy versions for this read only. Existing
    /// precise clocks cannot be overridden; the immutable history is not rewritten.
    pub fn prepare_read_at_with_legacy(
        &self,
        effective_at: i64,
        recorded_by: Option<i64>,
        legacy_publications: BTreeMap<StateId, crate::PublicationMetadata>,
    ) -> GraphResult<EffectiveReadPlan> {
        self.prepare_effective_read(
            effective_at,
            crate::state::effective_read::EffectiveReadOptions {
                recorded_by,
                legacy_publications,
                ..Default::default()
            },
        )
    }
    /// Prepare with explicit inputs for legacy clocks and unpinned references.
    pub fn prepare_effective_read(
        &self,
        effective_at: i64,
        options: crate::state::effective_read::EffectiveReadOptions,
    ) -> GraphResult<EffectiveReadPlan> {
        let crate::state::effective_read::EffectiveReadOptions {
            recorded_by,
            legacy_publications,
            group_sources,
        } = options;
        if self.has_uncommitted_changes() || (self.current_commit == 0 && !self.is_empty()) {
            return Err(GraphError::InvalidInput("Effective-time reads require a clean published graph; publish pending manual edits first".into()));
        }
        let context = self.history.effective_read_context(self.current_commit)?;
        context.validate_legacy(&legacy_publications)?;
        let legacy_clocks = legacy_publications
            .iter()
            .map(|(&id, time)| (id, (time.effective_at, time.recorded_at)))
            .collect();
        let mut group_revisions = BTreeMap::new();
        let mut groups = self.pool.borrow().retained_groups();
        if !group_sources.is_empty() {
            for (id, source) in group_sources {
                if groups.contains_key(&id) {
                    return Err(GraphError::InvalidInput(
                        "Cannot override a frozen group source".into(),
                    ));
                }
                let (nodes, edges, kind) = self.pool.borrow().get_subgraph(id)?;
                if source.nodes() != &nodes || source.edges() != &edges {
                    return Err(GraphError::InvalidInput(
                        "Legacy group source must have exactly the stored node and edge membership"
                            .into(),
                    ));
                }
                let group = crate::storage::retained_group::RetainedGroup::new(source, kind)?;
                group_revisions.insert(id, group.id());
                Arc::make_mut(&mut groups).insert(id, group);
            }
        }
        Ok(EffectiveReadPlan {
            context,
            graph_type: self.graph_type(),
            groups,
            legacy_publications,
            time: ReadTime {
                branch: self.current_branch.clone(),
                published_head: self.current_commit,
                effective_at,
                recorded_by,
                legacy_clocks,
                group_sources: group_revisions,
            },
        })
    }
    /// Reconstruct a read-only graph at T; optional U restricts acceptance time.
    pub fn read_view_at(
        &self,
        effective_at: i64,
        recorded_by: Option<i64>,
    ) -> GraphResult<crate::GraphView> {
        self.prepare_read_at(effective_at, recorded_by)?.read_view()
    }
    /// Reconstruct with explicit caller-assigned legacy publication clocks.
    pub fn read_view_at_with_legacy(
        &self,
        effective_at: i64,
        recorded_by: Option<i64>,
        legacy_publications: BTreeMap<StateId, crate::PublicationMetadata>,
    ) -> GraphResult<crate::GraphView> {
        self.prepare_read_at_with_legacy(effective_at, recorded_by, legacy_publications)?
            .read_view()
    }
}
