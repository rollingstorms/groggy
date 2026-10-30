//! One weak current default capture; explicit read_view still creates fresh provenance.
use super::*;

/// Opaque bound-view cache token. Construction can copy a branch name; matching is
/// allocation-free and includes all readable mappings and the current write origin.
#[derive(Debug, Clone)]
pub struct ViewRevision {
    data: crate::state::space::ReadRevision,
    branch: String,
    state: StateId,
    clean: bool,
}
impl Graph {
    /// Canonical immutable current default view. Related default selections share
    /// one capture while a caller owns it; the cache itself pins no values/groups.
    pub fn view(&self) -> GraphResult<crate::GraphView> {
        if let Some(state) = self.space.cached_default_capture() {
            if state.origin().is_some_and(|origin| {
                Arc::ptr_eq(&origin.history, self.history.identity())
                    && origin.branch == self.current_branch
                    && origin.state == self.current_commit
                    && origin.clean == self.current_view_clean()
            }) {
                return Ok(crate::GraphView::all(state));
            }
        }
        let state = self.pin_current_state()?;
        self.space.cache_default_capture(&state);
        Ok(crate::GraphView::all(state))
    }
    fn current_view_clean(&self) -> bool {
        !self.has_uncommitted_changes() && (self.current_commit != 0 || self.is_empty())
    }
    /// Capture a token for caching default read adapters without storing a graph pin.
    pub fn view_revision(&self) -> ViewRevision {
        ViewRevision {
            data: self.space.read_revision(),
            branch: self.current_branch.clone(),
            state: self.current_commit,
            clean: self.current_view_clean(),
        }
    }
    /// Check a cached token without copying IDs, attributes, or branch names.
    pub fn view_revision_is_current(&self, revision: &ViewRevision) -> bool {
        self.space.matches_read_revision(&revision.data)
            && self.current_commit == revision.state
            && self.current_branch == revision.branch
            && self.current_view_clean() == revision.clean
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn default_views_share_current_capture_but_explicit_reads_are_independent() {
        let mut graph = Graph::new();
        let node = graph.add_node().unwrap();
        let a = graph.view().unwrap();
        let b = graph.view().unwrap();
        assert!(a.read_state().shares_capture(b.read_state()));
        let explicit = graph.read_view().unwrap();
        assert!(!a.read_state().shares_capture(explicit.read_state()));
        let revision = graph.view_revision();
        assert!(graph.view_revision_is_current(&revision));
        graph
            .set_node_attr(node, "value".into(), AttrValue::Int(1))
            .unwrap();
        assert!(!graph.view_revision_is_current(&revision));
        let next = graph.view().unwrap();
        assert!(!a.read_state().shares_capture(next.read_state()));
        assert_eq!(a.node_attr(node, &"value".into()).unwrap(), None);
        assert_eq!(
            next.node_attr(node, &"value".into()).unwrap(),
            Some(AttrValue::Int(1))
        );
    }
    #[test]
    fn default_cache_does_not_pin_values_or_capture_identity() {
        let mut graph = Graph::new();
        graph.add_node().unwrap();
        let view = graph.view().unwrap();
        assert!(graph.space.cached_default_capture().is_some());
        drop(view);
        assert!(graph.space.cached_default_capture().is_none());
        assert!(graph.view().is_ok());
    }
}

#[cfg(test)]
mod catalog_tests {
    use super::*;
    #[test]
    fn retaining_a_group_refreshes_the_default_catalog_without_changing_old_reads() {
        let mut graph = Graph::new();
        graph.add_node().unwrap();
        let before = graph.view().unwrap();
        let revision = graph.view_revision();
        let id = graph.retain_group(&before, "source".into()).unwrap();
        assert!(!graph.view_revision_is_current(&revision));
        assert!(before.read_state().retained_group(id).is_err());
        assert!(graph
            .view()
            .unwrap()
            .read_state()
            .retained_group(id)
            .unwrap()
            .contains_node(0));
    }
}
