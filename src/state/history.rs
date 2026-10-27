//! History System - Git-like version control for graph evolution.
//!
//! ARCHITECTURE ROLE:
//! This is the version control backbone of the system. It manages immutable
//! snapshots of graph state over time, with branching and merging capabilities.
//!
//! DESIGN PHILOSOPHY:
//! - Immutable snapshots (states are never modified after creation)
//! - Content-addressed storage (deduplication via hashing)
//! - Git-like branching model (lightweight branches, merge capabilities)
//! - Efficient diff-based storage (only store what changed)

mod archive;
pub(crate) use archive::HistoryDocument;
mod effective_topology;
pub(crate) use effective_topology::TopologyPatch;

mod recorded_read;
pub use recorded_read::{CommitDiffPlan, RecordedReadPlan};

// Preserve legacy public names while sharing the snapshot diff representation.
pub use crate::state::change::{AttributeChange, EdgeEndpointChange, EntityType};

use crate::errors::{GraphError, GraphResult};
use crate::state::change_tracker::ChangeSet;
use crate::state::ref_manager::BranchInfo;
use crate::state::state::{GraphSnapshot, StateDiff, StateMetadata};
use crate::types::{AttrName, AttrValue, BranchName, EdgeId, GraphType, NodeId, StateId};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/*
=== HISTORY SYSTEM OVERVIEW ===

The history system is responsible for:
1. Creating immutable snapshots of graph state
2. Managing the DAG of state evolution (commits, branches, merges)
3. Providing time-travel capabilities (view graph at any point in history)
4. Handling branching and merging workflows
5. Garbage collection of unreachable states

KEY INSIGHTS:
- Store deltas (changes) rather than full snapshots for efficiency
- Use content addressing for automatic deduplication
- Separate metadata (who, when, why) from data (what changed)
- Support both linear history and complex branching workflows
*/

/// The main history management system
///
/// RESPONSIBILITIES:
/// - Store immutable snapshots of graph state over time
/// - Manage branching and merging operations
/// - Provide efficient diff-based storage
/// - Support time-travel queries
/// - Handle garbage collection of unreachable history
///
/// NOT RESPONSIBLE FOR:
/// - Current mutable state (that's GraphStore's job)
/// - Query processing (that's QueryEngine's job)
/// - Change tracking (that's ChangeTracker's job)
#[derive(Debug)]
pub struct HistoryForest {
    /// In-process provenance shared by snapshots, independent of numeric state IDs.
    identity: Arc<()>,
    graph_type: GraphType,
    read_context_cache:
        parking_lot::Mutex<Option<(StateId, Arc<crate::state::effective_read::ReadContext>)>>,
    effective_topology: parking_lot::Mutex<effective_topology::EffectiveTopology>,
    /*
    === COMMIT STORAGE ===
    The DAG of all commits in the system
    */
    /// All commits indexed by their state ID
    /// Each commit contains: parent_id, delta, metadata
    commits: HashMap<StateId, Arc<Commit>>,

    /// Content-addressed storage for deltas (automatic deduplication)
    /// If two commits have identical changes, they share the same delta
    deltas: HashMap<[u8; 32], Arc<Delta>>,
    // Rare exact-payload collisions retain legacy logical hash compatibility.
    delta_collisions: HashMap<[u8; 32], Vec<Arc<Delta>>>,

    /*
    === BRANCH MANAGEMENT ===
    Git-like lightweight branches
    */
    /// All branches: branch_name -> head_commit_id
    branches: HashMap<BranchName, StateId>,

    /// Tags (named references to specific commits)
    tags: HashMap<String, StateId>,

    /*
    === GRAPH STRUCTURE TRACKING ===
    Efficiently navigate the commit DAG
    */
    /// Parent-to-children index for efficient traversal
    /// Maps commit_id -> Vec<child_commit_ids>
    children: HashMap<StateId, Vec<StateId>>,

    /// Root commits (commits with no parent)
    roots: HashSet<StateId>,

    /*
    === ID MANAGEMENT ===
    */
    /// Next available state ID
    next_state_id: StateId,
    checkpoints: Vec<crate::Checkpoint>,
}

impl HistoryForest {
    /// Copy retained history into a new owner with distinct write provenance.
    /// Immutable commit/delta payloads are shared; references and caches are independent.
    pub(crate) fn detached_copy(&self) -> Self {
        Self {
            identity: Arc::new(()),
            graph_type: self.graph_type,
            read_context_cache: Default::default(),
            effective_topology: Default::default(),
            commits: self.commits.clone(),
            deltas: self.deltas.clone(),
            delta_collisions: self.delta_collisions.clone(),
            branches: self.branches.clone(),
            tags: self.tags.clone(),
            children: self.children.clone(),
            roots: self.roots.clone(),
            next_state_id: self.next_state_id,
            checkpoints: self.checkpoints.clone(),
        }
    }

    pub(crate) fn identity(&self) -> &Arc<()> {
        &self.identity
    }

    /// Reserved next acceptance ID while the caller holds exclusive graph ownership.
    pub(crate) fn retained_version_count(&self) -> usize {
        self.commits.len()
    }

    pub(crate) fn next_state_id(&self) -> StateId {
        self.next_state_id
    }

    /// Explicit checkpoints in acceptance order, independent of version IDs.
    pub fn checkpoints(&self) -> &[crate::Checkpoint] {
        &self.checkpoints
    }

    pub(crate) fn reserve_checkpoint(&mut self) -> crate::GraphResult<()> {
        self.checkpoints.try_reserve(1).map_err(|error| {
            GraphError::InvalidInput(format!("Cannot reserve checkpoint metadata: {error}"))
        })
    }
    pub(crate) fn record_checkpoint(&mut self, record: crate::Checkpoint) {
        self.checkpoints.push(record);
    }

    /// Create a new empty history system
    pub fn new() -> Self {
        Self::new_with_type(GraphType::default())
    }

    // Failure to prove exact identity means a separate payload, never a failed
    // ordinary publication after its storage writes have already executed.
    pub(crate) fn has_exact_delta_collisions(&self) -> bool {
        !self.delta_collisions.is_empty()
    }

    fn intern_delta(&mut self, delta: Delta) -> Arc<Delta> {
        let hash = delta.content_hash;
        let Some(existing) = self.deltas.get(&hash) else {
            let delta = Arc::new(delta);
            self.deltas.insert(hash, Arc::clone(&delta));
            return delta;
        };
        if matches!(exact_delta(existing, &delta), Ok(true)) {
            return Arc::clone(existing);
        }
        for existing in self.delta_collisions.get(&hash).into_iter().flatten() {
            if matches!(exact_delta(existing, &delta), Ok(true)) {
                return Arc::clone(existing);
            }
        }
        let delta = Arc::new(delta);
        self.delta_collisions
            .entry(hash)
            .or_default()
            .push(Arc::clone(&delta));
        delta
    }

    /// Create history for a graph with explicit directionality.
    pub fn new_with_type(graph_type: GraphType) -> Self {
        let mut branches = HashMap::new();
        branches.insert("main".to_string(), 0); // main branch starts at state 0

        Self {
            effective_topology: Default::default(),
            read_context_cache: Default::default(),
            identity: Arc::new(()),
            graph_type,
            commits: HashMap::new(),
            deltas: HashMap::new(),
            delta_collisions: HashMap::new(),
            branches,
            tags: HashMap::new(),
            children: HashMap::new(),
            roots: HashSet::new(),
            checkpoints: Vec::new(),
            next_state_id: 1, // 0 is reserved for empty state
        }
    }

    /*
    === COMMIT OPERATIONS ===
    Creating new points in history
    */

    /// Create a new commit from a set of changes
    ///
    /// ALGORITHM:
    /// 1. Create Delta from the changes
    /// 2. Compute content hash of delta for deduplication
    /// 3. Check if we already have this exact delta (hash collision -> reuse)
    /// 4. Create Commit with metadata (parent, message, author, timestamp)
    /// 5. Update parent-child relationships
    /// 6. Store everything and return new state ID
    pub fn create_commit(
        &mut self,
        changes: ChangeSet,
        message: String,
        author: String,
        parent: Option<StateId>,
    ) -> Result<StateId, GraphError> {
        self.create_commit_with_publication(changes, message, author, parent, None)
    }

    /// Create a version with optional engine publication/effective-time metadata.
    pub fn create_commit_with_publication(
        &mut self,
        changes: ChangeSet,
        message: String,
        author: String,
        parent: Option<StateId>,
        mut publication: Option<crate::PublicationMetadata>,
    ) -> Result<StateId, GraphError> {
        if let Some(metadata) = &mut publication {
            use crate::ChangeKey;
            let mut actual = std::collections::BTreeMap::new();
            let mut add = |key: ChangeKey| {
                actual.insert(
                    key.clone(),
                    metadata
                        .change_times
                        .get(&key)
                        .copied()
                        .unwrap_or(metadata.effective_at),
                );
            };
            for id in &changes.nodes_added {
                add(ChangeKey::NodeAdded(*id));
            }
            for id in &changes.nodes_removed {
                add(ChangeKey::NodeRemoved(*id));
            }
            for (id, _, _) in &changes.edges_added {
                add(ChangeKey::EdgeAdded(*id));
            }
            for id in &changes.edges_removed {
                add(ChangeKey::EdgeRemoved(*id));
            }
            for change in &changes.edge_endpoint_changes {
                add(ChangeKey::EdgeEndpoints(change.edge_id));
            }
            for (id, name, _, _) in &changes.node_attr_changes {
                add(ChangeKey::NodeAttribute(*id, name.clone()));
            }
            for (id, name, _, _) in &changes.edge_attr_changes {
                add(ChangeKey::EdgeAttribute(*id, name.clone()));
            }
            metadata.change_times = actual;
        }
        // 1. Create Delta from the changes
        let delta = Delta::from_changes(changes);

        // A legacy logical hash may normalize float bits. Reuse only an exact
        // payload match, not merely the matching bucket key.
        let delta_arc = self.intern_delta(delta);

        // 4. Create Commit with metadata
        let parents = match parent {
            Some(p) => vec![p],
            None => vec![],
        };
        let commit = Commit::with_publication(
            self.next_state_id,
            parents,
            delta_arc.clone(),
            message,
            author,
            publication,
        );

        // 5. Store commit
        let commit_id = self.next_state_id;
        self.effective_topology.get_mut().append_if_current(&commit);
        self.commits.insert(commit_id, Arc::new(commit));

        // 6. Update parent-child relationships
        if let Some(parent_id) = parent {
            self.children
                .entry(parent_id)
                .or_insert_with(Vec::new)
                .push(commit_id);
        } else {
            // This is a root commit
            self.roots.insert(commit_id);
        }

        // 7. Update state ID counter
        self.next_state_id += 1;

        // 8. Return new state ID
        Ok(commit_id)
    }

    /// Create a merge commit (commit with multiple parents)
    pub fn create_merge_commit(
        &mut self,
        changes: ChangeSet,
        message: String,
        author: String,
        parents: Vec<StateId>,
    ) -> Result<StateId, GraphError> {
        self.create_merge_commit_with_publication(changes, message, author, parents, None)
    }

    /// Accept an already resolved delta relative to the first parent. Conflict
    /// detection/policy belongs to the merge planner, independently of replay.
    pub fn create_merge_commit_with_publication(
        &mut self,
        changes: ChangeSet,
        message: String,
        author: String,
        parents: Vec<StateId>,
        publication: Option<crate::PublicationMetadata>,
    ) -> GraphResult<StateId> {
        if parents.len() < 2 {
            return Err(GraphError::InvalidInput(
                "Merge requires at least two parents".into(),
            ));
        }
        let mut seen = HashSet::new();
        for &parent in &parents {
            if !seen.insert(parent) {
                return Err(GraphError::InvalidInput("Duplicate merge parent".into()));
            }
            self.get_commit(parent)?;
        }
        let id = self.create_commit_with_publication(
            changes,
            message,
            author,
            Some(parents[0]),
            publication,
        )?;
        let old = self.get_commit(id)?;
        let commit = Commit::with_publication(
            id,
            parents.clone(),
            old.delta.clone(),
            old.message.clone(),
            old.author.clone(),
            old.publication.clone(),
        );
        self.commits.insert(id, Arc::new(commit));
        for &parent in &parents[1..] {
            self.children.entry(parent).or_default().push(id);
        }
        Ok(id)
    }

    /*
    === BRANCH OPERATIONS ===
    Git-like branch management
    */

    /// Create a new branch pointing to a specific commit
    pub fn create_branch(
        &mut self,
        name: BranchName,
        commit_id: StateId,
    ) -> Result<(), GraphError> {
        // 1. Validate that commit_id exists
        if commit_id != 0 && !self.commits.contains_key(&commit_id) {
            return Err(GraphError::InvalidInput(format!(
                "Commit {} does not exist",
                commit_id
            )));
        }

        // 2. Check that branch name doesn't already exist
        if self.branches.contains_key(&name) {
            return Err(GraphError::InvalidInput(format!(
                "Branch '{}' already exists",
                name
            )));
        }

        // 3. Create the branch
        self.branches.insert(name, commit_id);
        Ok(())
    }

    /// Delete a branch (but not the commits it pointed to)
    pub fn delete_branch(&mut self, name: &BranchName) -> Result<(), GraphError> {
        // 1. Check that branch exists
        if !self.branches.contains_key(name) {
            return Err(GraphError::InvalidInput(format!(
                "Branch '{}' does not exist",
                name
            )));
        }

        // 2. Check that it's not the main branch (safety check)
        if name == "main" {
            return Err(GraphError::InvalidInput(
                "Cannot delete main branch".to_string(),
            ));
        }

        // 3. Remove the branch
        self.branches.remove(name);
        Ok(())
    }

    /// Update a branch to point to a different commit (e.g., after new commit)
    pub fn update_branch_head(
        &mut self,
        name: &BranchName,
        new_head: StateId,
    ) -> Result<(), GraphError> {
        // 1. Validate branch exists and commit exists
        if !self.branches.contains_key(name) {
            return Err(GraphError::InvalidInput(format!(
                "Branch '{}' does not exist",
                name
            )));
        }
        if !self.commits.contains_key(&new_head) {
            return Err(GraphError::InvalidInput(format!(
                "Commit {} does not exist",
                new_head
            )));
        }

        // 2. Update branch head
        self.branches.insert(name.clone(), new_head);
        Ok(())
    }

    /// List all branches with their head commits
    pub fn list_branches(&self) -> Vec<BranchInfo> {
        self.branches
            .iter()
            .map(|(name, &head)| {
                BranchInfo {
                    name: name.clone(),
                    head,
                    description: None, // TODO: add descriptions to branches
                    created_at: 0,     // TODO: track creation time
                    created_by: "".to_string(), // TODO: track creator
                    is_default: name == "main",
                    is_current: false, // TODO: we need to track current branch in HistoryForest
                }
            })
            .collect()
    }

    /// Get the head commit of a branch
    pub fn get_branch_head(&self, name: &BranchName) -> Result<StateId, GraphError> {
        self.branches
            .get(name)
            .copied()
            .ok_or_else(|| GraphError::InvalidInput(format!("Branch '{}' not found", name)))
    }

    /*
    === COMMIT QUERIES ===
    Navigating and inspecting the commit history
    */

    /// Get a specific commit by ID
    pub fn get_commit(&self, state_id: StateId) -> Result<Arc<Commit>, GraphError> {
        self.commits
            .get(&state_id)
            .cloned()
            .ok_or_else(|| GraphError::InvalidInput(format!("Commit {} not found", state_id)))
    }

    /// Get all commits in chronological order
    pub fn get_commit_history(&self) -> Vec<Arc<Commit>> {
        let mut commits: Vec<Arc<Commit>> = self.commits.values().cloned().collect();
        commits.sort_by_key(|commit| (commit.timestamp, commit.id));
        commits
    }

    /// Find the commit whose timestamp is <= provided value and closest to it.
    pub fn commit_at_or_before(&self, timestamp: u64) -> Option<StateId> {
        self.commits
            .values()
            .filter(|commit| commit.timestamp <= timestamp)
            .max_by_key(|commit| (commit.timestamp, commit.id))
            .map(|commit| commit.id)
    }

    /// Select a recorded timestamp only on the retained first-parent lineage.
    /// Equal second-resolution timestamps use accepted state ID order.
    pub fn commit_at_or_before_on_lineage(
        &self,
        head: StateId,
        timestamp: u64,
    ) -> GraphResult<Option<StateId>> {
        Ok(self.read_context(head)?.commit_at_or_before(timestamp))
    }

    /// Get the commit history for a specific branch (following parent chain)
    pub fn get_branch_history(
        &self,
        branch_name: &BranchName,
    ) -> Result<Vec<Arc<Commit>>, GraphError> {
        // 1. Start from branch head
        let head_id = self.get_branch_head(branch_name)?;
        let mut history = Vec::new();
        let mut to_visit = vec![head_id];
        let mut visited = HashSet::new();

        // 2. Follow parent chain to roots using BFS
        while let Some(commit_id) = to_visit.pop() {
            if visited.contains(&commit_id) {
                continue;
            }
            visited.insert(commit_id);

            if let Some(commit) = self.commits.get(&commit_id) {
                history.push(commit.clone());
                // Add parents to visit queue
                for &parent_id in &commit.parents {
                    if !visited.contains(&parent_id) {
                        to_visit.push(parent_id);
                    }
                }
            }
        }

        // 3. Sort by timestamp (most recent first)
        history.sort_by_key(|commit| std::cmp::Reverse((commit.timestamp, commit.id)));
        Ok(history)
    }

    /// Get all children of a commit
    pub fn get_children(&self, commit_id: StateId) -> Vec<StateId> {
        self.children.get(&commit_id).cloned().unwrap_or_default()
    }

    /// Get the parent(s) of a commit
    pub fn get_parents(&self, commit_id: StateId) -> Result<Vec<StateId>, GraphError> {
        let commit = self.get_commit(commit_id)?;
        Ok(commit.parents.clone())
    }

    /// Check ancestry between retained published versions; unknown/zero IDs reject.
    pub fn is_ancestor(&self, ancestor: StateId, descendant: StateId) -> Result<bool, GraphError> {
        self.get_commit(ancestor)?;
        self.get_commit(descendant)?;
        if ancestor == descendant {
            return Ok(true);
        }

        let mut to_visit = vec![descendant];
        let mut visited = HashSet::new();

        // Traverse all parents; visit each retained version at most once.
        while let Some(commit_id) = to_visit.pop() {
            if visited.contains(&commit_id) {
                continue;
            }
            visited.insert(commit_id);

            if let Some(commit) = self.commits.get(&commit_id) {
                for &parent_id in &commit.parents {
                    if parent_id == ancestor {
                        return Ok(true);
                    }
                    if !visited.contains(&parent_id) {
                        to_visit.push(parent_id);
                    }
                }
            }
        }

        Ok(false)
    }

    /// All best common ancestors, sorted by state ID. A common ancestor is
    /// excluded when another common ancestor descends from it. Criss-cross merges
    /// can have several equally valid bases; none is selected by traversal order.
    /// Unknown states (including the unpublished zero root) reject explicitly.
    pub fn merge_bases(&self, left: StateId, right: StateId) -> GraphResult<Vec<StateId>> {
        let ancestors = |start: StateId| -> GraphResult<HashSet<StateId>> {
            self.get_commit(start)?;
            let mut seen = HashSet::new();
            let mut pending = vec![start];
            while let Some(id) = pending.pop() {
                if seen.insert(id) {
                    pending.extend(self.get_commit(id)?.parents.iter().copied());
                }
            }
            Ok(seen)
        };
        let left = ancestors(left)?;
        let right = ancestors(right)?;
        let common: HashSet<_> = left.intersection(&right).copied().collect();
        // Common ancestry is closed under parents, so each non-best ancestor is
        // the direct parent of another common ancestor. One pass avoids pairwise
        // ancestry searches even for long shared histories.
        let mut shadowed = HashSet::new();
        for &id in &common {
            for &parent in &self.get_commit(id)?.parents {
                if common.contains(&parent) {
                    shadowed.insert(parent);
                }
            }
        }
        let mut best: Vec<_> = common.difference(&shadowed).copied().collect();
        best.sort_unstable();
        Ok(best)
    }

    /// Return the unique best common ancestor, or None for unrelated roots.
    /// Ambiguous criss-cross bases reject so callers can choose an explicit base
    /// or implement a separately defined recursive-base policy.
    pub fn find_common_ancestor(
        &self,
        commit1: StateId,
        commit2: StateId,
    ) -> Result<Option<StateId>, GraphError> {
        let bases = self.merge_bases(commit1, commit2)?;
        match bases.as_slice() {
            [] => Ok(None),
            [base] => Ok(Some(*base)),
            _ => Err(GraphError::InvalidInput(format!(
                "Merge has multiple best common ancestors {bases:?}; choose an explicit base"
            ))),
        }
    }

    /*
    === DIFF OPERATIONS ===
    Comparing different points in history
    */

    /// Compute the changes between two commits
    pub fn diff_commits(&self, from: StateId, to: StateId) -> Result<CommitDiff, GraphError> {
        self.prepare_commit_diff(from, to)?.execute()
    }

    /// Get the delta (direct changes) introduced by a specific commit
    pub fn get_commit_delta(&self, commit_id: StateId) -> Result<Arc<Delta>, GraphError> {
        let commit = self.get_commit(commit_id)?;
        Ok(commit.delta.clone())
    }

    /*
    === STATE RECONSTRUCTION ===
    Building a graph state from history
    */

    /// Reconstruct the complete graph state at a specific commit
    /// This is expensive but necessary for time-travel functionality
    pub fn reconstruct_state_at(&self, commit_id: StateId) -> Result<GraphSnapshot, GraphError> {
        self.prepare_recorded_read(commit_id)?.execute()
    }

    /// Shared recorded/effective first-parent context and lazy event index.
    /// A merge's delta is its accepted resolution
    /// relative to the first parent; sibling mutations are not replayed twice.
    pub(crate) fn read_context(
        &self,
        head: StateId,
    ) -> GraphResult<Arc<crate::state::effective_read::ReadContext>> {
        let mut cached = self.read_context_cache.lock();
        if let Some((state, context)) = cached.as_ref() {
            if *state == head {
                return Ok(context.clone());
            }
        }
        let mut commits = Vec::new();
        let mut seen = HashSet::new();
        let mut state = head;
        while state != 0 {
            if !seen.insert(state) {
                return Err(GraphError::InvalidInput(
                    "Cycle in effective read ancestry".into(),
                ));
            }
            let commit = self.get_commit(state)?;
            state = commit.parents.first().copied().unwrap_or(0);
            commits.push(commit);
        }
        commits.reverse();
        let context = Arc::new(crate::state::effective_read::ReadContext::new(commits));
        *cached = Some((head, context.clone()));
        Ok(context)
    }
    /// Prepare a detached temporal read of a selected first-parent lineage.
    pub fn prepare_effective_read(
        &self,
        head: StateId,
        effective_at: i64,
        recorded_by: Option<i64>,
        legacy_publications: std::collections::BTreeMap<StateId, crate::PublicationMetadata>,
    ) -> GraphResult<crate::state::effective_read::EffectiveReadPlan> {
        let context = self.read_context(head)?;
        context.validate_legacy(&legacy_publications)?;
        let legacy_clocks = legacy_publications
            .iter()
            .map(|(&id, time)| (id, (time.effective_at, time.recorded_at)))
            .collect();
        Ok(crate::state::effective_read::EffectiveReadPlan {
            context,
            graph_type: self.graph_type,
            time: crate::state::effective_read::ReadTime {
                branch: "detached".into(),
                published_head: head,
                effective_at,
                recorded_by,
                legacy_clocks,
                group_sources: Default::default(),
            },
            groups: Default::default(),
            legacy_publications,
        })
    }

    /// Get the sequence of deltas needed to go from one commit to another
    pub fn get_delta_sequence(
        &self,
        from: StateId,
        to: StateId,
    ) -> Result<Vec<Arc<Delta>>, GraphError> {
        let _ = (from, to); // Silence unused warnings
                            // Basic implementation returns empty sequence
        Ok(Vec::new())
    }

    /*
    === GARBAGE COLLECTION ===
    Cleaning up unreachable history
    */

    /// Find all commits reachable from branch/tag references, including merge parents.
    pub fn find_reachable_commits(&self) -> HashSet<StateId> {
        self.reachable_commits(None, &HashSet::new())
    }

    fn reachable_commits(
        &self,
        excluded_branch: Option<&str>,
        extra: &HashSet<StateId>,
    ) -> HashSet<StateId> {
        let mut keep = HashSet::new();
        let mut pending: Vec<_> = self
            .branches
            .iter()
            .filter(|(name, _)| Some(name.as_str()) != excluded_branch)
            .map(|(_, &head)| head)
            .chain(self.tags.values().copied())
            .chain(extra.iter().copied())
            .collect();
        while let Some(state) = pending.pop() {
            if let Some(commit) = self.commits.get(&state) {
                if keep.insert(state) {
                    pending.extend(commit.parents.iter().copied());
                }
            }
        }
        keep
    }

    pub(crate) fn collection_plan(
        &self,
        excluded_branch: Option<&str>,
        extra: &HashSet<StateId>,
    ) -> (HashSet<StateId>, Vec<StateId>) {
        let keep = self.reachable_commits(excluded_branch, extra);
        let mut removed: Vec<_> = self
            .commits
            .keys()
            .filter(|state| !keep.contains(state))
            .copied()
            .collect();
        removed.sort_unstable();
        (keep, removed)
    }

    /// Explicit history collection. Branches/tags and ancestry of additional
    /// requested states are retained. Open immutable payloads keep their own Arcs.
    pub fn garbage_collect(&mut self, keep_commits: &HashSet<StateId>) -> usize {
        let keep = self.reachable_commits(None, keep_commits);
        self.retain_commits(&keep)
    }

    pub(crate) fn retain_commits(&mut self, keep: &HashSet<StateId>) -> usize {
        let before = self.commits.len();
        self.commits.retain(|state, _| keep.contains(state));
        let removed = before - self.commits.len();
        if removed == 0 {
            return 0;
        }
        let retained = &self.commits;
        self.deltas.clear();
        self.delta_collisions.clear();
        let mut seen = HashSet::new();
        for commit in retained.values() {
            if !seen.insert(Arc::as_ptr(&commit.delta)) {
                continue;
            }
            let hash = commit.delta.content_hash;
            match self.deltas.entry(hash) {
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(Arc::clone(&commit.delta));
                }
                std::collections::hash_map::Entry::Occupied(_) => {
                    self.delta_collisions
                        .entry(hash)
                        .or_default()
                        .push(Arc::clone(&commit.delta));
                }
            }
        }
        self.children.retain(|parent, children| {
            children.retain(|child| retained.contains_key(child));
            retained.contains_key(parent)
        });
        self.roots.retain(|state| retained.contains_key(state));
        // Checkpoint labels are metadata, not permanent retention pins (A09).
        self.checkpoints.retain(|checkpoint| {
            checkpoint.state_id == 0 || retained.contains_key(&checkpoint.state_id)
        });
        *self.effective_topology.get_mut() = Default::default();
        *self.read_context_cache.get_mut() = None;
        // Acceptance IDs are lifetime reservations and never rewind after collection.
        removed
    }

    /*
    === STATISTICS AND INTROSPECTION ===
    */

    /// Get statistics about the history system
    pub fn statistics(&self) -> HistoryStatistics {
        // Find oldest and newest commits
        let mut oldest_timestamp = u64::MAX;
        let mut newest_timestamp = 0;

        for commit in self.commits.values() {
            oldest_timestamp = oldest_timestamp.min(commit.timestamp);
            newest_timestamp = newest_timestamp.max(commit.timestamp);
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let oldest_age = if oldest_timestamp == u64::MAX {
            0
        } else {
            now.saturating_sub(oldest_timestamp)
        };
        let newest_age = now.saturating_sub(newest_timestamp);

        // Calculate storage efficiency (higher is better)
        let total_commits = self.commits.len();
        let unique_deltas =
            self.deltas.len() + self.delta_collisions.values().map(Vec::len).sum::<usize>();
        let storage_efficiency = if total_commits > 0 {
            unique_deltas as f64 / total_commits as f64
        } else {
            1.0
        };

        HistoryStatistics {
            total_commits,
            total_branches: self.branches.len(),
            total_tags: self.tags.len(),
            total_deltas: unique_deltas,
            storage_efficiency,
            oldest_commit_age: oldest_age,
            newest_commit_age: newest_age,
        }
    }

    /// List all commit IDs in the system
    pub fn list_all_commits(&self) -> Vec<StateId> {
        self.commits.keys().copied().collect()
    }

    /// Check if a commit exists
    pub fn has_commit(&self, commit_id: StateId) -> bool {
        self.commits.contains_key(&commit_id)
    }
}

/*
=== SUPPORTING DATA STRUCTURES ===
*/

/// A single commit in the history DAG
///
/// DESIGN: Immutable once created, reference-counted for sharing
#[derive(Debug, Clone)]
pub struct Commit {
    /// Unique identifier for this commit
    pub id: StateId,

    /// Parent commit(s) - None for root commits, Vec for merge commits
    pub parents: Vec<StateId>,

    /// The changes introduced by this commit
    pub delta: Arc<Delta>,

    /// Human-readable commit message
    pub message: String,

    /// Who created this commit
    pub author: String,

    /// When this commit was created (Unix timestamp)
    pub timestamp: u64,

    /// Automatic publication metadata; absent on legacy manually grouped writes.
    pub publication: Option<crate::PublicationMetadata>,

    /// Content hash for verification and deduplication
    pub content_hash: [u8; 32],
}

impl Commit {
    /// Create a new commit
    pub fn new(
        id: StateId,
        parents: Vec<StateId>,
        delta: Arc<Delta>,
        message: String,
        author: String,
    ) -> Self {
        Self::with_publication(id, parents, delta, message, author, None)
    }

    fn with_publication(
        id: StateId,
        parents: Vec<StateId>,
        delta: Arc<Delta>,
        message: String,
        author: String,
        publication: Option<crate::PublicationMetadata>,
    ) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        use std::time::{SystemTime, UNIX_EPOCH};

        // The legacy seconds field uses the same captured engine clock for batches.
        let timestamp = publication
            .as_ref()
            .map(|metadata| (metadata.recorded_at.max(0) as u64) / 1_000_000)
            .unwrap_or_else(|| {
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0)
            });

        // Compute content hash from all fields
        let mut hasher = DefaultHasher::new();
        id.hash(&mut hasher);
        parents.hash(&mut hasher);
        delta.content_hash.hash(&mut hasher);
        message.hash(&mut hasher);
        author.hash(&mut hasher);
        timestamp.hash(&mut hasher);
        publication.hash(&mut hasher);
        let hash_u64 = hasher.finish();

        // Convert u64 hash to [u8; 32] (simple but deterministic)
        let mut content_hash = [0u8; 32];
        let hash_bytes = hash_u64.to_le_bytes();
        for i in 0..4 {
            content_hash[i * 8..(i + 1) * 8].copy_from_slice(&hash_bytes);
        }

        Self {
            id,
            parents,
            delta,
            message,
            author,
            timestamp,
            publication,
            content_hash,
        }
    }

    /// Check if this is a root commit (no parents)
    pub fn is_root(&self) -> bool {
        self.parents.is_empty()
    }

    /// Check if this is a merge commit (multiple parents)
    pub fn is_merge(&self) -> bool {
        self.parents.len() > 1
    }
}

/// A delta represents the changes introduced by a single commit
///
/// DESIGN: Immutable, content-addressed for deduplication
#[derive(Debug, Clone)]
pub struct Delta {
    /// Hash of this delta's content (for deduplication)
    pub content_hash: [u8; 32],

    /// Nodes that were added in this commit
    pub nodes_added: Vec<NodeId>,

    /// Nodes that were removed in this commit  
    pub nodes_removed: Vec<NodeId>,

    /// Edges that were added in this commit
    pub edges_added: Vec<(EdgeId, NodeId, NodeId)>,

    /// Edges that were removed in this commit
    pub edges_removed: Vec<EdgeId>,

    /// Stable edge IDs with typed before/after endpoints.
    pub edge_endpoint_changes: Vec<EdgeEndpointChange>,

    /// Node attribute changes: (node_id, attr_name, old_value, new_value)
    pub node_attr_changes: Vec<(NodeId, AttrName, Option<AttrValue>, Option<AttrValue>)>,

    /// Edge attribute changes: (edge_id, attr_name, old_value, new_value)
    pub edge_attr_changes: Vec<(EdgeId, AttrName, Option<AttrValue>, Option<AttrValue>)>,
}

impl Delta {
    /// Create a delta from a change set
    pub fn from_changes(mut changes: ChangeSet) -> Self {
        // Stable hashes/replay must not depend on HashMap iteration seeds.
        changes.nodes_added.sort_unstable();
        changes.nodes_removed.sort_unstable();
        changes.edges_added.sort_unstable_by_key(|(id, _, _)| *id);
        changes.edges_removed.sort_unstable();
        changes
            .edge_endpoint_changes
            .sort_unstable_by_key(|c| c.edge_id);
        changes
            .node_attr_changes
            .sort_unstable_by(|a, b| (a.0, &a.1).cmp(&(b.0, &b.1)));
        changes
            .edge_attr_changes
            .sort_unstable_by(|a, b| (a.0, &a.1).cmp(&(b.0, &b.1)));
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Move owned buffers; hashing below only borrows them. No property/vector
        // payload is cloned during conversion to an immutable delta.
        let ChangeSet {
            nodes_added,
            nodes_removed,
            edges_added,
            edges_removed,
            edge_endpoint_changes,
            node_attr_changes,
            edge_attr_changes,
        } = changes;

        // Compute content hash from all changes
        let mut hasher = DefaultHasher::new();
        nodes_added.hash(&mut hasher);
        nodes_removed.hash(&mut hasher);
        edges_added.hash(&mut hasher);
        edges_removed.hash(&mut hasher);
        // Preserve existing assignment-only hashes. Absence gets a distinct
        // tag outside the AttrValue hash domain, never a fake Null value.
        for changes in [&node_attr_changes, &edge_attr_changes] {
            changes.len().hash(&mut hasher);
            for (id, name, old, new) in changes {
                id.hash(&mut hasher);
                name.hash(&mut hasher);
                old.hash(&mut hasher);
                match new {
                    Some(value) => value.hash(&mut hasher),
                    None => "groggy:unset:v1".hash(&mut hasher),
                }
            }
        }
        if !edge_endpoint_changes.is_empty() {
            "groggy:endpoint:v1".hash(&mut hasher);
            edge_endpoint_changes.hash(&mut hasher);
        }
        let hash_u64 = hasher.finish();

        // Convert u64 hash to [u8; 32]
        let mut content_hash = [0u8; 32];
        let hash_bytes = hash_u64.to_le_bytes();
        for i in 0..4 {
            content_hash[i * 8..(i + 1) * 8].copy_from_slice(&hash_bytes);
        }

        Self {
            content_hash,
            nodes_added,
            nodes_removed,
            edges_added,
            edges_removed,
            edge_endpoint_changes,
            node_attr_changes,
            edge_attr_changes,
        }
    }

    /// Check if this delta is empty (no changes)
    pub fn is_empty(&self) -> bool {
        self.nodes_added.is_empty()
            && self.nodes_removed.is_empty()
            && self.edges_added.is_empty()
            && self.edges_removed.is_empty()
            && self.edge_endpoint_changes.is_empty()
            && self.node_attr_changes.is_empty()
            && self.edge_attr_changes.is_empty()
    }

    /// Get a summary of what changed
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();

        if !self.nodes_added.is_empty() {
            parts.push(format!("+{} nodes", self.nodes_added.len()));
        }
        if !self.nodes_removed.is_empty() {
            parts.push(format!("-{} nodes", self.nodes_removed.len()));
        }
        if !self.edges_added.is_empty() {
            parts.push(format!("+{} edges", self.edges_added.len()));
        }
        if !self.edges_removed.is_empty() {
            parts.push(format!("-{} edges", self.edges_removed.len()));
        }

        if !self.edge_endpoint_changes.is_empty() {
            parts.push(format!(
                "{} endpoint changes",
                self.edge_endpoint_changes.len()
            ));
        }
        let total_attr_changes = self.node_attr_changes.len() + self.edge_attr_changes.len();
        if total_attr_changes > 0 {
            parts.push(format!("{} attr changes", total_attr_changes));
        }

        if parts.is_empty() {
            "no changes".to_string()
        } else {
            parts.join(", ")
        }
    }
}

/// Difference between two commits
#[derive(Debug, Clone)]
pub struct CommitDiff {
    pub from_commit: StateId,
    pub to_commit: StateId,
    pub nodes_added: Vec<NodeId>,
    pub nodes_removed: Vec<NodeId>,
    pub edges_added: Vec<(EdgeId, NodeId, NodeId)>,
    pub edges_removed: Vec<EdgeId>,
    /// Endpoint changes of edge IDs present in both selected states.
    pub edge_endpoint_changes: Vec<EdgeEndpointChange>,
    pub attr_changes: Vec<AttributeChange>,
}

/// Statistics about the history system
#[derive(Debug, Clone)]
pub struct HistoryStatistics {
    pub total_commits: usize,
    pub total_branches: usize,
    pub total_tags: usize,
    pub total_deltas: usize,
    pub storage_efficiency: f64, // How much deduplication we achieved
    pub oldest_commit_age: u64,  // Seconds since oldest commit
    pub newest_commit_age: u64,  // Seconds since newest commit
}

impl Default for HistoryForest {
    fn default() -> Self {
        Self::new()
    }
}

/*
=== HISTORICAL VIEW SYSTEM ===
Read-only views of the graph at specific points in history
*/

/// Read-only view of a graph at a specific point in history
///
/// DESIGN: This provides a read-only interface to a historical graph state.
/// It uses lazy loading to reconstruct the state only when needed, and
/// caches the results for subsequent access.
///
/// LIFETIME: The view holds a reference to the history system, so it
/// cannot outlive the system that created it.
///
/// PERFORMANCE:
/// - First access triggers state reconstruction: O(depth * changes)
/// - Subsequent accesses use cached data: O(1) or O(log n)
/// - Memory usage: Only stores what's been accessed
#[derive(Debug)]
pub struct HistoricalView<'a> {
    /*
    === HISTORY REFERENCE ===
    Connection to the history system for data access
    */
    /// Reference to the history system that contains the state data
    /// LIFETIME: View cannot outlive this reference
    history: &'a HistoryForest,

    /// The specific state ID this view represents
    /// IMMUTABLE: Views always represent the same state
    state_id: StateId,

    /*
    === CACHED RECONSTRUCTION ===
    Lazily-built snapshot of the graph at this state
    */
    /// Cached snapshot of the complete graph state
    /// LAZY: Only built when first accessed
    /// CACHING: Kept for subsequent operations
    cached_snapshot: Option<GraphSnapshot>,

    /*
    === ACCESS OPTIMIZATION ===
    Tracking what's been loaded for selective reconstruction
    */
    /// Which node attributes have been loaded
    /// OPTIMIZATION: Avoid loading unused attributes
    loaded_node_attrs: HashMap<AttrName, bool>,

    /// Which edge attributes have been loaded
    /// OPTIMIZATION: Avoid loading unused attributes  
    loaded_edge_attrs: HashMap<AttrName, bool>,
}

impl<'a> HistoricalView<'a> {
    /// Create a new historical view for a specific commit
    ///
    /// VALIDATION: Ensures the state exists in the history
    /// LAZY: Doesn't reconstruct the state until needed
    pub fn new(history: &'a HistoryForest, state_id: StateId) -> GraphResult<Self> {
        if !history.has_commit(state_id) {
            return Err(GraphError::InvalidInput(format!(
                "Commit {} does not exist in history",
                state_id
            )));
        }

        Ok(Self {
            history,
            state_id,
            cached_snapshot: None,
            loaded_node_attrs: HashMap::new(),
            loaded_edge_attrs: HashMap::new(),
        })
    }

    /*
    === SNAPSHOT MANAGEMENT ===
    Lazy loading and caching of the complete graph state
    */

    /// Get the complete snapshot, reconstructing if necessary
    ///
    /// ALGORITHM:
    /// 1. If snapshot is cached, return it
    /// 2. Otherwise, reconstruct from history deltas
    /// 3. Cache the result for future use
    /// 4. Return reference to cached snapshot
    ///
    /// PERFORMANCE: O(depth * changes) first time, O(1) after caching
    fn get_snapshot(&mut self) -> GraphResult<&GraphSnapshot> {
        if self.cached_snapshot.is_none() {
            let snapshot = self.history.reconstruct_state_at(self.state_id)?;
            self.cached_snapshot = Some(snapshot);
        }

        Ok(self.cached_snapshot.as_ref().unwrap())
    }

    /// Clear the cached snapshot to free memory
    ///
    /// USAGE: Call this if memory usage is a concern and the view
    /// won't be accessed again soon
    pub fn clear_cache(&mut self) {
        self.cached_snapshot = None;
        self.loaded_node_attrs.clear();
        self.loaded_edge_attrs.clear();
    }

    /// Check if the snapshot is currently cached
    pub fn is_cached(&self) -> bool {
        self.cached_snapshot.is_some()
    }

    /*
    === NODE OPERATIONS ===
    Read-only access to nodes and their attributes
    */

    /// Get all active node IDs at this state
    pub fn get_node_ids(&mut self) -> GraphResult<Vec<NodeId>> {
        let snapshot = self.get_snapshot()?;
        Ok(snapshot.active_nodes.clone())
    }

    /// Check if a specific node exists at this state
    pub fn has_node(&mut self, node: NodeId) -> GraphResult<bool> {
        let snapshot = self.get_snapshot()?;
        Ok(snapshot.contains_node(node))
    }

    /// Get the number of active nodes at this state
    pub fn node_count(&mut self) -> GraphResult<usize> {
        let snapshot = self.get_snapshot()?;
        Ok(snapshot.active_nodes.len())
    }

    /// Get a specific attribute value for a node
    ///
    /// OPTIMIZATION: This could potentially avoid loading the full snapshot
    /// by reconstructing only the requested attribute
    pub fn get_node_attribute(
        &mut self,
        node: NodeId,
        attr_name: &AttrName,
    ) -> GraphResult<Option<AttrValue>> {
        let snapshot = self.get_snapshot()?;
        if !snapshot.contains_node(node) {
            return Err(GraphError::NodeNotFound {
                node_id: node,
                operation: "get historical attribute".to_string(),
                suggestion: "Check if node exists in this historical state".to_string(),
            });
        }

        Ok(snapshot
            .node_attributes
            .get(&node)
            .and_then(|attrs| attrs.get(attr_name))
            .cloned())
    }

    /// Get all attributes for a specific node
    pub fn get_node_attributes(
        &mut self,
        node: NodeId,
    ) -> GraphResult<HashMap<AttrName, AttrValue>> {
        let snapshot = self.get_snapshot()?;
        if !snapshot.contains_node(node) {
            return Err(GraphError::NodeNotFound {
                node_id: node,
                operation: "get historical attributes".to_string(),
                suggestion: "Check if node exists in this historical state".to_string(),
            });
        }

        Ok(snapshot
            .node_attributes
            .get(&node)
            .cloned()
            .unwrap_or_default())
    }

    /*
    === EDGE OPERATIONS ===
    Read-only access to edges and their attributes
    */

    /// Get all active edge IDs at this state
    pub fn get_edge_ids(&mut self) -> GraphResult<Vec<EdgeId>> {
        let snapshot = self.get_snapshot()?;
        Ok(snapshot.edges.keys().cloned().collect())
    }

    /// Check if a specific edge exists at this state
    pub fn has_edge(&mut self, edge: EdgeId) -> GraphResult<bool> {
        let snapshot = self.get_snapshot()?;
        Ok(snapshot.contains_edge(edge))
    }

    /// Get the number of active edges at this state
    pub fn edge_count(&mut self) -> GraphResult<usize> {
        let snapshot = self.get_snapshot()?;
        Ok(snapshot.edges.len())
    }

    /// Get the endpoints of an edge
    pub fn get_edge_endpoints(&mut self, edge: EdgeId) -> GraphResult<(NodeId, NodeId)> {
        let snapshot = self.get_snapshot()?;
        snapshot
            .edges
            .get(&edge)
            .ok_or_else(|| GraphError::EdgeNotFound {
                edge_id: edge,
                operation: "get historical endpoints".to_string(),
                suggestion: "Check if edge exists in this historical state".to_string(),
            })
            .map(|&endpoints| endpoints)
    }

    /// Get a specific attribute value for an edge
    pub fn get_edge_attribute(
        &mut self,
        edge: EdgeId,
        attr_name: &AttrName,
    ) -> GraphResult<Option<AttrValue>> {
        let snapshot = self.get_snapshot()?;
        if !snapshot.contains_edge(edge) {
            return Err(GraphError::EdgeNotFound {
                edge_id: edge,
                operation: "get historical attribute".to_string(),
                suggestion: "Check if edge exists in this historical state".to_string(),
            });
        }

        Ok(snapshot
            .edge_attributes
            .get(&edge)
            .and_then(|attrs| attrs.get(attr_name))
            .cloned())
    }

    /// Get all attributes for a specific edge
    pub fn get_edge_attributes(
        &mut self,
        edge: EdgeId,
    ) -> GraphResult<HashMap<AttrName, AttrValue>> {
        let snapshot = self.get_snapshot()?;
        if !snapshot.contains_edge(edge) {
            return Err(GraphError::EdgeNotFound {
                edge_id: edge,
                operation: "get historical attributes".to_string(),
                suggestion: "Check if edge exists in this historical state".to_string(),
            });
        }

        Ok(snapshot
            .edge_attributes
            .get(&edge)
            .cloned()
            .unwrap_or_default())
    }

    /*
    === GRAPH TOPOLOGY OPERATIONS ===
    Structural queries about the graph
    */

    /// Get all neighbors of a node at this historical state
    /// NOTE: For current state, use Graph::neighbors() which is optimized with columnar topology
    pub fn get_neighbors(&mut self, node: NodeId) -> GraphResult<Vec<NodeId>> {
        let snapshot = self.get_snapshot()?;
        snapshot.get_neighbors(node)
    }

    /// Get the degree (number of incident edges) of a node
    pub fn get_degree(&mut self, node: NodeId) -> GraphResult<usize> {
        let neighbors = self.get_neighbors(node)?;
        Ok(neighbors.len())
    }

    /// Check if two nodes are connected by an edge
    pub fn are_connected(&mut self, node1: NodeId, node2: NodeId) -> GraphResult<bool> {
        let snapshot = self.get_snapshot()?;
        for &(source, target) in snapshot.edges.values() {
            if (source == node1 && target == node2) || (source == node2 && target == node1) {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /*
    === METADATA AND INTROSPECTION ===
    Information about the view and the state it represents
    */

    /// Get the state ID this view represents
    pub fn state_id(&self) -> StateId {
        self.state_id
    }

    /// Get metadata about the state this view represents
    pub fn get_state_metadata(&self) -> GraphResult<&StateMetadata> {
        let _commit = self.history.get_commit(self.state_id)?;
        // For now, create metadata from commit info
        // In a full implementation, we'd have proper StateMetadata stored
        Err(GraphError::NotImplemented {
            feature: "state metadata access".to_string(),
            tracking_issue: Some("Need to implement StateObject integration".to_string()),
        })
    }

    /// Check if this view represents a root state (no parent)
    pub fn is_root(&self) -> GraphResult<bool> {
        let commit = self.history.get_commit(self.state_id)?;
        Ok(commit.is_root())
    }

    /// Get the parent state ID, if any
    pub fn get_parent(&self) -> GraphResult<Option<StateId>> {
        let commit = self.history.get_commit(self.state_id)?;
        Ok(commit.parents.first().copied())
    }

    /// Get all child state IDs
    pub fn get_children(&self) -> Vec<StateId> {
        self.history.get_children(self.state_id)
    }

    /// Get a summary of this view's state
    pub fn summary(&mut self) -> GraphResult<ViewSummary> {
        let commit = self.history.get_commit(self.state_id)?;
        let node_count = self.node_count()?;
        let edge_count = self.edge_count()?;
        let children = self.get_children();

        Ok(ViewSummary {
            state_id: self.state_id,
            node_count,
            edge_count,
            label: commit.message.clone(),
            author: commit.author.clone(),
            timestamp: commit.timestamp,
            is_root: self.is_root()?,
            has_children: !children.is_empty(),
        })
    }

    /*
    === COMPARISON OPERATIONS ===
    Compare this view with other states
    */

    /// Compare this view with another view to find differences
    pub fn diff_with(&mut self, other: &mut HistoricalView) -> GraphResult<StateDiff> {
        let our_snapshot = self.get_snapshot()?;
        let their_snapshot = other.get_snapshot()?;

        our_snapshot.try_diff_with(their_snapshot)
    }

    /// Get the path of changes from this state to another state
    pub fn path_to(&self, target_state: StateId) -> GraphResult<Vec<StateId>> {
        // Find the shortest path through the commit DAG using BFS
        use std::collections::{HashSet, VecDeque};

        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        let mut parents: HashMap<StateId, StateId> = HashMap::new();

        queue.push_back(self.state_id);
        visited.insert(self.state_id);

        while let Some(current) = queue.pop_front() {
            if current == target_state {
                // Reconstruct path
                let mut path = Vec::new();
                let mut state = target_state;

                while state != self.state_id {
                    path.push(state);
                    state = parents[&state];
                }

                path.reverse();
                return Ok(path);
            }

            // Add children to explore
            for &child in &self.history.get_children(current) {
                if !visited.contains(&child) {
                    visited.insert(child);
                    parents.insert(child, current);
                    queue.push_back(child);
                }
            }
        }

        // No path found
        Ok(Vec::new())
    }
}

/// Summary information about a historical view
#[derive(Debug, Clone)]
pub struct ViewSummary {
    pub state_id: StateId,
    pub node_count: usize,
    pub edge_count: usize,
    pub label: String,
    pub author: String,
    pub timestamp: u64,
    pub is_root: bool,
    pub has_children: bool,
}

impl ViewSummary {
    /// Get a human-readable description of this view
    pub fn description(&self) -> String {
        format!(
            "State {}: '{}' by {} ({} nodes, {} edges)",
            self.state_id, self.label, self.author, self.node_count, self.edge_count
        )
    }

    /// Get the age of this state in seconds
    pub fn age_seconds(&self) -> u64 {
        let now = crate::utils::util::timestamp_now();
        now.saturating_sub(self.timestamp)
    }
}

/*
=== IMPLEMENTATION NOTES ===

STORAGE EFFICIENCY:
- Content-addressed deltas provide automatic deduplication
- Multiple commits with identical changes share the same Delta
- This is especially effective for automated commits or repeated patterns

PERFORMANCE CHARACTERISTICS:
- Creating commits: O(changes) - just store the delta
- Reconstructing state: O(depth) - need to apply all deltas from root
- Branch operations: O(1) - just update pointers
- Garbage collection: O(total_commits) - need to traverse the DAG

MEMORY MANAGEMENT:
- Use Arc<> for sharing deltas between commits
- Immutable data structures prevent accidental modification
- Garbage collection reclaims unreachable commits

BRANCHING MODEL:
- Lightweight branches (just pointers to commits)
- Support for merge commits (multiple parents)
- No special handling needed for branch creation/deletion

CONSISTENCY GUARANTEES:
- All operations are atomic (either succeed completely or not at all)
- Immutable commits ensure history can't be accidentally corrupted
- Content hashing detects corruption

FUTURE OPTIMIZATIONS:
- Snapshot caching for frequently accessed states
- Compressed storage for large deltas
- Incremental garbage collection
- Parallel state reconstruction
*/

/// Compare payload identity only after a hash match; unique mutations keep the
/// existing single-Arc fast path and allocate no collision bucket.
pub(crate) fn exact_delta(a: &Delta, b: &Delta) -> GraphResult<bool> {
    if a.nodes_added != b.nodes_added
        || a.nodes_removed != b.nodes_removed
        || a.edges_added != b.edges_added
        || a.edges_removed != b.edges_removed
        || a.edge_endpoint_changes != b.edge_endpoint_changes
    {
        return Ok(false);
    }
    for (left, right) in [
        (&a.node_attr_changes, &b.node_attr_changes),
        (&a.edge_attr_changes, &b.edge_attr_changes),
    ] {
        if left.len() != right.len() {
            return Ok(false);
        }
        for ((i, n, ao, an), (j, m, bo, bn)) in left.iter().zip(right) {
            if i != j || n != m {
                return Ok(false);
            }
            for (a, b) in [(ao, bo), (an, bn)] {
                match (a, b) {
                    (None, None) => {}
                    (Some(a), Some(b)) if super::change::same_value(a, b)? => {}
                    _ => return Ok(false),
                }
            }
        }
    }
    Ok(true)
}
