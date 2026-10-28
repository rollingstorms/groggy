//! Property timelines use the same sorted event positions and lifetime rules as replay.
use super::{EffectiveTimeIndex, Position};
use crate::{AttrValue, GraphResult};

/// One maximal half-open interval of an active entity's exact property value.
/// None means absent, while Some(AttrValue::Null) is explicit null. Inactive
/// periods have no rows. None at the end means the interval remains open.
#[derive(Debug, Clone)]
pub struct PropertyInterval {
    pub start: i64,
    pub end: Option<i64>,
    pub value: Option<AttrValue>,
}

fn same(a: Option<&AttrValue>, b: Option<&AttrValue>) -> GraphResult<bool> {
    match (a, b) {
        (None, None) => Ok(true),
        (Some(a), Some(b)) => crate::state::change::same_value(a, b),
        _ => Ok(false),
    }
}

impl EffectiveTimeIndex {
    /// Exact node-property intervals within structural active lifetimes.
    /// Recorded cutoffs are explicit; assignments outside a lifetime are ignored.
    pub fn node_property_intervals(
        &self,
        id: usize,
        name: &str,
        recorded_by: Option<i64>,
    ) -> GraphResult<Vec<PropertyInterval>> {
        self.property_intervals(true, id, name, recorded_by)
    }

    /// Exact edge-property intervals, including absence and explicit Null.
    /// These do not certify endpoint validity or infer domain-specific facts.
    pub fn edge_property_intervals(
        &self,
        id: usize,
        name: &str,
        recorded_by: Option<i64>,
    ) -> GraphResult<Vec<PropertyInterval>> {
        self.property_intervals(false, id, name, recorded_by)
    }

    fn property_intervals(
        &self,
        node: bool,
        id: usize,
        name: &str,
        cutoff: Option<i64>,
    ) -> GraphResult<Vec<PropertyInterval>> {
        let life = self
            .by_activation
            .get(&(node, id))
            .map_or(&[][..], Vec::as_slice);
        let properties = self
            .by_property
            .get(&(node, id))
            .and_then(|names| names.get(name))
            .map_or(&[][..], Vec::as_slice);
        // Merge two already sorted postings without allocating/sorting all of an
        // entity's events or scanning unrelated properties.
        let mut life = life.iter().copied().peekable();
        let mut properties = properties.iter().copied().peekable();
        let positions = std::iter::from_fn(move || match (life.peek(), properties.peek()) {
            (Some(a), Some(b)) if a < b => life.next(),
            (_, Some(_)) => properties.next(),
            (Some(_), None) => life.next(),
            (None, None) => None,
        });
        let mut positions = positions.peekable();
        let mut rows = Vec::new();
        let mut active = false;
        let mut value = None;
        let mut open = None;
        let mut previous = None;
        while let Some(ordinal) = positions.next() {
            let at = self.events.events[ordinal].0;
            let mut current = Some(ordinal);
            while let Some(ordinal) = current {
                let (_, accepted, _, position) = self.events.events[ordinal];
                let commit = &self.plan.context.commits[accepted];
                let metadata = commit
                    .publication
                    .as_ref()
                    .or_else(|| self.plan.legacy_publications.get(&commit.id));
                if metadata.is_some_and(|m| cutoff.is_none_or(|u| m.recorded_at <= u)) {
                    match position {
                        Position::NodeAdded(_) | Position::EdgeAdded(..) => {
                            active = true;
                            value = None;
                        }
                        Position::NodeRemoved(_) | Position::EdgeRemoved(_) => {
                            active = false;
                            value = None;
                        }
                        Position::NodeAttr(row) if active => {
                            value = commit.delta.node_attr_changes[row].3.as_ref();
                        }
                        Position::EdgeAttr(row) if active => {
                            value = commit.delta.edge_attr_changes[row].3.as_ref();
                        }
                        _ => {}
                    }
                }
                current = if positions
                    .peek()
                    .is_some_and(|&next| self.events.events[next].0 == at)
                {
                    positions.next()
                } else {
                    None
                };
            }
            if let Some(start) = open {
                if active && same(previous, value)? {
                    continue;
                }
                rows.push(PropertyInterval {
                    start,
                    end: Some(at),
                    value: previous.cloned(),
                });
                open = None;
            }
            if active {
                open = Some(at);
                previous = value;
            }
        }
        if let Some(start) = open {
            rows.push(PropertyInterval {
                start,
                end: None,
                value: previous.cloned(),
            });
        }
        Ok(rows)
    }
}
