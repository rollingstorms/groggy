//! Canonical bulk property validation independent of writable graph ownership.
use super::*;
pub(crate) fn collect(
    attr_name: &str,
    values: &[(usize, AttrValue)],
    is_node: bool,
    positions: Option<&[usize]>,
    contains: &mut impl FnMut(usize) -> bool,
    errors: &mut Vec<String>,
) {
    let kind = if is_node { "node" } else { "edge" };
    let mut first_positions = HashMap::new();
    for (offset, (id, value)) in values.iter().enumerate() {
        let id = *id;
        let position = positions
            .and_then(|positions| positions.get(offset))
            .copied()
            .unwrap_or(offset);
        if let Err(error) = kinds::validate_property(attr_name, value, is_node) {
            errors.push(format!("{attr_name}[{position}]: {error}"));
        }
        let exists = contains(id);
        if !exists {
            errors.push(format!(
                "{attr_name}[{position}]: {kind} {id} is not active"
            ));
        }
        match first_positions.entry(id) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(position);
            }
            std::collections::hash_map::Entry::Occupied(entry) => errors.push(format!(
                "{attr_name}[{position}]: duplicate {kind} {id} (first at {attr_name}[{}])",
                entry.get()
            )),
        }
    }
}
pub(crate) fn validate(
    attrs_values: &HashMap<AttrName, Vec<(usize, AttrValue)>>,
    is_node: bool,
    positions: Option<&HashMap<AttrName, Vec<usize>>>,
    mut contains: impl FnMut(usize) -> bool,
) -> GraphResult<()> {
    let mut attrs: Vec<_> = attrs_values.iter().collect();
    attrs.sort_unstable_by_key(|(name, _)| *name);
    let mut errors = Vec::new();
    for (attr_name, values) in attrs {
        if is_node && attr_name == "entity_type" && !values.is_empty() {
            errors.push("entity_type: system-managed node attribute cannot be changed".into());
        }
        let column_positions = positions
            .and_then(|positions| positions.get(attr_name))
            .map(Vec::as_slice);
        if attr_name == EXTERNAL_ID && !values.is_empty() {
            errors.push("external_id: system-managed attribute; use the external-ID API".into());
        }
        collect(
            attr_name,
            values,
            is_node,
            column_positions,
            &mut contains,
            &mut errors,
        );
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(GraphError::InvalidInput(format!(
            "Invalid bulk attribute assignments: {}",
            errors.join("; ")
        )))
    }
}
