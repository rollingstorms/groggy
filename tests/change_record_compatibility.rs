use groggy::{
    state::{change, history, state::GraphSnapshot},
    AttrValue,
};
use std::collections::HashMap;

// Legacy public paths must accept the same record without converting its fields.
fn legacy_to_canonical(record: history::AttributeChange) -> change::AttributeChange {
    let legacy: groggy::state::state::AttributeChange = record;
    legacy
}

#[test]
fn snapshot_and_history_records_share_identity_and_preserve_absence_and_null() {
    let mut before = GraphSnapshot::empty(1);
    before.active_nodes.push(42);
    let mut after = before.clone();
    after.state_id = 2;
    after
        .node_attributes
        .insert(42, HashMap::from([("p".into(), AttrValue::Null)]));
    let mut forward: Vec<history::AttributeChange> = before.diff_with(&after).attribute_changes;
    assert_eq!(forward.len(), 1);
    let record = legacy_to_canonical(forward.pop().unwrap());
    let _: history::EntityType = change::EntityType::Node;
    assert!(matches!(record.entity_type, change::EntityType::Node));
    assert_eq!(record.entity_id, 42);
    assert_eq!(record.attr_name, "p");
    assert!(record.old_value.is_none());
    assert_eq!(record.new_value, Some(AttrValue::Null));
    let mut reverse: Vec<history::AttributeChange> = after.diff_with(&before).attribute_changes;
    let reverse = legacy_to_canonical(reverse.pop().unwrap());
    assert_eq!(reverse.old_value, Some(AttrValue::Null));
    assert!(reverse.new_value.is_none());
}
