use groggy::ownership::OwnerCell;
use groggy::subgraphs::Subgraph;
use groggy::{Graph, GraphMatrix, GraphPool};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

fn require_send_sync<T: Send + Sync>() {}

#[test]
fn shared_graph_family_has_compiler_checked_thread_safe_ownership() {
    require_send_sync::<Graph>();
    require_send_sync::<GraphPool>();
    require_send_sync::<Subgraph>();
    require_send_sync::<GraphMatrix>();
    require_send_sync::<Arc<OwnerCell<Graph>>>();
}

struct CountDrop(Arc<AtomicUsize>);
impl Drop for CountDrop {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn final_owner_drops_immediately_on_another_thread_without_a_poll_queue() {
    let dropped = Arc::new(AtomicUsize::new(0));
    let cell = Arc::new(OwnerCell::new(CountDrop(dropped.clone())));
    std::thread::spawn(move || drop(cell)).join().unwrap();
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
}

#[test]
fn owner_exit_does_not_strand_a_native_value() {
    let dropped = Arc::new(AtomicUsize::new(0));
    let counter = dropped.clone();
    let cell = std::thread::spawn(move || Arc::new(OwnerCell::new(CountDrop(counter))))
        .join()
        .unwrap();
    assert_eq!(dropped.load(Ordering::SeqCst), 0);
    drop(cell);
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
}

#[test]
fn foreign_access_rejects_without_poisoning_the_creator_owner() {
    let cell = Arc::new(OwnerCell::new(7));
    let foreign = cell.clone();
    assert!(std::thread::spawn(move || {
        let _value = foreign.borrow();
    })
    .join()
    .is_err());
    assert_eq!(*cell.borrow(), 7);
    *cell.borrow_mut() = 8;
    assert_eq!(*cell.borrow(), 8);
}

#[test]
fn nested_read_borrows_work_and_conflicting_borrows_fail_immediately() {
    let cell = OwnerCell::new(7);
    let first = cell.borrow();
    let second = cell.borrow();
    assert_eq!(*first + *second, 14);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _write = cell.borrow_mut();
    }))
    .is_err());
    drop((first, second));
    *cell.borrow_mut() = 9;
    assert_eq!(*cell.borrow(), 9);
}
