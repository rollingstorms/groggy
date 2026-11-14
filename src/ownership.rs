//! Creator-thread borrowing with thread-safe ownership and final release.
//!
//! Python cyclic garbage collection can release the last native handle on a
//! different thread even when every graph operation stays on its creator. Arc
//! and checked lock guards make destruction safe without granting cross-thread
//! access, relying on the compiler's Send/Sync bounds rather than unsafe impls.

use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::thread::{self, ThreadId};

thread_local! {
    // Cache the immutable ID, not a graph or a thread handle. Repeated borrows
    // need no reference-counted std::thread::Thread clone.
    static CURRENT_ID: ThreadId = thread::current().id();
}

fn current_id() -> ThreadId {
    CURRENT_ID.with(|id| *id)
}

/// A borrow cell whose contents may only be accessed by its creating thread.
///
/// Shared immutable borrows are permitted; a conflicting mutable borrow fails
/// immediately, like RefCell, rather than blocking or introducing a deadlock.
/// Dropping the cell needs no creator-thread access. Send/Sync remain conditional
/// on the contents' actual Rust traits.
pub struct OwnerCell<T> {
    owner: ThreadId,
    value: RwLock<T>,
}

/// Shared owner-thread borrow of a value.
pub type OwnerReadGuard<'a, T> = RwLockReadGuard<'a, T>;
/// Exclusive owner-thread borrow of a value.
pub type OwnerWriteGuard<'a, T> = RwLockWriteGuard<'a, T>;

impl<T> OwnerCell<T> {
    /// Bind a value to the current thread.
    pub fn new(value: T) -> Self {
        Self {
            owner: current_id(),
            value: RwLock::new(value),
        }
    }

    /// Reject state access outside the creating thread, without borrowing data.
    pub fn assert_owner_thread(&self) {
        assert_eq!(
            self.owner,
            current_id(),
            "Graph state may only be accessed on its creating thread"
        );
    }

    /// Borrow on the creating thread, rejecting an outstanding mutable borrow.
    pub fn borrow(&self) -> OwnerReadGuard<'_, T> {
        self.assert_owner_thread();
        self.value
            .try_read()
            .expect("Graph state is already mutably borrowed")
    }

    /// Borrow exclusively on the creating thread, rejecting any other borrow.
    pub fn borrow_mut(&self) -> OwnerWriteGuard<'_, T> {
        self.assert_owner_thread();
        self.value
            .try_write()
            .expect("Graph state is already borrowed")
    }

    /// Access through an exclusive cell reference on its creating thread.
    pub fn get_mut(&mut self) -> &mut T {
        self.assert_owner_thread();
        self.value.get_mut()
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for OwnerCell<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut result = f.debug_struct("OwnerCell");
        result.field("owner", &self.owner);
        if self.owner == current_id() {
            if let Some(value) = self.value.try_read() {
                result.field("value", &*value);
            } else {
                result.field("value", &"<borrowed>");
            }
        } else {
            result.field("value", &"<creator-thread state>");
        }
        result.finish()
    }
}

impl<T: Default> Default for OwnerCell<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

impl<T: Clone> Clone for OwnerCell<T> {
    fn clone(&self) -> Self {
        Self::new(self.borrow().clone())
    }
}
