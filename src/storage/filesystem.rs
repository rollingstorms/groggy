//! Explicit host filesystem boundaries; file contents and directory entries differ.
use std::{io, path::Path};

/// Available implementation mechanisms, not a certification of a filesystem/device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StorageCapabilities {
    /// File contents are synchronized before a successful acknowledgement.
    pub file_sync: bool,
    /// Creation/backup additionally synchronize the containing directory on Unix.
    pub directory_sync: bool,
    /// Inode-checked replacement/maintenance is currently implemented on Unix.
    pub checked_replacement: bool,
}
/// Inspect platform support before choosing a storage/recovery policy.
pub const fn capabilities() -> StorageCapabilities {
    StorageCapabilities {
        file_sync: cfg!(any(unix, windows)),
        directory_sync: cfg!(unix),
        checked_replacement: cfg!(unix),
    }
}
/// Sync namespace changes where the host implementation supports directory fsync.
/// Windows callers retain file-sync acknowledgement only, reported by capabilities.
#[cfg(unix)]
pub(crate) fn sync_directory(path: &Path) -> io::Result<()> {
    std::fs::File::open(path)?.sync_all()
}
#[cfg(windows)]
pub(crate) fn sync_directory(_path: &Path) -> io::Result<()> {
    Ok(())
}
#[cfg(not(any(unix, windows)))]
pub(crate) fn sync_directory(_path: &Path) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "Unsupported storage platform",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn platform_mechanisms_are_explicit_and_directory_errors_propagate_on_unix() {
        let supported = capabilities();
        assert_eq!(supported.directory_sync, cfg!(unix));
        assert_eq!(supported.checked_replacement, cfg!(unix));
        #[cfg(unix)]
        {
            let folder = tempfile::tempdir().unwrap();
            sync_directory(folder.path()).unwrap();
            assert!(sync_directory(&folder.path().join("missing")).is_err());
        }
    }
}
