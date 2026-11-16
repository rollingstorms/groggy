//! Aligned numerical storage, guarded live buffers and owned backend read snapshots.
#![doc = include_str!("../../../docs/api/matrix-memory-ownership.md")]

use crate::storage::matrix::numeric_type::NumericType;
use std::alloc::{alloc, alloc_zeroed, dealloc, Layout};
use std::any::TypeId;
use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};

/// Errors related to memory management
#[derive(Debug, Clone)]
pub enum MemoryError {
    AllocationFailed,
    InsufficientMemory,
    InvalidAlignment,
    BufferInUse,
    InvalidSize,
    BackendSyncFailed(String),
}

impl std::fmt::Display for MemoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryError::AllocationFailed => write!(f, "Memory allocation failed"),
            MemoryError::InsufficientMemory => write!(f, "Insufficient memory available"),
            MemoryError::InvalidAlignment => write!(f, "Invalid memory alignment"),
            MemoryError::BufferInUse => write!(f, "Buffer is currently in use"),
            MemoryError::InvalidSize => write!(f, "Invalid buffer size"),
            MemoryError::BackendSyncFailed(msg) => {
                write!(f, "Backend synchronization failed: {}", msg)
            }
        }
    }
}

impl std::error::Error for MemoryError {}

pub type MemoryResult<T> = Result<T, MemoryError>;

/// Identifies different compute backends for memory synchronization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendId {
    Native,
    NumPy,
    BLAS,
    CUDA,
    OpenCL,
}

/// Memory layout information for matrices
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixLayout {
    RowMajor,    // C-style: elements are stored row by row
    ColumnMajor, // Fortran-style: elements are stored column by column
    Blocked,     // Block layout for cache efficiency
}

/// SIMD alignment requirements
pub const SIMD_ALIGNMENT: usize = 64; // AVX-512 alignment

/// Align size to SIMD boundary
pub fn align_to_simd_boundary(size: usize) -> usize {
    (size + SIMD_ALIGNMENT - 1) & !(SIMD_ALIGNMENT - 1)
}

/// Get the required alignment for SIMD operations
pub fn get_simd_alignment<T: NumericType>() -> usize {
    std::cmp::max(std::mem::align_of::<T>(), SIMD_ALIGNMENT)
}

/// Aligned memory allocation with custom deleter
pub struct AlignedBuffer<T: NumericType> {
    ptr: NonNull<T>,
    layout: Layout,
    capacity: usize,
}

impl<T: NumericType> std::fmt::Debug for AlignedBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlignedBuffer")
            .field("ptr", &self.ptr.as_ptr())
            .field("layout", &self.layout)
            .field("capacity", &self.capacity)
            .finish()
    }
}

impl<T: NumericType> AlignedBuffer<T> {
    /// Allocate aligned memory for the given number of elements
    fn new(capacity: usize) -> MemoryResult<Self> {
        // Keep a valid SIMD-aligned allocation even for a logically empty
        // buffer. Its exposed slices still have length zero.
        if std::mem::size_of::<T>() == 0 {
            return Err(MemoryError::InvalidSize);
        }
        let size = capacity
            .max(1)
            .checked_mul(std::mem::size_of::<T>())
            .ok_or(MemoryError::InvalidSize)?;
        let align = get_simd_alignment::<T>();

        let layout =
            Layout::from_size_align(size, align).map_err(|_| MemoryError::InvalidAlignment)?;

        // NumericType is a safe, externally implementable trait: its zero need
        // not have an all-zero representation (and all-zero bits may be invalid).
        // Keep calloc for the concrete built-ins whose representations are known.
        let builtin_zero = [
            TypeId::of::<f64>(),
            TypeId::of::<f32>(),
            TypeId::of::<i64>(),
            TypeId::of::<i32>(),
            TypeId::of::<i128>(),
            TypeId::of::<bool>(),
        ]
        .contains(&TypeId::of::<T>());
        let zero = (!builtin_zero && capacity != 0).then(T::zero);
        let ptr = unsafe {
            if builtin_zero {
                alloc_zeroed(layout)
            } else {
                alloc(layout)
            }
        } as *mut T;
        let ptr = NonNull::new(ptr).ok_or(MemoryError::AllocationFailed)?;
        if let Some(zero) = zero {
            for index in 0..capacity {
                // Each logical element is initialized before any slice exists.
                unsafe {
                    ptr.as_ptr().add(index).write(zero);
                }
            }
        }

        Ok(Self {
            ptr,
            layout,
            capacity,
        })
    }

    /// Get a slice to the allocated memory
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.capacity) }
    }

    /// Get a mutable slice to the allocated memory
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.capacity) }
    }
}

impl<T: NumericType> Drop for AlignedBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr.as_ptr() as *mut u8, self.layout);
        }
    }
}

unsafe impl<T: NumericType> Send for AlignedBuffer<T> {}
unsafe impl<T: NumericType> Sync for AlignedBuffer<T> {}

/// Backend-specific view of shared data
pub trait BackendView<T: NumericType>: Send + Sync {
    /// Get backend identifier
    fn backend_id(&self) -> BackendId;

    /// Borrow stable read data. Implementations must retain a lock or owned
    /// snapshot for the complete returned lifetime; a released lock is insufficient.
    fn data(&self) -> &[T];

    /// Get mutable access to data (if supported)
    fn data_mut(&mut self) -> Option<&mut [T]>;

    /// Synchronize data from this backend to main memory
    fn sync_to_main(&self) -> MemoryResult<()>;

    /// Synchronize data from main memory to this backend
    fn sync_from_main(&mut self, data: &[T]) -> MemoryResult<()>;
}

/// Owned native read snapshot. Live zero-copy access uses SharedBuffer::data's
/// read guard; the object-safe BackendView slice cannot carry that guard.
pub struct NativeView<T: NumericType> {
    data: Vec<T>,
}

impl<T: NumericType> BackendView<T> for NativeView<T> {
    fn backend_id(&self) -> BackendId {
        BackendId::Native
    }

    fn data(&self) -> &[T] {
        &self.data
    }

    fn data_mut(&mut self) -> Option<&mut [T]> {
        None
    }

    fn sync_to_main(&self) -> MemoryResult<()> {
        Err(MemoryError::BackendSyncFailed(
            "Native read snapshots have no writeback source; use SharedBuffer::data_mut".into(),
        ))
    }

    fn sync_from_main(&mut self, data: &[T]) -> MemoryResult<()> {
        if data.len() != self.data.len() {
            return Err(MemoryError::InvalidSize);
        }
        self.data.copy_from_slice(data);
        Ok(())
    }
}

/// Shared buffer that can be viewed by multiple backends
pub struct SharedBuffer<T: NumericType> {
    buffer: Arc<RwLock<AlignedBuffer<T>>>,
    layout: MatrixLayout,
    shape: (usize, usize),
    modification_count: Arc<AtomicUsize>,
}

impl<T: NumericType> std::fmt::Debug for SharedBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedBuffer")
            .field("layout", &self.layout)
            .field("shape", &self.shape)
            .field(
                "modification_count",
                &self
                    .modification_count
                    .load(std::sync::atomic::Ordering::Relaxed),
            )
            .field("buffer", &"<shared buffer>")
            .finish()
    }
}

impl<T: NumericType> SharedBuffer<T> {
    /// Create a new shared buffer with the given shape and layout
    pub fn new(shape: (usize, usize), layout: MatrixLayout) -> MemoryResult<Self> {
        let (rows, cols) = shape;
        let capacity = rows.checked_mul(cols).ok_or(MemoryError::InvalidSize)?;

        let buffer = AlignedBuffer::new(capacity)?;

        Ok(Self {
            buffer: Arc::new(RwLock::new(buffer)),
            layout,
            shape,
            modification_count: Arc::new(AtomicUsize::new(0)),
        })
    }

    /// Create from existing data
    pub fn from_data(
        data: Vec<T>,
        shape: (usize, usize),
        layout: MatrixLayout,
    ) -> MemoryResult<Self> {
        let size = shape
            .0
            .checked_mul(shape.1)
            .ok_or(MemoryError::InvalidSize)?;
        if data.len() != size {
            return Err(MemoryError::InvalidSize);
        }
        let buffer = Self::new(shape, layout)?;

        // Copy data into the aligned buffer
        {
            let mut buf = buffer.buffer.write().unwrap();
            let slice = buf.as_slice_mut();
            let len = data.len().min(slice.len());
            slice[..len].copy_from_slice(&data[..len]);
        }

        buffer.mark_modified();
        Ok(buffer)
    }

    /// Get the shape of the buffer
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// Get the memory layout
    pub fn layout(&self) -> MatrixLayout {
        self.layout
    }

    /// Get the total number of elements
    pub fn len(&self) -> usize {
        self.shape.0 * self.shape.1
    }

    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get read-only access to the data
    pub fn data(&self) -> MemoryResult<std::sync::RwLockReadGuard<'_, AlignedBuffer<T>>> {
        self.buffer.read().map_err(|_| MemoryError::BufferInUse)
    }

    /// Get mutable access to the data
    pub fn data_mut(&self) -> MemoryResult<std::sync::RwLockWriteGuard<'_, AlignedBuffer<T>>> {
        let guard = self.buffer.write().map_err(|_| MemoryError::BufferInUse)?;
        self.mark_modified();
        Ok(guard)
    }

    /// Capture an owned native read snapshot. Other backends reject explicitly.
    /// Use data()/data_mut() guards for live access without copying.
    pub fn view_for_backend(&self, backend: BackendId) -> MemoryResult<Box<dyn BackendView<T>>> {
        match backend {
            BackendId::Native => {
                let source = self.data()?;
                let mut data = Vec::new();
                data.try_reserve_exact(source.as_slice().len())
                    .map_err(|_| MemoryError::AllocationFailed)?;
                data.extend_from_slice(source.as_slice());
                Ok(Box::new(NativeView { data }))
            }
            _ => {
                // TODO: Implement views for other backends
                Err(MemoryError::BackendSyncFailed(format!(
                    "Backend {:?} not implemented yet",
                    backend
                )))
            }
        }
    }

    /// Validate native residency. No external backend owns a buffer in this
    /// implementation; unavailable synchronization must not report success.
    pub fn sync_from_backend(&self, source: BackendId) -> MemoryResult<()> {
        if source != BackendId::Native {
            return Err(MemoryError::BackendSyncFailed(format!(
                "Backend {source:?} synchronization is unavailable"
            )));
        }
        drop(self.data()?);
        Ok(())
    }

    /// Mark the buffer as modified
    fn mark_modified(&self) {
        self.modification_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Get the modification count (for cache invalidation)
    pub fn modification_count(&self) -> usize {
        self.modification_count.load(Ordering::Relaxed)
    }

    /// Clone the data to a new vector
    pub fn to_vec(&self) -> MemoryResult<Vec<T>> {
        let buffer = self.data()?;
        Ok(buffer.as_slice().to_vec())
    }
}

impl<T: NumericType> Clone for SharedBuffer<T> {
    fn clone(&self) -> Self {
        // Create a new buffer and copy the data
        let new_buffer = Self::new(self.shape, self.layout).unwrap();

        {
            let src = self.data().unwrap();
            let mut dst = new_buffer.data_mut().unwrap();
            dst.as_slice_mut().copy_from_slice(src.as_slice());
        }

        new_buffer
    }
}

/// Advanced memory pool for efficient allocation and reuse
#[derive(Debug)]
pub struct AdvancedMemoryPool<T: NumericType> {
    // Buckets for different sizes (power of 2 sizes)
    size_buckets: Mutex<HashMap<usize, Vec<AlignedBuffer<T>>>>,
    // Large blocks that don't fit in buckets
    large_blocks: Mutex<Vec<AlignedBuffer<T>>>,
    // Statistics
    peak_usage: AtomicUsize,
    current_usage: AtomicUsize,
    total_allocations: AtomicUsize,
    cache_hits: AtomicUsize,
}

impl<T: NumericType> AdvancedMemoryPool<T> {
    /// Create a new memory pool
    pub fn new() -> Self {
        Self {
            size_buckets: Mutex::new(HashMap::new()),
            large_blocks: Mutex::new(Vec::new()),
            peak_usage: AtomicUsize::new(0),
            current_usage: AtomicUsize::new(0),
            total_allocations: AtomicUsize::new(0),
            cache_hits: AtomicUsize::new(0),
        }
    }

    /// Allocate a buffer with the given number of elements
    pub fn allocate(&self, size: usize) -> MemoryResult<AlignedBuffer<T>> {
        self.total_allocations.fetch_add(1, Ordering::Relaxed);

        let aligned_size = align_to_simd_boundary(size);
        let bucket_size = self.get_bucket_size(aligned_size);

        // Try to reuse from appropriate bucket
        if let Some(mut buffer) = self.try_reuse_from_bucket(bucket_size) {
            self.cache_hits.fetch_add(1, Ordering::Relaxed);

            // Zero out the reused buffer
            buffer.as_slice_mut().fill(T::zero());
            return Ok(buffer);
        }

        // Allocate new buffer
        let buffer = AlignedBuffer::new(aligned_size)?;

        // Update statistics
        let new_usage = self
            .current_usage
            .fetch_add(aligned_size * std::mem::size_of::<T>(), Ordering::Relaxed)
            + aligned_size * std::mem::size_of::<T>();

        // Update peak usage
        self.peak_usage.fetch_max(new_usage, Ordering::Relaxed);

        Ok(buffer)
    }

    /// Return a buffer to the pool for reuse
    pub fn deallocate(&self, buffer: AlignedBuffer<T>) {
        let size = buffer.capacity * std::mem::size_of::<T>();
        self.current_usage.fetch_sub(size, Ordering::Relaxed);

        let bucket_size = self.get_bucket_size(buffer.capacity);

        // Return to appropriate bucket if it's not too large
        if bucket_size <= 1024 * 1024 {
            // 1MB limit for buckets
            let mut buckets = self.size_buckets.lock().unwrap();
            buckets
                .entry(bucket_size)
                .or_insert_with(Vec::new)
                .push(buffer);
        } else {
            // Store large blocks separately
            let mut large_blocks = self.large_blocks.lock().unwrap();
            large_blocks.push(buffer);
        }
    }

    /// Allocate a shared buffer
    pub fn allocate_shared_buffer(
        &self,
        shape: (usize, usize),
        layout: MatrixLayout,
    ) -> MemoryResult<SharedBuffer<T>> {
        SharedBuffer::new(shape, layout)
    }

    /// Get bucket size (next power of 2)
    fn get_bucket_size(&self, size: usize) -> usize {
        if size <= 1 {
            return 1;
        }
        let mut bucket_size = 1;
        while bucket_size < size {
            bucket_size <<= 1;
        }
        bucket_size
    }

    /// Try to reuse a buffer from the appropriate bucket
    fn try_reuse_from_bucket(&self, bucket_size: usize) -> Option<AlignedBuffer<T>> {
        let mut buckets = self.size_buckets.lock().unwrap();

        if let Some(bucket) = buckets.get_mut(&bucket_size) {
            return bucket.pop();
        }

        // Try larger buckets if available - collect keys first to avoid borrow conflicts
        let keys: Vec<usize> = buckets
            .keys()
            .copied()
            .filter(|&size| size > bucket_size)
            .collect();

        for size in keys {
            if let Some(bucket) = buckets.get_mut(&size) {
                if let Some(buffer) = bucket.pop() {
                    return Some(buffer);
                }
            }
        }

        None
    }

    /// Prefetch memory patterns for predictable access
    pub fn prefetch_for_operation(&self, _op_type: &str, _size: (usize, usize)) {
        // TODO: Implement prefetching based on operation patterns
        // This could pre-allocate commonly used buffer sizes
    }

    /// Clear the pool and free all cached buffers
    pub fn clear(&self) {
        let mut buckets = self.size_buckets.lock().unwrap();
        buckets.clear();

        let mut large_blocks = self.large_blocks.lock().unwrap();
        large_blocks.clear();

        self.current_usage.store(0, Ordering::Relaxed);
    }

    /// Get memory usage statistics
    pub fn stats(&self) -> MemoryPoolStats {
        MemoryPoolStats {
            current_usage_bytes: self.current_usage.load(Ordering::Relaxed),
            peak_usage_bytes: self.peak_usage.load(Ordering::Relaxed),
            total_allocations: self.total_allocations.load(Ordering::Relaxed),
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            cache_hit_rate: if self.total_allocations.load(Ordering::Relaxed) > 0 {
                self.cache_hits.load(Ordering::Relaxed) as f64
                    / self.total_allocations.load(Ordering::Relaxed) as f64
            } else {
                0.0
            },
        }
    }
}

impl<T: NumericType> Default for AdvancedMemoryPool<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory pool statistics
#[derive(Debug, Clone)]
pub struct MemoryPoolStats {
    pub current_usage_bytes: usize,
    pub peak_usage_bytes: usize,
    pub total_allocations: usize,
    pub cache_hits: usize,
    pub cache_hit_rate: f64,
}

/// Global memory pool instance
static GLOBAL_MEMORY_POOL_F64: std::sync::OnceLock<AdvancedMemoryPool<f64>> =
    std::sync::OnceLock::new();
static GLOBAL_MEMORY_POOL_F32: std::sync::OnceLock<AdvancedMemoryPool<f32>> =
    std::sync::OnceLock::new();
static GLOBAL_MEMORY_POOL_I64: std::sync::OnceLock<AdvancedMemoryPool<i64>> =
    std::sync::OnceLock::new();
static GLOBAL_MEMORY_POOL_I32: std::sync::OnceLock<AdvancedMemoryPool<i32>> =
    std::sync::OnceLock::new();

/// Get the global memory pool for a specific type
pub fn get_memory_pool<T: NumericType>() -> &'static AdvancedMemoryPool<T> {
    // This is a simplified implementation - in practice we'd use trait objects
    // or a more sophisticated type system
    panic!("Generic memory pool access not implemented - use specific type pools")
}

/// Get the global f64 memory pool
pub fn get_f64_memory_pool() -> &'static AdvancedMemoryPool<f64> {
    GLOBAL_MEMORY_POOL_F64.get_or_init(AdvancedMemoryPool::new)
}

/// Get the global f32 memory pool
pub fn get_f32_memory_pool() -> &'static AdvancedMemoryPool<f32> {
    GLOBAL_MEMORY_POOL_F32.get_or_init(AdvancedMemoryPool::new)
}

/// Get the global i64 memory pool
pub fn get_i64_memory_pool() -> &'static AdvancedMemoryPool<i64> {
    GLOBAL_MEMORY_POOL_I64.get_or_init(AdvancedMemoryPool::new)
}

/// Get the global i32 memory pool
pub fn get_i32_memory_pool() -> &'static AdvancedMemoryPool<i32> {
    GLOBAL_MEMORY_POOL_I32.get_or_init(AdvancedMemoryPool::new)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_backend_slices_remain_stable_during_concurrent_source_writes() {
        let buffer = Arc::new(
            SharedBuffer::from_data(vec![1.0, 2.0], (1, 2), MatrixLayout::RowMajor).unwrap(),
        );
        let view = buffer.view_for_backend(BackendId::Native).unwrap();
        let slice = view.data();
        let writer = buffer.clone();
        std::thread::spawn(move || {
            for value in 0..128 {
                writer.data_mut().unwrap().as_slice_mut().fill(value as f64);
            }
        })
        .join()
        .unwrap();
        assert_eq!(slice, [1.0, 2.0]);
        assert_eq!(buffer.to_vec().unwrap(), [127.0, 127.0]);
        drop(buffer);
        let values = std::thread::spawn(move || view.data().to_vec())
            .join()
            .unwrap();
        assert_eq!(values, [1.0, 2.0]);
    }

    #[test]
    fn native_snapshot_refresh_is_explicit_exact_sized_and_has_no_implicit_writeback() {
        let values = [f64::from_bits(0x7ff8_0000_0000_1234), -0.0, 1.0000000001];
        let buffer =
            SharedBuffer::from_data(values.to_vec(), (1, 3), MatrixLayout::ColumnMajor).unwrap();
        let count = buffer.modification_count();
        let mut view = buffer.view_for_backend(BackendId::Native).unwrap();
        assert_eq!(view.backend_id(), BackendId::Native);
        assert_eq!(
            view.data().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            values.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert!(view.data_mut().is_none());
        assert!(matches!(
            view.sync_from_main(&[4.0]),
            Err(MemoryError::InvalidSize)
        ));
        assert_eq!(view.data()[2], values[2]);
        view.sync_from_main(&[4.0, 5.0, 6.0]).unwrap();
        assert_eq!(view.data(), [4.0, 5.0, 6.0]);
        assert!(matches!(
            view.sync_to_main(),
            Err(MemoryError::BackendSyncFailed(_))
        ));
        assert_eq!(buffer.modification_count(), count);
        assert_eq!(
            buffer
                .to_vec()
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            values.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn unavailable_backend_views_and_synchronization_reject_without_mutation() {
        let buffer =
            SharedBuffer::from_data(vec![3_i128, i128::MAX], (2, 1), MatrixLayout::RowMajor)
                .unwrap();
        let count = buffer.modification_count();
        for backend in [
            BackendId::NumPy,
            BackendId::BLAS,
            BackendId::CUDA,
            BackendId::OpenCL,
        ] {
            for _ in 0..2 {
                assert!(matches!(
                    buffer.view_for_backend(backend),
                    Err(MemoryError::BackendSyncFailed(_))
                ));
                assert!(matches!(
                    buffer.sync_from_backend(backend),
                    Err(MemoryError::BackendSyncFailed(_))
                ));
            }
        }
        buffer.sync_from_backend(BackendId::Native).unwrap();
        assert_eq!(buffer.modification_count(), count);
        assert_eq!(buffer.to_vec().unwrap(), [3, i128::MAX]);
    }

    #[test]
    fn poisoned_live_storage_rejects_views_and_does_not_advance_failed_write_generation() {
        let buffer = Arc::new(SharedBuffer::<f64>::new((1, 1), MatrixLayout::RowMajor).unwrap());
        let writer = buffer.clone();
        assert!(std::thread::spawn(move || {
            let _guard = writer.data_mut().unwrap();
            panic!("deliberate poison");
        })
        .join()
        .is_err());
        let count = buffer.modification_count();
        assert!(matches!(buffer.data_mut(), Err(MemoryError::BufferInUse)));
        assert!(matches!(
            buffer.view_for_backend(BackendId::Native),
            Err(MemoryError::BufferInUse)
        ));
        assert!(matches!(
            buffer.sync_from_backend(BackendId::Native),
            Err(MemoryError::BufferInUse)
        ));
        assert_eq!(buffer.modification_count(), count);
    }

    #[test]
    fn empty_native_snapshots_and_builtin_zeros_retain_exact_logical_lengths() {
        fn check<T: NumericType>() {
            for shape in [(0, 3), (3, 0), (2, 3)] {
                let buffer = SharedBuffer::<T>::new(shape, MatrixLayout::RowMajor).unwrap();
                let mut view = buffer.view_for_backend(BackendId::Native).unwrap();
                assert_eq!(view.data().len(), shape.0 * shape.1);
                assert!(view.data().iter().all(|v| *v == T::zero()));
                let values = vec![T::one(); shape.0 * shape.1];
                view.sync_from_main(&values).unwrap();
                assert_eq!(view.data(), values);
                assert!(buffer.to_vec().unwrap().iter().all(|v| *v == T::zero()));
            }
        }
        check::<f64>();
        check::<f32>();
        check::<i64>();
        check::<i32>();
        check::<i128>();
        check::<bool>();
    }

    // A valid safe NumericType whose mathematical zero has a nonzero bit pattern.
    #[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
    struct EncodedZero<const N: usize>([std::num::NonZeroU8; N]);
    impl<const N: usize> NumericType for EncodedZero<N> {
        type Accumulator = Self;
        type Wide = Self;
        const DTYPE: crate::storage::numeric::types::DType =
            crate::storage::numeric::types::DType::Int8;
        const BYTE_SIZE: usize = N;
        fn zero() -> Self {
            Self([std::num::NonZeroU8::new(1).unwrap(); N])
        }
        fn one() -> Self {
            Self([std::num::NonZeroU8::new(2).unwrap(); N])
        }
        fn from_f64(value: f64) -> Option<Self> {
            (value.is_finite() && (0.0..=254.0).contains(&value) && value.fract() == 0.0)
                .then(|| Self([std::num::NonZeroU8::new(value as u8 + 1).unwrap(); N]))
        }
        fn to_f64(self) -> f64 {
            self.0
                .first()
                .map_or(0.0, |value| f64::from(value.get() - 1))
        }
        fn is_finite(self) -> bool {
            true
        }
        fn abs(self) -> Self {
            self
        }
        fn add(self, other: Self) -> Self {
            Self::from_f64(self.to_f64() + other.to_f64()).unwrap()
        }
        fn sub(self, other: Self) -> Self {
            Self::from_f64(self.to_f64() - other.to_f64()).unwrap()
        }
        fn mul(self, other: Self) -> Self {
            Self::from_f64(self.to_f64() * other.to_f64()).unwrap()
        }
        fn div(self, other: Self) -> Self {
            Self::from_f64(self.to_f64() / other.to_f64()).unwrap()
        }
        fn simd_reduce_sum(values: &[Self]) -> Self {
            values.iter().copied().fold(Self::zero(), Self::add)
        }
        fn simd_reduce_max(values: &[Self]) -> Self {
            values
                .iter()
                .copied()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or_else(Self::zero)
        }
        fn simd_reduce_min(values: &[Self]) -> Self {
            values
                .iter()
                .copied()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or_else(Self::zero)
        }
    }

    #[test]
    fn safe_custom_numeric_types_are_initialized_with_their_own_valid_zero() {
        let buffer = SharedBuffer::<EncodedZero<1>>::new((2, 3), MatrixLayout::RowMajor).unwrap();
        assert!(buffer
            .data()
            .unwrap()
            .as_slice()
            .iter()
            .all(|v| *v == EncodedZero::<1>::zero()));
        let view = buffer.view_for_backend(BackendId::Native).unwrap();
        assert_eq!(view.data(), [EncodedZero::<1>::zero(); 6]);
        let copied = buffer.clone();
        assert_eq!(copied.to_vec().unwrap(), [EncodedZero::<1>::zero(); 6]);
    }

    #[test]
    fn zero_sized_custom_numeric_types_reject_before_requesting_allocator_storage() {
        assert!(matches!(
            SharedBuffer::<EncodedZero<0>>::new((2, 3), MatrixLayout::RowMajor),
            Err(MemoryError::InvalidSize)
        ));
        assert!(matches!(
            SharedBuffer::<EncodedZero<0>>::new((0, 0), MatrixLayout::RowMajor),
            Err(MemoryError::InvalidSize)
        ));
    }

    #[test]
    fn test_aligned_buffer_allocation() {
        let buffer = AlignedBuffer::<f64>::new(100).unwrap();
        assert_eq!(buffer.capacity, 100);

        // Check alignment
        let ptr = buffer.as_slice().as_ptr() as usize;
        assert_eq!(ptr % get_simd_alignment::<f64>(), 0);
    }

    #[test]
    fn test_shared_buffer_creation() {
        let buffer = SharedBuffer::<f64>::new((10, 20), MatrixLayout::RowMajor).unwrap();
        assert_eq!(buffer.shape(), (10, 20));
        assert_eq!(buffer.len(), 200);
        assert_eq!(buffer.layout(), MatrixLayout::RowMajor);
    }

    #[test]
    fn test_memory_pool_allocation() {
        let pool = AdvancedMemoryPool::<f64>::new();

        let buffer1 = pool.allocate(100).unwrap();
        let buffer2 = pool.allocate(100).unwrap();

        assert_eq!(buffer1.capacity, align_to_simd_boundary(100));
        assert_eq!(buffer2.capacity, align_to_simd_boundary(100));

        // Return buffers to pool
        pool.deallocate(buffer1);
        pool.deallocate(buffer2);

        // Allocate again - should reuse
        let _buffer3 = pool.allocate(100).unwrap();
        let stats = pool.stats();

        assert!(stats.cache_hit_rate > 0.0);
    }

    #[test]
    fn test_simd_alignment() {
        assert_eq!(align_to_simd_boundary(10), 64);
        assert_eq!(align_to_simd_boundary(64), 64);
        assert_eq!(align_to_simd_boundary(65), 128);
    }

    #[test]
    fn test_shared_buffer_from_data() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let buffer = SharedBuffer::from_data(data.clone(), (2, 3), MatrixLayout::RowMajor).unwrap();

        let retrieved_data = buffer.to_vec().unwrap();
        assert_eq!(retrieved_data[..6], data[..]);
    }
}
