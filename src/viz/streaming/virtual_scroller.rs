//! Virtual scrolling manager with LRU caching for large datasets
//!
//! Handles efficient data loading and caching for streaming tables with
//! millions of rows, ensuring responsive user experience.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use super::data_source::{DataSource, DataWindow, WindowKey};

/// Virtual scroll manager for handling large datasets efficiently
#[derive(Debug, Clone)]
pub struct VirtualScrollManager {
    /// Visible rows in viewport
    window_size: usize,

    /// Preloaded rows for smooth scrolling
    buffer_size: usize,

    /// Current scroll position
    current_offset: usize,

    /// LRU cache for data windows
    data_cache: Arc<RwLock<LRUCache<WindowKey, DataWindow>>>,

    /// Configuration
    config: VirtualScrollConfig,
}

impl VirtualScrollManager {
    /// Create new virtual scroll manager
    pub fn new(config: VirtualScrollConfig) -> Self {
        Self {
            window_size: config.window_size,
            buffer_size: config.buffer_size,
            current_offset: 0,
            data_cache: Arc::new(RwLock::new(LRUCache::new(config.cache_size))),
            config,
        }
    }

    /// Get visible window at current offset
    pub fn get_visible_window(
        &self,
        data_source: &dyn DataSource,
    ) -> VirtualScrollResult<DataWindow> {
        self.get_window_at_offset(data_source, self.current_offset)
    }

    /// Get window at specific offset with caching
    pub fn get_window_at_offset(
        &self,
        data_source: &dyn DataSource,
        offset: usize,
    ) -> VirtualScrollResult<DataWindow> {
        let cache_key = data_source.get_cache_key(offset, self.window_size);

        // Try cache first
        if let Some(cached_window) = self.get_from_cache(&cache_key) {
            return Ok(cached_window);
        }

        // Load from data source
        let start_time = std::time::Instant::now();
        let mut window = data_source.get_window(offset, self.window_size);
        let load_time = start_time.elapsed().as_millis() as u64;

        window.set_load_time(load_time);

        // Cache the result
        self.store_in_cache(cache_key, window.clone());

        Ok(window)
    }

    /// Handle scroll to new offset
    pub fn handle_scroll(
        &mut self,
        new_offset: usize,
        data_source: &dyn DataSource,
    ) -> VirtualScrollResult<UpdateResult> {
        let old_offset = self.current_offset;
        self.current_offset = new_offset;

        // Check if we need to load new data
        let needs_update = self.needs_data_update(old_offset, new_offset);

        if needs_update {
            // Preload buffer around new position
            self.preload_buffer(data_source, new_offset)?;

            let window = self.get_window_at_offset(data_source, new_offset)?;

            Ok(UpdateResult {
                updated: true,
                new_window: Some(window),
                old_offset,
                new_offset,
                cache_hits: self.get_cache_stats().hits,
                cache_misses: self.get_cache_stats().misses,
            })
        } else {
            Ok(UpdateResult {
                updated: false,
                new_window: None,
                old_offset,
                new_offset,
                cache_hits: self.get_cache_stats().hits,
                cache_misses: self.get_cache_stats().misses,
            })
        }
    }

    /// Preload buffer around given offset for smooth scrolling
    pub fn preload_buffer(
        &self,
        data_source: &dyn DataSource,
        center_offset: usize,
    ) -> VirtualScrollResult<()> {
        let total_rows = data_source.total_rows();

        // Calculate buffer range
        let buffer_start = center_offset.saturating_sub(self.buffer_size / 2);
        let buffer_end = std::cmp::min(center_offset + self.buffer_size, total_rows);

        // Load buffer in chunks
        let chunk_size = self.window_size;
        let mut current_pos = buffer_start;

        while current_pos < buffer_end {
            let chunk_end = std::cmp::min(current_pos + chunk_size, buffer_end);
            let chunk_count = chunk_end - current_pos;

            let cache_key = data_source.get_cache_key(current_pos, chunk_count);

            // Only load if not in cache
            if !self.is_in_cache(&cache_key) {
                let window = data_source.get_window(current_pos, chunk_count);
                self.store_in_cache(cache_key, window);
            }

            current_pos += chunk_size;
        }

        Ok(())
    }

    /// Check if scroll requires data update
    fn needs_data_update(&self, old_offset: usize, new_offset: usize) -> bool {
        let threshold = self.window_size / 4; // Update when scrolled 25% of window
        (new_offset as i64 - old_offset as i64).unsigned_abs() as usize > threshold
    }

    /// Get window from cache
    fn get_from_cache(&self, key: &WindowKey) -> Option<DataWindow> {
        self.data_cache.write().ok()?.get(key).map(|v| v.clone())
    }

    /// Store window in cache
    fn store_in_cache(&self, key: WindowKey, window: DataWindow) {
        if let Ok(mut cache) = self.data_cache.write() {
            let mut cached_window = window;
            cached_window.mark_cached();
            cache.put(key, cached_window);
        }
    }

    /// Check if key exists in cache
    fn is_in_cache(&self, key: &WindowKey) -> bool {
        self.data_cache
            .read()
            .map(|cache| cache.contains(key))
            .unwrap_or(false)
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> CacheStats {
        self.data_cache
            .read()
            .map(|cache| cache.get_stats())
            .unwrap_or_default()
    }

    /// Clear cache
    pub fn clear_cache(&self) {
        if let Ok(mut cache) = self.data_cache.write() {
            cache.clear();
        }
    }

    /// Get current configuration
    pub fn get_config(&self) -> &VirtualScrollConfig {
        &self.config
    }
}

/// Configuration for virtual scrolling
#[derive(Debug, Clone)]
pub struct VirtualScrollConfig {
    /// Visible rows in viewport (default: 50)
    pub window_size: usize,

    /// Preloaded rows for smooth scrolling (default: 200)
    pub buffer_size: usize,

    /// Maximum cached windows (default: 100)
    pub cache_size: usize,

    /// Auto-preload threshold (default: true)
    pub auto_preload: bool,

    /// Cache timeout in seconds (default: 300 = 5 minutes)
    pub cache_timeout_secs: u64,
}

impl Default for VirtualScrollConfig {
    fn default() -> Self {
        Self {
            window_size: 50,
            buffer_size: 200,
            cache_size: 100,
            auto_preload: true,
            cache_timeout_secs: 300,
        }
    }
}

/// Result of scroll update
#[derive(Debug, Clone)]
pub struct UpdateResult {
    pub updated: bool,
    pub new_window: Option<DataWindow>,
    pub old_offset: usize,
    pub new_offset: usize,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

/// Cache statistics
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub size: usize,
    pub capacity: usize,
    pub hit_rate: f64,
}

/// Simple LRU Cache implementation
#[derive(Debug)]
pub struct LRUCache<K, V> {
    capacity: usize,
    cache: HashMap<K, CacheEntry<V>>,
    access_order: Vec<K>,
    hits: u64,
    misses: u64,
}

#[derive(Debug, Clone)]
struct CacheEntry<V> {
    value: V,
    last_accessed: std::time::Instant,
}

impl<K: Clone + std::hash::Hash + Eq, V: Clone> LRUCache<K, V> {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            cache: HashMap::new(),
            access_order: Vec::new(),
            hits: 0,
            misses: 0,
        }
    }

    pub fn get(&mut self, key: &K) -> Option<V> {
        let value = if let Some(entry) = self.cache.get_mut(key) {
            entry.last_accessed = std::time::Instant::now();
            Some(entry.value.clone())
        } else {
            None
        };

        if value.is_some() {
            self.update_access_order(key.clone());
            self.hits += 1;
        } else {
            self.misses += 1;
        }

        value
    }

    pub fn put(&mut self, key: K, value: V) {
        if self.cache.len() >= self.capacity && !self.cache.contains_key(&key) {
            self.evict_lru();
        }

        let entry = CacheEntry {
            value,
            last_accessed: std::time::Instant::now(),
        };

        self.cache.insert(key.clone(), entry);
        self.update_access_order(key);
    }

    pub fn contains(&self, key: &K) -> bool {
        self.cache.contains_key(key)
    }

    pub fn clear(&mut self) {
        self.cache.clear();
        self.access_order.clear();
    }

    pub fn get_stats(&self) -> CacheStats {
        let total_requests = self.hits + self.misses;
        let hit_rate = if total_requests > 0 {
            self.hits as f64 / total_requests as f64
        } else {
            0.0
        };

        CacheStats {
            hits: self.hits,
            misses: self.misses,
            size: self.cache.len(),
            capacity: self.capacity,
            hit_rate,
        }
    }

    fn update_access_order(&mut self, key: K) {
        // Remove key if it exists
        self.access_order.retain(|k| k != &key);
        // Add to end (most recently used)
        self.access_order.push(key);
    }

    fn evict_lru(&mut self) {
        if let Some(lru_key) = self.access_order.first().cloned() {
            self.cache.remove(&lru_key);
            self.access_order.remove(0);
        }
    }
}

/// Error types for virtual scrolling
#[derive(Debug)]
pub enum VirtualScrollError {
    DataSource(String),
    Cache(String),
    InvalidOffset { offset: usize, max: usize },
    Config(String),
}

impl std::fmt::Display for VirtualScrollError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VirtualScrollError::DataSource(msg) => write!(f, "Data source error: {}", msg),
            VirtualScrollError::Cache(msg) => write!(f, "Cache error: {}", msg),
            VirtualScrollError::InvalidOffset { offset, max } => {
                write!(f, "Invalid offset: {} (max: {})", offset, max)
            }
            VirtualScrollError::Config(msg) => write!(f, "Configuration error: {}", msg),
        }
    }
}

impl std::error::Error for VirtualScrollError {}

pub type VirtualScrollResult<T> = Result<T, VirtualScrollError>;
