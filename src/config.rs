//! Configuration Management System - Central settings and performance tuning.
//!
//! ARCHITECTURE ROLE:
//! This module provides centralized configuration for all system components.
//! It handles performance tuning, memory management, and feature toggles.
//!
//! DESIGN PHILOSOPHY:
//! - Single source of truth for all configuration
//! - Environment-aware defaults (production vs development)
//! - Runtime configuration validation
//! - Performance profile presets for common use cases

/*
=== CONFIGURATION SYSTEM OVERVIEW ===

The configuration system provides:
1. PERFORMANCE TUNING: Memory limits, cache sizes, optimization levels
2. FEATURE TOGGLES: Enable/disable optional features
3. OPERATIONAL SETTINGS: Logging levels, monitoring, debugging
4. STORAGE CONFIGURATION: Persistence, compression, backup settings

KEY DESIGN DECISIONS:
- Immutable configuration objects (changes require restart)
- Validation at creation time to catch errors early
- Profile-based presets for common scenarios
- Environment variable overrides for deployment flexibility
*/

use crate::core::strategies::StorageStrategyType;
use crate::types::GraphType;

/// The main configuration structure for the entire graph system
///
/// RESPONSIBILITIES:
/// - Define all configurable parameters in one place
/// - Provide validation for configuration values
/// - Support different performance profiles
/// - Enable/disable features consistently across components
///
/// CONFIGURATION CATEGORIES:
/// - Temporal Storage: Strategy selection for different workloads
/// - Memory Management: Limits, garbage collection thresholds
/// - Performance Tuning: Cache sizes, optimization levels
/// - History System: Snapshot frequency, retention policies
/// - Query Engine: Cache settings, timeout limits
/// - Storage: Compression, persistence, backup settings
#[derive(Debug, Clone)]
pub struct GraphConfig {
    /*
    === GRAPH STRUCTURE ===
    Fundamental graph properties that affect the entire system
    */
    /// Graph directionality - affects adjacency, traversal, and matrix operations
    /// DESIGN: This is the most fundamental property and affects all other operations
    pub graph_type: GraphType,

    /*
    === MEMORY MANAGEMENT ===
    Control memory usage across all components
    */
    /// Maximum total memory usage before triggering cleanup (bytes)
    /// When this limit is approached, the system will:
    /// 1. Clear query caches
    /// 2. Trigger garbage collection
    /// 3. Compress old history data
    pub max_memory_usage: usize,

    /// Memory threshold for starting proactive cleanup (percentage of max)
    /// Example: 80 means start cleanup when at 80% of max_memory_usage
    pub memory_pressure_threshold: u8,

    /// Enable automatic garbage collection of unreachable history
    pub enable_auto_gc: bool,

    /// How often to run garbage collection (in commits)
    pub gc_frequency: u32,

    /*
    === TEMPORAL STORAGE STRATEGY ===
    Configuration for different temporal storage approaches
    */
    /// Which temporal storage strategy to use
    /// Different strategies optimize for different workloads:
    /// - IndexDeltas: Efficient for frequent small changes (default)
    /// - Future: FullSnapshots, Hybrid, Compressed, etc.
    pub temporal_storage_strategy: StorageStrategyType,

    /*
    === HISTORY SYSTEM CONFIGURATION ===
    Settings for version control and state management
    */
    /// How often to create full snapshots instead of deltas (every N commits)
    /// Snapshots speed up state reconstruction but use more memory
    /// Trade-off: Lower values = faster access, higher memory usage
    pub snapshot_frequency: u32,

    /// Maximum length of delta chains before forcing a snapshot
    /// Long chains slow down state reconstruction
    /// This provides a hard limit even if snapshot_frequency hasn't been reached
    pub max_delta_chain: u32,

    /// Maximum number of states to keep in history
    /// Older states beyond this limit will be garbage collected
    /// None = unlimited history
    pub max_history_size: Option<usize>,

    /// Enable content-based deduplication of deltas
    /// Identical changes across different commits will share storage
    /// Trade-off: Saves memory but adds computational overhead
    pub enable_deduplication: bool,

    /*
    === QUERY ENGINE CONFIGURATION ===
    Settings for read-only operations and caching
    */
    /// Enable query result caching
    /// Repeated queries will return cached results
    pub enable_query_cache: bool,

    /// Maximum memory to use for query result cache (bytes)
    pub query_cache_size: usize,

    /// Query timeout in milliseconds
    /// Queries taking longer than this will be cancelled
    pub query_timeout_ms: u64,

    /// Maximum number of results to return from a single query
    /// Prevents accidentally loading huge result sets
    pub max_query_results: usize,

    /*
    === STORAGE AND PERSISTENCE ===
    Settings for data storage and compression
    */
    /// Enable compression of historical deltas
    /// Reduces storage size but adds CPU overhead
    pub enable_compression: bool,

    /// Compression level (1-9, higher = better compression, slower)
    pub compression_level: u8,

    /// Enable automatic backup of critical data
    pub enable_auto_backup: bool,

    /// How often to create backups (in commits)
    pub backup_frequency: u32,

    /*
    === PERFORMANCE OPTIMIZATION ===
    Settings for various performance optimizations
    */
    /// Enable adjacency list caching for fast neighbor lookups
    /// Trade-off: Faster queries but more memory usage
    pub enable_adjacency_cache: bool,

    /// Enable attribute indexing for faster filtered queries
    /// Trade-off: Faster attribute-based queries but more memory
    pub enable_attribute_indexing: bool,

    /// Number of worker threads for parallel operations
    /// 0 = auto-detect based on CPU cores
    pub worker_threads: usize,

    /// Enable parallel processing where possible
    pub enable_parallel_processing: bool,

    /*
    === DEBUGGING AND MONITORING ===
    Settings for development and troubleshooting
    */
    /// Enable detailed performance metrics collection
    pub enable_metrics: bool,

    /// Enable debug logging (warning: very verbose)
    pub enable_debug_logging: bool,

    /// Enable validation checks (slower but catches bugs)
    pub enable_validation: bool,

    /// Enable crash recovery mechanisms
    pub enable_crash_recovery: bool,
}

impl GraphConfig {
    /// Create a new configuration with reasonable defaults
    pub fn new() -> Self {
        // TODO: Initialize all other fields with balanced default values
        // For now, create a minimal config with the default strategy
        Self {
            graph_type: GraphType::default(),
            temporal_storage_strategy: StorageStrategyType::default(),
            // TODO: Initialize remaining fields with reasonable defaults
            max_memory_usage: 1024 * 1024 * 1024, // 1GB placeholder
            memory_pressure_threshold: 80,
            enable_auto_gc: true,
            gc_frequency: 100,
            snapshot_frequency: 100,
            max_delta_chain: 50,
            max_history_size: None,
            enable_deduplication: true,
            enable_query_cache: true,
            query_cache_size: 64 * 1024 * 1024, // 64MB
            query_timeout_ms: 30000,            // 30 seconds
            max_query_results: 100000,
            enable_compression: false,
            compression_level: 6,
            enable_auto_backup: false,
            backup_frequency: 1000,
            enable_adjacency_cache: true,
            enable_attribute_indexing: true,
            worker_threads: 0, // auto-detect
            enable_parallel_processing: true,
            enable_metrics: false,
            enable_debug_logging: false,
            enable_validation: false,
            enable_crash_recovery: true,
        }
    }

    /// Create a configuration optimized for low memory usage
    ///
    /// USE CASE: Embedded systems, resource-constrained environments
    /// OPTIMIZATIONS:
    /// - Index-based deltas (minimal memory overhead)
    /// - Aggressive garbage collection
    /// - Minimal caching
    /// - High compression
    /// - Frequent snapshots to keep delta chains short
    pub fn memory_optimized() -> Self {
        // TODO: Implement with all fields
        let mut config = Self::new();
        config.temporal_storage_strategy = StorageStrategyType::IndexDeltas; // Most memory efficient
        config
    }

    /// Create a configuration optimized for maximum performance
    ///
    /// USE CASE: High-performance computing, real-time applications
    /// OPTIMIZATIONS:
    /// - Index-based deltas (fastest commits)
    /// - Large memory buffers
    /// - Extensive caching
    /// - No compression
    /// - Parallel processing
    /// - Minimal garbage collection
    pub fn performance_optimized() -> Self {
        // TODO: Implement with all fields
        let mut config = Self::new();
        config.temporal_storage_strategy = StorageStrategyType::IndexDeltas; // Fastest commits
        config
    }

    /// Create a configuration optimized for development/debugging
    ///
    /// USE CASE: Development, testing, debugging
    /// FEATURES:
    /// - Extensive validation
    /// - Debug logging
    /// - Crash recovery
    /// - Metrics collection
    /// - Conservative resource usage
    pub fn development_optimized() -> Self {
        GraphConfig {
            max_memory_usage: 1024 * 1024 * 512, // 512MB
            memory_pressure_threshold: 80,
            enable_auto_gc: true,
            gc_frequency: 100,
            enable_query_cache: true,
            query_cache_size: 1024 * 1024, // 1MB for query cache
            enable_debug_logging: true,
            enable_validation: true,
            enable_metrics: true,
            worker_threads: 2,
            ..Self::default()
        }
    }

    /// Create a configuration for production deployment
    ///
    /// USE CASE: Production systems, stable deployments
    /// FEATURES:
    /// - Index-based deltas (proven and stable)
    /// - Balanced performance and reliability
    /// - Automatic backup and recovery
    /// - Moderate resource usage
    /// - Error handling without debug overhead
    pub fn production_optimized() -> Self {
        // TODO: Implement with all fields
        let mut config = Self::new();
        config.temporal_storage_strategy = StorageStrategyType::IndexDeltas; // Proven and stable
        config
    }

    /*
    === CONFIGURATION VALIDATION ===
    Ensure configuration values are reasonable
    */

    /// Validate the configuration and return errors if invalid
    pub fn validate(&self) -> Result<(), ConfigError> {
        // Basic validation of critical settings
        if self.max_memory_usage == 0 {
            return Err(ConfigError::InvalidConfiguration {
                setting: "max_memory_usage".to_string(),
                value: "0".to_string(),
                reason: "Memory limit must be greater than 0".to_string(),
                valid_values: vec!["Any positive integer".to_string()],
            });
        }

        if self.query_cache_size == 0 && self.enable_query_cache {
            return Err(ConfigError::InvalidConfiguration {
                setting: "query_cache_size".to_string(),
                value: "0".to_string(),
                reason: "Query cache size must be greater than 0 when caching is enabled"
                    .to_string(),
                valid_values: vec!["Any positive integer".to_string()],
            });
        }

        Ok(())
    }

    /// Get the effective number of worker threads
    /// (resolves 0 to actual CPU count)
    pub fn effective_worker_threads(&self) -> usize {
        if self.worker_threads == 0 {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        } else {
            self.worker_threads
        }
    }

    /// Calculate the memory threshold for triggering cleanup
    pub fn memory_cleanup_threshold(&self) -> usize {
        (self.max_memory_usage * self.memory_pressure_threshold as usize) / 100
    }

    /*
    === CONFIGURATION UPDATES ===
    Methods for modifying configuration (returns new instance)
    */

    /// Create a new configuration with updated memory limit
    pub fn with_memory_limit(mut self, limit: usize) -> Self {
        self.max_memory_usage = limit;
        self
    }

    /// Create a new configuration with updated cache settings
    pub fn with_cache_settings(mut self, enable_query_cache: bool, cache_size: usize) -> Self {
        self.enable_query_cache = enable_query_cache;
        self.query_cache_size = cache_size;
        self
    }

    /// Create a new configuration with updated worker thread count
    pub fn with_worker_threads(mut self, threads: usize) -> Self {
        self.worker_threads = threads;
        self
    }

    /// Create a new configuration with specific temporal storage strategy
    pub fn with_storage_strategy(mut self, strategy: StorageStrategyType) -> Self {
        self.temporal_storage_strategy = strategy;
        self
    }

    /*
    === ENVIRONMENT INTEGRATION ===
    Load configuration from environment variables
    */

    /// Load configuration overrides from environment variables
    ///
    /// SUPPORTED VARIABLES:
    /// - GROGGY_MAX_MEMORY: Maximum memory usage in bytes
    /// - GROGGY_WORKER_THREADS: Number of worker threads
    /// - GROGGY_ENABLE_DEBUG: Enable debug logging (true/false)
    /// - GROGGY_COMPRESSION_LEVEL: Compression level (1-9)
    pub fn from_environment(mut self) -> Self {
        if let Ok(memory) = std::env::var("GROGGY_MAX_MEMORY") {
            if let Ok(bytes) = memory.parse::<usize>() {
                self.max_memory_usage = bytes;
            }
        }

        if let Ok(threads) = std::env::var("GROGGY_WORKER_THREADS") {
            if let Ok(count) = threads.parse::<usize>() {
                self.worker_threads = count;
            }
        }

        if let Ok(debug) = std::env::var("GROGGY_ENABLE_DEBUG") {
            self.enable_debug_logging = debug.to_lowercase() == "true";
        }

        if let Ok(level) = std::env::var("GROGGY_COMPRESSION_LEVEL") {
            if let Ok(compression) = level.parse::<u8>() {
                if (1..=9).contains(&compression) {
                    self.compression_level = compression;
                }
            }
        }

        self
    }

    /// Save current configuration to environment variables
    pub fn to_environment(&self) {
        // TODO: Set environment variables based on current config
        // Useful for passing configuration to child processes
    }
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Errors that can occur during configuration validation
#[derive(Debug, Clone)]
pub enum ConfigError {
    /// Invalid memory limit (too low or too high)
    InvalidMemoryLimit {
        provided: usize,
        min: usize,
        max: usize,
    },

    /// Invalid compression level
    InvalidCompressionLevel { provided: u8, min: u8, max: u8 },

    /// Invalid thread count
    InvalidThreadCount {
        provided: usize,
        max_supported: usize,
    },

    /// Invalid percentage value
    InvalidPercentage { field: String, provided: u8 },

    /// Incompatible configuration options
    IncompatibleOptions {
        option1: String,
        option2: String,
        reason: String,
    },

    /// Invalid configuration setting
    InvalidConfiguration {
        setting: String,
        value: String,
        reason: String,
        valid_values: Vec<String>,
    },
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::InvalidMemoryLimit { provided, min, max } => {
                write!(
                    f,
                    "Invalid memory limit: {} bytes (must be between {} and {} bytes)",
                    provided, min, max
                )
            }
            ConfigError::InvalidCompressionLevel { provided, min, max } => {
                write!(
                    f,
                    "Invalid compression level: {} (must be between {} and {})",
                    provided, min, max
                )
            }
            ConfigError::InvalidThreadCount {
                provided,
                max_supported,
            } => {
                write!(
                    f,
                    "Invalid thread count: {} (maximum supported: {})",
                    provided, max_supported
                )
            }
            ConfigError::InvalidPercentage { field, provided } => {
                write!(
                    f,
                    "Invalid percentage for {}: {}% (must be between 0% and 100%)",
                    field, provided
                )
            }
            ConfigError::IncompatibleOptions {
                option1,
                option2,
                reason,
            } => {
                write!(
                    f,
                    "Incompatible configuration: {} and {} cannot both be enabled ({})",
                    option1, option2, reason
                )
            }
            ConfigError::InvalidConfiguration {
                setting,
                value,
                reason,
                valid_values,
            } => {
                write!(
                    f,
                    "Invalid configuration for {}: '{}' - {} (valid values: {})",
                    setting,
                    value,
                    reason,
                    valid_values.join(", ")
                )
            }
        }
    }
}

impl std::error::Error for ConfigError {}

/*
=== IMPLEMENTATION NOTES ===

CONFIGURATION PHILOSOPHY:
- Immutable configuration objects prevent runtime surprises
- Validation at creation time catches problems early
- Profile-based presets make common configurations easy
- Environment variable overrides support deployment flexibility

PERFORMANCE CONSIDERATIONS:
- Configuration validation should be fast (done once at startup)
- Environment variable reading should be cached
- Profile methods should be const-time operations

EXTENSIBILITY:
- New configuration options can be added without breaking existing code
- Profile methods can be customized for specific deployment scenarios
- Validation can be extended with custom rules

INTEGRATION WITH COMPONENTS:
- All system components should accept &GraphConfig in their constructors
- Components should validate config options they care about
- Components should respect memory limits and other constraints

TESTING STRATEGY:
- Test all profile methods produce valid configurations
- Test environment variable parsing handles edge cases
- Test validation catches all invalid combinations
- Test configuration serialization/deserialization
*/
