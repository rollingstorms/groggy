use xxhash_rust::xxh3::xxh3_64;

/// Fast hash function using xxHash
pub fn fast_hash(data: &[u8]) -> u64 {
    xxh3_64(data)
}

/// Hash a string 
pub fn hash_string(s: &str) -> u64 {
    fast_hash(s.as_bytes())
}

/// Create a hex string from hash
pub fn hash_to_hex(hash: u64) -> String {
    format!("{:016x}", hash)
}
