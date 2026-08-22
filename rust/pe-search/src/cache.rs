//! A cache of network evaluations, keyed by the exact bytes of the model input.
//!
//! Self-play re-reaches the same position constantly: hundreds of games drawn
//! from one start pool share their openings, trees transpose, and a search
//! revisits ground its predecessor already covered. Every one of those is a
//! forward pass the GPU does not need to run twice.
//!
//! The key is a 128-bit hash of the encoded board rather than a Zobrist hash of
//! the position. The encoding carries repetition flags and the halfmove clock,
//! which a bare position hash does not, so keying on the position would return a
//! neighbour's evaluation for a board the network sees differently. Hashing the
//! encoding makes "same key" mean "same model input" by construction, which is
//! the only property that makes a hit safe.
//!
//! The table is direct-mapped and overwrites on collision. A cache miss costs a
//! forward pass that would have happened anyway, so eviction accuracy is worth
//! far less than a bounded, allocation-free lookup.

/// A 128-bit digest of one encoded board.
pub type EvalKey = (u64, u64);

/// Hash an encoded board.
///
/// Two accumulators run with different constants, orders, and a position-mixed
/// term, so the halves do not degenerate into the same function of the input.
/// 128 bits matters here: a run evaluating a few hundred million positions has a
/// percent-order chance of a 64-bit collision, and a false hit would silently
/// teach the network one position's policy under another's board.
pub fn hash_encoded(encoded: &[u8]) -> EvalKey {
    let mut low: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut high: u64 = 0xBF58_476D_1CE4_E5B9;
    for (index, chunk) in encoded.chunks_exact(8).enumerate() {
        let word = u64::from_le_bytes(chunk.try_into().expect("chunks_exact(8) yields 8 bytes"));
        low = (low ^ word).wrapping_mul(0x0000_0100_0000_01B3).rotate_left(31);
        high = (high.rotate_left(27) ^ word.wrapping_add(index as u64))
            .wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
    // The encoding is mostly zeros, so finish both halves with a strong
    // avalanche rather than trusting the accumulation to have spread the bits.
    (splitmix64(low), splitmix64(high ^ low))
}

fn splitmix64(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

struct Entry {
    key: EvalKey,
    value: f32,
    /// Logits for the position's legal actions, in ascending action order. The
    /// key pins the position, so this matches the caller's legal list exactly.
    logits: Box<[f32]>,
}

/// A direct-mapped table of network evaluations.
pub struct EvalCache {
    entries: Vec<Option<Entry>>,
    mask: usize,
    hits: u64,
    misses: u64,
}

impl EvalCache {
    /// Build a table holding at least `capacity` entries, rounded up to a power
    /// of two. Zero disables the cache, and every lookup then misses.
    pub fn new(capacity: usize) -> Self {
        if capacity == 0 {
            return Self {
                entries: Vec::new(),
                mask: 0,
                hits: 0,
                misses: 0,
            };
        }
        let slots = capacity.next_power_of_two();
        Self {
            entries: (0..slots).map(|_| None).collect(),
            mask: slots - 1,
            hits: 0,
            misses: 0,
        }
    }

    pub fn is_enabled(&self) -> bool {
        !self.entries.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.entries.len()
    }

    pub fn hits(&self) -> u64 {
        self.hits
    }

    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// Look one evaluation up, counting the outcome.
    ///
    /// The full key is compared, so a slot collision misses rather than
    /// returning the evaluation of whatever position landed there first.
    pub fn get(&mut self, key: EvalKey) -> Option<(&[f32], f32)> {
        if self.entries.is_empty() {
            self.misses += 1;
            return None;
        }
        let slot = key.0 as usize & self.mask;
        match &self.entries[slot] {
            Some(entry) if entry.key == key => {
                self.hits += 1;
                Some((&entry.logits, entry.value))
            }
            _ => {
                self.misses += 1;
                None
            }
        }
    }

    /// Record one evaluation, displacing whatever shared its slot.
    pub fn insert(&mut self, key: EvalKey, logits: &[f32], value: f32) {
        if self.entries.is_empty() {
            return;
        }
        let slot = key.0 as usize & self.mask;
        self.entries[slot] = Some(Entry {
            key,
            value,
            logits: logits.into(),
        });
    }
}

impl std::fmt::Debug for EvalCache {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("EvalCache")
            .field("capacity", &self.entries.len())
            .field("hits", &self.hits)
            .field("misses", &self.misses)
            .finish()
    }
}
