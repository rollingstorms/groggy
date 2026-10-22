//! Exact finite f64 products accumulated in fixed base-2^32 bins, rounded once.
//! Covers exponents -2148..2047 plus 64 carry bits for a usize-sized dot product.
const LIMBS: usize = 136;
const MASK: u128 = (1_u128 << 32) - 1;

pub(super) struct Dot {
    integer: Option<i128>,
    positive: [u128; LIMBS],
    negative: [u128; LIMBS],
    nan: bool,
    positive_infinity: bool,
    negative_infinity: bool,
}
impl Dot {
    pub(super) fn new() -> Self {
        Self {
            integer: Some(0),
            positive: [0; LIMBS],
            negative: [0; LIMBS],
            nan: false,
            positive_infinity: false,
            negative_infinity: false,
        }
    }
    pub(super) fn add(&mut self, a: f64, b: f64) {
        if !a.is_finite() || !b.is_finite() {
            let product = a * b;
            self.nan |= product.is_nan();
            self.positive_infinity |= product == f64::INFINITY;
            self.negative_infinity |= product == f64::NEG_INFINITY;
            return;
        }
        if a == 0.0 || b == 0.0 {
            return;
        }
        if let Some(integer) = self.integer {
            if a.abs() <= 9223372036854775808.0
                && b.abs() <= 9223372036854775808.0
                && a.trunc() == a
                && b.trunc() == b
            {
                if let Some(next) = (a as i128)
                    .checked_mul(b as i128)
                    .and_then(|v| integer.checked_add(v))
                {
                    self.integer = Some(next);
                    return;
                }
            }
            // Preserve every bit when switching from the integer fast path to bins.
            let bins = if integer < 0 {
                &mut self.negative
            } else {
                &mut self.positive
            };
            let mut magnitude = integer.unsigned_abs();
            let mut limb = 2148 / 32;
            while magnitude != 0 {
                bins[limb] += (magnitude & MASK) << (2148 % 32);
                magnitude >>= 32;
                limb += 1;
            }
            self.integer = None;
        }
        let parts = |value: f64| {
            let bits = value.to_bits();
            let exponent = ((bits >> 52) & 0x7ff) as i32;
            let fraction = bits & ((1_u64 << 52) - 1);
            if exponent == 0 {
                (u128::from(fraction), -1074)
            } else {
                (u128::from(fraction | (1_u64 << 52)), exponent - 1075)
            }
        };
        let ((a_mantissa, a_exp), (b_mantissa, b_exp)) = (parts(a), parts(b));
        let mut product = a_mantissa * b_mantissa;
        let shift = (a_exp + b_exp + 2148) as usize;
        let (mut limb, offset) = (shift / 32, shift % 32);
        let bins = if a.is_sign_negative() ^ b.is_sign_negative() {
            &mut self.negative
        } else {
            &mut self.positive
        };
        while product != 0 {
            bins[limb] += (product & MASK) << offset;
            product >>= 32;
            limb += 1;
        }
    }
    pub(super) fn total(mut self) -> f64 {
        if self.nan || (self.positive_infinity && self.negative_infinity) {
            return f64::NAN;
        }
        if self.positive_infinity {
            return f64::INFINITY;
        }
        if self.negative_infinity {
            return f64::NEG_INFINITY;
        }
        if let Some(integer) = self.integer {
            return integer as f64;
        }
        for bins in [&mut self.positive, &mut self.negative] {
            for i in 0..LIMBS - 1 {
                bins[i + 1] += bins[i] >> 32;
                bins[i] &= MASK;
            }
        }
        let negative = self
            .positive
            .iter()
            .rev()
            .cmp(self.negative.iter().rev())
            .is_lt();
        let (large, small) = if negative {
            (&self.negative, &self.positive)
        } else {
            (&self.positive, &self.negative)
        };
        let mut magnitude = [0_u32; LIMBS];
        let mut borrow = 0_i64;
        for i in 0..LIMBS {
            let difference = large[i] as i64 - small[i] as i64 - borrow;
            borrow = i64::from(difference < 0);
            magnitude[i] = difference as u32;
        }
        let Some(last) = magnitude.iter().rposition(|&v| v != 0) else {
            return 0.0;
        };
        let highest = last * 32 + 31 - magnitude[last].leading_zeros() as usize;
        let shift = highest.saturating_sub(52).max(1074);
        let bit = |index: usize| (magnitude[index / 32] >> (index % 32)) & 1;
        let mut significand = 0_u64;
        for index in (shift..=highest).rev() {
            significand = (significand << 1) | u64::from(bit(index));
        }
        let halfway = bit(shift - 1) != 0;
        let discarded = shift - 1;
        let whole = discarded / 32;
        let remainder = discarded % 32;
        let sticky = magnitude[..whole].iter().any(|&limb| limb != 0)
            || magnitude[whole] & ((1_u32 << remainder) - 1) != 0;
        if halfway && (sticky || significand & 1 != 0) {
            significand += 1;
        }
        let sign = u64::from(negative) << 63;
        if significand < 1_u64 << 52 {
            return f64::from_bits(sign | significand);
        }
        let mut exponent = shift as i32 - 2148 + 52;
        if significand == 1_u64 << 53 {
            significand >>= 1;
            exponent += 1;
        }
        if exponent > 1023 {
            return f64::from_bits(sign | f64::INFINITY.to_bits());
        }
        f64::from_bits(
            sign | (((exponent + 1023) as u64) << 52) | (significand & ((1_u64 << 52) - 1)),
        )
    }
}
