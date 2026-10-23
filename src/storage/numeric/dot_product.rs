//! Exact finite f64 products accumulated in fixed base-2^32 bins, rounded once.
//! Covers exponents -2148..2047 plus 64 carry bits for a usize-sized dot product.
const LIMBS: usize = 136;
const MASK: u128 = (1_u128 << 32) - 1;

#[derive(Clone)]
pub(crate) struct Dot {
    integer: Option<i128>,
    positive: [u128; LIMBS],
    negative: [u128; LIMBS],
    nan: bool,
    positive_infinity: bool,
    negative_infinity: bool,
}
impl Dot {
    pub(crate) fn new() -> Self {
        Self {
            integer: Some(0),
            positive: [0; LIMBS],
            negative: [0; LIMBS],
            nan: false,
            positive_infinity: false,
            negative_infinity: false,
        }
    }
    fn flush_integer(&mut self) {
        if let Some(integer) = self.integer.take() {
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
        }
    }
    pub(crate) fn add(&mut self, a: f64, b: f64) {
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
            self.flush_integer();
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
    pub(crate) fn total(self) -> f64 {
        self.total_divided_by(1)
    }
    /// Divide the exact accumulator by a positive count before rounding once.
    pub(crate) fn total_divided_by(self, divisor: usize) -> f64 {
        self.total_scaled_divided_by(divisor, 0)
    }
    /// Population moment root without requiring its squared value to fit in f64.
    pub(crate) fn sqrt_divided_by(self, divisor: usize) -> f64 {
        let variance = self.clone().total_divided_by(divisor);
        if variance.is_finite() {
            return variance.max(0.0).sqrt();
        }
        if variance.is_nan() {
            return f64::NAN;
        }
        self.total_scaled_divided_by(divisor, -1024).sqrt() * 2f64.powi(512)
    }
    fn total_scaled_divided_by(mut self, divisor: usize, exponent_adjust: i32) -> f64 {
        if divisor == 0 {
            return f64::NAN;
        }
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
            if divisor == 1 && exponent_adjust == 0 {
                return integer as f64;
            }
            if exponent_adjust == 0
                && integer.unsigned_abs() <= 1_u128 << 53
                && divisor as u128 <= 1_u128 << 53
            {
                return integer as f64 / divisor as f64;
            }
            self.flush_integer();
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
        let mut division_remainder = 0_u128;
        if divisor != 1 {
            for limb in magnitude.iter_mut().rev() {
                let numerator = (division_remainder << 32) | u128::from(*limb);
                *limb = (numerator / divisor as u128) as u32;
                division_remainder = numerator % divisor as u128;
            }
        }
        let Some(last) = magnitude.iter().rposition(|&v| v != 0) else {
            return 0.0;
        };
        let highest = last * 32 + 31 - magnitude[last].leading_zeros() as usize;
        let minimum_shift = (1074 - exponent_adjust).max(0) as usize;
        let shift = highest.saturating_sub(52).max(minimum_shift);
        let bit = |index: usize| (magnitude[index / 32] >> (index % 32)) & 1;
        let mut significand = 0_u64;
        for index in (shift..=highest).rev() {
            significand = (significand << 1) | u64::from(bit(index));
        }
        let halfway = bit(shift - 1) != 0;
        let discarded = shift - 1;
        let whole = discarded / 32;
        let remainder = discarded % 32;
        let sticky = division_remainder != 0
            || magnitude[..whole].iter().any(|&limb| limb != 0)
            || magnitude[whole] & ((1_u32 << remainder) - 1) != 0;
        if halfway && (sticky || significand & 1 != 0) {
            significand += 1;
        }
        let sign = u64::from(negative) << 63;
        if significand < 1_u64 << 52 {
            return f64::from_bits(sign | significand);
        }
        let mut exponent = shift as i32 - 2148 + 52 + exponent_adjust;
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

#[cfg(test)]
mod division_tests {
    use super::Dot;
    #[test]
    fn divide_before_rounding_preserves_representable_means_and_remainder_sticky_bits() {
        let mut total = Dot::new();
        total.add(f64::MAX, 1.0);
        total.add(f64::MAX, 1.0);
        assert_eq!(total.clone().total(), f64::INFINITY);
        assert_eq!(total.total_divided_by(2), f64::MAX);
        let mut integer = Dot::new();
        integer.add(1.0, 1.0);
        assert_eq!(integer.total_divided_by(3), 1.0 / 3.0);
        for sign in [1.0, -1.0] {
            let tiny = f64::from_bits(1);
            let mut boundary = Dot::new();
            boundary.add(sign * tiny, 1.5);
            boundary.add(sign * tiny, tiny);
            assert_eq!(
                boundary.total_divided_by(3).to_bits(),
                (sign * tiny).to_bits()
            );
        }
        let mut tie = Dot::new();
        tie.add(-f64::from_bits(1), 1.0);
        assert_eq!(tie.total_divided_by(2).to_bits(), (-0.0_f64).to_bits());
    }
}
