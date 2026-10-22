//! Shared typed NumArray kernels. Preserve f64/Option outputs and IEEE propagation.
use std::cmp::Ordering;

/// Scalar conversion for typed reduction kernels; integer hints preserve differences.
pub trait Scalar: Copy {
    /// Convert the final scalar or a scaled working value to f64.
    fn as_f64(self) -> f64;
    /// Exact integer value when applicable.
    fn integer(self) -> Option<i128> {
        None
    }
    /// Numeric order, used only after NaNs have been screened.
    fn numeric_cmp(self, other: Self) -> Ordering {
        let (a, b) = (self.as_f64(), other.as_f64());
        if a == b {
            Ordering::Equal
        } else {
            a.total_cmp(&b)
        }
    }
}
macro_rules! integers { ($($ty:ty),*) => {$(impl Scalar for $ty {
    fn as_f64(self)->f64 {self as f64}
    fn integer(self)->Option<i128> {Some(i128::from(self))}
    fn numeric_cmp(self,other:Self)->Ordering {self.cmp(&other)}
})*}; }
integers!(i32, i64);
impl Scalar for bool {
    fn as_f64(self) -> f64 {
        if self {
            1.0
        } else {
            0.0
        }
    }
    fn integer(self) -> Option<i128> {
        Some(i128::from(self))
    }
    fn numeric_cmp(self, other: Self) -> Ordering {
        self.cmp(&other)
    }
}
impl Scalar for f32 {
    fn as_f64(self) -> f64 {
        f64::from(self)
    }
}
impl Scalar for f64 {
    fn as_f64(self) -> f64 {
        self
    }
}
#[derive(Default)]
struct Sum {
    value: f64,
    correction: f64,
}
impl Sum {
    fn add(&mut self, x: f64) {
        let next = self.value + x;
        self.correction += if self.value.abs() >= x.abs() {
            (self.value - next) + x
        } else {
            (x - next) + self.value
        };
        self.value = next;
    }
    fn total(self) -> f64 {
        self.value + self.correction
    }
}
fn aggregate<S: Scalar, I: ExactSizeIterator<Item = S> + Clone>(data: I, average: bool) -> f64 {
    let n = data.len();
    let mut integer = Some(0_i128);
    let mut sum = Sum::default();
    let mut scale = 0.0_f64;
    let (mut pos_inf, mut neg_inf) = (false, false);
    for value in data.clone() {
        integer = match (integer, value.integer()) {
            (Some(a), Some(b)) => a.checked_add(b),
            _ => None,
        };
        let x = value.as_f64();
        if x.is_nan() {
            return f64::NAN;
        }
        if x == f64::INFINITY {
            pos_inf = true;
        } else if x == f64::NEG_INFINITY {
            neg_inf = true;
        } else {
            sum.add(x);
            scale = scale.max(x.abs());
        }
    }
    if let Some(integer) = integer {
        return if average {
            integer as f64 / n as f64
        } else {
            integer as f64
        };
    }
    match (pos_inf, neg_inf) {
        (true, true) => return f64::NAN,
        (true, false) => return f64::INFINITY,
        (false, true) => return f64::NEG_INFINITY,
        _ => {}
    }
    let total = sum.total();
    if total.is_finite() {
        return if average { total / n as f64 } else { total };
    }
    if scale == 0.0 {
        return 0.0;
    }
    // Avoid intermediate overflow when the final sum or mean is representable.
    let mut scaled = Sum::default();
    for x in data {
        scaled.add(x.as_f64() / scale);
    }
    if average {
        (scaled.total() / n as f64).clamp(-1.0, 1.0) * scale
    } else {
        scaled.total() * scale
    }
}
/// Sum with an exact integer subtotal or compensated floating accumulation; empty=0.
pub fn sum<S: Scalar>(data: &[S]) -> f64 {
    sum_iter(data.iter().copied())
}
pub(crate) fn sum_iter<S: Scalar, I: ExactSizeIterator<Item = S> + Clone>(data: I) -> f64 {
    aggregate(data, false)
}
/// Mean; empty=None, NaNs propagate, opposite infinities produce NaN.
pub fn mean<S: Scalar>(data: &[S]) -> Option<f64> {
    mean_iter(data.iter().copied())
}
pub(crate) fn mean_iter<S: Scalar, I: ExactSizeIterator<Item = S> + Clone>(data: I) -> Option<f64> {
    if data.len() == 0 {
        None
    } else {
        Some(aggregate(data, true))
    }
}
fn dispersion<S: Scalar, I: ExactSizeIterator<Item = S> + Clone>(
    data: I,
    about: Option<f64>,
) -> Option<(f64, f64)> {
    let n = data.len();
    let origin = data.clone().next()?;
    if data.clone().any(|x| !x.as_f64().is_finite()) || about.is_some_and(|x| !x.is_finite()) {
        return Some((f64::NAN, 1.0));
    }
    if n == 1 {
        return Some((0.0, 1.0));
    }
    let offset = |x: S| match about {
        Some(mean) => x.as_f64() - mean,
        None => match (x.integer(), origin.integer()) {
            (Some(a), Some(b)) => a
                .checked_sub(b)
                .map_or_else(|| x.as_f64() - origin.as_f64(), |d| d as f64),
            _ => x.as_f64() - origin.as_f64(),
        },
    };
    let mut scale = 0.0_f64;
    let mut overflow = false;
    for x in data.clone() {
        let d = offset(x);
        overflow |= !d.is_finite();
        scale = scale.max(d.abs());
    }
    if overflow {
        scale = data
            .clone()
            .map(|x| x.as_f64().abs())
            .fold(about.unwrap_or(0.0).abs(), f64::max);
    }
    if scale == 0.0 {
        return Some((0.0, 1.0));
    }
    let normalized = |x: S| {
        if overflow {
            x.as_f64() / scale - about.unwrap_or_else(|| origin.as_f64()) / scale
        } else {
            offset(x) / scale
        }
    };
    let center = if about.is_some() {
        0.0
    } else {
        let mut s = Sum::default();
        for x in data.clone() {
            s.add(normalized(x));
        }
        s.total() / n as f64
    };
    let mut squares = Sum::default();
    for x in data {
        let delta = normalized(x) - center;
        squares.add(delta * delta);
    }
    Some((squares.total() / (n - 1) as f64, scale))
}
/// Sample variance; empty=None, finite singleton=0, nonfinite input=NaN.
pub fn variance<S: Scalar>(data: &[S]) -> Option<f64> {
    variance_iter(data.iter().copied())
}
pub(crate) fn variance_iter<S: Scalar, I: ExactSizeIterator<Item = S> + Clone>(
    data: I,
) -> Option<f64> {
    dispersion(data, None).map(|(factor, scale)| (factor * scale) * scale)
}
/// Population counterpart for matrix column APIs; empty=None, finite singleton=0.
pub(crate) fn population_variance_iter<S: Scalar, I: ExactSizeIterator<Item = S> + Clone>(
    data: I,
) -> Option<f64> {
    let n = data.len();
    dispersion(data, None).map(|(factor, scale)| {
        let factor = if n > 1 {
            factor * ((n - 1) as f64 / n as f64)
        } else {
            factor
        };
        (factor * scale) * scale
    })
}
/// Standard deviation computed before rescaling, even when variance overflows.
pub fn std_dev<S: Scalar>(data: &[S]) -> Option<f64> {
    std_iter(data.iter().copied())
}
pub(crate) fn std_iter<S: Scalar, I: ExactSizeIterator<Item = S> + Clone>(data: I) -> Option<f64> {
    dispersion(data, None).map(|(factor, scale)| factor.sqrt() * scale)
}
/// Compatibility moment around the caller's explicit center, with N-1 denominator.
/// Retains the legacy zero result for zero/one observation.
pub fn variance_with_mean(data: &[f64], center: f64) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    dispersion(data.iter().copied(), Some(center))
        .map(|(factor, scale)| (factor * scale) * scale)
        .unwrap_or(0.0)
}
/// Minimum/maximum return f64; propagate any NaN and retain the first tied scalar.
pub fn extremum<S: Scalar>(data: &[S], minimum: bool) -> Option<f64> {
    extremum_iter(data.iter().copied(), minimum)
}
pub(crate) fn extremum_iter<S: Scalar, I: Iterator<Item = S>>(
    data: I,
    minimum: bool,
) -> Option<f64> {
    let mut best: Option<S> = None;
    for x in data {
        if x.as_f64().is_nan() {
            return Some(f64::NAN);
        }
        if let Some(previous) = best {
            let order = x.numeric_cmp(previous);
            if (minimum && order == Ordering::Less) || (!minimum && order == Ordering::Greater) {
                best = Some(x);
            }
        } else {
            best = Some(x);
        }
    }
    best.map(Scalar::as_f64)
}
/// Safe midpoint for finite extremes, infinities, and exact integer pairs.
fn midpoint<S: Scalar>(a: S, b: S) -> f64 {
    if let (Some(x), Some(y)) = (a.integer(), b.integer()) {
        if let Some(total) = x.checked_add(y) {
            return total as f64 / 2.0;
        }
    }
    let (a, b) = (a.as_f64(), b.as_f64());
    if a == b {
        return a;
    }
    if !a.is_finite() || !b.is_finite() {
        return if a.is_infinite() && b.is_infinite() {
            f64::NAN
        } else if a.is_infinite() {
            a
        } else {
            b
        };
    }
    if a.is_sign_negative() == b.is_sign_negative() {
        a + (b - a) * 0.5
    } else {
        (a + b) * 0.5
    }
}
/// Median, preserving typed ordering until final f64 output; NaNs propagate.
pub fn median<S: Scalar>(data: &[S]) -> Option<f64> {
    median_iter(data.iter().copied())
}
pub(crate) fn median_iter<S: Scalar, I: Iterator<Item = S>>(data: I) -> Option<f64> {
    let mut sorted: Vec<S> = data.collect();
    if sorted.is_empty() {
        return None;
    }
    if sorted.iter().any(|x| x.as_f64().is_nan()) {
        return Some(f64::NAN);
    }
    sorted.sort_by(|a, b| a.numeric_cmp(*b));
    let middle = sorted.len() / 2;
    Some(if sorted.len().is_multiple_of(2) {
        midpoint(sorted[middle - 1], sorted[middle])
    } else {
        sorted[middle].as_f64()
    })
}
/// In-place median selector with bounded stack use; NaN input is not partitioned.
pub fn select_median(data: &mut [f64]) -> Option<f64> {
    if data.is_empty() {
        return None;
    }
    if data.iter().any(|x| x.is_nan()) {
        return Some(f64::NAN);
    }
    let odd = data.len() % 2 == 1;
    let middle = data.len() / 2;
    let (left, value, _) = data.select_nth_unstable_by(middle, |a, b| {
        if a == b {
            Ordering::Equal
        } else {
            a.total_cmp(b)
        }
    });
    let high = *value;
    if odd {
        Some(high)
    } else {
        let low = left.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Some(midpoint(low, high))
    }
}

/// Linear percentile at rank p*(N-1); invalid p/empty=None, NaNs propagate.
/// Source values are not modified. Output remains approximate f64.
pub fn percentile<S: Scalar>(data: &[S], p: f64) -> Option<f64> {
    percentile_iter(data.iter().copied(), p)
}
pub(crate) fn percentile_iter<S: Scalar, I: Iterator<Item = S>>(data: I, p: f64) -> Option<f64> {
    if !(0.0..=1.0).contains(&p) {
        return None;
    }
    let mut sorted: Vec<S> = data.collect();
    if sorted.is_empty() {
        return None;
    }
    if sorted.iter().any(|x| x.as_f64().is_nan()) {
        return Some(f64::NAN);
    }
    sorted.sort_by(|a, b| a.numeric_cmp(*b));
    let rank = p * (sorted.len() - 1) as f64;
    // Bound indices even when usize->f64 rounding puts the last rank above N-1.
    let low = (rank.floor() as usize).min(sorted.len() - 1);
    let high = (rank.ceil() as usize).min(sorted.len() - 1);
    let a = sorted[low];
    let b = sorted[high];
    if low == high {
        return Some(a.as_f64());
    }
    let weight = rank - low as f64;
    if weight == 0.5 {
        return Some(midpoint(a, b));
    }
    let (a, b) = (a.as_f64(), b.as_f64());
    Some(if a == b {
        a
    } else if !a.is_finite() || !b.is_finite() {
        a * (1.0 - weight) + b * weight
    } else if a.is_sign_negative() == b.is_sign_negative() {
        a + (b - a) * weight
    } else {
        a * (1.0 - weight) + b * weight
    })
}

/// Pearson correlation; mismatched/short/constant input=None, nonfinite=NaN.
/// Independent normalization avoids overflow/underflow of squared magnitudes.
pub fn correlate<X: Scalar, Y: Scalar>(x: &[X], y: &[Y]) -> Option<f64> {
    correlate_iter(x.iter().copied(), y.iter().copied())
}
fn centered<S: Scalar, I: ExactSizeIterator<Item = S> + Clone>(data: I) -> Vec<f64> {
    let Some(origin) = data.clone().next() else {
        return Vec::new();
    };
    let difference = |x: S| match (x.integer(), origin.integer()) {
        (Some(a), Some(b)) => a
            .checked_sub(b)
            .map_or_else(|| x.as_f64() - origin.as_f64(), |d| d as f64),
        _ => x.as_f64() - origin.as_f64(),
    };
    let overflow = data.clone().any(|x| !difference(x).is_finite());
    let scale = data
        .clone()
        .map(|x| {
            if overflow {
                x.as_f64().abs()
            } else {
                difference(x).abs()
            }
        })
        .fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return vec![0.0; data.len()];
    }
    let mut values: Vec<f64> = data
        .map(|x| {
            if overflow {
                x.as_f64() / scale - origin.as_f64() / scale
            } else {
                difference(x) / scale
            }
        })
        .collect();
    let mut total = Sum::default();
    for &x in &values {
        total.add(x);
    }
    let mean = total.total() / values.len() as f64;
    let mut residual_scale = 0.0_f64;
    for x in &mut values {
        *x -= mean;
        residual_scale = residual_scale.max(x.abs());
    }
    if residual_scale != 0.0 {
        for x in &mut values {
            *x /= residual_scale;
        }
    }
    values
}
pub(crate) fn correlate_iter<
    X: Scalar,
    Y: Scalar,
    I: ExactSizeIterator<Item = X> + Clone,
    J: ExactSizeIterator<Item = Y> + Clone,
>(
    x: I,
    y: J,
) -> Option<f64> {
    if x.len() != y.len() || x.len() < 2 {
        return None;
    }
    if x.clone().any(|v| !v.as_f64().is_finite()) || y.clone().any(|v| !v.as_f64().is_finite()) {
        return Some(f64::NAN);
    }
    let (x, y) = (centered(x), centered(y));
    let (mut xy, mut xx, mut yy) = (Sum::default(), Sum::default(), Sum::default());
    for (&a, &b) in x.iter().zip(&y) {
        xy.add(a * b);
        xx.add(a * a);
        yy.add(b * b);
    }
    let (xx, yy) = (xx.total(), yy.total());
    if xx == 0.0 || yy == 0.0 {
        return None;
    }
    Some((xy.total() / xx.sqrt() / yy.sqrt()).clamp(-1.0, 1.0))
}
