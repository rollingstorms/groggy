//! Bounded dense complex shifted-QR eigensolver for finite real matrices.
//! Full similarities deliberately favor a reviewable reference over throughput.
use crate::{GraphError, GraphResult};
use std::ops::{Add, Div, Mul, Neg, Sub};
#[derive(Clone, Copy, Debug, Default)]
struct C {
    re: f64,
    im: f64,
}
impl C {
    fn real(re: f64) -> Self {
        Self { re, im: 0. }
    }
    fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
    fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }
    fn sqrt(self) -> Self {
        let norm = self.abs();
        if norm == 0. {
            return Self::default();
        }
        let u = (norm / 2. + self.re.abs() / 2.).sqrt();
        if self.re >= 0. {
            Self {
                re: u,
                im: self.im / (2. * u),
            }
        } else {
            Self {
                re: self.im.abs() / (2. * u),
                im: u.copysign(self.im),
            }
        }
    }
    fn finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
}
impl Add for C {
    type Output = Self;
    fn add(self, b: Self) -> Self {
        Self {
            re: self.re + b.re,
            im: self.im + b.im,
        }
    }
}
impl Sub for C {
    type Output = Self;
    fn sub(self, b: Self) -> Self {
        Self {
            re: self.re - b.re,
            im: self.im - b.im,
        }
    }
}
impl Neg for C {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }
}
impl Mul for C {
    type Output = Self;
    fn mul(self, b: Self) -> Self {
        Self {
            re: self.re.mul_add(b.re, -self.im * b.im),
            im: self.re.mul_add(b.im, self.im * b.re),
        }
    }
}
impl Div for C {
    type Output = Self;
    fn div(self, b: Self) -> Self {
        if b.re.abs() >= b.im.abs() {
            let ratio = b.im / b.re;
            let denominator = b.re + b.im * ratio;
            Self {
                re: (self.re + self.im * ratio) / denominator,
                im: (self.im - self.re * ratio) / denominator,
            }
        } else {
            let ratio = b.re / b.im;
            let denominator = b.im + b.re * ratio;
            Self {
                re: (self.re * ratio + self.im) / denominator,
                im: (self.im * ratio - self.re) / denominator,
            }
        }
    }
}
struct Dot {
    re: super::dot_product::Dot,
    im: super::dot_product::Dot,
}
impl Default for Dot {
    fn default() -> Self {
        Self {
            re: super::dot_product::Dot::new(),
            im: super::dot_product::Dot::new(),
        }
    }
}
impl Dot {
    fn add(&mut self, a: C, b: C) {
        self.re.add(a.re, b.re);
        self.re.add(-a.im, b.im);
        self.im.add(a.re, b.im);
        self.im.add(a.im, b.re);
    }
    fn total(self) -> C {
        C {
            re: self.re.total(),
            im: self.im.total(),
        }
    }
}
fn invalid(message: &str) -> GraphError {
    GraphError::InvalidInput(message.into())
}
fn zeros(rows: usize, cols: usize) -> GraphResult<Vec<C>> {
    let count = rows
        .checked_mul(cols)
        .filter(|n| {
            n.checked_mul(std::mem::size_of::<C>())
                .is_some_and(|b| b <= isize::MAX as usize)
        })
        .ok_or_else(|| invalid("Complex eigen dimensions overflow"))?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| invalid("Complex eigen allocation failed"))?;
    values.resize(count, C::default());
    Ok(values)
}
fn identity(n: usize) -> GraphResult<Vec<C>> {
    let mut a = zeros(n, n)?;
    for i in 0..n {
        a[i * n + i] = C::real(1.);
    }
    Ok(a)
}
fn qr_q(mut a: Vec<C>, n: usize) -> GraphResult<Vec<C>> {
    let mut q = identity(n)?;
    for col in 0..n {
        let mut v: Vec<_> = (col..n).map(|row| a[row * n + col]).collect();
        let norm = v.iter().fold(0.0_f64, |s, v| s.hypot(v.abs()));
        if norm == 0. {
            continue;
        }
        let first = v[0].abs();
        let phase = if first == 0. {
            C::real(1.)
        } else {
            v[0] / C::real(first)
        };
        v[0] = v[0] + phase * C::real(norm);
        let norm = v.iter().fold(0.0_f64, |s, v| s.hypot(v.abs()));
        for value in &mut v {
            *value = *value / C::real(norm);
        }
        for j in col..n {
            let mut dot = Dot::default();
            for row in col..n {
                dot.add(v[row - col].conj(), a[row * n + j]);
            }
            let projection = dot.total() * C::real(2.);
            for row in col..n {
                a[row * n + j] = a[row * n + j] - v[row - col] * projection;
            }
        }
        for row in 0..n {
            let mut dot = Dot::default();
            for j in col..n {
                dot.add(q[row * n + j], v[j - col]);
            }
            let projection = dot.total() * C::real(2.);
            for j in col..n {
                q[row * n + j] = q[row * n + j] - projection * v[j - col].conj();
            }
        }
    }
    if q.iter().any(|v| !v.finite()) {
        return Err(invalid("Complex QR exceeds working range"));
    }
    Ok(q)
}
// Apply an active Q by full similarity, including coupling to deflated blocks.
fn similarity(h: &mut Vec<C>, z: &mut Vec<C>, q: &[C], n: usize, active: usize) -> GraphResult<()> {
    let mut right = h.clone();
    let mut vectors = z.clone();
    for row in 0..n {
        for col in 0..active {
            let mut hd = Dot::default();
            let mut zd = Dot::default();
            for k in 0..active {
                hd.add(h[row * n + k], q[k * active + col]);
                zd.add(z[row * n + k], q[k * active + col]);
            }
            right[row * n + col] = hd.total();
            vectors[row * n + col] = zd.total();
        }
    }
    let mut result = right.clone();
    for row in 0..active {
        for col in 0..n {
            let mut dot = Dot::default();
            for k in 0..active {
                dot.add(q[k * active + row].conj(), right[k * n + col]);
            }
            result[row * n + col] = dot.total();
        }
    }
    if result.iter().chain(&vectors).any(|v| !v.finite()) {
        return Err(invalid("Complex similarity exceeds working range"));
    }
    *h = result;
    *z = vectors;
    Ok(())
}
fn shift(h: &[C], n: usize, m: usize, iteration: usize) -> C {
    let d = h[(m - 1) * n + m - 1];
    if iteration % 32 == 31 {
        let bump = h[(m - 1) * n + m - 2].abs();
        return d + C {
            re: 0.75 * bump,
            im: 0.25 * bump,
        };
    }
    let a = h[(m - 2) * n + m - 2];
    let b = h[(m - 2) * n + m - 1];
    let c = h[(m - 1) * n + m - 2];
    // Scale the trailing pair independently: global scaling cannot protect tiny blocks.
    let scale = a.abs().max(b.abs()).max(c.abs()).max(d.abs());
    if scale == 0. {
        return d;
    }
    let divisor = C::real(scale);
    let (a, b, c, d) = (a / divisor, b / divisor, c / divisor, d / divisor);
    let midpoint = (a + d) * C::real(0.5);
    let half = (a - d) * C::real(0.5);
    let root = (half * half + b * c).sqrt();
    let first = midpoint + root;
    let second = midpoint - root;
    let nearest = if (first - d).abs() < (second - d).abs() {
        first
    } else {
        second
    };
    nearest * divisor
}
pub(super) struct Eigen {
    pub values_real: Vec<f64>,
    pub values_imag: Vec<f64>,
    pub vectors_real: Vec<f64>,
    pub vectors_imag: Vec<f64>,
}
pub(super) fn eigen(input: Vec<f64>, n: usize) -> GraphResult<Eigen> {
    eigen_with_limit(input, n, 256)
}
fn eigen_with_limit(mut input: Vec<f64>, n: usize, limit: usize) -> GraphResult<Eigen> {
    let scale = super::factorization::normalize(&mut input)?;
    let mut h: Vec<_> = input.iter().copied().map(C::real).collect();
    let mut z = identity(n)?;
    let tolerance = 64. * f64::EPSILON * n.max(1) as f64;
    let mut active = n;
    let mut iteration = 0;
    while active > 1 {
        let bottom_diagonal = h[(active - 1) * n + active - 1].abs();
        let deflatable = (0..active - 1).all(|j| {
            h[(active - 1) * n + j].abs() <= tolerance * (h[j * n + j].abs() + bottom_diagonal)
        });
        if deflatable {
            for col in 0..active - 1 {
                h[(active - 1) * n + col] = C::default();
            }
            active -= 1;
            iteration = 0;
            continue;
        }
        if iteration >= limit {
            return Err(invalid(
                "Complex eigen QR did not converge within its bounded deflation budget",
            ));
        }
        let shift = shift(&h, n, active, iteration);
        let mut block = zeros(active, active)?;
        for row in 0..active {
            for col in 0..active {
                block[row * active + col] =
                    h[row * n + col] - if row == col { shift } else { C::default() };
            }
        }
        let q = qr_q(block, active)?;
        similarity(&mut h, &mut z, &q, n, active)?;
        iteration += 1;
    }
    let mut vectors = zeros(n, n)?;
    let norm = input.iter().fold(0.0_f64, |s, v| s.hypot(*v));
    for col in 0..n {
        let lambda = h[col * n + col];
        let mut y = vec![C::default(); n];
        y[col] = C::real(1.);
        for row in (0..col).rev() {
            let mut dot = Dot::default();
            for k in row + 1..=col {
                dot.add(-h[row * n + k], y[k]);
            }
            let numerator = dot.total();
            let pivot = h[row * n + row] - lambda;
            if pivot.abs() <= tolerance * (h[row * n + row].abs() + lambda.abs()) {
                let row_scale = (row..n).fold(lambda.abs(), |s, k| s.max(h[row * n + k].abs()));
                if numerator.abs() > tolerance * row_scale {
                    return Err(invalid(
                        "Complex eigenbasis is defective or numerically unresolved",
                    ));
                }
            } else {
                y[row] = numerator / pivot;
            }
            let maximum = y.iter().fold(0.0_f64, |s, v| s.max(v.abs()));
            if maximum > 1. {
                for value in &mut y {
                    *value = *value / C::real(maximum);
                }
            }
        }
        let mut v = vec![C::default(); n];
        for row in 0..n {
            let mut dot = Dot::default();
            for k in 0..=col {
                dot.add(z[row * n + k], y[k]);
            }
            v[row] = dot.total();
        }
        let length = v.iter().fold(0.0_f64, |s, v| s.hypot(v.abs()));
        if !length.is_finite() || length == 0. {
            return Err(invalid("Complex eigenvector exceeds working range"));
        }
        for value in &mut v {
            *value = *value / C::real(length);
        }
        let pivot = (0..n).max_by(|&i, &j| v[i].abs().total_cmp(&v[j].abs()));
        if let Some(pivot) = pivot {
            let phase = v[pivot].conj() / C::real(v[pivot].abs());
            for value in &mut v {
                *value = *value * phase;
            }
        }
        let mut residual = 0.0_f64;
        for row in 0..n {
            let mut dot = Dot::default();
            for k in 0..n {
                dot.add(C::real(input[row * n + k]), v[k]);
            }
            dot.add(-lambda, v[row]);
            residual = residual.hypot(dot.total().abs());
        }
        if !residual.is_finite() || residual > 8. * tolerance * (norm + lambda.abs()) {
            return Err(invalid("Complex eigenpair residual failed verification"));
        }
        for row in 0..n {
            vectors[row * n + col] = v[row];
        }
    }
    // Reject duplicate/near-dependent eigenvectors rather than promise a full basis.
    let mut basis = vectors.clone();
    for col in 0..n {
        for _ in 0..2 {
            for previous in 0..col {
                let mut dot = Dot::default();
                for row in 0..n {
                    dot.add(basis[row * n + previous].conj(), basis[row * n + col]);
                }
                let projection = dot.total();
                for row in 0..n {
                    basis[row * n + col] =
                        basis[row * n + col] - basis[row * n + previous] * projection;
                }
            }
        }
        let length = (0..n).fold(0.0_f64, |s, row| s.hypot(basis[row * n + col].abs()));
        if length <= tolerance || !length.is_finite() {
            return Err(invalid(
                "Complex eigenbasis is defective or too ill-conditioned",
            ));
        }
        for row in 0..n {
            basis[row * n + col] = basis[row * n + col] / C::real(length);
        }
    }
    let mut order: Vec<_> = (0..n).collect();
    order.sort_by(|&i, &j| {
        h[j * n + j]
            .re
            .total_cmp(&h[i * n + i].re)
            .then(h[j * n + j].im.total_cmp(&h[i * n + i].im))
            .then(i.cmp(&j))
    });
    let mut real = Vec::with_capacity(n);
    let mut imag = Vec::with_capacity(n);
    let mut vr = super::factorization::zeros(n, n)?;
    let mut vi = super::factorization::zeros(n, n)?;
    for (col, &old) in order.iter().enumerate() {
        let value = h[old * n + old] * C::real(scale);
        if !value.finite() {
            return Err(invalid("Complex eigenvalue exceeds finite range"));
        }
        real.push(value.re);
        imag.push(value.im);
        for row in 0..n {
            vr[row * n + col] = vectors[row * n + old].re;
            vi[row * n + col] = vectors[row * n + old].im;
        }
    }
    Ok(Eigen {
        values_real: real,
        values_imag: imag,
        vectors_real: vr,
        vectors_imag: vi,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exhausted_budget_is_an_explicit_error() {
        assert!(eigen_with_limit(vec![0., -1., 1., 0.], 2, 0).is_err());
    }
}
