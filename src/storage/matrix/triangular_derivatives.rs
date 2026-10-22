//! Economy QR and fixed-pivot LU derivatives on resolved smooth branches.
use super::derivative_linalg::*;
use crate::GraphResult;
fn nonsingular(a: &[f64], n: usize) -> GraphResult<()> {
    let norm = a.iter().fold(0.0_f64, |s, v| s.max(v.abs()));
    let tolerance = 64. * f64::EPSILON * n.max(1) as f64 * norm;
    for i in 0..n {
        if a[i * n + i].abs() <= tolerance {
            return Err(invalid(
                "Factor derivative requires numerically resolved full rank",
            ));
        }
    }
    Ok(())
}
struct Qr {
    a: Vec<f64>,
    q: Vec<f64>,
    lead: Vec<f64>,
    alpha: f64,
    k: usize,
}
fn qr(mut a: Vec<f64>, m: usize, n: usize) -> GraphResult<Qr> {
    let alpha = normalize(&mut a)?;
    let (q, r) = super::factorization::qr_derivative_chart(a.clone(), m, n)?;
    let k = m.min(n);
    let lead = thin_columns(&r, k, n, k)?;
    nonsingular(&lead, k)?;
    Ok(Qr {
        a,
        q,
        lead,
        alpha,
        k,
    })
}
pub(super) fn qr_jvp(
    a: Vec<f64>,
    mut h: Vec<f64>,
    m: usize,
    n: usize,
) -> GraphResult<(Vec<f64>, Vec<f64>)> {
    let Qr {
        a,
        q,
        lead,
        alpha,
        k,
        ..
    } = qr(a, m, n)?;
    let gamma = normalize(&mut h)?;
    let x = right_solve(&thin_columns(&h, m, n, k)?, m, &lead, k, false, false)?;
    let middle = mul(&transpose(&q, m, k)?, &x, k, m, k)?;
    let mut omega = zeros(k, k)?;
    for i in 0..k {
        for j in 0..i {
            omega[i * k + j] = middle[i * k + j];
            omega[j * k + i] = -middle[i * k + j];
        }
    }
    let correction = combine(&middle, &omega, -1.)?;
    let mut dq = combine(&x, &mul(&q, &correction, m, k, k)?, -1.)?;
    let mut dr = combine(
        &mul(&transpose(&dq, m, k)?, &a, k, m, n)?,
        &mul(&transpose(&q, m, k)?, &h, k, m, n)?,
        1.,
    )?;
    for i in 0..k {
        for j in 0..i {
            dr[i * n + j] = 0.;
        }
    }
    scale(&mut dq, gamma, alpha)?;
    scale(&mut dr, gamma, 1.)?;
    Ok((dq, dr))
}
fn qr_pull(qr: &Qr, gq: &[f64], gr: &[f64], m: usize, n: usize) -> GraphResult<Vec<f64>> {
    let k = qr.k;
    let b = combine(gq, &mul(&qr.a, &transpose(gr, k, n)?, m, n, k)?, 1.)?;
    let middle = mul(&transpose(&qr.q, m, k)?, &b, k, m, k)?;
    let mut lower = zeros(k, k)?;
    for i in 0..k {
        for j in 0..i {
            lower[i * k + j] = finite(middle[i * k + j] - middle[j * k + i])?;
        }
    }
    let p = combine(
        &b,
        &mul(&qr.q, &combine(&lower, &middle, -1.)?, m, k, k)?,
        1.,
    )?;
    let lead = right_solve(&p, m, &qr.lead, k, false, true)?;
    let mut out = mul(&qr.q, gr, m, k, n)?;
    for i in 0..m {
        for j in 0..k {
            out[i * n + j] = finite(out[i * n + j] + lead[i * k + j])?;
        }
    }
    Ok(out)
}
pub(super) fn qr_vjp(
    a: Vec<f64>,
    mut gq: Vec<f64>,
    mut gr: Vec<f64>,
    m: usize,
    n: usize,
) -> GraphResult<Vec<f64>> {
    let qr = qr(a, m, n)?;
    let k = qr.k;
    if gr.iter().any(|v| !v.is_finite()) {
        return Err(invalid("QR cotangents must be finite"));
    }
    for i in 0..k {
        for j in 0..i {
            gr[i * n + j] = 0.;
        }
    }
    let beta = normalize(&mut gq)?;
    let gamma = normalize(&mut gr)?;
    let mut qpart = qr_pull(&qr, &gq, &zeros(k, n)?, m, n)?;
    scale(&mut qpart, beta, qr.alpha)?;
    let mut rpart = qr_pull(&qr, &zeros(m, k)?, &gr, m, n)?;
    scale(&mut rpart, gamma, 1.)?;
    combine(&qpart, &rpart, 1.)
}
struct Lu {
    l: Vec<f64>,
    u: Vec<f64>,
    p: Vec<usize>,
    alpha: f64,
}
fn lu(mut a: Vec<f64>, n: usize) -> GraphResult<Lu> {
    let alpha = normalize(&mut a)?;
    let factor = super::factorization::lu(a, n)?;
    nonsingular(&factor.upper, n)?;
    let tolerance = 64. * f64::EPSILON * n.max(1) as f64;
    for i in 0..n {
        for j in 0..i {
            if factor.lower[i * n + j].abs() >= 1. - tolerance {
                return Err(invalid(
                    "LU derivative has an unresolved pivot-selection tie",
                ));
            }
        }
    }
    Ok(Lu {
        l: factor.lower,
        u: factor.upper,
        p: factor.permutation,
        alpha,
    })
}
pub(super) fn lu_jvp(a: Vec<f64>, mut h: Vec<f64>, n: usize) -> GraphResult<(Vec<f64>, Vec<f64>)> {
    let Lu { l, u, p, alpha } = lu(a, n)?;
    let gamma = normalize(&mut h)?;
    let mut b = zeros(n, n)?;
    for i in 0..n {
        b[i * n..(i + 1) * n].copy_from_slice(&h[p[i] * n..(p[i] + 1) * n]);
    }
    let x = solve_triangular(&l, b, n, n, true, false)?;
    let y = right_solve(&x, n, &u, n, false, false)?;
    let mut lower = zeros(n, n)?;
    for i in 0..n {
        for j in 0..i {
            lower[i * n + j] = y[i * n + j];
        }
    }
    let mut dl = mul(&l, &lower, n, n, n)?;
    let mut du = mul(&combine(&y, &lower, -1.)?, &u, n, n, n)?;
    for i in 0..n {
        for j in 0..n {
            if i <= j {
                dl[i * n + j] = 0.;
            }
            if i > j {
                du[i * n + j] = 0.;
            }
        }
    }
    scale(&mut dl, gamma, alpha)?;
    scale(&mut du, gamma, 1.)?;
    Ok((dl, du))
}
fn lu_pull(lu: &Lu, gl: &[f64], gu: &[f64], n: usize) -> GraphResult<Vec<f64>> {
    let mut y = mul(gu, &transpose(&lu.u, n, n)?, n, n, n)?;
    let lower = mul(&transpose(&lu.l, n, n)?, gl, n, n, n)?;
    for i in 0..n {
        for j in 0..i {
            y[i * n + j] = lower[i * n + j];
        }
    }
    let x = right_solve(&y, n, &lu.u, n, false, true)?;
    let b = solve_triangular(&lu.l, x, n, n, true, true)?;
    let mut out = zeros(n, n)?;
    for i in 0..n {
        out[lu.p[i] * n..(lu.p[i] + 1) * n].copy_from_slice(&b[i * n..(i + 1) * n]);
    }
    Ok(out)
}
pub(super) fn lu_vjp(
    a: Vec<f64>,
    mut gl: Vec<f64>,
    mut gu: Vec<f64>,
    n: usize,
) -> GraphResult<Vec<f64>> {
    let lu = lu(a, n)?;
    if gl.iter().chain(&gu).any(|v| !v.is_finite()) {
        return Err(invalid("LU cotangents must be finite"));
    }
    for i in 0..n {
        for j in 0..n {
            if i <= j {
                gl[i * n + j] = 0.;
            }
            if i > j {
                gu[i * n + j] = 0.;
            }
        }
    }
    let beta = normalize(&mut gl)?;
    let gamma = normalize(&mut gu)?;
    let mut lpart = lu_pull(&lu, &gl, &zeros(n, n)?, n)?;
    scale(&mut lpart, beta, lu.alpha)?;
    let mut upart = lu_pull(&lu, &zeros(n, n)?, &gu, n)?;
    scale(&mut upart, gamma, 1.)?;
    combine(&lpart, &upart, 1.)
}
