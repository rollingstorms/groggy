//! Distinct-spectrum real symmetric eigen and thin-SVD derivatives.
use super::derivative_linalg::*;
use crate::GraphResult;
fn separated(s: &[f64], dimension: usize, positive: bool) -> GraphResult<()> {
    let norm = s.iter().fold(0.0_f64, |a, x| a.max(x.abs()));
    let tolerance = 64. * f64::EPSILON * dimension.max(1) as f64 * norm;
    for (i, &v) in s.iter().enumerate() {
        if positive && v <= tolerance {
            return Err(invalid(
                "SVD derivative requires numerically resolved full rank",
            ));
        }
        for &other in &s[..i] {
            if (v - other).abs() <= tolerance {
                return Err(invalid(
                    "Spectral derivative requires separated distinct values",
                ));
            }
        }
    }
    Ok(())
}
pub(super) fn eigen_jvp(
    mut a: Vec<f64>,
    mut h: Vec<f64>,
    n: usize,
) -> GraphResult<(Vec<f64>, Vec<f64>)> {
    let alpha = normalize(&mut a)?;
    let gamma = normalize(&mut h)?;
    symmetric(&mut h, n)?;
    let (values, v) = super::decomposition::symmetric_eigen(a, n)?;
    separated(&values, n, false)?;
    let vt = transpose(&v, n, n)?;
    let middle = mul(&mul(&vt, &h, n, n, n)?, &v, n, n, n)?;
    let mut ds: Vec<_> = (0..n).map(|i| middle[i * n + i]).collect();
    scale(&mut ds, gamma, 1.)?;
    let mut omega = zeros(n, n)?;
    for i in 0..n {
        for j in 0..n {
            if i != j {
                omega[i * n + j] = finite(middle[i * n + j] / (values[j] - values[i]))?;
            }
        }
    }
    let mut dv = mul(&v, &omega, n, n, n)?;
    scale(&mut dv, gamma, alpha)?;
    Ok((ds, dv))
}
pub(super) fn eigen_vjp(
    mut a: Vec<f64>,
    mut gs: Vec<f64>,
    mut gv: Vec<f64>,
    n: usize,
) -> GraphResult<Vec<f64>> {
    let alpha = normalize(&mut a)?;
    let beta = normalize(&mut gs)?;
    let gamma = normalize(&mut gv)?;
    let (values, v) = super::decomposition::symmetric_eigen(a, n)?;
    separated(&values, n, false)?;
    let vt = transpose(&v, n, n)?;
    let b = mul(&vt, &gv, n, n, n)?;
    let mut middle = zeros(n, n)?;
    let mut diagonal = zeros(n, n)?;
    for i in 0..n {
        diagonal[i * n + i] = gs[i];
        for j in 0..n {
            if i != j {
                middle[i * n + j] = finite(b[i * n + j] / (values[j] - values[i]))?;
            }
        }
    }
    symmetrize(&mut middle, n);
    let mut out = mul(&mul(&v, &middle, n, n, n)?, &vt, n, n, n)?;
    scale(&mut out, gamma, alpha)?;
    let mut ds = mul(&mul(&v, &diagonal, n, n, n)?, &vt, n, n, n)?;
    scale(&mut ds, beta, 1.)?;
    let mut result = combine(&out, &ds, 1.)?;
    symmetrize(&mut result, n);
    Ok(result)
}
struct Thin {
    u: Vec<f64>,
    v: Vec<f64>,
    s: Vec<f64>,
    alpha: f64,
}
fn thin(mut a: Vec<f64>, m: usize, n: usize) -> GraphResult<Thin> {
    let alpha = normalize(&mut a)?;
    let factor = super::decomposition::svd_economy(a, m, n)?;
    let k = m.min(n);
    separated(&factor.singular, m.max(n), true)?;
    let u = factor.u;
    let v = transpose(&factor.vt, k, n)?;
    Ok(Thin {
        u,
        v,
        s: factor.singular,
        alpha,
    })
}
/// dU(m*k), dS(k), dVt(k*n), for positive distinct singular values.
pub(super) fn svd_jvp(
    a: Vec<f64>,
    mut h: Vec<f64>,
    m: usize,
    n: usize,
) -> GraphResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let Thin { u, v, s, alpha } = thin(a, m, n)?;
    let k = s.len();
    let gamma = normalize(&mut h)?;
    let ut = transpose(&u, m, k)?;
    let hv = mul(&h, &v, m, n, k)?;
    let middle = mul(&ut, &hv, k, m, k)?;
    let mut ou = zeros(k, k)?;
    let mut ov = zeros(k, k)?;
    for i in 0..k {
        for j in 0..k {
            if i != j {
                let denominator = (s[j] - s[i]) * (s[j] + s[i]);
                ou[i * k + j] =
                    finite((s[j] * middle[i * k + j] + s[i] * middle[j * k + i]) / denominator)?;
                ov[i * k + j] =
                    finite((s[i] * middle[i * k + j] + s[j] * middle[j * k + i]) / denominator)?;
            }
        }
    }
    let mut pu = combine(&hv, &mul(&u, &middle, m, k, k)?, -1.)?;
    let htu = mul(&transpose(&h, m, n)?, &u, n, m, k)?;
    let mut pv = combine(&htu, &mul(&v, &transpose(&middle, k, k)?, n, k, k)?, -1.)?;
    for i in 0..m {
        for j in 0..k {
            pu[i * k + j] = finite(pu[i * k + j] / s[j])?;
        }
    }
    for i in 0..n {
        for j in 0..k {
            pv[i * k + j] = finite(pv[i * k + j] / s[j])?;
        }
    }
    let mut du = combine(&mul(&u, &ou, m, k, k)?, &pu, 1.)?;
    let mut dv = combine(&mul(&v, &ov, n, k, k)?, &pv, 1.)?;
    let mut ds: Vec<_> = (0..k).map(|i| middle[i * k + i]).collect();
    scale(&mut du, gamma, alpha)?;
    scale(&mut dv, gamma, alpha)?;
    scale(&mut ds, gamma, 1.)?;
    Ok((du, ds, transpose(&dv, n, k)?))
}
pub(super) fn svd_vjp(
    a: Vec<f64>,
    mut gu: Vec<f64>,
    mut gs: Vec<f64>,
    gvt: Vec<f64>,
    m: usize,
    n: usize,
) -> GraphResult<Vec<f64>> {
    let Thin { u, v, s, alpha } = thin(a, m, n)?;
    let k = s.len();
    let beta = normalize(&mut gs)?;
    let mut joined = gu;
    joined.extend(gvt);
    let gamma = normalize(&mut joined)?;
    let gvt = joined.split_off(m * k);
    gu = joined;
    let gv = transpose(&gvt, k, n)?;
    let ut = transpose(&u, m, k)?;
    let vt = transpose(&v, n, k)?;
    let au = mul(&ut, &gu, k, m, k)?;
    let av = mul(&vt, &gv, k, n, k)?;
    let mut middle = zeros(k, k)?;
    let mut diagonal = zeros(k, k)?;
    for i in 0..k {
        diagonal[i * k + i] = gs[i];
        for j in 0..k {
            if i != j {
                middle[i * k + j] = finite(
                    ((au[i * k + j] - au[j * k + i]) * s[j]
                        + (av[i * k + j] - av[j * k + i]) * s[i])
                        / ((s[j] - s[i]) * (s[j] + s[i])),
                )?;
            }
        }
    }
    let mut pu = combine(&gu, &mul(&u, &au, m, k, k)?, -1.)?;
    let mut pv = combine(&gv, &mul(&v, &av, n, k, k)?, -1.)?;
    for i in 0..m {
        for j in 0..k {
            pu[i * k + j] = finite(pu[i * k + j] / s[j])?;
        }
    }
    for i in 0..n {
        for j in 0..k {
            pv[i * k + j] = finite(pv[i * k + j] / s[j])?;
        }
    }
    let first = mul(&mul(&u, &middle, m, k, k)?, &vt, m, k, n)?;
    let second = mul(&pu, &vt, m, k, n)?;
    let third = mul(&u, &transpose(&pv, n, k)?, m, k, n)?;
    let mut vector = combine(&combine(&first, &second, 1.)?, &third, 1.)?;
    scale(&mut vector, gamma, alpha)?;
    let mut singular = mul(&mul(&u, &diagonal, m, k, k)?, &vt, m, k, n)?;
    scale(&mut singular, beta, 1.)?;
    combine(&vector, &singular, 1.)
}
