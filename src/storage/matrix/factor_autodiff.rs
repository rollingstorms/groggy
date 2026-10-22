//! Checked factor output routes for the existing native computation tape.
use super::*;
impl<T: NumericType> GraphMatrix<T> {
    /// Economy SVD with matrix-valued singular values (k x 1) for tape seeding.
    /// All three outputs own data and can independently backpropagate to this input.
    pub fn svd_tracked(&self) -> GraphResult<(Self, Self, Self)> {
        let (m, n) = self.shape();
        let k = m.min(n);
        let factors =
            crate::storage::matrix::decomposition::svd(self.decomposition_input()?, m, n)?;
        let u = (0..m)
            .flat_map(|row| factors.u[row * m..row * m + k].iter().copied())
            .collect();
        let vt = factors.vt[..k * n].to_vec();
        let u = Self::factor_matrix(u, m, k)?;
        let s = Self::factor_matrix(factors.singular, k, 1)?;
        let vt = Self::factor_matrix(vt, k, n)?;
        Ok((
            self.track_factor(
                u,
                Operation::FactorOutput {
                    part: FactorOutput::SvdU,
                },
            )?,
            self.track_factor(
                s,
                Operation::FactorOutput {
                    part: FactorOutput::SvdValues,
                },
            )?,
            self.track_factor(
                vt,
                Operation::FactorOutput {
                    part: FactorOutput::SvdVt,
                },
            )?,
        ))
    }
    /// Symmetric eigenpairs with matrix-valued eigenvalues (n x 1) for tape seeding.
    pub fn eigen_tracked(&self) -> GraphResult<(Self, Self)> {
        let (n, cols) = self.shape();
        if n != cols {
            return Err(GraphError::InvalidInput(
                "Symmetric eigen requires square input".into(),
            ));
        }
        let (values, vectors) =
            crate::storage::matrix::decomposition::symmetric_eigen(self.decomposition_input()?, n)?;
        Ok((
            self.track_factor(
                Self::factor_matrix(values, n, 1)?,
                Operation::FactorOutput {
                    part: FactorOutput::EigenValues,
                },
            )?,
            self.track_factor(
                Self::factor_matrix(vectors, n, n)?,
                Operation::FactorOutput {
                    part: FactorOutput::EigenVectors,
                },
            )?,
        ))
    }
    pub(crate) fn factor_output(&self, part: FactorOutput) -> GraphResult<Self> {
        match part {
            FactorOutput::QrQ => self.qr_decomposition().map(|(q, _)| q),
            FactorOutput::QrR => self.qr_decomposition().map(|(_, r)| r),
            FactorOutput::LuL => self.lu_decomposition().map(|(_, l, _)| l),
            FactorOutput::LuU => self.lu_decomposition().map(|(_, _, u)| u),
            FactorOutput::EigenValues => self.eigen_tracked().map(|(v, _)| v),
            FactorOutput::EigenVectors => self.eigen_tracked().map(|(_, v)| v),
            FactorOutput::SvdU => self.svd_tracked().map(|(u, _, _)| u),
            FactorOutput::SvdValues => self.svd_tracked().map(|(_, s, _)| s),
            FactorOutput::SvdVt => self.svd_tracked().map(|(_, _, v)| v),
        }
    }
    pub(crate) fn factor_output_vjp(
        &self,
        part: FactorOutput,
        cotangent: &Self,
    ) -> GraphResult<Self> {
        let (m, n) = self.shape();
        let k = m.min(n);
        match part {
            FactorOutput::QrQ => self.qr_vjp(cotangent, &Self::arithmetic_zeros(k, n)?),
            FactorOutput::QrR => self.qr_vjp(&Self::arithmetic_zeros(m, k)?, cotangent),
            FactorOutput::LuL => self.lu_vjp(cotangent, &Self::arithmetic_zeros(n, n)?),
            FactorOutput::LuU => self.lu_vjp(&Self::arithmetic_zeros(n, n)?, cotangent),
            FactorOutput::EigenValues => {
                if cotangent.shape() != (n, 1) {
                    return Err(GraphError::InvalidInput(
                        "Eigenvalue cotangent requires n x 1 shape".into(),
                    ));
                }
                self.eigen_vjp(
                    &cotangent.decomposition_input()?,
                    &Self::arithmetic_zeros(n, n)?,
                )
            }
            FactorOutput::EigenVectors => self.eigen_vjp(&vec![0.; n], cotangent),
            FactorOutput::SvdU => {
                self.svd_vjp(cotangent, &vec![0.; k], &Self::arithmetic_zeros(k, n)?)
            }
            FactorOutput::SvdVt => {
                self.svd_vjp(&Self::arithmetic_zeros(m, k)?, &vec![0.; k], cotangent)
            }
            FactorOutput::SvdValues => {
                if cotangent.shape() != (k, 1) {
                    return Err(GraphError::InvalidInput(
                        "Singular value cotangent requires k x 1 shape".into(),
                    ));
                }
                self.svd_vjp(
                    &Self::arithmetic_zeros(m, k)?,
                    &cotangent.decomposition_input()?,
                    &Self::arithmetic_zeros(k, n)?,
                )
            }
        }
    }
}
