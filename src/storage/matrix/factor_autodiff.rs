//! Checked factor output routes for the existing native computation tape.
use super::*;
impl<T: NumericType> GraphMatrix<T> {
    /// Economy SVD with matrix-valued singular values (k x 1) for tape seeding.
    /// All three outputs own data and can independently backpropagate to this input.
    pub fn svd_tracked(&self) -> GraphResult<(Self, Self, Self)> {
        let (u, values, vt) = self.thin_svd_parts()?;
        let k = values.len();
        let mut s = Self::factor_matrix(values, k, 1)?;
        s.row_labels = Some(u.column_names.clone());
        s.column_names = vec!["singular_value".into()];
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
    /// Economy SVD with owned singular-value list and tracked U/Vt when enabled.
    pub fn svd_thin(&self) -> GraphResult<(Self, Vec<f64>, Self)> {
        let (u, singular, vt) = self.thin_svd_parts()?;
        let u = self.track_factor(
            u,
            Operation::FactorOutput {
                part: FactorOutput::SvdU,
            },
        )?;
        let vt = self.track_factor(
            vt,
            Operation::FactorOutput {
                part: FactorOutput::SvdVt,
            },
        )?;
        Ok((u, singular, vt))
    }
    fn thin_svd_parts(&self) -> GraphResult<(Self, Vec<f64>, Self)> {
        let (m, n) = self.shape();
        let k = m.min(n);
        if self.column_names.len() != n
            || self
                .row_labels
                .as_ref()
                .is_some_and(|labels| labels.len() != m)
        {
            return Err(GraphError::InvalidInput(
                "SVD labels do not match shape".into(),
            ));
        }
        let factors =
            crate::storage::matrix::decomposition::svd_economy(self.decomposition_input()?, m, n)?;
        let mut u = Self::factor_matrix(factors.u, m, k)?;
        let mut vt = Self::factor_matrix(factors.vt, k, n)?;
        let basis: Vec<_> = (0..k).map(|i| format!("svd_{i}")).collect();
        u.row_labels = self.row_labels.clone();
        u.column_names = basis.clone();
        vt.row_labels = Some(basis);
        vt.column_names = self.column_names.clone();
        Ok((u, factors.singular, vt))
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
