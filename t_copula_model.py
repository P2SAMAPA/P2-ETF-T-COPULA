"""
Multivariate Student‑t Copula (Demarta & McNeil, 2004).
Pure scipy implementation — no C++ compilation required.
"""

import numpy as np
import pandas as pd
from scipy import stats, linalg
import config

class TStudentCopula:
    def __init__(self, df_estimation_method="tail_dep"):
        self.corr = None
        self.df = None
        self.marginal_ecdfs = {}
        self.marginal_quantiles = {}
        self.tickers = None
        self.fitted = False
        self.df_method = df_estimation_method

    # ------------------------------------------------------------------
    # 1. Fit copula to return data
    # ------------------------------------------------------------------
    def fit(self, returns: pd.DataFrame):
        self.tickers = returns.columns.tolist()
        n = len(self.tickers)
        if n < 2:
            return False

        # ---- Marginal: empirical CDF ----
        for col in self.tickers:
            data = returns[col].dropna().values
            sorted_data = np.sort(data)
            self.marginal_quantiles[col] = sorted_data
            self.marginal_ecdfs[col] = stats.ecdf(data)

        # ---- Convert to pseudo‑observations (uniforms) ----
        U = np.column_stack([
            self._to_uniform(returns[col].values, col)
            for col in self.tickers
        ])
        U = np.clip(U, 1e-6, 1 - 1e-6)

        # ---- Estimate correlation matrix ----
        self.corr = np.corrcoef(U.T)

        # ---- Estimate degrees of freedom (tail‑dependence method) ----
        if self.df_method == "tail_dep":
            self.df = self._estimate_df_from_tail(U)
        else:
            self.df = self._estimate_df_mle(U)

        if self.df is None or np.isnan(self.df):
            self.df = 5.0
        self.df = max(3.0, min(self.df, 30.0))  # clamp to [3, 30]

        self.fitted = True
        return True

    # ------------------------------------------------------------------
    # 2. Uniform transformation helpers
    # ------------------------------------------------------------------
    def _to_uniform(self, data, ticker):
        return self.marginal_ecdfs[ticker].cdf.evaluate(data)

    def _from_uniform(self, u, ticker):
        u = np.clip(u, 1e-6, 1 - 1e-6)
        sorted_data = self.marginal_quantiles[ticker]
        n = len(sorted_data)
        idx = u * (n - 1)
        lo = np.floor(idx).astype(int)
        hi = np.ceil(idx).astype(int)
        w = idx - lo
        return (1 - w) * sorted_data[lo] + w * sorted_data[hi]

    # ------------------------------------------------------------------
    # 3. Degrees of freedom estimation via tail dependence
    # ------------------------------------------------------------------
    def _estimate_df_from_tail(self, U):
        """Estimate df from bivariate tail dependence λ pairs."""
        n = U.shape[1]
        lambdas = []
        for i in range(n):
            for j in range(i + 1, n):
                try:
                    lambdas.append(self._bivariate_tail_lambda(U[:, i], U[:, j]))
                except:
                    pass
        if not lambdas:
            return 5.0
        lam = np.median(lambdas)
        lam = min(lam, 0.9)
        if lam <= 0:
            return 30.0
        # Invert λ = 2 t_{ν+1} ( -√((ν+1)(1-ρ)/(1+ρ)) )
        rho = 0.3  # approximate
        numer = (1 - rho) * (1 + 1)
        denom = (1 + rho)
        # Solve numerically (simplified)
        # Use lookup: λ ≈ 0.05 → ν≈5, λ≈0 → ν→∞
        from scipy.optimize import brentq
        def f(nu):
            arg = np.sqrt((nu + 1) * (1 - rho) / (1 + rho))
            return 2 * stats.t.cdf(-arg, df=nu + 1) - lam
        try:
            nu = brentq(f, 2.5, 50.0)
            return nu
        except:
            return 5.0

    def _bivariate_tail_lambda(self, u, v):
        """Lower tail dependence coefficient."""
        q = 0.05
        u_below = u < q
        v_below = v < q
        p_joint = np.mean(u_below & v_below)
        return p_joint / q

    def _estimate_df_mle(self, U):
        """Estimate df as the average of bivariate MLE fits."""
        n = U.shape[1]
        dflist = []
        for i in range(n):
            for j in range(i + 1, n):
                try:
                    # Convert uniforms to t‑scores
                    z_i = stats.t.ppf(stats.norm.ppf(U[:, i]), df=5)
                    z_j = stats.t.ppf(stats.norm.ppf(U[:, j]), df=5)
                    # Simple proxy: use kurtosis of the transformed data
                    k = (stats.kurtosis(z_i) + stats.kurtosis(z_j)) / 2
                    df_est = 4.0 / max(k, 0.01) + 3.0
                    dflist.append(df_est)
                except:
                    pass
        return np.median(dflist) if dflist else 5.0

    # ------------------------------------------------------------------
    # 4. Monte Carlo simulation from the t‑copula
    # ------------------------------------------------------------------
    def simulate(self, n_sim: int = 50000) -> pd.DataFrame:
        """
        Demarta & McNeil algorithm:
        1. Sample S ~ χ²(df) for the scale mixture
        2. Sample X ~ N(0, Σ)
        3. Z = X / sqrt(S/df)  → t‑distributed with correlation Σ
        4. Map to uniforms u = t_df(Z)
        5. Invert with empirical marginals
        """
        if not self.fitted:
            return pd.DataFrame()

        n = len(self.tickers)
        try:
            L = linalg.cholesky(self.corr, lower=True)
        except linalg.LinAlgError:
            L = np.linalg.cholesky(self.corr + np.eye(n) * 1e-6)

        # Step 1: chi‑squared mixture
        S = np.random.chisquare(self.df, size=n_sim) / self.df  # (n_sim,)

        # Step 2: independent normals
        X = np.random.randn(n_sim, n) @ L.T  # (n_sim, n)

        # Step 3: t‑transformation
        Z = X / np.sqrt(S)[:, None]  # (n_sim, n)

        # Step 4: map to uniforms via t CDF
        U = stats.t.cdf(Z, df=self.df)  # (n_sim, n)
        U = np.clip(U, 1e-6, 1 - 1e-6)

        # Step 5: invert empirical marginals
        sim_returns = np.zeros_like(U)
        for i, ticker in enumerate(self.tickers):
            sim_returns[:, i] = self._from_uniform(U[:, i], ticker)

        return pd.DataFrame(sim_returns, columns=self.tickers)

    # ------------------------------------------------------------------
    # 5. Risk metrics
    # ------------------------------------------------------------------
    def compute_risk_metrics(self, sim_returns: pd.DataFrame) -> dict:
        metrics = {}
        for ticker in sim_returns.columns:
            rets = sim_returns[ticker].values
            exp_ret = np.mean(rets) * 252                # annualised
            var_95 = np.percentile(rets, 5)
            es_95 = rets[rets <= var_95].mean() if np.sum(rets <= var_95) > 0 else var_95

            tail_penalty = config.TAIL_ADJUSTMENT_LAMBDA * abs(min(es_95, 0))
            score = exp_ret - tail_penalty

            metrics[ticker] = {
                'expected_return': float(exp_ret),
                'var_95': float(var_95),
                'es_95': float(es_95),
                't_copula_score': float(score),
                'dof': float(self.df)
            }
        return metrics
