"""
Multivariate Student‑t Copula (Demarta & McNeil, 2004).
scipy/numpy implementation with optional GARCH+skew‑t marginals.
"""

import numpy as np
import pandas as pd
from scipy import stats, linalg
from sklearn.utils import resample
import config

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    print("Warning: arch not installed. Using empirical marginals.")


class TStudentCopula:
    def __init__(self, df_estimation_method="tail_dep", use_garch=config.USE_GARCH):
        self.corr = None
        self.df = None
        self.marginal_params = {}
        self.tickers = None
        self.fitted = False
        self.df_method = df_estimation_method
        self.use_garch = use_garch and ARCH_AVAILABLE

    # ------------------------------------------------------------------
    # 1. Fit copula to returns
    # ------------------------------------------------------------------
    def fit(self, returns: pd.DataFrame):
        self.tickers = returns.columns.tolist()
        n = len(self.tickers)
        if n < 2:
            return False

        # ---- Fit marginals: GARCH(1,1) + skew‑t on the EXACT data passed ----
        if self.use_garch:
            self._fit_garch_marginals(returns)
            # Build uniforms from standardized residuals
            U = np.column_stack([
                self._to_uniform_parametric(returns[col], col)
                for col in self.tickers
            ])
        else:
            # Empirical marginals
            self.marginal_ecdfs = {}
            self.marginal_quantiles = {}
            for col in self.tickers:
                data = returns[col].dropna().values
                self.marginal_quantiles[col] = np.sort(data)
                self.marginal_ecdfs[col] = stats.ecdf(data)
            U = np.column_stack([
                self._to_uniform(returns[col].values, col)
                for col in self.tickers
            ])
        U = np.clip(U, 1e-6, 1 - 1e-6)

        # ---- Estimate correlation matrix ----
        self.corr = np.corrcoef(U.T)

        # ---- Estimate degrees of freedom ----
        if self.df_method == "tail_dep":
            self.df = self._estimate_df_from_tail(U)
        else:
            self.df = self._estimate_df_mle(U)
        if self.df is None or np.isnan(self.df):
            self.df = 5.0
        self.df = max(3.0, min(self.df, 30.0))

        self.fitted = True
        return True

    # ------------------------------------------------------------------
    # 2. GARCH + skew‑t marginal fitting
    # ------------------------------------------------------------------
    def _fit_garch_marginals(self, returns):
        """Fit GARCH(1,1) with skew‑t innovations to each ETF column in returns."""
        for col in self.tickers:
            ret = returns[col].dropna() * 100   # scale to percentage for numerical stability
            try:
                model = arch_model(ret, mean='constant', vol='Garch',
                                  p=config.GARCH_P, q=config.GARCH_Q,
                                  dist=config.GARCH_DIST)
                res = model.fit(disp='off')
                # Extract standardized residuals (aligned to ret)
                std_resid = res.std_resid
                # Fit skew‑t to standardized residuals
                skewt_params = stats.jf_skew_t.fit(std_resid.dropna())
                self.marginal_params[col] = {
                    'garch_result': res,
                    'skewt_params': skewt_params,
                }
            except Exception as e:
                print(f"    GARCH fit failed for {col}: {e}. Falling back to empirical.")
                self._fallback_empirical(returns[col], col)

    def _fallback_empirical(self, series, col):
        data = series.dropna().values
        self.marginal_quantiles[col] = np.sort(data)
        self.marginal_ecdfs[col] = stats.ecdf(data)
        self.marginal_params[col] = None

    # ------------------------------------------------------------------
    # 3. Uniform conversion helpers
    # ------------------------------------------------------------------
    def _to_uniform(self, data, ticker):
        if self.use_garch and self.marginal_params.get(ticker) is not None:
            return self._to_uniform_parametric(data, ticker)
        return self.marginal_ecdfs[ticker].cdf.evaluate(data)

    def _to_uniform_parametric(self, data, ticker):
        """
        For a pandas Series `data` (the returns used to fit the GARCH),
        compute uniform via the fitted skew‑t CDF applied to standardized residuals.
        """
        params = self.marginal_params.get(ticker)
        if params is None:
            return stats.ecdf(data).cdf.evaluate(data.values)
        res = params['garch_result']
        std_resid = res.std_resid   # pandas Series aligned with data
        skewt_dist = stats.jf_skew_t(*params['skewt_params'])
        return skewt_dist.cdf(std_resid.values)

    def _from_uniform(self, u, ticker):
        u = np.clip(u, 1e-6, 1 - 1e-6)
        if self.use_garch and self.marginal_params.get(ticker) is not None:
            return self._from_uniform_parametric(u, ticker)
        sorted_data = self.marginal_quantiles[ticker]
        n = len(sorted_data)
        idx = u * (n - 1)
        lo = np.floor(idx).astype(int)
        hi = np.ceil(idx).astype(int)
        w = idx - lo
        return (1 - w) * sorted_data[lo] + w * sorted_data[hi]

    def _from_uniform_parametric(self, u, ticker):
        """Inverse skew‑t CDF, then scale by GARCH forecast."""
        params = self.marginal_params.get(ticker)
        if params is None:
            sorted_data = self.marginal_quantiles[ticker]
            n = len(sorted_data)
            idx = u * (n - 1)
            lo = np.floor(idx).astype(int)
            hi = np.ceil(idx).astype(int)
            w = idx - lo
            return (1 - w) * sorted_data[lo] + w * sorted_data[hi]
        skewt_dist = stats.jf_skew_t(*params['skewt_params'])
        z = skewt_dist.ppf(u)
        forecast = params['garch_result'].forecast(horizon=1, reindex=False)
        cond_vol_forecast = forecast.variance.iloc[-1, 0] ** 0.5 / 100
        cond_mean = params['garch_result'].params['mu'] / 100
        return cond_mean + z * cond_vol_forecast

    # ------------------------------------------------------------------
    # 4. Degrees of freedom estimation (unchanged)
    # ------------------------------------------------------------------
    def _estimate_df_from_tail(self, U):
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
        from scipy.optimize import brentq
        def f(nu):
            arg = np.sqrt((nu + 1) * 0.7 / 1.3)
            return 2 * stats.t.cdf(-arg, df=nu + 1) - lam
        try:
            nu = brentq(f, 2.5, 50.0)
            return nu
        except:
            return 5.0

    def _bivariate_tail_lambda(self, u, v):
        q = 0.05
        u_below = u < q
        v_below = v < q
        p_joint = np.mean(u_below & v_below)
        return p_joint / q

    def _estimate_df_mle(self, U):
        n = U.shape[1]
        dflist = []
        for i in range(n):
            for j in range(i + 1, n):
                try:
                    z_i = stats.t.ppf(stats.norm.ppf(U[:, i]), df=5)
                    z_j = stats.t.ppf(stats.norm.ppf(U[:, j]), df=5)
                    k = (stats.kurtosis(z_i) + stats.kurtosis(z_j)) / 2
                    df_est = 4.0 / max(k, 0.01) + 3.0
                    dflist.append(df_est)
                except:
                    pass
        return np.median(dflist) if dflist else 5.0

    # ------------------------------------------------------------------
    # 5. Monte Carlo simulation
    # ------------------------------------------------------------------
    def simulate(self, n_sim: int = 50000) -> pd.DataFrame:
        if not self.fitted:
            return pd.DataFrame()

        n = len(self.tickers)
        try:
            L = linalg.cholesky(self.corr, lower=True)
        except linalg.LinAlgError:
            L = np.linalg.cholesky(self.corr + np.eye(n) * 1e-6)

        S = np.random.chisquare(self.df, size=n_sim) / self.df
        X = np.random.randn(n_sim, n) @ L.T
        Z = X / np.sqrt(S)[:, None]
        U = stats.t.cdf(Z, df=self.df)
        U = np.clip(U, 1e-6, 1 - 1e-6)

        sim_returns = np.zeros_like(U)
        for i, ticker in enumerate(self.tickers):
            sim_returns[:, i] = self._from_uniform(U[:, i], ticker)

        return pd.DataFrame(sim_returns, columns=self.tickers)

    # ------------------------------------------------------------------
    # 6. Risk metrics with bootstrap confidence intervals
    # ------------------------------------------------------------------
    def compute_risk_metrics(self, sim_returns: pd.DataFrame) -> dict:
        metrics = {}
        for ticker in sim_returns.columns:
            rets = sim_returns[ticker].values
            exp_ret = np.mean(rets) * 252
            var_95, es_95 = self._var_es_bootstrap(rets, confidence=0.95, n_boot=config.BOOTSTRAP_SAMPLES)
            tail_penalty = config.TAIL_ADJUSTMENT_LAMBDA * abs(min(es_95['point'], 0))
            score = exp_ret - tail_penalty
            metrics[ticker] = {
                'expected_return': float(exp_ret),
                'var_95': var_95,
                'es_95': es_95,
                't_copula_score': float(score),
                'dof': float(self.df)
            }
        return metrics

    def _var_es_bootstrap(self, data, confidence=0.95, n_boot=1000, alpha=0.05):
        """Bootstrap VaR and ES with confidence intervals."""
        var_pt = np.percentile(data, alpha * 100)
        es_pt = data[data <= var_pt].mean() if np.sum(data <= var_pt) > 0 else var_pt

        var_boot = []
        es_boot = []
        for _ in range(n_boot):
            boot_sample = resample(data, replace=True, n_samples=len(data))
            v = np.percentile(boot_sample, alpha * 100)
            var_boot.append(v)
            es_boot.append(boot_sample[boot_sample <= v].mean() if np.sum(boot_sample <= v) > 0 else v)

        var_ci = (np.percentile(var_boot, 2.5), np.percentile(var_boot, 97.5))
        es_ci = (np.percentile(es_boot, 2.5), np.percentile(es_boot, 97.5))

        return {
            'point': float(var_pt),
            'lower': float(var_ci[0]),
            'upper': float(var_ci[1])
        }, {
            'point': float(es_pt),
            'lower': float(es_ci[0]),
            'upper': float(es_ci[1])
        }
