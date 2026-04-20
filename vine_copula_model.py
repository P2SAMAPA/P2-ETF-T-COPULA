"""
Vine copula modeling using pyvinecopulib.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

try:
    import vinecopulib as vc
    VINECOP_AVAILABLE = True
except ImportError:
    VINECOP_AVAILABLE = False
    print("Warning: vinecopulib not installed. Using fallback Gaussian copula.")


class VineCopulaModel:
    def __init__(self, margin_model='empirical'):
        self.margin_model = margin_model
        self.fitted = False
        self.marginals = {}
        self.vine = None
        self.tickers = None
        self.scaler = StandardScaler()
        
    def fit(self, returns: pd.DataFrame):
        """Fit vine copula to multivariate returns."""
        self.tickers = returns.columns.tolist()
        n = len(self.tickers)
        if n < 2:
            return False
        
        # Store marginal models (empirical CDF by default)
        self.marginals = {}
        for col in self.tickers:
            data = returns[col].values
            if self.margin_model == 'empirical':
                self.marginals[col] = {
                    'ecdf': stats.ecdf(data),
                    'mean': np.mean(data),
                    'std': np.std(data)
                }
            else:
                # Fit skew-t (simplified as t for now)
                params = stats.t.fit(data)
                self.marginals[col] = {'dist': stats.t, 'params': params}
        
        # Transform to uniform pseudo-observations
        u_data = np.column_stack([
            self._to_uniform(returns[col].values, col) for col in self.tickers
        ])
        
        if VINECOP_AVAILABLE:
            # Fit R-vine copula
            controls = vc.FitControlsVinecop(family_set=[vc.BicopFamily.tll])
            self.vine = vc.Vinecop(u_data, controls=controls)
        else:
            # Fallback: Gaussian copula correlation matrix
            self.vine = np.corrcoef(u_data.T)
        
        self.fitted = True
        return True
    
    def _to_uniform(self, data, ticker):
        """Convert data to uniform using fitted marginal."""
        if self.margin_model == 'empirical':
            return self.marginals[ticker]['ecdf'].cdf.evaluate(data)
        else:
            dist = self.marginals[ticker]['dist']
            params = self.marginals[ticker]['params']
            return dist.cdf(data, *params)
    
    def _from_uniform(self, u, ticker):
        """Convert uniform back to original scale."""
        if self.margin_model == 'empirical':
            return self.marginals[ticker]['ecdf'].cdf.quantile(u)
        else:
            dist = self.marginals[ticker]['dist']
            params = self.marginals[ticker]['params']
            return dist.ppf(u, *params)
    
    def simulate(self, n_sim: int = 10000) -> pd.DataFrame:
        """Simulate next-day returns from fitted vine copula."""
        if not self.fitted:
            return pd.DataFrame()
        
        if VINECOP_AVAILABLE and isinstance(self.vine, vc.Vinecop):
            u_sim = self.vine.simulate(n_sim)
        else:
            # Fallback: multivariate normal on uniform scores
            mean = np.zeros(len(self.tickers))
            u_sim = stats.multivariate_normal.rvs(mean=mean, cov=self.vine, size=n_sim)
            u_sim = stats.norm.cdf(u_sim)
            u_sim = np.clip(u_sim, 1e-6, 1-1e-6)
        
        sim_returns = np.zeros_like(u_sim)
        for i, ticker in enumerate(self.tickers):
            sim_returns[:, i] = self._from_uniform(u_sim[:, i], ticker)
        
        return pd.DataFrame(sim_returns, columns=self.tickers)
    
    def compute_risk_metrics(self, sim_returns: pd.DataFrame) -> dict:
        """Compute expected return, VaR, ES, and combined score per ETF."""
        metrics = {}
        for ticker in sim_returns.columns:
            rets = sim_returns[ticker].values
            exp_ret = np.mean(rets)
            var_95 = np.percentile(rets, 5)
            es_95 = rets[rets <= var_95].mean() if np.sum(rets <= var_95) > 0 else var_95
            
            # Combined score: expected return minus lambda * tail risk (negative ES)
            tail_penalty = config.TAIL_ADJUSTMENT_LAMBDA * abs(min(es_95, 0))
            score = exp_ret - tail_penalty
            
            metrics[ticker] = {
                'expected_return': exp_ret,
                'var_95': var_95,
                'es_95': es_95,
                'combined_score': score
            }
        return metrics
