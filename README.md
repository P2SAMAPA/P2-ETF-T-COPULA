# P2-ETF-T-COPULA

**Multivariate Student‑t Copula – Tail‑Dependence‑Aware ETF Ranking with GARCH Marginals**

[![Daily Run](https://github.com/P2SAMAPA/P2-ETF-T-COPULA/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-T-COPULA/actions/workflows/daily_run.yml)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-p2--etf--t--copula--results-blue)](https://huggingface.co/datasets/P2SAMAPA/p2-etf-t-copula-results)

## Overview

`P2-ETF-T-COPULA` fits an `n`‑dimensional **Student‑t copula** (Demarta & McNeil algorithm) to ETF returns. Unlike a Gaussian copula, the t‑copula explicitly captures **tail dependence**—the tendency of assets to crash together. The engine enriches the standard t‑copula with **GARCH(1,1) + skew‑t marginal models**, **bootstrap confidence intervals** for VaR₉₅ and ES₉₅, and a forward‑looking **momentum‑adjusted ranking**. Three views are produced per universe: Daily (504d), Global (2008‑YTD), and Shrinking Windows Consensus.

## Methodology

1. **Marginal modelling:** GARCH(1,1) with skew‑t innovations (or empirical CDF as fallback).
2. **Copula:** Multivariate Student‑t; degrees of freedom estimated from bivariate lower‑tail dependence coefficients.
3. **Simulation:** Cholesky decomposition + χ²/ν scaling → correlated t‑variates → mapped to uniforms → inverted via marginal quantiles.
4. **Risk metrics:** VaR₉₅ and ES₉₅ with bootstrap CI.
5. **Scoring:** `Adj Score = 21‑day annualised momentum − λ × |ES₉₅|`.

## Dashboard

- **Daily (504d):** Trained on the most recent 2 years to capture the current regime.
- **Global (2008‑YTD):** Trained on the entire available history for long‑term tail estimation.
- **Shrinking Windows Consensus:** The most frequently selected ETF across 17 rolling windows (2008‑2024).
- **VaR/ES Confidence Intervals:** Bootstrap‑based 95% CI displayed.

## Universe
FI/Commodities, Equity Sectors, Combined (23 ETFs)

## Usage

```bash
pip install -r requirements.txt
python trainer.py
streamlit run app.py
Why t‑Copula Instead of Vine Copula
The original VINE‑COPULA engine depended on vinecopulib, a C++ library that cannot be installed on GitHub Actions. The t‑copula is implemented in pure Python (SciPy/NumPy), installs natively, and captures the essential tail‑dependence feature that was missing from the Gaussian fallback.

text

Replace the files with these versions. The dashboard now visualizes bootstrap VaR/ES confidence intervals, and the README accurately reflects the improved methodology.
