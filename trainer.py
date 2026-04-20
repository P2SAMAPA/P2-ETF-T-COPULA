"""
Main training script for Vine Copula engine.
Fits vine copula, simulates scenarios, computes risk-adjusted returns, and ranks ETFs.
"""

import json
import pandas as pd
import numpy as np

import config
import data_manager
from vine_copula_model import VineCopulaModel
import push_results

def run_vine_copula():
    print(f"=== P2-ETF-VINE-COPULA Run: {config.TODAY} ===")
    df_master = data_manager.load_master_data()
    
    all_results = {}
    top_picks = {}
    
    # Combined universe for global vine (used for all universes)
    returns_all = data_manager.prepare_returns_matrix(df_master, config.ALL_TICKERS)
    if len(returns_all) < config.MIN_OBSERVATIONS:
        print("Insufficient data for combined universe.")
        return
    
    recent_all = returns_all.iloc[-config.LOOKBACK_WINDOW:]
    model = VineCopulaModel(margin_model=config.MARGIN_MODEL)
    success = model.fit(recent_all)
    if not success:
        print("Vine copula fitting failed.")
        return
    
    sim_returns = model.simulate(n_sim=config.N_SIMULATIONS)
    risk_metrics = model.compute_risk_metrics(sim_returns)
    
    # Process each universe using the global metrics (subset of tickers)
    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n--- Processing Universe: {universe_name} ---")
        universe_metrics = {t: risk_metrics[t] for t in tickers if t in risk_metrics}
        all_results[universe_name] = universe_metrics
        
        # Rank by combined_score
        sorted_tickers = sorted(universe_metrics.items(), key=lambda x: x[1]['combined_score'], reverse=True)
        top_picks[universe_name] = [
            {'ticker': t, **m} for t, m in sorted_tickers[:3]
        ]
    
    # Shrinking windows (simplified: fit vine on each window and store top pick)
    shrinking_results = {}
    for start_year in config.SHRINKING_WINDOW_START_YEARS:
        start_date = pd.Timestamp(f"{start_year}-01-01")
        window_label = f"{start_year}-{config.TODAY[:4]}"
        mask = df_master['Date'] >= start_date
        df_window = df_master[mask].copy()
        if len(df_window) < config.MIN_OBSERVATIONS:
            continue
        returns_win = data_manager.prepare_returns_matrix(df_window, config.ALL_TICKERS)
        if len(returns_win) < config.MIN_OBSERVATIONS:
            continue
        recent_win = returns_win.iloc[-config.LOOKBACK_WINDOW:]
        win_model = VineCopulaModel(margin_model=config.MARGIN_MODEL)
        if not win_model.fit(recent_win):
            continue
        win_sim = win_model.simulate(n_sim=config.N_SIMULATIONS//2)  # fewer sims for speed
        win_metrics = win_model.compute_risk_metrics(win_sim)
        window_top = {}
        for universe_name, tickers in config.UNIVERSES.items():
            best_ticker = max(tickers, key=lambda t: win_metrics.get(t, {}).get('combined_score', -np.inf))
            window_top[universe_name] = {
                'ticker': best_ticker,
                'combined_score': win_metrics[best_ticker]['combined_score']
            }
        shrinking_results[window_label] = {
            'start_year': start_year,
            'top_picks': window_top
        }
    
    output_payload = {
        "run_date": config.TODAY,
        "config": {
            "lookback_window": config.LOOKBACK_WINDOW,
            "n_simulations": config.N_SIMULATIONS,
            "margin_model": config.MARGIN_MODEL,
            "tail_adjustment_lambda": config.TAIL_ADJUSTMENT_LAMBDA
        },
        "daily_trading": {
            "universes": all_results,
            "top_picks": top_picks
        },
        "shrinking_windows": shrinking_results
    }
    
    push_results.push_daily_result(output_payload)
    print("\n=== Run Complete ===")

if __name__ == "__main__":
    run_vine_copula()
