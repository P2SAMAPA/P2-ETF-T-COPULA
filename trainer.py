"""
Main training script — T‑COPULA with Daily, Global, and Shrinking modes.
"""

import json
import pandas as pd
import numpy as np

import config
import data_manager
from t_copula_model import TStudentCopula
import push_results

def compute_mode_results(returns, mode_name, mode_label):
    """Fit copula, simulate, and compute metrics for a given data slice."""
    copula = TStudentCopula(df_estimation_method="tail_dep")
    success = copula.fit(returns)
    if not success:
        return None

    sim_returns = copula.simulate(n_sim=min(config.N_SIMULATIONS, len(returns) * 50))
    all_metrics = copula.compute_risk_metrics(sim_returns)

    # Conditional expected return: 21‑day momentum (forward‑looking)
    momentum = returns.iloc[-config.MOMENTUM_WINDOW:].mean().to_dict()

    # Build results
    universe_results = {}
    for ticker in returns.columns:
        raw_ret = momentum.get(ticker, 0.0)
        copula_info = all_metrics.get(ticker, {})
        copula_score = copula_info.get('t_copula_score', 0.0)
        es95 = copula_info.get('es_95', 0.0)

        # T‑copula adjusted score: momentum minus tail penalty
        tail_penalty = config.TAIL_ADJUSTMENT_LAMBDA * abs(min(es95, 0))
        adj_score = raw_ret * 252 - tail_penalty

        universe_results[ticker] = {
            'ticker': ticker,
            'expected_return_raw': float(raw_ret),
            'copula_score': float(copula_score),
            'es_95': float(es95),
            'dof': float(copula_info.get('dof', 0)),
            't_copula_adj_score': float(adj_score)
        }

    sorted_tickers = sorted(universe_results.items(),
                            key=lambda x: x[1]['t_copula_adj_score'], reverse=True)
    top_picks = [{"ticker": t, **d} for t, d in sorted_tickers[:3]]

    return {
        'mode_name': mode_label,
        'top_picks': top_picks,
        'universes': universe_results,
        'training_start': str(returns.index[0].date()),
        'training_end': str(returns.index[-1].date()),
        'n_observations': len(returns)
    }


def run_shrinking_windows(df_master, tickers):
    """Shrinking windows consensus."""
    results = []
    for start_year in config.SHRINKING_WINDOW_START_YEARS:
        start_date = pd.Timestamp(f"{start_year}-01-01")
        end_date = pd.Timestamp("2024-12-31")
        mask = (df_master.index >= start_date) & (df_master.index <= end_date)
        window_returns = df_master[mask][tickers]
        if len(window_returns) < config.MIN_OBSERVATIONS:
            continue

        copula = TStudentCopula()
        copula.fit(window_returns)
        sim = copula.simulate(n_sim=min(config.N_SIMULATIONS, len(window_returns) * 50))
        metrics = copula.compute_risk_metrics(sim)

        momentum = window_returns.iloc[-config.MOMENTUM_WINDOW:].mean().to_dict()
        best_ticker = max(tickers, key=lambda t: momentum.get(t, 0) - abs(min(metrics.get(t, {}).get('es_95', 0), 0)))

        results.append({
            'window_start': start_year,
            'window_end': 2024,
            'ticker': best_ticker,
            'expected_return': float(momentum.get(best_ticker, 0))
        })

    # Consensus
    if results:
        vote = {}
        for r in results:
            t = r['ticker']
            vote[t] = vote.get(t, 0) + 1
        pick = max(vote, key=vote.get)
        conviction = vote[pick] / len(results) * 100
        consensus = {
            'ticker': pick,
            'conviction': conviction,
            'num_windows': len(results),
            'num_pick_windows': vote[pick],
            'windows': results
        }
        return consensus
    return None


def run_t_copula():
    print(f"=== P2-ETF-T-COPULA Run: {config.TODAY} ===")
    df_master = data_manager.load_master_data()
    df_master.index = pd.to_datetime(df_master.index)
    df_master = df_master[df_master.index >= config.GLOBAL_TRAIN_START]

    all_results = {}

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n--- Processing Universe: {universe_name} ---")
        returns = data_manager.prepare_returns_matrix(df_master, tickers)
        if len(returns) < config.MIN_OBSERVATIONS:
            continue

        universe_output = {}
        # Daily mode
        daily_returns = returns.iloc[-config.DAILY_LOOKBACK:]
        if len(daily_returns) >= config.MIN_OBSERVATIONS:
            daily = compute_mode_results(daily_returns, "daily", "Daily (504d)")
            if daily:
                universe_output["daily"] = daily
                print(f"  Daily top: {daily['top_picks'][0]['ticker']}")

        # Global mode
        if len(returns) >= config.GLOBAL_MIN_OBSERVATIONS:
            global_ = compute_mode_results(returns, "global", "Global (2008‑YTD)")
            if global_:
                universe_output["global"] = global_
                print(f"  Global top: {global_['top_picks'][0]['ticker']}")

        # Shrinking windows
        shrinking = run_shrinking_windows(df_master, tickers)
        if shrinking:
            universe_output["shrinking"] = shrinking
            print(f"  Shrinking consensus: {shrinking['ticker']} ({shrinking['conviction']:.0f}%)")

        all_results[universe_name] = universe_output

    output_payload = {
        "run_date": config.TODAY,
        "config": {k: v for k, v in config.__dict__.items() if not k.startswith("_") and k.isupper() and k != "HF_TOKEN"},
        "universes": all_results
    }

    push_results.push_daily_result(output_payload)
    print("\n=== Run Complete ===")


if __name__ == "__main__":
    run_t_copula()
