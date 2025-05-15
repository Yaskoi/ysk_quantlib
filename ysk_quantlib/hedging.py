import numpy as np
import pandas as pd
from .pricing import black_scholes_price
from .bs_greeks import delta, gamma, vega

def simulate_delta_hedge(
    S_series, K, T, r, sigma,
    option_type='call',
    cost_per_trade=0.01
):
    """
    Simulates a discrete delta hedging strategy over the life of a European option.

    Parameters:
    - S_series: pd.Series of underlying prices (indexed by date)
    - K: strike price
    - T: time to maturity in years (float)
    - r: risk-free interest rate
    - sigma: constant volatility (or use implied)
    - option_type: 'call' or 'put'
    - cost_per_trade: transaction cost per rebalancing (% of notional)

    Returns:
    - pd.DataFrame with columns: price, delta, position, hedge_cost, portfolio_value, pnl
    """

    n = len(S_series)
    dt = T / n
    dates = S_series.index

    df = pd.DataFrame(index=dates)
    df['price'] = S_series.values
    df['time_to_maturity'] = np.linspace(T, 0, n)

    df['delta'] = df.apply(lambda row: delta(row['price'], K, row['time_to_maturity'], r, sigma, option_type), axis=1)
    df['position'] = df['delta']  # units of underlying held at each step
    df['position_shift'] = df['position'].diff().fillna(0)

    # Cost = number of shares adjusted * price * cost per trade
    df['hedge_cost'] = np.abs(df['position_shift']) * df['price'] * cost_per_trade
    df['option_value'] = df.apply(lambda row: black_scholes_price(row['price'], K, row['time_to_maturity'], r, sigma, option_type), axis=1)

    # PnL: value of option + hedge - initial cost
    df['hedge_value'] = df['position'] * df['price']
    df['portfolio_value'] = df['hedge_value'] - df['hedge_cost'].cumsum()
    df['pnl'] = df['portfolio_value'] + df['option_value'] - df['option_value'].iloc[0]

    return df


def compute_hedging_error(final_portfolio_value, actual_payoff):
    """
    Returns the replication error of the hedging strategy.

    Parameters:
    - final_portfolio_value: value of the delta-hedged portfolio at maturity
    - actual_payoff: true payoff of the option (e.g., max(S_T - K, 0))

    Returns:
    - float: replication error
    """
    return final_portfolio_value - actual_payoff


def hedging_summary(pnl_series):
    """
    Computes summary statistics for the delta hedging performance.

    Parameters:
    - pnl_series: pd.Series of daily PnL

    Returns:
    - dict with final PnL, mean, std, and Sharpe ratio
    """
    sharpe = pnl_series.mean() / pnl_series.std() * np.sqrt(252) if pnl_series.std() > 0 else np.nan
    return {
        'final_pnl': pnl_series.iloc[-1],
        'mean_daily_pnl': pnl_series.mean(),
        'std_daily_pnl': pnl_series.std(),
        'sharpe_ratio': sharpe
    }


def simulate_greeks_over_time(S_series, K, T, r, sigma, option_type='call'):
    """
    Computes the time series of Delta, Gamma, and Vega for an option.

    Parameters:
    - S_series: pd.Series of underlying prices
    - K: strike
    - T: time to maturity (in years)
    - r: interest rate
    - sigma: volatility
    - option_type: 'call' or 'put'

    Returns:
    - pd.DataFrame with greeks at each point in time
    """
    n = len(S_series)
    dt = T / n
    time_grid = np.linspace(T, 0, n)
    dates = S_series.index

    df = pd.DataFrame(index=dates)
    df['price'] = S_series.values
    df['time_to_maturity'] = time_grid

    df['delta'] = df.apply(lambda row: delta(row['price'], K, row['time_to_maturity'], r, sigma, option_type), axis=1)
    df['gamma'] = df.apply(lambda row: gamma(row['price'], K, row['time_to_maturity'], r, sigma), axis=1)
    df['vega']  = df.apply(lambda row: vega(row['price'], K, row['time_to_maturity'], r, sigma), axis=1)

    return df