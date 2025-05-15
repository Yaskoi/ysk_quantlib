def PortfolioAnalysis(returns, weights=None, risk_free_rate=0.01):
    """
    Perform portfolio analysis.
    """
    import numpy as np
    import pandas as pd

    # Calculate portfolio returns and volatility
    portfolio_returns = np.dot(returns, weights) if weights is not None else returns.mean(axis=1)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights))) if weights is not None else returns.std(axis=1)
    portfolio_sharpe_ratio = (portfolio_returns - risk_free_rate) / portfolio_volatility
    portfolio_cumulative_returns = (1 + portfolio_returns).cumprod() - 1
    portfolio_drawdown = (portfolio_cumulative_returns - portfolio_cumulative_returns.cummax()) / (1 + portfolio_cumulative_returns.cummax())
    return {
        'returns': portfolio_returns,
        'volatility': portfolio_volatility,
        'sharpe_ratio': portfolio_sharpe_ratio,
        'cumulative_returns': portfolio_cumulative_returns,
        'drawdown': portfolio_drawdown
    }

# End of file portfolio.py
