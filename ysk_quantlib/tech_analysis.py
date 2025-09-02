import pandas as pd
import numpy as np

def SMA(series, period=14, mode = 'mean'):
    """
    Calculate a Simple Moving Average (SMA) or variations (max, min, std).

    Parameters
    ----------
    series : pandas.Series
        Price data.
    period : int, optional
        Rolling window period, by default 14.
    mode : {'mean', 'upper', 'down', 'std'}, optional
        Type of calculation:
        - 'mean': Simple Moving Average
        - 'upper': Rolling maximum
        - 'down': Rolling minimum
        - 'std': Rolling standard deviation

    Returns
    -------
    pandas.Series
        Series containing the computed rolling values.
    """

    valid_modes = {'mean', 'upper', 'down', 'std'}
    
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode: {mode}. Choose from {valid_modes}")


    if mode == 'mean':
        return series.rolling(window=period).mean()
    elif mode == 'upper':
        return series.rolling(window=period).max()
    elif mode == 'down':
        return series.rolling(window=period).min()
    elif mode == 'std':
        return series.rolling(window=period).std()
    

def RSI(series, period=14):
    """
    Compute the Relative Strength Index (RSI).

    Parameters
    ----------
    series : pandas.Series
        Price data.
    period : int, optional
        Period for RSI calculation, by default 14.

    Returns
    -------
    pandas.Series
        RSI values between 0 and 100.
    """
    
    change = series.diff()
    gain = np.where(change > 0, change, 0)
    loss = np.where(change < 0, -change, 0)

    gain = pd.Series(gain, index=series.index)
    loss = pd.Series(loss, index=series.index)

    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()

    RS = avg_gain / avg_loss

    RSI = 100 - (100 / (1 + RS))

    return RSI


def MACD(series, fast_period=12, slow_period=26, signal_period=9):
    """
    Compute Moving Average Convergence Divergence (MACD).

    Parameters
    ----------
    series : pandas.Series
        Price data.
    fast_period : int, optional
        Fast EMA period, by default 12.
    slow_period : int, optional
        Slow EMA period, by default 26.
    signal_period : int, optional
        Signal line EMA period, by default 9.

    Returns
    -------
    tuple of pandas.Series
        macd : MACD line
        signal : Signal line
        histogram : MACD histogram
    """
    macd = series.ewm(span=fast_period, adjust=False).mean() - series.ewm(span=slow_period, adjust=False).mean()
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    histogram = macd - signal
    
    return macd, signal, histogram


def BBands(series, period=20, k=2):
    """
    Compute Bollinger Bands.

    Parameters
    ----------
    series : pandas.Series
        Price data.
    period : int, optional
        Rolling window period, by default 20.
    k : int or float, optional
        Number of standard deviations, by default 2.

    Returns
    -------
    tuple of pandas.Series
        upper : Upper Bollinger Band
        middle : Rolling mean
        lower : Lower Bollinger Band
    """
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()

    upper = middle + (k * std)
    lower = middle - (k * std)
    
    return upper, middle, lower

def ATR(series, period=14):
    """
    Compute the Average True Range (ATR).

    Parameters
    ----------
    series : pandas.DataFrame
        OHLC data with columns ['high', 'low', 'close'].
    period : int, optional
        Period for ATR calculation, by default 14.

    Returns
    -------
    pandas.Series
        ATR values.
    """
    
    high_low = series['high'] - series['low']

    high_close = abs(series['high'] - series['close'].shift(1))
    low_close = abs(series['low'] - series['close'].shift(1))

    TR = np.maximum.reduce([high_low, high_close, low_close])

    ATR = pd.Series(TR, index=series.index).ewm(span=period, adjust=False).mean()
    
    return ATR

def KAMA(series, n=14, fastest=2, slowest=30):
    """
    Compute the Kaufman Adaptive Moving Average (KAMA).

    Parameters
    ----------
    series : pandas.Series
        Price data.
    n : int, optional
        Lookback period, by default 14.
    fastest : int, optional
        Fastest EMA constant, by default 2.
    slowest : int, optional
        Slowest EMA constant, by default 30.

    Returns
    -------
    pandas.Series
        KAMA values.
    """
    change = series - series.shift(n)
    volatility = series.diff().abs().rolling(n).sum()
    ER = change.abs() / volatility
    ER = ER.fillna(0)

    fastest_sc = 2 / (fastest + 1)
    slowest_sc = 2 / (slowest + 1)
    smoothing_constant = (ER * (fastest_sc - slowest_sc) + slowest_sc) ** 2

    kama = pd.Series(index=series.index, dtype=float)
    kama.iloc[n] = series.iloc[n]

    for t in range(n+1, len(series)):
        kama.iloc[t] = kama.iloc[t-1] + smoothing_constant.iloc[t] * (series.iloc[t] - kama.iloc[t-1])

    return kama

def ADX (series, period=14):
    """
    Compute the Average Directional Index (ADX).

    Parameters
    ----------
    series : pandas.DataFrame
        OHLC data with columns ['high', 'low', 'close'].
    period : int, optional
        Period for ADX calculation, by default 14.

    Returns
    -------
    tuple of pandas.Series
        ADX : Average Directional Index
        DI_up : Positive Directional Indicator
        DI_down : Negative Directional Indicator
    """
    
    high_low = series['high'] - series['low']
    high_close = abs(series['high'] - series['close'].shift(1))
    low_close = abs(series['low'] - series['close'].shift(1))

    TR = np.maximum.reduce([high_low, high_close, low_close])

    DM_up = np.where((series['high'] - series['high'].shift(1)) > (series['low'].shift(1) - series['low']),
                     series['high'] - series['high'].shift(1), 0.0)
    DM_down = np.where((series['low'].shift(1) - series['low']) > (series['high'] - series['high'].shift(1)),
                       series['low'].shift(1) - series['low'], 0.0)
    
    DI_up = 100 * (DM_up / TR).ewm(span=period, adjust=False).mean()
    DI_down = 100 * (DM_down / TR).ewm(span=period, adjust=False).mean()

    DX = (100 * abs(DI_up - DI_down) / (DI_up + DI_down)).fillna(0)

    ADX = DX.ewm(span=period, adjust=False).mean()

    return ADX, DI_up, DI_down

def Parabolic_SAR(series, acceleration=0.02, maximum=0.2):
    """
    Compute the Parabolic Stop and Reverse (Parabolic SAR).

    Parameters
    ----------
    series : pandas.DataFrame
        OHLC data with columns ['high', 'low', 'close'].
    acceleration : float, optional
        Acceleration factor increment, by default 0.02.
    maximum : float, optional
        Maximum acceleration factor, by default 0.2.

    Returns
    -------
    pandas.Series
        Parabolic SAR values.
    """
    
    high = series['high'].to_numpy()
    low = series['low'].to_numpy()
    close = series['close'].to_numpy()
    n = len(series)

    psar = np.zeros(n)
    trend_up = True
    af = acceleration
    ep = high[0]
    psar[0] = low[0]


    for i in range(1, n):
        prev_psar = psar[i-1]
        if trend_up:
            psar[i] = prev_psar + af * (ep - prev_psar)
            psar[i] = min(psar[i], low[i-1], low[i])

            if high[i] > ep:
                ep = high[i]
                af = min(af + acceleration, maximum)

            if close[i] < psar[i]:
                trend_up = False
                psar[i] = ep
                ep = low[i]
                af = acceleration
        else:
            psar[i] = prev_psar + af * (ep - prev_psar)
            psar[i] = max(psar[i], high[i-1], high[i])

            if low[i] < ep:
                ep = low[i]
                af = min(af + acceleration, maximum)

            if close[i] > psar[i]:
                trend_up = True
                psar[i] = ep
                ep = high[i]
                af = acceleration

    return pd.Series(psar, index=series.index)

def stoch_oscillator(series, period=14, d=3):
    """
    Compute the Stochastic Oscillator (%K and %D).

    Parameters
    ----------
    series : pandas.DataFrame
        OHLC data with columns ['high', 'low', 'close'].
    period : int, optional
        Lookback period for %K, by default 14.
    d : int, optional
        Smoothing period for %D, by default 3.

    Returns
    -------
    tuple of pandas.Series
        K : %K line
        D : %D line (smoothed)
    """
    
    high = series['high']
    low = series['low']
    close = series['close']

    rolling_low = low.rolling(window=period).min()
    rolling_high = high.rolling(window=period).max()

    K = 100 * (close - rolling_low) / (rolling_high - rolling_low)

    D = K.ewm(span=d, adjust=False).mean()

    return K, D

def CCI(series, n=20):
    """
    Compute the Commodity Channel Index (CCI).

    Parameters
    ----------
    series : pandas.DataFrame
        OHLC data with columns ['high', 'low', 'close'].
    n : int, optional
        Lookback period, by default 20.

    Returns
    -------
    pandas.Series
        CCI values.
    """
    
    typical_price = (series['high'] + series['low'] + series['close']) / 3
    ma = typical_price.rolling(window=n).mean()

    md = (typical_price - ma).abs().rolling(window=n).mean()

    cci = (typical_price - ma) / (0.015 * md)

    return cci

def VWAP(series):
    """
    Compute the Volume Weighted Average Price (VWAP).

    Parameters
    ----------
    series : pandas.DataFrame
        Data with columns ['close', 'volume'].

    Returns
    -------
    pandas.Series
        VWAP values.
    """
    
    price = series['close']
    volume = series['volume']

    vwap = (price * volume).cumsum() / volume.cumsum()

    return pd.Series(vwap, index=series.index)

def VWAP_intraday(series):
    """
    Compute the intraday Volume Weighted Average Price (VWAP).

    Parameters
    ----------
    series : pandas.DataFrame
        OHLCV data with columns ['high', 'low', 'close', 'volume'].
        The index should be datetime-like.

    Returns
    -------
    pandas.Series
        Intraday VWAP values, resetting daily.
    """
    
    typical_price = (series['high'] + series['low'] + series['close']) / 3
    volume = series['volume']

    df = series.copy()
    df['typical_price'] = typical_price
    df['cum_vol'] = volume.groupby(df.index.date).cumsum()
    df['cum_pv'] = (typical_price * volume).groupby(df.index.date).cumsum()

    vwap = df['cum_pv'] / df['cum_vol']
    return pd.Series(vwap, index=series.index)

def Ichimoku(series, n1=9, n2=26, n3=52):
    """
    Compute the Ichimoku Cloud indicator.

    Parameters
    ----------
    series : pandas.DataFrame
        OHLC data with columns ['high', 'low', 'close'].
    n1 : int, optional
        Period for Tenkan-sen, by default 9.
    n2 : int, optional
        Period for Kijun-sen and Chikou Span shift, by default 26.
    n3 : int, optional
        Period for Senkou Span B, by default 52.

    Returns
    -------
    tuple of pandas.Series
        tenkan : Tenkan-sen
        kijun : Kijun-sen
        span_a : Senkou Span A
        span_b : Senkou Span B
        chikou : Chikou Span
    """
    #Lecture rapide
    #Prix > Cloud → tendance haussière
    #Prix < Cloud → tendance baissière
    #Prix dans Cloud → marché neutre ou consolidation
    #SpanA > SpanB → nuage haussier
    #SpanA < SpanB → nuage baissier
    #Chikou Span → confirme la force de la tendance

    high = series['high']
    low = series['low']
    close = series['close']
    
    tenkan = (high.rolling(window=n1).max() + low.rolling(window=n1).min()) / 2
    kijun = (high.rolling(window=n2).max() + low.rolling(window=n2).min()) / 2

    span_a = ((tenkan + kijun) / 2).shift(-n2)
    span_b = ((high.rolling(window=n3).max() + low.rolling(window=n3).min()) / 2).shift(-n2)

    chikou = close.shift(n2)

    return tenkan, kijun, span_a, span_b, chikou


