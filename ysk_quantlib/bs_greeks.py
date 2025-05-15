from scipy.stats import norm
import numpy as np

def delta(S, K, T, r, sigma, option_type = 'x'):
    if option_type == 'call':
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return norm.cdf(d1)
    elif option_type == 'put':
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return norm.cdf(d1) - 1
    else:
        raise ValueError("option_type must be 'call' or 'put'")

def gamma(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def vega(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T) / 100

def theta(S, K, T, r, sigma, option_type = 'x'):
    if option_type == 'call':
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    elif option_type == 'put':
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

def rho(S, K, T, r, sigma, option_type = 'x'):
    if option_type == 'call':
        d2 = (np.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    elif option_type == 'put':
        d2 = (np.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

def volga(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    vega = S * norm.pdf(d1) * np.sqrt(T)

    return vega * d1 * d2 / sigma

def charm(S, K, T, r, sigma, option_type = 'x'):
    if T <= 0:
        return 0.0  # At maturity, charm tends to 0

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    pdf_d1 = norm.pdf(d1)

    common_term = - pdf_d1 * (2 * r * T - d2 * sigma * sqrt_T) / (2 * T * sigma * sqrt_T)

    if option_type == 'call':
        return common_term
    elif option_type == 'put':
        return -common_term
    else:
        raise ValueError("option_type must be either 'call' or 'put'")