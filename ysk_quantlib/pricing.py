from scipy.stats import norm
import numpy as np
from numpy import log, exp, sqrt, real
from scipy.integrate import quad

def black_scholes_price(S, K, T, r, sigma, option_type='x'):

    """
    Calculate the Black-Scholes price for a European call or put option.

    S : action price
    K : exercice price
    T : time to maturity (in years)
    r : risk-free interest rate
    sigma : implied volatility
    option_type : 'call' or 'put'

    """
    if T <= 0:
        if option_type == 'call':
            return max(S - K, 0)
        elif option_type == 'put':
            return max(K - S, 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def binomial_tree_pricing(S, K, T, r, sigma, n, option_type='x'):
    """
    Computes the European option price using the Cox-Ross-Rubinstein binomial tree.

    Parameters:
    S : initial stock price
    K : strike price
    T : time to maturity (in years)
    r : risk-free interest rate
    sigma : volatility
    n : number of time steps
    option_type : 'call' or 'put'

    Returns: option price
    """

    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))    # up factor
    d = 1 / u                          # down factor
    p = (np.exp(r * dt) - d) / (u - d) # risk-neutral probability
    discount = np.exp(-r * dt)

    # Terminal payoffs
    option = np.zeros(n + 1)
    for i in range(n + 1):
        ST = S * (u ** i) * (d ** (n - i))
        if option_type == 'call':
            option[i] = max(ST - K, 0)
        elif option_type == 'put':
            option[i] = max(K - ST, 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

    # Backward induction
    for j in range(n - 1, -1, -1):
        for i in range(j + 1):
            option[i] = discount * (p * option[i + 1] + (1 - p) * option[i])

    return option[0]


def monte_carlo_pricing(S, K, T, r, sigma, n_simulations, option_type='x'):
    """
    Calculate the price of a European call or put option using the Monte Carlo method.
    Parameters :
    S : spot price of the underlying asset
    K : exercise price of the option
    T : time to maturity (in years)
    r : risk-free interest rate
    sigma : constant volatility of the underlying asset
    option_type : 'call' or 'put'
    """
    dt = T / 252  # Assuming 252 trading days in a year
    discount_factor = np.exp(-r * T)
    payoffs = np.zeros(n_simulations)
    for i in range(n_simulations):
        ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * np.random.normal())
        if option_type == 'call':
            payoffs[i] = max(ST - K, 0)
        elif option_type == 'put':
            payoffs[i] = max(K - ST, 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
    option_price = discount_factor * np.mean(payoffs)
    return option_price

    plot_paths(paths, title="Simulated Asset Price Paths")
    return paths


def heston_price(S, K, T, r, v0, kappa, theta, sigma, rho, option_type='x'):
    """
    Calculate the Heston model price for a European call or put option using the Fourier transform method.

    Parameters :
    S           : spot price of the underlying asset
    K           : exercise price of the option
    T           : time to maturity (in years)
    r           : risk-free interest rate
    v0          : initial variance
    kappa       : rate at which variance reverts to long-term mean
    theta       : long-term mean of variance
    sigma       : volatility of the variance (vol-of-vol)
    rho         : correlation between the asset and the variance
    option_type : 'call' or 'put'

    Retour :
    Option price
    """

    def integrand(phi, j):
        if j == 1:
            u = 0.5
            b = kappa - rho * sigma
        else:
            u = -0.5
            b = kappa

        a = kappa * theta
        x = log(S)
        d = np.sqrt((rho * sigma * 1j * phi - b)**2 - sigma**2 * (2 * u * 1j * phi - phi**2))
        g = (b - rho * sigma * 1j * phi + d) / (b - rho * sigma * 1j * phi - d)

        C = (r * 1j * phi * T + 
             a / sigma**2 * ((b - rho * sigma * 1j * phi + d) * T - 
             2 * np.log((1 - g * np.exp(d * T)) / (1 - g))))
        D = ((b - rho * sigma * 1j * phi + d) / sigma**2) * ((1 - np.exp(d * T)) / (1 - g * np.exp(d * T)))

        return real(np.exp(C + D * v0 + 1j * phi * x - 1j * phi * log(K)) / (1j * phi))

    P1 = 0.5 + (1 / np.pi) * quad(lambda phi: integrand(phi, 1), 1e-6, 100)[0]
    P2 = 0.5 + (1 / np.pi) * quad(lambda phi: integrand(phi, 2), 1e-6, 100)[0]

    price = S * P1 - K * exp(-r * T) * P2

    if option_type == 'call':
        return price
    elif option_type == 'put':
        return price - S + K * exp(-r * T)  # put-call parity
    else:
        raise ValueError("option_type must be 'call' or 'put'")