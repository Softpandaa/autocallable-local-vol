"""Black-Scholes pricing, implied volatility by bisection, and the implied dividend."""

import numpy as np
from scipy.stats import norm


def get_ATMF(chain):
    """
    This function estimates the values of C_ATMF and K_ATMF based on the given chain, the methodology can be found in the word doc
    chain: raw option chain
    
    Return: C_ATMF, K_ATMF
    """
    C = chain["Call"]
    P = chain["Put"]
    K = chain["Strike"]
    n = len(C)
    for i in range(1, n):
        if C[i] >= P[i] and C[i+1] <= P[i+1]:
            C_K1, P_K1, K1 = C[i], P[i], K[i]
            C_K2, P_K2, K2 = C[i+1], P[i+1], K[i+1]
            break
    C_ATMF = (C_K1+P_K1)/2
    K_ATMF = (C_ATMF-C_K1)/(C_K2-C_K1) * (K2-K1) + K1
    return C_ATMF, K_ATMF

def get_implied_div(S, K, r, T):
    """
    This functions estimated the continuous dividend yield based on a forward price
    S: Spot price of underlying
    K: Forward price of underlying
    r: risk-free rate
    T: Year to maturity
    
    Return: implied dividend yield
    """
    q = r - np.log(K/S) / T
    return max(q, 0) # Prevent implied dividend yield < 0, which doesn't make sense

def BS_Euro_Pricer_F(F, K, r, T, sig):
    """
    This function computes the European option price based on a forward price
    F: forward price at T
    K: Strike
    r: risk-free rate (in decimal)
    T: Year to maturity
    sig: Volatility of the underlying (in decimal)
    
    Return: European option price
    """
    d1 = (np.log(F/K) + ((sig**2)/2)*T) / (sig*np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    return np.exp(-r*T) * (F * norm.cdf(d1) - K * norm.cdf(d2))

def BS_Euro_Pricer(isCall, S, K, r, q, T, sig):
    """
    This function computes the European option price
    isCall: True for Call; False for Put
    S: Spot price of underlying
    K: Strike
    r: risk-free rate (in decimal)
    q: continuous dividend yield (in decimal)
    T: Year to maturity
    sig: Volatility of the underlying (in decimal)
    
    Return: European option price
    """
    d1 = (np.log(S/K) + (r - q + (sig**2) / 2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    if isCall:
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

def IV_Bisect(P, tol, isCall, S, K, r, q, T):
    """
    This function estimates the implied volatility of the given option
    P: Observed price of the option
    tol: Maximum error of approximation
    isCall: True for Call; False for Put
    S: Spot price of underlying
    K: Strike
    r: risk-free rate (in decimal)
    q: continuous dividend yield (in decimal)
    T: Year to maturity
    
    Return: implied volatility (in sigma)
    """
    sig_left = 0.0001
    sig_right = 0.0001
    while BS_Euro_Pricer(isCall, S, K, r, q, T, sig_right) - P < 0:
        sig_right += 0.2
    sig_temp = (sig_left + sig_right) / 2
    while abs(BS_Euro_Pricer(isCall, S, K, r, q, T, sig_temp) - P) > tol:
        if BS_Euro_Pricer(isCall, S, K, r, q, T, sig_temp) - P > 0:
            sig_right = sig_temp
        else:
            sig_left = sig_temp
        sig_temp = (sig_left + sig_right) / 2
    return sig_temp
