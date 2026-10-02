import numpy as np

def participation_ratio(singular_values):
    s = np.asarray(singular_values, float)
    e = s*s
    d = np.sum(e*e)
    return float(np.sum(e)**2/d) if d > 0 else float("nan")

def entropy_rank(singular_values):
    s = np.asarray(singular_values, float)
    total = np.sum(s)
    if total <= 0:
        return float("nan")
    p = s/total
    p = p[p > 0]
    return float(np.exp(-np.sum(p*np.log(p))))

def stable_rank(singular_values):
    s = np.asarray(singular_values, float)
    if s.size == 0 or s[0] <= 0:
        return float("nan")
    return float(np.sum(s*s)/(s[0]**2))
