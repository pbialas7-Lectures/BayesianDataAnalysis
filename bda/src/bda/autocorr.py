import numpy as np

def ac_and_tau_int(series, c=10, maxlen=200):
    mu = np.mean(series)
    var = np.mean((series - mu) * (series - mu))
    out = [1.0]
    tau_int = 0.5
    for t in range(1, maxlen):
        cor = np.mean((series[:-t] - mu) * (series[t:] - mu)) / var
        tau_int += cor
        out.append(cor)
        if t > c * tau_int:
            break
    return tau_int, np.asarray(out)