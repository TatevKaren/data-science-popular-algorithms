import numpy as np
from scipy.stats import norm

X_A = np.random.randint(100, size = [25])
X_B = np.random.randint(100, size = [25])

n_A = len(X_A)
n_B = len(X_B)

mean_A = np.mean(X_A)
mean_B = np.mean(X_B)

variance_A = np.var(X_A)
variance_B = np.var(X_B)

Z_stat = (mean_A - mean_B) / np.sqrt(variance_A/n_A + variance_B/n_B)

p_value = norm.sf(Z)

print("Z-score: ", Z_stat)
print("P_value: ", p_value)


