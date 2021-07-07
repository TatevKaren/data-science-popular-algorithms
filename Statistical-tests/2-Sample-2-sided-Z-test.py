import numpy as np
from scipy.stats import norm

N_con = 60
N_exp = 60
alpha = 0.05

X_A = np.random.randint(100, size = N_con)
X_B = np.random.randint(100, size = N_exp)

mu_con = np.mean(X_A)
mu_exp = np.mean(X_B)

variance_con = np.var(X_A)
varaince_exp = np.var(X_B)

pooled_variance = np.sqrt(variance_con/N_con + varaince_exp/N_exp)

Z_stat = (mu_con-mu_exp)/np.sqrt(variance_con/N_con + variance_con/N_con)
p_value_Z_test = norm.sf(Z_stat)

Z_critical  = norm.ppf(1-alpha/2)

m = Z_critical*pooled_variance

CI = [(mu_con - mu_exp) - m, (mu_con - mu_exp) + m]

print("Z-score: ", Z_stat)
print("Z-critical: ", Z_critical)
print("P_value: ", p_value_Z_test)
print("Confidence Interval of 2 sample Z-test: ", np.round(CI,2))