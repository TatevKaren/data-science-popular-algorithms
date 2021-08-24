import numpy as np
from scipy.stats import t

N_con = 20
df_con = N_con - 1
N_exp = 20
df_exp = N_exp - 1

alpha = 0.05

X_con = np.random.standard_t(df_con,N_con)
X_exp = np.random.standard_t(df_exp,N_exp)

mu_con = np.mean(X_con)
mu_exp = np.mean(X_exp)

sigma_sqr_con = np.var(X_con)
sigma_sqr_exp = np.var(X_exp)

pooled_variance_t_test = ((N_con-1)*sigma_sqr_con + (N_exp -1) * sigma_sqr_exp)/(N_con + N_exp-2)*(1/N_con + 1/N_exp)
SE_t_test = np.sqrt(pooled_variance_t_test)

t_stat = (mu_con-mu_exp)/SE_t_test
#two sided 2 sample t-test
t_crit = t.ppf(1-alpha/2, N_con + N_exp - 2)

p_value_T_test = t.sf(t_stat, N_con + N_exp - 2)
margin_error = t_crit * SE_t_test

CI = [(mu_con-mu_exp) - margin_error, (mu_con-mu_exp) + margin_error]


print("Z-score: ", t_stat)
print("Z-critical: ", t_crit)
print("P_value: ", p_value_T_test)
print("Confidence Interval of 2 sample Z-test: ", np.round(CI,2))
