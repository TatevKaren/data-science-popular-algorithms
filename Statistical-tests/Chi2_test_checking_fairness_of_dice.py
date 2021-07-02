import numpy as np
from scipy.stats import chi2


def Chi_2_test(X1,X2,X3,X4,X5,X6):
    X_observed_sides = np.array([X1, X2, X3, X4, X5, X6])
    # sum of all outcomes= number of dice tosses
    N = X_observed_sides.sum()
    Exp_num_occ = N / len(X_observed_sides)

    Chi_2_stat = (np.square(X_observed_sides  - Exp_num_occ)/Exp_num_occ).sum()
    df = N-1
    return(Chi_2_stat,df)



Chi_2_stat = Chi_2_test(0, 0, 0, 0, 40, 5)[0]
df = Chi_2_test(0, 0, 0, 0, 40, 5)[1]
Chi_2_crit_value = chi2.ppf(0.95,df)

p_value_Chi2test = chi2.sf(Chi_2_stat,df)

print(Chi_2_stat)
print(Chi_2_crit_value)
print(p_value_Chi2test)

x_values = np.arange(0,100,0.01)
y_values = chi2.pdf(x_values,df)

# import matplotlib.pyplot as plt
# plt.plot(x_values,y_values,alpha=0.6)
# plt.axvline(Chi_2_crit_value, 0,1,color = 'green',linewidth = 2)
# plt.show()


chi_sq_upper_crit = chi2.ppf(1-0.05/2,19)
chi_sq_lower_crit = chi2.ppf(0.05/2,19)

print(chi_sq_upper_crit)
print(chi_sq_lower_crit)