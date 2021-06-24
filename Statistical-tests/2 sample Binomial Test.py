import numpy as np
from scipy.stats import norm

X1 = 1242
N1 = 9886
X2 = 974
N2 = 10072

p_pooled = (X1+X2)/(N1+N2)
s2_pooled = p_pooled*(1-p_pooled)
SE_p_pooled = np.sqrt(s2_pooled*(1/N1 + 1/N2))

p_1_hat = X1/N1
p_2_hat = X2/N2

d_hat = p_1_hat - p_2_hat

z_critical_value = norm.ppf(1-0.05/2,0,1)

margin_error = z_critical_value*SE_p_pooled

test_stat = d_hat/SE_p_pooled
p_value = round(norm.sf(test_stat, 0,1),3)

if test_stat >= z_critical_value:
    print("reject the null")
    print(p_value)


CI = [d_hat - margin_error, d_hat + margin_error]
print(CI[1]-CI[0])