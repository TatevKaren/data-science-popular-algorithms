import numpy as np
from scipy.stats import chi2

#Observed
O = np.array([90, 80, 5910,3920])
#Theoretical
T = np.array([102,68,5898, 3932])

# Squared_relative_distance: D statistics
def calculate_D(O,T):
    D_sum = 0
    for i in range(len(O)):
        D_sum += (O[i] - T[i])**2/T[i]
    return(D_sum)

D_stat = calculate_D(O,T)
p_value = chi2.sf(D_stat, df = 1)


import matplotlib.pyplot as plt

# Step 1: pick a x-axis range
d = np.arange(0,5,0.1)
# Step 2: drawing the initial pdf of chi-2 with df = 1 and x-axis d range we just created
plt.plot(d, chi2.pdf(d, df = 1), label = 'Chi2 Distribution',color = 'purple',linewidth = 2.5)
# Step 3: filling in the rejection region
plt.fill_between(d[d>D_stat], chi2.pdf(d[d>D_stat], df = 1),label = 'Chi2 Rejection Region',color = 'y')
plt.title("Chi2 Test")
plt.show()