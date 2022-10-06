import pandas as pd
from PyNonpar import twosample

df = pd.read_csv("Prussion Horse-Kick Data.csv")
#print(df.drop(["Year"],axis = 1).sum().sum()/(df.drop(["Year"],axis = 1).shape[0] *df.drop(["Year"],axis = 1).shape[1] ) )
print("median per col:", df.drop(["Year"],1).median(axis = 0))
print("median per year:", df.drop(["Year"],1).median(axis = 1))
median_deaths = list(df.drop(["Year"],1).median(axis = 1))

print("---------------Mann-Whitney Test---------------")
df = df.drop(["Year"],1)
cols = df.columns
print(cols)
for i in range(len(cols)):
    print("\n {}".format(cols[i]))
    num_death = list(df[cols[i]])
    print(num_death)
    print("Mann Whitney Test Stat",twosample.wilcoxon_mann_whitney_test(num_death, median_deaths, method = "exact"))









