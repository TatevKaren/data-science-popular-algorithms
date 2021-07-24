import numpy as np
import pandas as pd

X_cont = pd.Series([30221,44330,42198,39907,50000])
X_exp = pd.Series([40324,41034,52404])

#add a lebel
X_c = pd.concat([X_cont, pd.Series(np.repeat("cont",len(X_cont)))], axis = 1)
X_e = pd.concat([X_exp,pd.Series(np.repeat("exp",len(X_exp)))], axis = 1)
X = pd.concat([X_c,X_e])
X.columns = ["Sales","group"]


def Mann_Whitney_U_test_large_Samples(data):
    #sorting the data on Sales
    data_sorted = data.sort_values(by = "Sales",ascending = True).reset_index().drop("index",axis = 1)
    # create a rank variable based on Sales, dont forget +1
    data_sorted["rank"] = np.arange(1,len(data_sorted)+1,1)
    n1 = data_sorted.groupby("group").apply(len)["cont"]
    n2 =data_sorted.groupby("group").apply(len)["exp"]
    R1 = data_sorted.groupby("group")["rank"].apply(sum)["cont"]
    R2 = data_sorted.groupby("group")["rank"].apply(sum)["exp"]

    U1 = R1  - n1*(n1+1)/2
    U2 = R2 - n2*(n2+1)/2
    U = U1+U2
    return(U)

print(Mann_Whitney_U_test(X))