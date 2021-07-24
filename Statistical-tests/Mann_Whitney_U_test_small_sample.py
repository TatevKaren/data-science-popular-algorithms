import numpy as np
import pandas as pd

X_cont = pd.Series([30221,44330,42198,39907,50000])
X_exp = pd.Series([40324,41034,52404])

#add a lebel
X_c = pd.concat([X_cont, pd.Series(np.repeat("cont",len(X_cont)))], axis = 1)
X_e = pd.concat([X_exp,pd.Series(np.repeat("exp",len(X_exp)))], axis = 1)
X = pd.concat([X_c,X_e])
X.columns = ["Sales","group"]


def Mann_Whitney_U_test(data):
    #sorting the data on Sales
    data_sorted = data.sort_values(by = "Sales",ascending = True).reset_index().drop("index",axis = 1)
    # create a rank variable based on Sales, dont forget +1
    data_sorted["rank"] = np.arange(1,len(data_sorted)+1,1)
    # per group compute the sum rank
    sum_rank = pd.DataFrame(data_sorted.groupby("group")["rank"].sum()).reset_index()
    sum_rank.columns = ["group","sumrank"]
    smaller_sum_rank_group = sum_rank[sum_rank.sumrank == sum_rank.sumrank.min()]["group"].tolist()

    for group in smaller_sum_rank_group:
        smaller_group_ranks = data_sorted[data_sorted.group == group]["rank"].tolist()
        larger_group = data_sorted[data_sorted.group != group]["rank"].tolist()

        counts = []
        # for each value in smaller rank group checking number of pints in larger group smaller then that
        for r in smaller_group_ranks:
            count = 0
            for i in larger_group:
                if i < r:
                    count+=1
            counts.append(count)

        U = sum(counts)

    return(U)

print(Mann_Whitney_U_test(X))
