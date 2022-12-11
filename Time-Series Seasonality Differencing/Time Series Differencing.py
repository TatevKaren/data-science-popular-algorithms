from datetime import datetime
import pandas as pd

def transform_to_data(x):
     date_string = "190"+x
     date = datetime.strptime(date_string,'%Y-%m')
     return(date)
time_series = pd.read_csv('shampoo.txt')
time_series["date"] = time_series["Month"].apply(transform_to_data)
time_series = time_series[["date","Sales"]]
non_stationary_data = time_series

def differenced_data(df,d):
     # the difference of the first period is 0
     differenced_ = [0]
     t = len(df)
     for i in range(d,t):
           delta_vb = df[i]-df[i-d]
           differenced_.append(delta_vb)
           differenced_data = pd.Series(differenced_)
     return(differenced_data)
# we need the values only from the pd dataframe and only the sales
stationary_data = differenced_data(df = non_stationary_data["Sales"].values, d = 1)


import matplotlib.pyplot as plt
plt.plot(time_series["date"], time_series["Sales"],label = "Non-Stationary Sales",color = 'r')
plt.plot(time_series["date"], stationary_data, label = "Differenced -> Stationary Sales")
plt.title("Differencing to get Stationarity in Time Series")
plt.ylabel("Sales")
plt.xlabel("dates")
plt.legend()
plt.show()