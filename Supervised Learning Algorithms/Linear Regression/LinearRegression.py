import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample Data
stamps_bought = np.array([1, 3, 5, 7, 9]).reshape((-1, 1))  # Reshaping to make it a 2D array
amount_spent = np.array([2, 6, 8, 12, 18])

# Creating a Linear Regression Model
model = LinearRegression()

# Training the Model
model.fit(stamps_bought, amount_spent)

# Predictions
next_month_stamps = 10
predicted_spend = model.predict([[next_month_stamps]])

# Plotting
plt.scatter(stamps_bought, amount_spent, color='forestgreen')
plt.plot(stamps_bought, model.predict(stamps_bought), color='darkred')
plt.title('Stamps Bought vs Amount Spent')
plt.xlabel('Stamps Bought')
plt.ylabel('Amount Spent ($)')
plt.title('Regression Line')
plt.show()

# Displaying Prediction
print(f"If Alex buys {next_month_stamps} stamps next month, they will likely spend ${predicted_spend[0]:.2f}.")
