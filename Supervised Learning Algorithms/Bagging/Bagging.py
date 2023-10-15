import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree  # Ensure plot_tree is imported
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Sample Data
clients_data = np.array([[2000, 60], [2500, 45], [1800, 75], [2200, 50], [2100, 62], [2300, 70], [1900, 55], [2000, 65]])
weight_loss = np.array([3, 2, 4, 3, 3.5, 4.5, 3.7, 4.2])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(clients_data, weight_loss, test_size=0.25, random_state=42)

# Creating a Bagging Model
base_estimator = DecisionTreeRegressor(max_depth=4)
model = BaggingRegressor(base_estimator=base_estimator, n_estimators=10, random_state=42)

# Training the Model
model.fit(X_train, y_train)

# Prediction & Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Displaying Prediction and Evaluation
print(f"True weight loss: {y_test}")
print(f"Predicted weight loss: {y_pred}")
print(f"Mean Squared Error: {mse:.2f}")

# Visualizing One of the Base Estimators (if desired)
plt.figure(figsize=(12, 8))
tree = model.estimators_[0]
plt.title('One of the Base Decision Trees from Bagging')
plot_tree(tree, filled=True, rounded=True, feature_names=["Calorie Intake", "Workout Duration"])
plt.show()
