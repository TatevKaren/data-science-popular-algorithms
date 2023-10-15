import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Sample Data
# [size, sweetness]
fruits_features = np.array([[3, 7], [2, 8], [3, 6], [4, 7], [1, 4], [2, 3], [3, 2], [4, 3]])
fruits_likes = np.array([1, 1, 1, 1, 0, 0, 0, 0])  # 1: Like, 0: Dislike

# Creating an LDA Model
model = LinearDiscriminantAnalysis()

# Training the Model
model.fit(fruits_features, fruits_likes)

# Prediction
new_fruit = np.array([[2.5, 6]])  # [size, sweetness]
predicted_like = model.predict(new_fruit)

# Plotting
plt.scatter(fruits_features[:, 0], fruits_features[:, 1], c=fruits_likes, cmap='viridis', marker='o')
plt.scatter(new_fruit[:, 0], new_fruit[:, 1], color='darkred', marker='x')
plt.title('Fruits Enjoyment Based on Size and Sweetness')
plt.xlabel('Size')
plt.ylabel('Sweetness')
plt.show()

# Displaying Prediction
print(f"Sarah will {'like' if predicted_like[0] == 1 else 'not like'} a fruit of size {new_fruit[0, 0]} and sweetness {new_fruit[0, 1]}.")
