import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

# Sample Data
# [movie_length, genre_code] (assuming genre is coded as: 0 for Action, 1 for Romance, etc.)
movies_features = np.array([[120, 0], [150, 1], [90, 0], [140, 1], [100, 0], [80, 1], [110, 0], [130, 1]])
movies_likes = np.array([1, 1, 0, 1, 0, 1, 0, 1])  # 1: Like, 0: Dislike

# Creating a Naive Bayes Model
model = GaussianNB()

# Training the Model
model.fit(movies_features, movies_likes)

# Prediction
new_movie = np.array([[100, 1]])  # [movie_length, genre_code]
predicted_like = model.predict(new_movie)

# Plotting
plt.scatter(movies_features[:, 0], movies_features[:, 1], c=movies_likes, cmap='viridis', marker='o')
plt.scatter(new_movie[:, 0], new_movie[:, 1], color='darkred', marker='x')
plt.title('Movie Likes Based on Length and Genre')
plt.xlabel('Movie Length (min)')
plt.ylabel('Genre Code')
plt.show()

# Displaying Prediction
print(f"Tom will {'like' if predicted_like[0] == 1 else 'not like'} a {new_movie[0, 0]}-min long movie of genre code {new_movie[0, 1]}.")
