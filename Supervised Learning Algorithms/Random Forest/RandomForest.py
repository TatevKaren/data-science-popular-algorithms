import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Expanded Data
plants_features = np.array([
    [3, 1], [2, 2], [4, 1], [3, 2], [5, 1], [2, 2], [4, 1], [5, 2],
    [3, 1], [4, 2], [5, 1], [3, 2], [2, 1], [4, 2], [3, 1], [4, 2],
    [5, 1], [2, 2], [3, 1], [4, 2], [2, 1], [5, 2], [3, 1], [4, 2]
])
plants_species = np.array([
    0, 1, 0, 1, 0, 1, 0, 1,
    0, 1, 0, 1, 0, 1, 0, 1,
    0, 1, 0, 1, 0, 1, 0, 1
])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(plants_features, plants_species, test_size=0.25, random_state=42)

# Creating a Random Forest Model
model = RandomForestClassifier(n_estimators=10, random_state=42)

# Training the Model
model.fit(X_train, y_train)

# Prediction & Evaluation
y_pred = model.predict(X_test)
classification_rep = classification_report(y_test, y_pred)

# Displaying Prediction and Evaluation
print("Classification Report:")
print(classification_rep)


# Scatter Plot Visualizing Classes
plt.figure(figsize=(8, 4))
for species, marker, color in zip([0, 1], ['o', 's'], ['forestgreen', 'darkred']):
    plt.scatter(plants_features[plants_species == species, 0],
                plants_features[plants_species == species, 1],
                marker=marker, color=color, label=f'Species {species}')
plt.xlabel('Leaf Size')
plt.ylabel('Flower Color (coded)')
plt.title('Scatter Plot of Species')
plt.legend()
plt.tight_layout()
plt.show()

# Visualizing Feature Importances
plt.figure(figsize=(8, 4))
features_importance = model.feature_importances_
features = ["Leaf Size", "Flower Color"]
plt.barh(features, features_importance, color = "darkred")
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.show()