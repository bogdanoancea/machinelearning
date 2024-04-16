import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load the California Housing dataset
data = fetch_california_housing()
X = data.data
y = data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the base estimator
base_estimator = DecisionTreeRegressor(max_depth=7, random_state=42)

# Initialize AdaBoost with Decision Trees as the base estimator
ada_boost_regressor = AdaBoostRegressor(estimator=base_estimator,
                                        n_estimators=150,  # Number of trees
                                        random_state=42)

# Fit the model on the training data
ada_boost_regressor.fit(X_train, y_train)

# Predict on the test data
y_pred = ada_boost_regressor.predict(X_test)

# Calculate the overall R-squared score
overall_r2 = r2_score(y_test, y_pred)
print(f"Overall R-squared of AdaBoost Regressor: {overall_r2:.3f}")


