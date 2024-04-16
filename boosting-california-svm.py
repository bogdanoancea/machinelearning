from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_california_housing

# Load the California Housing dataset
data = fetch_california_housing()
X = data.data
y = data.target
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize SVM with probability estimates turned on
svm = SVR( kernel='rbf')

# Initialize AdaBoost with SVM as the base estimator
ada_boost = AdaBoostRegressor(estimator=svm, n_estimators=50, random_state=42)

# Fit AdaBoost model
ada_boost.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = ada_boost.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")