from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing

# Load the California Housing dataset
data = fetch_california_housing()
X = data.data
y = data.target
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize SVM with probability estimates turned on
svm = SVR( kernel='rbf', C = 2)
lr = LinearRegression(n_jobs=8)
# Initialize AdaBoost with SVM as the base estimator
ada_boost = AdaBoostRegressor(estimator=svm, n_estimators=10, random_state=42)
ada_boost2 = AdaBoostRegressor(estimator=lr, n_estimators=10, random_state=42)

# Fit AdaBoost model
ada_boost.fit(X_train, y_train)
ada_boost2.fit(X_train, y_train)
# Predict and evaluate the model
y_pred = ada_boost.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"r2: {r2:.3f}")

y_pred2 = ada_boost2.predict(X_test)
r22 = r2_score(y_test, y_pred2)
print(f"r2: {r22:.3f}")