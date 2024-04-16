
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# Load the California Housing dataset
data = fetch_california_housing()
X = data.data
y = data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with preprocessing and Gradient Boosting regressor
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Feature scaling
    ('pca', PCA(n_components=0.95)),  # Dimensionality reduction
    ('gbr', GradientBoostingRegressor(random_state=42))
])

# Parameters grid for GridSearchCV
param_grid = {
    'gbr__n_estimators': [100, 200],
    'gbr__learning_rate': [0.1, 0.05],
    'gbr__max_depth': [3, 4, 5],
    'gbr__min_samples_split': [2, 4]
}

# Grid search to find the best model parameters
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='r2', verbose=2)
grid_search.fit(X_train, y_train)

# Predict on the test data using the best model
y_pred = grid_search.predict(X_test)

# Calculate the overall R-squared score
overall_r2 = r2_score(y_test, y_pred)
print(f"Optimized R-squared of Gradient Boosting Regressor: {overall_r2:.3f}")
