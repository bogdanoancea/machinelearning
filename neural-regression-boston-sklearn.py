# Importing necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

data = pd.read_csv(filepath_or_buffer="http://lib.stat.cmu.edu/datasets/boston",delim_whitespace=True, skiprows=22,header=None)
columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
values_w_nulls = data.values.flatten()
all_values = values_w_nulls[~np.isnan(values_w_nulls)]

# Reshape the values to have 14 columns and make a new df out of them
data = pd.DataFrame(data=all_values.reshape(-1, len(columns)), columns=columns)
column_sels = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
X = data.loc[:,column_sels]
y = data['MEDV']

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Standardizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Creating the MLP neural network model
mlp_regressor = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=1)

# Training the model with the training data
mlp_regressor.fit(X_train, y_train)

# Making predictions on the test data
predictions = mlp_regressor.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")
