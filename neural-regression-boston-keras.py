# Importing necessary libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

# Creating the Keras model
model = Sequential()
#model.add(Dense(50, input_dim=X_train.shape[1], activation='relu'))  # Input layer + first hidden layer
model.add(Input(shape=(13,)))
model.add(Dense(50, activation='relu'))  # Second hidden layer
model.add(Dense(50, activation='relu'))  # Second hidden layer
model.add(Dense(1))  # Output layer

# Compiling the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Training the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=0)

# Making predictions on the test data
predictions = model.predict(X_test).flatten()

# Evaluating the model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")
