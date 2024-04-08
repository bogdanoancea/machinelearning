# Importing necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Loading the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Standardizing the features
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Creating the MLP neural network model
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)

# Training the model with the training data
mlp.fit(X_train, y_train)

# Making predictions on the test data
predictions = mlp.predict(X_test)

# Evaluating the model
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
