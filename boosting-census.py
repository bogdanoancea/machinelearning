import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
column_names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
                "hours-per-week", "native-country", "income"]
data = pd.read_csv(url, names=column_names, na_values=" ?", skipinitialspace=True)

# Handle missing values
data.dropna(inplace=True)

# Encode categorical features and target variable
for col in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

# Define features and target
X = data.drop('income', axis=1)
y = data['income']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Gradient Boosting Classifier
gradient_boosting_classifier = GradientBoostingClassifier(n_estimators=100,
                                                          learning_rate=0.1,
                                                          max_depth=3,
                                                          random_state=42)

# Fit the model on the training data
gradient_boosting_classifier.fit(X_train, y_train)

# Predict on the test data
y_pred = gradient_boosting_classifier.predict(X_test)

# Calculate the overall accuracy
overall_accuracy = accuracy_score(y_test, y_pred)
print(f"Overall accuracy of Gradient Boosting Classifier: {overall_accuracy:.3f}")
