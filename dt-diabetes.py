# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import export_graphviz
from six import StringIO

from IPython.display import display, Image
import pydotplus
import matplotlib.pyplot as plt
import cv2


# load dataset
#https://www.kaggle.com/uciml/pima-indians-diabetes-database
pima = pd.read_csv("diabetes.csv")
pima.head()

#split dataset in features and target variable
feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = pima[feature_cols] # Features
y = pima.Outcome # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Confusion matrix:",metrics.confusion_matrix(y_test, y_pred))


#pip install graphviz
#pip install pydotplus
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue().replace("\n", ""))
graph.write_png('diabetes.png')
img = Image(graph.create_png())
display(img)


# with matplotlib
img = cv2.imread('diabetes.png')
plt.imshow(img)
plt.show()


# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Confusion matrix:",metrics.confusion_matrix(y_test, y_pred))

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue().replace("\n", ""))
graph.write_png('diabetes2.png')


img = cv2.imread('diabetes2.png')
plt.matshow(img)
plt.show()

#subprocess.call(['dot', '-Tpdf', 'dot_data.dot', '-o' 'tree.pdf'])