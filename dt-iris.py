from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
# Visualizing the decision tree structure
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus
import matplotlib.pyplot as plt
import cv2

iris = load_iris()
print('Classes to predict: ', iris.target_names)

X = iris.data
y = iris.target
print('Number of examples in the data:', X.shape[0])
#First four rows in the variable 'X'
X[:4]

# without train test splitting
tree_clf = DecisionTreeClassifier(criterion = 'entropy', max_depth=4)
tree_clf.fit(X, y)


#Predict the response for train dataset
y_pred = tree_clf.predict(X)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y, y_pred))


export_graphviz(
        tree_clf,
        out_file='iris_tree.dot',
        feature_names=iris.feature_names[2:],
        class_names=iris.target_names,
        rounded=True,
        filled=True
    )
#dot -Tpng iris_tree.dot -o

dot_data = StringIO()
export_graphviz(tree_clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,feature_names = iris.feature_names, class_names=iris.target_names)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue().replace("\n", ""))
graph.write_png('iris.png')
Image(graph.create_png())

img = cv2.imread('iris.png')
plt.matshow(img)
plt.show()

# predictions
tree_clf.predict_proba([[5, 1.5, 1.4, 0.2]])
tree_clf.predict(np.array([5, 1.5, 1.4, 0.2]).reshape(1,-1))


# with train test splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
tree_clf = DecisionTreeClassifier(criterion='entropy')

# Train Decision Tree Classifer
tree_clf = tree_clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = tree_clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Evaluating the model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#optimization
clf = DecisionTreeClassifier(criterion='entropy', min_samples_split=50)
clf.fit(X_train, y_train)
print('Accuracy Score on train data: ', accuracy_score(y_true=y_train, y_pred=clf.predict(X_train)))
print('Accuracy Score on the test data: ', accuracy_score(y_true=y_test, y_pred=clf.predict(X_test)))
