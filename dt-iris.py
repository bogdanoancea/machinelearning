from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

# Visualizing the decision tree structure
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus


iris = load_iris()
X = iris.data[:, 2:] # petal length and width
y = iris.target
# without train test splitting
tree_clf = DecisionTreeClassifier(max_depth=4)
tree_clf.fit(X, y)


#Predict the response for train dataset
y_pred = tree_clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


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
                special_characters=True,feature_names = iris.feature_names[2:], class_names=iris.target_names)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('iris.png')
Image(graph.create_png())


# predictions
tree_clf.predict_proba([[5, 1.5]])
#array([[0, 0.90740741, 0.09259259]])
tree_clf.predict([[5, 1.5]])
#array([1])


# with train test splitting

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
tree_clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
tree_clf = tree_clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = tree_clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Evaluating the model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))