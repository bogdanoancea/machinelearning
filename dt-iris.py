from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz

iris = load_iris()
X = iris.data[:, 2:] # petal length and width
y = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)


export_graphviz(
        tree_clf,
        out_file='iris_tree.dot',
        feature_names=iris.feature_names[2:],
        class_names=iris.target_names,
        rounded=True,
        filled=True
    )
#dot -Tpng iris_tree.dot -o

tree_clf.predict_proba([[5, 1.5]])
#array([[0, 0.90740741, 0.09259259]])
tree_clf.predict([[5, 1.5]])
#array([1])



tree_reg = DecisionTreeRegressor(max_depth=2)
tree_reg.fit(X, y)