# Importing the libraries
import numpy as np
import pandas as pd


# scikit-learn modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

# Visualizing the decision tree structure
import matplotlib.pyplot as plt
import cv2
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus


# Reading the data
dataset = pd.read_csv("petroleum_consumption.csv")
dataset.head()

x = dataset.drop('Petrol_Consumption', axis = 1) # Features
y = dataset['Petrol_Consumption']  # Target

# Splitting the dataset into training and testing set (75/25)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 2)

# Initializing the Decision Tree Regression model
model = DecisionTreeRegressor(random_state = 0)

# Fitting the Decision Tree Regression model to the data
model.fit(x_train, y_train)

# Predicting the target values of the test set
y_pred = model.predict(x_test)

# RMSE (Root Mean Square Error)
rmse = float(format(np.sqrt(mean_squared_error(y_test, y_pred)),'.3f'))
print("\nRMSE:",rmse)



# export the decision tree model to a tree_structure.dot file
# paste the contents of the file to webgraphviz.com
export_graphviz(model, out_file ='tree_structure.dot',
               feature_names = ['Petrol_tax', 'Average_income', 'Paved_Highways', 'Population_Driver_licence(%)'])


dot_data = StringIO()
export_graphviz(model, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,feature_names = ['Petrol_tax', 'Average_income', 'Paved_Highways', 'Population_Driver_licence(%)'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue().replace("\n", ""))
graph.write_png('petroleum.png')
Image(graph.create_png())

img = cv2.imread('petroleum.png')
plt.matshow(img)
plt.show()