import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.preprocessing import MinMaxScaler
from ucimlrepo import fetch_ucirepo
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from sklearn.metrics import classification_report
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

RANDOM_SEED = np.random.seed(0)
data = pd.read_csv(filepath_or_buffer="http://lib.stat.cmu.edu/datasets/boston",delim_whitespace=True, skiprows=22,header=None)
columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
values_w_nulls = data.values.flatten()
all_values = values_w_nulls[~np.isnan(values_w_nulls)]

# Reshape the values to have 14 columns and make a new df out of them
data = pd.DataFrame(data=all_values.reshape(-1, len(columns)), columns=columns)

data


column_sels = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
x = data.loc[:,column_sels]
y = data['MEDV']
min_max_scaler = MinMaxScaler()
x = pd.DataFrame(data=min_max_scaler.fit_transform(x), columns=column_sels)

regressor = linear_model.LinearRegression()
regressor.fit(x,y)
y_hat = regressor.predict(x)

mse = (y-y_hat)**2

print(f"MSE: {mse.mean():0.2f} (+/- {mse.std():0.2f})")

mae = np.abs(y-y_hat)
print(f"MAE: {mae.mean():0.2f} (+/- {mae.std():0.2f})")

mse = (y-y_hat)**2
rmse = np.sqrt(mse.mean())
print(f"RMSE: {rmse:0.2f}")


# R^2 coefficient of determination
SE_line = sum((y-y_hat)**2)
SE_mean = sum((y-y.mean())**2)
r2 = 1-(SE_line/SE_mean)
print(f"R^2 coefficient of determination: {r2*100:0.2f}%")

##Classification

# fetch dataset
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

# data (as pandas dataframes)
X = breast_cancer_wisconsin_diagnostic.data.features
y = breast_cancer_wisconsin_diagnostic.data.targets

# metadata
print(breast_cancer_wisconsin_diagnostic.metadata)

# variable information
print(breast_cancer_wisconsin_diagnostic.variables)

tmp = [X, y]
df = pd.concat(tmp, axis=1)
df = df.replace("?", np.nan)
df.dropna(inplace=True)
df.Diagnosis.value_counts().plot.bar()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
clf_svc = SVC(C=1.0,
                kernel='rbf',
                degree=3,
                gamma='scale',
                coef0=0.0,
                shrinking=True,
                probability=False,
                tol=0.001,
                cache_size=200,
                class_weight=None,
                verbose=False,
                max_iter=-1,
                decision_function_shape='ovr',
                break_ties=False,
                random_state=None)

clf_svc.fit(X_train,y_train)
y_hat = clf_svc.predict(X_test)
print(f'Accuracy Score is {accuracy_score(y_test,y_hat)}')


def find_TP(y, y_hat):
   # counts the number of true positives (y = 1, y_hat = 1)
   return sum((y == 'M') & (y_hat == 'M'))

def find_FN(y, y_hat):
   # counts the number of false negatives (y = 1, y_hat = 0) Type-II error
   return sum((y == 'M') & (y_hat == 'B'))

def find_FP(y, y_hat):
   # counts the number of false positives (y = 0, y_hat = 1) Type-I error
   return sum((y == 'B') & (y_hat == 'M'))

def find_TN(y, y_hat):
   # counts the number of true negatives (y = 0, y_hat = 0)
   return sum((y == 'B') & (y_hat == 'B'))


clf_1 = LogisticRegression(C=1.0, class_weight={'B':100,'M':0.2}, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)

clf_2 = LogisticRegression(C=1.0, class_weight={'B':0.001,'M':90000}, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)

#Precision
clf_1.fit(X,y)
y_hat = clf_1.predict(X)

TP = find_TP(y.values, y_hat.reshape(-1,1))
FN = find_FN(y.values, y_hat.reshape(-1,1))
FP = find_FP(y.values, y_hat.reshape(-1,1))
TN = find_TN(y.values, y_hat.reshape(-1,1))
print('TP:',TP)
print('FN:',FN)
print('FP:',FP)
print('TN:',TN)
precision = TP/(TP+FP)
print('Precision:',precision[0])

clf_2.fit(X,y)
y_hat = clf_2.predict(X)

TP = find_TP(y.values, y_hat.reshape(-1,1))
FN = find_FN(y.values, y_hat.reshape(-1,1))
FP = find_FP(y.values, y_hat.reshape(-1,1))
TN = find_TN(y.values, y_hat.reshape(-1,1))
print('TP:',TP)
print('FN:',FN)
print('FP:',FP)
print('TN:',TN)
precision = TP/(TP+FP)
print('Precision:',precision[0])


#Recall
clf_1.fit(X,y)
y_hat = clf_1.predict(X)

TP = find_TP(y.values, y_hat.reshape(-1,1))
FN = find_FN(y.values, y_hat.reshape(-1,1))
FP = find_FP(y.values, y_hat.reshape(-1,1))
TN = find_TN(y.values, y_hat.reshape(-1,1))
print('TP:',TP)
print('FN:',FN)
print('FP:',FP)
print('TN:',TN)
recall = TP/(TP+FN)
precision = TP/(TP+FP)
print('Recall:',recall[0])
f1score = 2*((precision*recall)/(precision+recall))
print('F1 score: %f' % f1score)

clf_2.fit(X,y)
y_hat = clf_2.predict(X)

TP = find_TP(y.values, y_hat.reshape(-1,1))
FN = find_FN(y.values, y_hat.reshape(-1,1))
FP = find_FP(y.values, y_hat.reshape(-1,1))
TN = find_TN(y.values, y_hat.reshape(-1,1))
print('TP:',TP)
print('FN:',FN)
print('FP:',FP)
print('TN:',TN)
recall = TP/(TP+FN)
precision = TP/(TP+FP)
print('recall:',recall[0])

f1score = 2*((precision*recall)/(precision+recall))
print('F1 score: %f' % f1score)


ns_probs = [0 for _ in range(len(y))]
# predict probabilities
lr_probs = clf_1.predict_proba(X)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(y, ns_probs)
lr_auc = roc_auc_score(y, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y, ns_probs, pos_label='M')
lr_fpr, lr_tpr, _ = roc_curve(y, lr_probs, pos_label='M')
# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
pyplot.show()

# all in one
report = classification_report(y, y_hat)
print(report)