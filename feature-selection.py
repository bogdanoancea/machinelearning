import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE

wine_data = load_wine()
wine_df=pd.DataFrame(data=wine_data.data,  columns=wine_data.feature_names)

wine_df['target'] = wine_data.target
plt.figure()
sns.boxplot(x = wine_df['target'], y = wine_df['alcohol'])
plt.show()

plt.figure()
sns.boxplot(x = wine_df['target'], y = wine_df['ash'])
plt.show()
plt.figure()
sns.boxplot(x = wine_df['target'], y = wine_df['magnesium'])
plt.show()



X = wine_df.drop(['target'], axis=1)
y = wine_df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, shuffle=True, stratify=y,  random_state=42)


X_train_v1 = X_train.copy()
X_train_v1.var(axis=0)


norm = Normalizer().fit(X_train_v1)
norm_X_train = norm.transform(X_train_v1)
norm_X_train.var(axis=0)

selector = VarianceThreshold(threshold = 1e-6)
selected_features = selector.fit_transform(norm_X_train)
selected_features.shape




dt = DecisionTreeClassifier(random_state=500)
#Classifier with all features
dt.fit(X_train, y_train)
preds = dt.predict(X_test)
f1_score_all = round(f1_score(y_test, preds, average='weighted'),3)
# Classifier with selected features with variance threshold
X_train_sel = X_train.drop(['hue', 'nonflavanoid_phenols'], axis=1)
X_test_sel = X_test.drop(['hue', 'nonflavanoid_phenols'], axis=1)
dt.fit(X_train_sel, y_train)
preds_sel = dt.predict(X_test_sel)
f1_score_sel = round(f1_score(y_test, preds_sel, average='weighted'), 3)

X_train_v2, X_test_v2, y_train_v2, y_test_v2 = X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()
f1_score_list = []
for k in range(1, 14):
    selector = SelectKBest(chi2, k=k)
    selector.fit(X_train_v2, y_train_v2)

    sel_X_train_v2 = selector.transform(X_train_v2)
    sel_X_test_v2 = selector.transform(X_test_v2)

    dt.fit(sel_X_train_v2, y_train_v2)
    kbest_preds = dt.predict(sel_X_test_v2)
    f1_score_kbest = round(f1_score(y_test, kbest_preds, average='weighted'), 3)
    f1_score_list.append(f1_score_kbest)

print(f1_score_list)

fig, ax = plt.subplots(figsize=(12, 6))
x = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']
y = f1_score_list
ax.bar(x, y, width=0.4)
ax.set_xlabel('Number of features (selected using chi2 test)')
ax.set_ylabel('F1-Score (weighted)')
ax.set_ylim(0, 1.2)
for index, value in enumerate(y):
    plt.text(x=index, y=value + 0.05, s=str(value), ha='center')

plt.tight_layout()


X_train_v3, X_test_v3, y_train_v3, y_test_v3 = X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()
RFE_selector = RFE(estimator=dt, n_features_to_select=4, step=1)
RFE_selector.fit(X_train_v3, y_train_v3)
X_train_v3.columns[RFE_selector.support_]


sel_X_train_v3 = RFE_selector.transform(X_train_v3)
sel_X_test_v3 = RFE_selector.transform(X_test_v3)
dt.fit(sel_X_train_v3, y_train_v3)
RFE_preds = dt.predict(sel_X_test_v3)
rfe_f1_score = round(f1_score(y_test_v3, RFE_preds, average='weighted'),3)
print(rfe_f1_score)
