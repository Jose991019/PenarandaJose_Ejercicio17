import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # para leer datos
import sklearn.ensemble # para el random forest
import sklearn.model_selection # para split train-test
import sklearn.metrics # para calcular el f1-score

from scipy.io import arff
import pandas as pd

data1 = arff.loadarff('1year.arff')
df1 = pd.DataFrame(data1[0])
df1 = np.array(df1)

data2 = arff.loadarff('2year.arff')
df2 = pd.DataFrame(data2[0])
df2 = np.array(df2)

data3 = arff.loadarff('3year.arff')
df3 = pd.DataFrame(data3[0])
df3 = np.array(df3)

data4 = arff.loadarff('4year.arff')
df4 = pd.DataFrame(data4[0])
df4 = np.array(df4)

data5 = arff.loadarff('5year.arff')
df5 = pd.DataFrame(data5[0])
df5 = np.array(df5)

dfs = [df2,df3,df4,df5]
df = df1 
for i in range(4):
    df = np.concatenate((df,dfs[i]), axis = 0)
    
df = np.array(df,dtype = np.float)
nans = np.where(np.isnan(df))
nans = np.array(list(set(nans[0])))
df = np.delete(df,nans,axis = 0)

x = np.array(df[:,:-1])
y = np.array(df[:,-1])

x_train, x_split, y_train, y_split = sklearn.model_selection.train_test_split(x, y, test_size=0.5)
x_test, x_validation, y_test, y_validation = sklearn.model_selection.train_test_split(x_split, y_split, test_size=0.6)

clf = sklearn.ensemble.RandomForestClassifier(n_estimators=10, max_features='sqrt')

n_trees = np.arange(1,200,10)
f1_train = []
f1_test = []

for i, n_tree in enumerate(n_trees):
    clf = sklearn.ensemble.RandomForestClassifier(n_estimators=n_tree, max_features='sqrt')
    clf.fit(x_train, y_train)
    f1_train.append(sklearn.metrics.f1_score(y_train, clf.predict(x_train)))
    f1_test.append(sklearn.metrics.f1_score(y_test, clf.predict(x_test)))
    
plt.figure(figsize = (10,10))
plt.scatter(n_trees, f1_test)
plt.xlabel('Número de árboles')
plt.ylabel('F1-score')
plt.savefig('mejor_Arbol.png')

mejor_numero = n_trees[np.where(f1_test == np.max(f1_test))]
clf = sklearn.ensemble.RandomForestClassifier(n_estimators=mejor_numero[0], max_features='sqrt')
clf.fit(x_train, y_train)
F1_validation = sklearn.metrics.f1_score(y_validation, clf.predict(x_validation))
feature_importance = clf.feature_importances_

predictors = [ 'net profit / total assets'
 ,'total liabilities / total assets'
,'working capital / total assets'
,'current assets / short-term liabilities'
,'[(cash + short-term securities + receivables - short-term liabilities) / (operating expenses - depreciation)] * 365'
,'retained earnings / total assets'
,'EBIT / total assets'
,'book value of equity / total liabilities'
,'sales / total assets'
 ,'equity / total assets'
 ,'(gross profit + extraordinary items + financial expenses) / total assets'
 ,'gross profit / short-term liabilities'
 ,'(gross profit + depreciation) / sales'
 ,'(gross profit + interest) / total assets'
 ,'(total liabilities * 365) / (gross profit + depreciation)'
 ,'(gross profit + depreciation) / total liabilities'
 ,'total assets / total liabilities'
 ,'gross profit / total assets'
 ,'gross profit / sales'
 ,'(inventory * 365) / sales'
 ,'sales (n) / sales (n-1)'
 ,'profit on operating activities / total assets'
 ,'net profit / sales'
 ,'gross profit (in 3 years) / total assets'
 ,'(equity - share capital) / total assets'
 ,'(net profit + depreciation) / total liabilities'
 ,'profit on operating activities / financial expenses'
 ,'working capital / fixed assets'
 ,'logarithm of total assets'
 ,'(total liabilities - cash) / sales'
 ,'(gross profit + interest) / sales'
 ,'(current liabilities * 365) / cost of products sold'
 ,'operating expenses / short-term liabilities'
 ,'operating expenses / total liabilities'
 ,'profit on sales / total assets'
 ,'total sales / total assets'
 ,'(current assets - inventories) / long-term liabilities'
 ,'constant capital / total assets'
 ,'profit on sales / sales'
 ,'(current assets - inventory - receivables) / short-term liabilities'
 ,'total liabilities / ((profit on operating activities + depreciation) * (12/365))'
 ,'profit on operating activities / sales'
 ,'rotation receivables + inventory turnover in days'
 ,'(receivables * 365) / sales'
 ,'net profit / inventory'
 ,'(current assets - inventory) / short-term liabilities'
 ,'(inventory * 365) / cost of products sold'
 ,'EBITDA (profit on operating activities - depreciation) / total assets'
 ,'EBITDA (profit on operating activities - depreciation) / sales'
 ,'current assets / total liabilities'
 ,'short-term liabilities / total assets'
 ,'(short-term liabilities * 365) / cost of products sold)'
 ,'equity / fixed assets'
 ,'constant capital / fixed assets'
 ,'working capital'
 ,'(sales - cost of products sold) / sales'
 ,'(current assets - inventory - short-term liabilities) / (sales - gross profit - depreciation)'
 ,'total costs /total sales'
 ,'long-term liabilities / equity'
 ,'sales / inventory'
 ,'sales / receivables'
 ,'(short-term liabilities *365) / sales'
 ,'sales / short-term liabilities'
 ,'sales / fixed assets']
plt.figure(figsize = (30,8))
a = pd.Series(feature_importance, index=predictors)
a.nlargest().plot(kind='barh')
plt.xlabel('Average Feature Importance')
plt.title('{} arbol(es), F1-score = {:.3f}'.format(mejor_numero[0],F1_validation))
plt.savefig('features.png')