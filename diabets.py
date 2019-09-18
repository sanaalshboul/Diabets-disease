# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 10:38:16 2019

@author: Administrator
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix



#read Diabetes dataset and rename the columns' names
diabets=pd.read_csv('C:\\Users\\Administrator.Omnisec421_05\\Downloads\\Diabetes.csv')
diabets.columns=['Pregnancies','Glucose','BloodPressure','SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age','diabetes']


diabets.Insulin.replace(0, np.nan, inplace=True)
diabets.SkinThickness.replace(0, np.nan, inplace=True)
diabets.BMI.replace(0, np.nan, inplace=True)



diabets['Insulin'] = diabets['Insulin'].fillna(diabets['Insulin'].mean())
diabets['SkinThickness'] = diabets['SkinThickness'].fillna(diabets['SkinThickness'].mean())
diabets['BMI'] = diabets['BMI'].fillna(diabets['BMI'].mean())

diabets=diabets.dropna()

#plot EDA
features=list(diabets.columns)
fig, ax = plt.subplots(figsize=(8,8))
scatter_matrix(diabets, alpha=1, ax=ax)
plt.savefig('Scatter Matrix.png')
#******************************************************************************
#fining the outliers
out=diabets.describe().T
IQR = out['75%'] - out['25%']

# create an outliers column which is either 3 IQRs below the first quartile or
# 3 IQRs above the third quartile
out['outliers'] = (out['min']<(out['25%']-(3*IQR)))|(out['max'] > (out['75%']+3*IQR))

# just display the features containing extreme outliers
out.ix[out.outliers,]
#******************************************************************************
#calculating the Pearson correlation coefficient (r) is a measurement of the amount of linear correlation between equal length arrays which outputs a value ranging -1 to 1
print(diabets.corr()[['diabetes']].sort_values('diabetes'))


#******************************************************************************
#splitting the dataset to train, validation, testing sets
y=diabets['diabetes']

X=diabets.drop(columns=['diabetes'], axis=1)

X_temp, X_test, y_temp, y_test=train_test_split(X,y, test_size=.2, random_state=42)

X_train, X_vald, y_train, y_vald=train_test_split(X_temp, y_temp, test_size=0.1, random_state=42)
#******************************************************************************
#Calculate a Pearson correlation coefficient 
features=diabets.columns
for i in features:
    print(i,pearsonr(diabets[i],y))
#******************************************************************************
#GradientBoostingClassifier
gb=GradientBoostingClassifier(n_estimators=20, learning_rate = 0.25, max_features=2, max_depth = 1, random_state = 0)

gb.fit(X_train, y_train)

prediction=gb.predict(X_vald)
print('Gradient training score: ', gb.score(X_train,y_train))
print('Gradient validation score: ', gb.score(X_vald, y_vald))
print("Gradient testing score: %.2f" % gb.score(X_test, y_test))
#******************************************************************************
#logistic regeression
logreg=LogisticRegression()

paramters={'random_state':[0,1,2,3,4,5,10], 'solver':['newton-cg', 'lbfgs',  'sag', 'saga'], 'multi_class':['ovr', 'multinomial']}
grid3=RandomizedSearchCV(logreg,paramters)
grid3.fit(X_train,y_train)
print('Best Parameters: ', grid3.best_params_)
print('Best score: ', grid3.best_score_)

logreg=LogisticRegression(random_state=10, solver='newton-cg', multi_class='ovr')
logreg.fit(X_train,y_train)

logreg.predict(X_vald)
print("Logistic regression score after tuning: ", logreg.score(X_test, y_test))
print("Logistic regression score after tuning: ", logreg.score(X_train, y_train))

#plot ROC 
y_pred=logreg.predict(X_test)
y_prop_pred=logreg.predict_proba(X_test)[:,1]
fpr,tpr,threshold=roc_curve(y_test,y_prop_pred)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.show();

print('logistic regression validation score: ', logreg.score(X_vald, y_vald))
print('logistic regression test score: ', logreg.score(X_test, y_test))
print('ROC_auc_score: ', roc_auc_score(y_test,y_prop_pred))

#print the confusion matrix
print('confusion matrix: ', confusion_matrix(y_test,y_pred))

#Diagnosing classification predictions
precision=81/(81+20)
Recall=81/(81+18)
F1_score=2*(precision*Recall)/(precision+Recall)
print('Precision: ', precision)
print('Recall: ',Recall)
print('F1 score: ',F1_score )