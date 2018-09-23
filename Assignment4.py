#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 09:04:29 2018

@author: sherry
"""
import pandas as pd
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('https://raw.githubusercontent.com/rasbt/' 'python-machine-learning-book-2nd-edition' '/master/code/ch10/housing.data.txt', sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS','NOX', 'RM', 'AGE', 'DIS', 'RAD','TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
#basic information
print(df.head())
print(df.tail())
summary=df.describe()
print(summary)
#pairplot
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
sns.pairplot(df[cols], size=2.5)
plt.tight_layout()
plt.show()
#heatmap
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size': 15},yticklabels=cols,xticklabels=cols)
plt.show()
#hist
sns.set()
_ = plt.hist(df['MEDV'])
_ = plt.xlabel('The price of housing') 
_ = plt.ylabel('The number of housing') 
plt.show()
#warmplot
_ = sns.swarmplot(x='RM', y='MEDV', data=df) 
_ = plt.xlabel('Average number of Room') 
_ = plt.ylabel('Price') 
plt.show()

#Box plot
_=sns.boxplot(x='RM',y='MEDV',data=df)
_=plt.xlabel('RM')
_=plt.ylabel('MEDV')
plt.show()

#ECDF
x = np.sort(df['MEDV'])
y = np.arange(1, len(x)+1) / len(x) 
_ = plt.plot(x, y, marker='.', linestyle='none') 
_ = plt.xlabel('The price') 
_ = plt.ylabel('ECDF') 
plt.margins(0.02) 
plt.show()

#Regression Model
x = df['RM'].values.reshape(-1,1)
y = df['MEDV'].values.reshape(-1,1)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.linear_model import LinearRegression
slr = LinearRegression()
slr.fit(x_train, y_train)
print('Slope: %.3f' % slr.coef_[0]) 
print('Intercept: %.3f' % slr.intercept_)

def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)
    return None

lin_regplot(x_train, y_train, slr)
plt.xlabel('Average number of rooms [RM] ')
plt.ylabel('Price in $1000s [MEDV] ')
plt.show()

Xb = np.hstack((np.ones((x.shape[0], 1)), x))
w = np.zeros(x.shape[1])
z = np.linalg.inv(np.dot(Xb.T, Xb))
w = np.dot(z, np.dot(Xb.T, y))
print('Slope: %.3f' % w[1])
print('Intercept: %.3f' % w[0])

y_train_pred = slr.predict(x_train)
y_test_pred = slr.predict(x_test)

#plot the error
plt.scatter(y_train_pred, y_train_pred - y_train,c='steelblue', marker='o', edgecolor='white', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='limegreen', marker='s', edgecolor='white', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.show()

#compute MSE
from sklearn.metrics import mean_squared_error
print('MSE train: %.3f, test: %.3f' % 
      (mean_squared_error(y_train, y_train_pred),mean_squared_error(y_test, y_test_pred)))

#compute R^2
from sklearn.metrics import r2_score
print('R^2 train: %.3f, test: %.3f' % 
      (r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))

#Ridge
from sklearn.linear_model import Ridge
for i in [1.0,2.0,5.0,8.0]:
    ridge = Ridge(alpha=i)
    ridge.fit(x_train, y_train)
    print('Ridge Slope: %.3f' % ridge.coef_[0]) 
    print('Ridge Intercept: %.3f' % ridge.intercept_)
    y_train_pred = ridge.predict(x_train)
    y_test_pred = ridge.predict(x_test)

    plt.scatter(y_train_pred, y_train_pred - y_train,c='steelblue', marker='o', edgecolor='white', label='Training data')
    plt.scatter(y_test_pred, y_test_pred - y_test, c='limegreen', marker='s', edgecolor='white', label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
    plt.xlim([-10, 50])
    plt.show()
    print('Ridge MSE train: %.3f, test: %.3f' % 
      (mean_squared_error(y_train, y_train_pred),mean_squared_error(y_test, y_test_pred)))
    print('Ridge R^2 train: %.3f, test: %.3f' % 
      (r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))



#Lasso
from sklearn.linear_model import Lasso
for i in [1.0,2.0,5.0,8.0]:
    lasso = Lasso(alpha=i)
    lasso.fit(x_train, y_train)
    print('Lasso Slope: %.3f' % lasso.coef_[0]) 
    print('Lasso Intercept: %.3f' % lasso.intercept_)
    y_train_pred = lasso.predict(x_train).reshape(-1,1)
    y_test_pred = lasso.predict(x_test).reshape(-1,1)

    plt.scatter(y_train_pred, y_train_pred - y_train,c='steelblue', marker='o', edgecolor='white', label='Training data')
    plt.scatter(y_test_pred, y_test_pred - y_test, c='limegreen', marker='s', edgecolor='white', label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
    plt.xlim([-10, 50])
    plt.show()
    print('Lasso MSE train: %.3f, test: %.3f' % 
      (mean_squared_error(y_train, y_train_pred),mean_squared_error(y_test, y_test_pred)))
    print('Lasso R^2 train: %.3f, test: %.3f' % 
      (r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))


#Elanet
from sklearn.linear_model import ElasticNet
for i in [0.1,0.2,0.5,1]:
    elanet = ElasticNet(alpha=1.0, l1_ratio=i)
    elanet.fit(x_train, y_train)
    print('Elanet Slope: %.3f' % elanet.coef_[0]) 
    print('Elanet Intercept: %.3f' % elanet.intercept_)
    y_train_pred = elanet.predict(x_train).reshape(-1,1)
    y_test_pred = elanet.predict(x_test).reshape(-1,1)

    plt.scatter(y_train_pred, y_train_pred - y_train,c='steelblue', marker='o', edgecolor='white', label='Training data')
    plt.scatter(y_test_pred, y_test_pred - y_test, c='limegreen', marker='s', edgecolor='white', label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
    plt.xlim([-10, 50])
    plt.show()
    print('Elanet MSE train: %.3f, test: %.3f' % 
      (mean_squared_error(y_train, y_train_pred),mean_squared_error(y_test, y_test_pred)))
    print('Elanet R^2 train: %.3f, test: %.3f' % 
      (r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))

print('My name is Sihan Li')
print('My NetId is sihanl2')
print('I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.')

