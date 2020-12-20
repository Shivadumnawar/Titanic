# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 01:15:52 2020

@author: rahul dumu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

train_df= pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

train = train_df.copy()
test= test_df.copy()

print(train.head())
print(test.head())

print(train.info())
print(test.info())
# it gives information of dataset

print(train.describe())
print(test.describe())
# it gives statistical details

# check null values
print(train.isnull().sum())
print(test.isnull().sum())

# there are null values in train dataset and test dataset 

print(train.columns)
print(test.columns)

train.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test.drop(columns= ['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace= True)

train['Age'].median()
train['Embarked'].mode()[0]

train['Age'].fillna(train['Age'].median(), inplace=True)
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)

print(train.isnull().sum())
# there are no null values in train dataset

test['Age'].median()
test['Fare'].median()

test['Age'].fillna(test['Age'].median(), inplace=True)
test['Fare'].fillna(test['Fare'].median(), inplace=True)

print(test.isnull().sum())
# there are no null values in test dataset

train['Survived'].value_counts()
train['Pclass'].value_counts()
train['Sex'].value_counts()
train['SibSp'].value_counts()
train['Parch'].value_counts()
train['Embarked'].value_counts()

test['Pclass'].value_counts()
test['Sex'].value_counts()
test['SibSp'].value_counts()
test['Parch'].value_counts()
test['Embarked'].value_counts()

# data visualization

plt.figure(figsize=(8,6))
sns.countplot(x='Survived', data= train)

plt.figure(figsize=(8,6))
sns.countplot(x='Sex', data= train)

plt.figure(figsize=(8,6))
sns.countplot(x='Survived', hue='Sex', data= train)

plt.figure(figsize=(8,6))
sns.countplot(x='Survived', hue='Pclass', data= train)

plt.figure(figsize=(8,6))
sns.boxplot(x='Survived', y= 'Age', hue='Sex', data= train)

plt.figure(figsize=(8,6))
sns.boxplot(x='Pclass', y= 'Fare', data= train)


plt.figure(figsize=(8,6))
sns.countplot(x='Sex', data= test)

plt.figure(figsize=(8,6))
sns.boxplot(x='Pclass', y= 'Fare', data= test)

# check outliers

train.plot(kind='box', figsize= (10,8))
# there are outliers

cols= ['Age', 'SibSp', 'Parch', 'Fare']

train[cols]= train[cols].clip(lower= train[cols].quantile(0.15), upper= train[cols].quantile(0.85), axis=1)

train.drop(columns=['Parch'], axis=1, inplace=True)

train.plot(kind='box', figsize= (10,8)) # no outliers 


test.plot(kind='box', figsize= (10,8))
# there are outliers

test[cols]= test[cols].clip(lower= test[cols].quantile(0.15), upper= test[cols].quantile(0.85), axis=1)

test.drop(columns=['Parch'], axis=1, inplace=True)

test.plot(kind='box', figsize= (10,8))  # no outliers

# one hot encoding

train= pd.get_dummies(train, columns=['Pclass', 'Sex', 'Embarked' ], drop_first= True)

test= pd.get_dummies(test, columns=['Pclass', 'Sex', 'Embarked' ], drop_first= True)

X_train= train.iloc[:, 1:]
y_train= train['Survived'].values.reshape(-1,1)

X_test= test

# scaling
from sklearn.preprocessing import StandardScaler
ss= StandardScaler()

features= ['Age', 'SibSp', 'Fare']

X_train[features]= ss.fit_transform(X_train[features])
X_test[features]= ss.fit_transform(X_test[features])

from sklearn.linear_model import LogisticRegression

clf= LogisticRegression()

clf.fit(X_train, y_train.ravel())

predictions= clf.predict(X_test)

print(clf.score(X_train, y_train))

submission= pd.DataFrame({'PassengerId' : test_df['PassengerId'], 'Survived': predictions })

print(submission.head())

filename= 'titanic predictions.csv'
submission.to_csv(filename, index=False)