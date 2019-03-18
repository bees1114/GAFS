from base import GeneticFeatureSelection


import pandas as pd
import sys

FILE_PATH = 'D:/git/GAFS/dataset/titanic/'

data = pd.read_csv(FILE_PATH + 'train.csv')

y = data[['Survived']]
X = data.drop(['Survived'], axis=1)

# One hot Encoding
Pclass = pd.get_dummies(X['Pclass'], 'Pclass')
Sex = pd.get_dummies(X['Sex'], columns=['M', 'W'])
SibSp = pd.get_dummies(X['SibSp'], 'SibSp')
Parch = pd.get_dummies(X['Parch'], 'Parch')
Embarked = pd.get_dummies(X['Embarked'], 'Embarked')
Cabin = pd.DataFrame([1 if pd.notnull(i) else 0 for i in X['Cabin']], columns=['new_Cabin'])

# Make features
X = pd.concat([X, Pclass, Sex, SibSp, Parch, Embarked, Cabin], axis=1)
X = X.drop(['PassengerId', 'Pclass', 'Sex', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis=1)
X = X.fillna(0)

print(X.head())
print(X.columns)

gfs = GeneticFeatureSelection(generations=2, population_size=5)
gfs.fit(X, y)
