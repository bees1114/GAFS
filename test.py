from sklearn import datasets
from base import GeneticFeatureSelection

import pandas as pd
import sys

FILE_PATH = 'D:/git/GAFS/dataset/titanic/'

data = pd.read_csv(FILE_PATH + 'train.csv')
#print(data.isnull().sum())
#
y = data[['Survived']]
X = data.drop(['Survived'], axis=1)
print(X.columns)
gfs = GeneticFeatureSelection(generations=2, population_size=5)
gfs.fit(X, y)