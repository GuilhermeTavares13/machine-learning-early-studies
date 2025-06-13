import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv('titanic.csv')


x = dataset.iloc[:,10].values

print(x)


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[4])],remainder='passthrough')

x = ct.fit_transform(x)

print(x)
