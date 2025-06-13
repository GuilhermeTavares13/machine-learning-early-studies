# Importing the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer

# Load the dataset
dataset = pd.read_csv('pima-indians-diabetes.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Identify missing data (assumes that missing data is represented as NaN)
missing_data = dataset.isnull().sum()

# Print the number of missing entries in each column
print("Missing data: \n",missing_data)

# Configure an instance of the SimpleImputer class
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# Fit the imputer on the DataFrame
imputer.fit(dataset)

# Apply the transform to the DataFrame
dataset_imputed = imputer.transform(dataset)

#Print your updated matrix of features
print("Update matrix of features: \n",dataset_imputed)
