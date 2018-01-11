import pandas as pd
import numpy as np
from io import StringIO
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from itertools import combinations
import matplotlib.pyplot as plt

# for sklearn 0.18's alternative syntax
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version
if Version(sklearn_version) < '0.18':
    from sklearn.grid_search import train_test_split
else:
    from sklearn.model_selection import train_test_split

#############################################################################
print(50 * '=')
print('Section: Dealing with missing data')
print(50 * '-')

csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

# If you are using Python 2.7, you need
# to convert the string to unicode:
# csv_data = unicode(csv_data)

df = pd.read_csv(StringIO(csv_data))
print(df)
print('\n\nExecuting df.isnull().sum():')
print(df.isnull().sum())


#############################################################################
print(50 * '=')
print('Section: Eliminating samples or features with missing values')
print(50 * '-')

print('\n\nExecuting df.dropna()')
print(df.dropna())

print('\n\nExecuting df.dropna(axis=1)')
print(df.dropna(axis=1))

print("\n\nExecuting df.dropna(thresh=4)")
print("(drop rows that have not at least 4 non-NaN values)")
print(df.dropna(thresh=4))

print("\n\nExecuting df.dropna(how='all')")
print("(only drop rows where all columns are NaN)")
print(df.dropna(how='all'))

print("\n\nExecuting df.dropna(subset=['C'])")
print("(only drop rows where NaN appear in specific columns (here: 'C'))")
print(df.dropna(subset=['C']))


#############################################################################
print(50 * '=')
print('Section: Imputing missing values')
print(50 * '-')

imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr = imr.fit(df)
imputed_data = imr.transform(df.values)

print('Input Array:\n', df.values)
print('Imputed Data:\n', imputed_data)


#############################################################################
print(50 * '=')
print('Section: Handling categorical data')
print(50 * '-')

df = pd.DataFrame([['green', 'M', 10.1, 'class1'],
                   ['red', 'L', 13.5, 'class2'],
                   ['blue', 'XL', 15.3, 'class1']])

df.columns = ['color', 'size', 'price', 'classlabel']
print('Input Array:\n', df)


#############################################################################
print(50 * '=')
print('Section: Mapping ordinal features')
print(50 * '-')

size_mapping = {'XL': 3,
                'L': 2,
                'M': 1}

df['size'] = df['size'].map(size_mapping)
print('Mapping:\n', df)

inv_size_mapping = {v: k for k, v in size_mapping.items()}
df_inv = df['size'].map(inv_size_mapping)
print('\nInverse mapping:\n', df_inv)


#############################################################################
print(50 * '=')
print('Section: Encoding class labels')
print(50 * '-')

class_mapping = {label: idx for idx, label
                 in enumerate(np.unique(df['classlabel']))}
print('\nClass mapping:\n', class_mapping)

df['classlabel'] = df['classlabel'].map(class_mapping)
print('Mapping:\n', df)

inv_class_mapping = {v: k for k, v in class_mapping.items()}
df_inv = df['classlabel'] = df['classlabel'].map(inv_class_mapping)
print('\nInverse mapping:\n', df_inv)

class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
print('Label encoder tansform:\n', y)

y_inv = class_le.inverse_transform(y)
print('Label encoder inverse tansform:\n', y_inv)


#############################################################################
print(50 * '=')
print('Section: Performing one hot encoding on nominal features')
print(50 * '-')

X = df[['color', 'size', 'price']].values

color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
print("Input array:\n", X)

ohe = OneHotEncoder(categorical_features=[0])
X_onehot = ohe.fit_transform(X).toarray()
print("Encoded array:\n", X_onehot)

df_dummies = pd.get_dummies(df[['price', 'color', 'size']])
print("Pandas get_dummies alternative:\n", df_dummies)
