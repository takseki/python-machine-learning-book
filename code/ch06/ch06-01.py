import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from scipy import interp

# for sklearn 0.18's alternative syntax
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version
if Version(sklearn_version) < '0.18':
    from sklearn.grid_search import train_test_split
    from sklearn.cross_validation import StratifiedKFold
    from sklearn.cross_validation import cross_val_score
    from sklearn.learning_curve import learning_curve
    from sklearn.learning_curve import validation_curve
    from sklearn.grid_search import GridSearchCV
else:
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import learning_curve
    from sklearn.model_selection import validation_curve
    from sklearn.model_selection import GridSearchCV

#############################################################################
print(50 * '=')
print('Section: Loading the Breast Cancer Wisconsin dataset')
print(50 * '-')

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases'
                 '/breast-cancer-wisconsin/wdbc.data', header=None)
print('Breast Cancer dataset excerpt:\n\n')
print(df.head())

print('Breast Cancer dataset dimensions:\n\n')
print(df.shape)

X = df.loc[:, 2:].values
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)
y_enc = le.transform(['M', 'B'])
print("Label encoding example, le.transform(['M', 'B'])")
print(le.transform(['M', 'B']))

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.20, random_state=1)


#############################################################################
print(50 * '=')
print('Section: Combining transformers and estimators in a pipeline')
print(50 * '-')


pipe_lr = Pipeline([('scl', StandardScaler()),
                    ('pca', PCA(n_components=2)),
                    ('clf', LogisticRegression(random_state=1))])

pipe_lr.fit(X_train, y_train)
print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))
y_pred = pipe_lr.predict(X_test)
