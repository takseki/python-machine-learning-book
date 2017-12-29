import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import export_graphviz
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import warnings
from plot_decision_regions import *

# for sklearn 0.18's alternative syntax
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version
if Version(sklearn_version) < '0.18':
    from sklearn.grid_search import train_test_split
else:
    from sklearn.model_selection import train_test_split

#############################################################################
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
print('Class labels:', np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

#############################################################################
print(50 * '=')
print('Section: Training a logistic regression model with scikit-learn')
print(50 * '-')


lr = LogisticRegression(C=1000.0, random_state=0)
#lr = LogisticRegression(C=0.01, random_state=0)
lr.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined,
                      classifier=lr, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
# plt.tight_layout()
# plt.savefig('./figures/logistic_regression.png', dpi=300)
plt.show()

print('Predicted probabilities', lr.predict_proba(X_test_std[0, :]
                                                  .reshape(1, -1)))

# seki: confirms that probability is equal to sigmoid(w^t x)
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

print('coef_', lr.coef_)
print('intercept_', lr.intercept_)
print('X_test_std[0]', X_test_std[0, :])
# [[-7.34015187 -6.64685581]
#   [ 2.54373335 -2.3421979 ]
#   [ 9.46617627  6.44380858]]
# [-9.31757401 -0.89462847 -8.85765974]
# [ 0.70793846  1.50872803]

w = np.c_[lr.intercept_.reshape(3,1), lr.coef_]
print('W', w)
x = np.r_[np.array([1]), X_test_std[0, :]]
print('x', x)
print('W^t x', np.dot(w, x))
print('sigmoid(W^t, x)', sigmoid(np.dot(w, x)))
