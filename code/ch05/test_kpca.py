import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.decomposition import KernelPCA
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
from matplotlib.ticker import FormatStrFormatter
from plot_decision_regions import *
from rbf_kernel_pca import *
from numpy import cos, sin
from numpy import pi

# for sklearn 0.18's alternative syntax
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version
if Version(sklearn_version) < '0.18':
    from sklearn.grid_search import train_test_split
    from sklearn.lda import LDA
else:
    from sklearn.model_selection import train_test_split
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

#############################################################################

# 2次元正規分布に従う乱数を生成

# 平均
mu = np.array([0, 0])
# 共分散
cov_d = np.array([[8, 0],
                  [0, 1]])
c = cos(pi/3)
s = sin(pi/3)
R = np.array([[c, -s],
              [s, c]])
cov = np.dot(np.dot(R, cov_d), R.T)
print(cov)

N = 1000
np.random.seed(seed=1)
X = np.random.multivariate_normal(mu, cov, N)
print(X.shape)

plt.figure(1)
plt.axis('equal')
plt.scatter(X[:,0], X[:,1], color='r', marker='x')


# scikit-learn の KernelPCA で確認してみる
# 自前実装版と比べると、値のスケールが違う
scikit_kpca = KernelPCA(n_components=2, kernel='linear')

# ガウスカーネルだとイマイチ結果の意味がわからないため線形カーネル使う
g = 100
#scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=g)

X_skernpca = scikit_kpca.fit_transform(X)


# 自前実装版
X_kpca = linear_kernel_pca(X, 2)
#X_kpca = rbf_kernel_pca(X, g, 2)

plt.figure(2)
plt.axis('equal')
plt.scatter(X_skernpca[:, 0], X_skernpca[:, 1], color='b', marker='x')

plt.figure(3)
plt.axis('equal')
plt.scatter(X_kpca[:, 0], X_kpca[:, 1], color='g', marker='x')

plt.show()
