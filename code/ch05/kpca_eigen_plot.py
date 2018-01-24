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
from rbf_kernel_pca import *

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
# 計算する主成分の数
nc = 3

# 半月 or 円
X, y = make_moons(n_samples=100, random_state=123)
#X, y = make_circles(n_samples=100, random_state=123, noise=0.01, factor=0.3)

# 半月はガウスカーネルの第1主成分で分離できる
scikit_kpca = KernelPCA(n_components=nc, kernel='rbf', gamma=15)
# 円は2次多項式カーネルで分離できるが、分離可能な空間が第3主成分になる
#scikit_kpca = KernelPCA(n_components=nc, kernel='poly', degree=2, coef0=0)

X_kpca = scikit_kpca.fit_transform(X)

# PRML 12章にあるような元の空間での基底関数のプロットをしてみる
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# 175,250 (resolution=0.02のとき)
resolution=0.02
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                       np.arange(x2_min, x2_max, resolution))
# 175*250,2
mesh = np.array([xx1.ravel(), xx2.ravel()]).T

z = scikit_kpca.transform(mesh)

# デフォルトのcontour色だとデータ点と重なった時見づらい
colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
cmap = ListedColormap(colors[:len(np.unique(y))])

# 上位成分を元空間上に等高線図でプロット
for i in range(nc):
    plt.figure(i)
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='^', alpha=0.5)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='o', alpha=0.5)
    z_i = z[:,i].reshape(xx1.shape)
    plt.contourf(xx1, xx2, z_i, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(nc)
ax = Axes3D(fig)
ax.scatter(X_kpca[y == 0, 0], X_kpca[y == 0, 1], X_kpca[y == 0, 2],
            color='red', marker='^', alpha=0.5)
ax.scatter(X_kpca[y == 1, 0], X_kpca[y == 1, 1], X_kpca[y == 1, 2], 
            color='blue', marker='o', alpha=0.5)

plt.show()
