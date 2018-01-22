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

# 円状の分布
## 自分で作らなくてもmake_circlesというのがある
np.random.seed(seed=1)
N = 500
r = np.random.normal(loc=1, scale=0.01, size=N)
th = 2 * np.pi * np.random.rand(N)

x = r * np.cos(th)
y = r * np.sin(th)

print(x.shape)
print(y.shape)

r = np.random.normal(loc=0.3, scale=0.01, size=N)
th = 2 * np.pi * np.random.rand(N)
x = np.append(x, r * np.cos(th))
y = np.append(y, r * np.sin(th))

print(x.shape)
print(y.shape)

X = np.column_stack((x, y))
print(X.shape)

plt.figure(1)
plt.axis('equal')
plt.scatter(X[0:N-1,0], X[0:N-1,1], color='r', marker='x')
plt.scatter(X[N:2*N:,0], X[N:2*N,1], color='b', marker='x')


# scikit-learn の KernelPCA で確認してみる
# 自前実装版と比べると、値のスケールが違う
# 線形カーネル
#scikit_kpca = KernelPCA(n_components=2, kernel='linear')
# ガウスカーネル
# gammaによって結果が全然違う, 線形分離可能になりそうなgammaの範囲は限られている
g = 4
#scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=g)
# 多項式カーネル
# x^2+y^2に相当する成分が第3主成分に現れるようなので、3成分までを計算
scikit_kpca = KernelPCA(n_components=3, kernel='poly', degree=2, coef0=0)

X_skernpca = scikit_kpca.fit_transform(X)


# 第3主成分までを3d散布図で
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(2)
ax = Axes3D(fig)
ax.scatter(X_skernpca[0:N-1, 0], X_skernpca[0:N-1, 1], X_skernpca[0:N-1, 2], color='r', marker='x')
ax.scatter(X_skernpca[N:2*N, 0], X_skernpca[N:2*N, 1], X_skernpca[N:2*N:, 2], color='b', marker='.')


plt.show()
