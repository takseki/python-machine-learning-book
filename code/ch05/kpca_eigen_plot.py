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
print(50 * '=')
print('Section: Projecting new data points')
print(50 * '-')

X, y = make_moons(n_samples=100, random_state=123)

#plt.figure(1)
#plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='^', alpha=0.5)
#plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='o', alpha=0.5)
#plt.show()

alphas, lambdas = rbf_kernel_pca2(X, gamma=15, n_components=2)


x_new = X[25]
print('New data point x_new:', x_new)

#x_proj = alphas[25]  # original projection

# rbf_kernel_pca.py にコメントで書いたように
# ここは固有ベクトルaに特異値sqrt(lambda)を掛けるべき
x_proj = np.sqrt(lambdas[0]) * alphas[25]
print('Original projection x_proj:', x_proj)

def project_x(x_new, X, gamma, alphas, lambdas):
    pair_dist = np.array([np.sum((x_new - row)**2) for row in X])
    k = np.exp(-gamma * pair_dist)
    
    # ここではsqrt(lambda_i)で割るのが正しい気がする
    # return k.dot(alphas / lambdas)
    return k.dot(alphas / np.sqrt(lambdas))

    # データx_newを主成分に直すにはv方向への射影をとって
    # x'_i = x_new^t v_i とする必要がある
    # ここで正規化された特異ベクトルの間の関係を使う。
    #   X^t a_i = sigma_i v_i      (sigma_i = sqrt(lambda_i))
    # これを代入して
    #   x'_i = x_new^t X^t a_i / sigma_i
    #        =  \sum_n k(x_new, x_n) a_i[n] / sqrt(lambda(i))

    # 本文にあるように
    # v_i = X^t a_i と定義していたらこの規格化項は出てこない
    # 一方、コード上では v_i, a_i がともにノルム1で規格化されているので、
    # v_i = X^t a_i / sigma_i という関係になっている

    # もともとのコードでは
    # X v_i / sigma_i をデータとしているので、余分にsigma_iで割る必要があり
    #   x'_i = \sum_n k(x_new, x_n) a_i[n] / lambda(i)
    # となる

# PRML 12章にあるような元の空間での基底関数のプロットをしてみる
#### ここから
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# 175,250 (resolution=0.02のとき)
resolution=0.05
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                       np.arange(x2_min, x2_max, resolution))
# 175*250,2
mesh = np.array([xx1.ravel(), xx2.ravel()]).T
# 175*250
z = np.zeros((mesh.shape[0], 2))
# ここが時間かかる
for i in range(mesh.shape[0]):
    z[i] = project_x(mesh[i], X, gamma=15, alphas=alphas, lambdas=lambdas)
# 175,250
print(z.shape)

for i in range(2):
    plt.figure(i)
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='^', alpha=0.5)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='o', alpha=0.5)
    z_i = z[:,i].reshape(xx1.shape)
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    plt.contourf(xx1, xx2, z_i, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

plt.figure(2)
plt.scatter(alphas[y == 0, 0], alphas[y == 0, 1], 
            color='red', marker='^', alpha=0.5)
plt.scatter(alphas[y == 1, 0], alphas[y == 1, 1], 
            color='blue', marker='o', alpha=0.5)

plt.show()
