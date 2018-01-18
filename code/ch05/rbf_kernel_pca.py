import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh

def rbf_kernel_pca(X, gamma, n_components):
    """
    RBF kernel PCA implementation.

    Parameters
    ------------
    X: {NumPy ndarray}, shape = [n_samples, n_features]

    gamma: float
      Tuning parameter of the RBF kernel

    n_components: int
      Number of principal components to return

    Returns
    ------------
     X_pc: {NumPy ndarray}, shape = [n_samples, k_features]
       Projected dataset

    """
    # Calculate pairwise squared Euclidean distances
    # in the MxN dimensional dataset.
    sq_dists = pdist(X, 'sqeuclidean')

    # Convert pairwise distances into a square matrix.
    mat_sq_dists = squareform(sq_dists)

    # Compute the symmetric kernel matrix.
    K = exp(-gamma * mat_sq_dists)

    # Center the kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtaining eigenpairs from the centered kernel matrix
    # numpy.eigh returns them in sorted order
    eigvals, eigvecs = eigh(K)

    # Collect the top k eigenvectors (projected samples)
    X_pc = np.column_stack((eigvecs[:, -i]
                            for i in range(1, n_components + 1)))

    # scikit-learnの結果と比べてみても, たぶんこれが正しい気がする
    # ただ結局各成分にスケール因子が入るだけなので、
    # 学習という意味ではどちらでも良いのかもしれない
    #X_pc = np.column_stack((np.sqrt(eigvals[-i]) * eigvecs[:, -i]
    #                        for i in range(1, n_components + 1)))

    # PCA固有ベクトルvをデータサンプルに直すには X v とする必要がある
    # ここで正規化された特異ベクトルの間の関係を使う。
    #   X v_i = sigma_i a_i      (sigma_i = sqrt(lambda_i))
    # よって sqrt(lambda_i) a_i で主成分方向に基底変換したデータサンプルになる。

    return X_pc
