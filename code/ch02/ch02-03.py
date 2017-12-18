from adaline_gd import *
from plot_decision_regions import *
import matplotlib.pyplot as plt
import numpy as np

# ### Reading-in the Iris data
import pandas as pd

df = pd.read_csv('https://archive.ics.uci.edu/ml/'
        'machine-learning-databases/iris/iris.data', header=None)
df.tail()

# select setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values


### 正規化してから勾配法つかう
# standardize features
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()


# In[20]:


#ada = AdalineGD(n_iter=15, eta=0.01)
ada = AdalineGD(n_iter=100, eta=0.01)
ada.fit(X_std, y)

plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('./adaline_2.png', dpi=300)
plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')

plt.tight_layout()
# plt.savefig('./adaline_3.png', dpi=300)
plt.show()


### 解析的に最小二乗解求められるので計算してみる
# データを正規化しているからバイアス項は最初から無視して計算
#print(X_std.shape)  # 100 x 2
R = np.dot(X_std.T, X_std)
p = np.dot(X_std.T, y)
w_opt = np.linalg.solve(R, p)
#print(w_opt)
