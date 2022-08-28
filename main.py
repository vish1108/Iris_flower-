import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler



data_set = load_iris()
# print(load_iris().DESCR)
X = data_set.data
y = data_set.target
plt.plot(X[:, 0][y == 0]*X[:, 1][y == 0], X[:, 1][y == 0]*X[:, 2][y == 0], 'r.', label='Satosa')
plt.plot(X[:, 0][y == 1]*X[:, 1][y == 1], X[:, 1][y == 1]*X[:, 2][y == 1], 'g.', label='Versicolour')
plt.plot(X[:, 0][y == 2]*X[:, 1][y == 2], X[:, 1][y == 2]*X[:, 2][y == 2], 'b.', label='Verginica')
plt.legend()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y)
log_reg = LogisticRegression()
log_reg.fit(X_train, y_test)
print(log_reg.fit(X_train, y_test))

