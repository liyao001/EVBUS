from EVBUS import EVBUS
from sklearn.datasets import load_boston
import sklearn.model_selection as xval

boston = load_boston()
Y = boston.data[:, 12]
X = boston.data[:, 0:12]

bos_X_train, bos_X_test, bos_y_train, bos_y_test = xval.train_test_split(X, Y, test_size=0.3)
evbus = EVBUS.varU(bos_X_train, bos_y_train, bos_X_test)

v = evbus.calculate_variance()
print(v)
