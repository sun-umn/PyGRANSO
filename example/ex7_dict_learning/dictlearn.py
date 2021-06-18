import numpy as np
from sklearn.datasets import make_sparse_coded_signal
from sklearn.decomposition import DictionaryLearning

X, dictionary, code = make_sparse_coded_signal(
 n_samples=100, n_components=15, n_features=20, n_nonzero_coefs=10,
  random_state=42,
 )

dict_learner = DictionaryLearning(
   n_components=15, transform_algorithm='lasso_lars', random_state=42, verbose=2
 )

X_transformed = dict_learner.fit_transform(X)

X_hat = X_transformed @ dict_learner.components_ # U * V

ans = np.mean(np.sum((X_hat - X) ** 2, axis=1) / np.sum(X ** 2, axis=1))

print("ans = " , ans)