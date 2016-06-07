import pandas as pd
import numpy as np

from sklearn import preprocessing

X = pd.read_csv('../data/train.csv')

#is_cat = np.bool([0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1])

enc = preprocessing.OneHotEncoder(categorical_features=is_cat)
#enc = preprocessing.OneHotEncoder()
X_end = enc.fit_transform(X)

