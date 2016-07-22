import pandas as pd
import numpy as np

from sklearn import preprocessing

i = 5

train_mice = '../data/five_imps/train_mice_%d.csv' % i
all_mice = '../data/five_imps/all_mice_%d.csv' % i

train_hot = '../data/five_imps/train_mice_hot_%d.csv' % i
test_hot = '../data/five_imps/test_mice_hot_%d.csv' % i

# Read dataframe
X = pd.read_csv(train_mice)
X_all = pd.read_csv(all_mice)

y = np.ravel(X[['y']])
X.drop('y', axis=1, inplace=True)        


# Specify categorical columns
is_cat  = ['workclass', 'edu',  'married', 'occupation', 'relationship',
           'race', 'sex', 'country']

# Onehot hot encoding    
#X_enc =  pd.get_dummies(X, columns=is_cat) 
X_all_enc = pd.get_dummies(X_all, columns=is_cat) 

X_train_enc = X_all_enc.ix[0:len(X)-1, :]
X_train_enc['y'] = y
X_train_enc.to_csv(train_hot, index=False)

X_test_enc = X_all_enc.ix[(len(X)):len(X_all), :]     
X_test_enc.to_csv(test_hot, index=False)


