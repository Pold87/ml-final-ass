import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation
from sklearn import linear_model, svm
from sklearn.ensemble import VotingClassifier
from xgboost.sklearn import XGBClassifier

do_cross_validation = 1

X = pd.read_csv('../data/train_mice_hot.csv')
X_test = pd.read_csv('../data/test_mice_hot.csv')
y = np.ravel(X[['y']])

X.drop('y', axis=1, inplace=True)        

cfr = linear_model.LogisticRegression(penalty= 'l2', max_iter=10000)
cfr = VotingClassifier(estimators=[('clf1', clf1),
                                    ('clf2', clf2),
                                    ('clf3', clf3)],
                                    voting='hard',
                                    weights=[1,1,1])




if do_cross_validation:
    cv = cross_validation.KFold(len(X), n_folds=5, shuffle=True)


    #iterate through the training and test cross validation segments and
    #run the classifier on each one, aggregating the results into a list
    results = []
    for train_idx, test_idx in cv:
            X_np = X.values
            cl = cfr.fit(X_np[train_idx, :], y[train_idx])
            pred =  cl.predict(X_np[test_idx, :])
            loss = np.sum(np.fabs (y[test_idx] - pred))      
            print(1 - (loss / len(test_idx)))      

else:
    X_train_np = X.values
    X_test_np = X_test.values
    cl = cfr.fit(X_train_np, y)
    preds = cl.predict(X_test_np)

    # Make submission

    ids = np.arange(1, len(X_test) + 1)

    submission = pd.DataFrame({'Id': ids, 'Prediction': preds})

    submission.to_csv('submissions/sub1.csv', index=False)
    