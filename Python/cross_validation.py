import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation
from sklearn import linear_model, svm
from xgboost.sklearn import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from frameworks.SelfLearning import *
from frameworks.CPLELearning import CPLELearningModel
#from sklearn.ensemble import VotingClassifier

do_cross_validation = 1

for i in range(1, 6):
    
    X = pd.read_csv('../data/five_imps/train_mice_hot_%d.csv' % i)
    X_test = pd.read_csv('../data/five_imps/test_mice_hot_%d.csv' % i)
    y = np.ravel(X[['y']])

    X.drop('y', axis=1, inplace=True)

    #scaler = RobustScaler()
    #X = scaler.fit_transform(X)
    #X_test = scaler.transform(X_test)

        

    #cfr = linear_model.Ridge(fit_intercept=True, normalize=False, alpha=3000)
    #cfr = linear_model.LogisticRegression(penalty= 'l2', max_iter=10000)
    # cfr = svm.SVC(probability=True) 
    #cfr     =       GradientBoostingClassifier(max_depth=7, subsample=0.8, n_estimators=    500,
    #                                          learning_rate =       0.05 )
    # Create and fit an AdaBoosted decision tree
    #cfr = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),

    #                         algorithm="SAMME",
    #                         n_estimators=500)

    #cfr = RandomForestClassifier(n_estimators=1000)

    #cfr = VotingClassifier(estimators=[('clf1', clf1),
    #                                    ('clf2', clf2),
    #                                    ('clf3', clf3)],
    #                                    voting='hard',
    #                                    weights=[1,1,1])
    #cfr = XGBClassifier(seed=27)
    cfr = KNeighborsClassifier(n_neighbors=30, weights='distance')

    #cfr = SelfLearningModel(cfr)
    cfr = CPLELearningModel(cfr)

    if do_cross_validation:
        cv = cross_validation.KFold(len(X), n_folds=5, shuffle=True)

        #iterate through the training and test cross validation segments and
        #run the classifier on each one, aggregating the results into a list
        results = []
        X_test_np = X_test.values
        for train_idx, test_idx in cv:
            X_np = X.values
            X_all = np.concatenate((X_np[train_idx, :], X_test_np), axis = 0)
            y_unl = -1 * np.ones(len(X_test_np))
            y_all = np.concatenate((y[train_idx], y_unl), axis=0)
            cl = cfr.fit(X_all, y_all)
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

        submission.to_csv('submissions/sub_mult_%d.csv' % i, index=False)
    
