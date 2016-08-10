from __future__ import division

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation
from sklearn import linear_model, svm
from xgboost.sklearn import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.semi_supervised import LabelSpreading, LabelPropagation

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from frameworks.SelfLearning import *
from frameworks.CPLELearning import CPLELearningModel
from sklearn.linear_model.stochastic_gradient import SGDClassifier
#from sklearn.ensemble import VotingClassifier
from methods.scikitWQDA import WQDA
from methods import scikitTSVM
from sklearn.metrics import confusion_matrix

use_scaling = 0

do_cross_validation = 1

use_transductive = 0

all_preds = [None] * 5

write_specs = 1

#cfr = SGDClassifier(loss='log')

#cfr = WQDA()

#cfr = linear_model.Ridge(fit_intercept=True, normalize=False, alpha=3000)
#cfr = linear_model.LogisticRegression(penalty= 'l2', max_iter=10000)

#cfr     =       GradientBoostingClassifier(max_depth=7, subsample=0.8, n_estimators=    500,
#                                          learning_rate =       0.05 )
# Create and fit an AdaBoosted decision tree

#cfr = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=100)

cfr = RandomForestClassifier(n_estimators=100)

#cfr = VotingClassifier(estimators=[('clf1', clf1),
#                                    ('clf2', clf2),
#                                    ('clf3', clf3)],
#                                    voting='hard',
#                                    weights=[1,1,1])
#cfr = XGBClassifier(seed=42)
#cfr = KNeighborsClassifier(n_neighbors=30, weights='distance')

#cfr = svm.SVC() 

if do_cross_validation:
    X = pd.read_csv('../data/five_imps/train_mice_hot_%d.csv' % 1)
    X.drop('y', axis=1, inplace=True)    
    # Transform to numpy
    X = X.values
    # Get cross validation indeces
    cv = cross_validation.KFold(len(X), n_folds=5, shuffle=True)

for i in range(1, 6):
    
    X = pd.read_csv('../data/five_imps/train_mice_hot_%d.csv' % i)
    y = np.ravel(X[['y']])
    X.drop('y', axis=1, inplace=True)    
    # Transform to numpy
    X = X.values

    if use_scaling:
        scaler = StandardScaler()
        #scaler = RobustScaler()
        #scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

    if use_transductive:
        #cfr = CPLELearningModel(cfr, verbose=2)
        cfr = scikitTSVM.SKTSVM(kernel="rbf")
        #cfr = SelfLearningModel(cfr)

    if do_cross_validation:

        #iterate through the training and test cross validation segments and
        #run the classifier on each one, aggregating the results into a list
        results = []
        X_cv_train = X

        for j in range(1, 6):
            X_cv_test = pd.read_csv('../data/five_imps/train_mice_hot_%d.csv' % j)
            X_cv_test.drop('y', axis=1, inplace=True)    
            # Transform to numpy
            X_cv_test = X_cv_test.values

            if use_scaling:
                X_cv_test = scaler.transform(X_cv_test)
            k = 0
            for train_idx, test_idx in cv:

                if use_transductive:
                    X_all = np.concatenate((X_cv_train[train_idx, :], X_cv_test[test_idx, :]), axis = 0)
                    y_unl = -1 * np.ones(len(X_cv_test[test_idx, :]))
                    y_all = np.concatenate((y[train_idx], y_unl), axis=0)

                    #lpm = LabelPropagation()
                    #lp = lpm.fit(X_all, y_all)
                    #y_semi = lp.predict(X_all)
                    #print y_semi
                    # For CPLR:
                    cl = cfr.fit(X_all, y_all.astype(int))
                    #cl = cfr.fit(X_all, y_semi)
                else:
                    cl = cfr.fit(X_cv_train[train_idx, :], y[train_idx])
                    pred =  cl.predict(X_cv_test[test_idx, :])
                    print pred

                    if (i == 1) and (j == 1):
                        all_preds[k] = pred
                    else:
                        all_preds[k] += pred

                    print pd.crosstab(y[test_idx], pred)
                    print 1 - (np.sum(np.fabs(y[test_idx] - pred)) / len(pred))
                k += 1
                
    else:

        for j in range(1, 6):
            X_test = pd.read_csv('../data/five_imps/test_mice_hot_%d.csv' % j)
            X_test = X_test.values
            if use_scaling:
                X_test = scaler.transform(X_test)
            X_train_np = X
            X_test_np = X_test
            X_all = np.concatenate((X_train_np, X_test_np), axis = 0)
            y_unl = -1 * np.ones(len(X_test_np))
            y_all = np.concatenate((y, y_unl), axis=0)
                
            if use_transductive:
                cl = cfr.fit(X_all, y_all)
            else:
                cl = cfr.fit(X_train_np, y)            

            preds = cl.predict(X_test_np)

            # Make submission

            ids = np.arange(1, len(X_test) + 1)

            submission = pd.DataFrame({'Id': ids, 'Prediction': preds})
                
            k = (i - 1) * 5 + j
            print k
            submission.to_csv('submissions/sub_xgb_%d.csv' % k, index=False)

theta = -1

if write_specs:
    with open('specs.csv', 'w') as fp:
        fp.write("theta,sensitivity,specificity,accuracy\n")

    for theta in range(1, 26):
        k = 0
        for train_idx, test_idx in cv:
            print all_preds[k]
            s_sub_again = np.array(all_preds[k] >= theta)
            s_sub_again = s_sub_again.astype(int)

            y_actual = pd.Series(y[test_idx], name='Actual')
            y_pred = pd.Series(s_sub_again, name='Predicted')

            if k == 0:
                df_confusion = pd.crosstab(y_actual, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
            else:
                df_confusion += pd.crosstab(y_actual, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)

            #print df_confusion
            #loss = np.sum(np.fabs (y[test_idx] - s_sub_again))      
            #print "Final accuracy", 1 - (loss / len(test_idx))
            k += 1
        print df_confusion

        # Calculate sensitivity, specificity, and accuracy
        sensitivity = df_confusion.iloc[1, 1] / df_confusion.iloc[1, 2]
        specificity = df_confusion.iloc[0, 0] / df_confusion.iloc[0, 2]
        accuracy = (df_confusion.iloc[0, 0] + df_confusion.iloc[1, 1]) / df_confusion.iloc[2, 2]

        print theta
        print sensitivity
        print specificity
        print accuracy

        with open('specs.csv', 'a') as fp:
            fp.write('%d,%f,%f,%f\n' % (theta, sensitivity, specificity, accuracy))
