library(caret)
library(randomForest)
library(e1071)
library(gmum.r)

m <- 5
X.all <- list()
X.train <- list()
X.test <- list()
y <- list()

## Read multiple datasets
for (i in 1:5) {
    ## Read all
    fn.all <- sprintf("../data/five_imps/all_mice_%d.csv", i)
    X.all[[i]] <-  read.csv(fn.all, header = T,
                            colClasses=c('numeric', 'factor', 'factor',
                                         'numeric', 'factor', 'factor',
                                         'factor', 'factor', 'factor',
                                         'numeric', 'numeric', 'numeric',
                                         'factor'))
    ## Read train
    fn.train <- sprintf("../data/five_imps/train_mice_%d.csv", i)
    X.train.tmp <- read.csv(fn.train, header = T,
                            colClasses=c('numeric', 'factor', 'factor',
                                         'numeric', 'factor', 'factor',
                                         'factor', 'factor', 'factor',
                                         'numeric', 'numeric', 'numeric',
                                         'factor', 'factor'))
    #y <- X.train.tmp[, "y"]
    #X.train.tmp$y <- NULL
    X.train[[i]] <- X.train.tmp
    ## Read test set
    fn.test <-  sprintf("../data/five_imps/test_mice_%d.csv", i)
    X.test[[i]] <- read.csv(fn.test, header = T,
                            colClasses=c('numeric', 'factor', 'factor',
                                         'numeric', 'factor', 'factor',
                                         'factor', 'factor', 'factor',
                                         'numeric', 'numeric', 'numeric',
                                         'factor'))
}


## Number of folds
k = 10
## Generate cross-validation index
train.idx.all <- createFolds(1:nrow(X.train[[1]]),
                        k = k,
                        returnTrain = TRUE)


## Split data into train and test
for (i in 1:k) {
    ## For all imputations
    for (j in 1:m) {
        ## Get current imputation
        X.cur1 <- X.train[[j]]
        ## Split into train and test
        X.train1.cv <- X.cur1[train.idx.all[[i]], ]
        X.test1.cv <- X.cur1[-train.idx.all[[i]], ]
        actual <- X.test1.cv$y
        X.test1.cv$y <- NULL
        preds <- matrix(0, nrow(X.test1.cv), m)
        ## For all train sets
        for (k in 1:m) {
            ## Get current imputation
            X.cur2 <- X.train[[k]]
            ## Split into train and test
            X.train2.cv <- X.cur2[train.idx.all[[i]], ]
            X.test2.cv <- X.cur2[-train.idx.all[[i]], ]
            ##
            ## Add non-labeled feature vector
            ##
            ## Split labels from datasets
            train.y <- X.train2.cv$y
            train.y <- as.numeric(train.y)
            X.train2.cv$y <- NULL
            ##
            X.test2.cv$y <- NULL
            test.y <- rep(0, nrow(X.test[[1]]))
            ##
            X.trans.cv <- rbind(X.train2.cv, X.test[[1]])
            trans.y <- as.factor(c(train.y, test.y))
            ##
            ## Fit model and predict for test set
            ##mdl <- glm(y ~ ., data = X.train2.cv, family='binomial')
            ## Normal SVM:
            ##mdl <- svm(y ~ ., data = X.train2.cv, probability=TRUE)
            ## Tranductive:
            mdl <- SVM(x=X.trans.cv, y=trans.y,
                        transductive.learning=TRUE,
                       kernel="rbf", core="svmlight")
            ##svm.transduction.pred <- predict(svm.transduction, test$x)
            ##mdl <- randomForest(y ~ ., data = X.train2.cv)
        ## preds[[j]] <- predict(mdl, X.test.cv, type = 'response')
            ## for log
            ##preds.cur <- predict(mdl, X.test1.cv, type = 'response')
            ##preds.cur <- predict(mdl, X.test1.cv, probability=TRUE)
            preds.cur <- predict(mdl, X.test1.cv, probability=TRUE)
            ##print(preds.cur)
            print(svm.accuracy(preds.cur, actual))
            ##preds[,k] <- attr(preds.cur, "prob")[, 2]
            #print(preds)
            ## TODO: move one step deeper
        }
        ##preds.combined <- rowSums(preds) / m
        ##errs <- sum(abs((preds.combined > 0.5) - (as.numeric(actual) - 1)))
        ##print(1 - (errs / length(X.test1.cv$y)))
    }
}
