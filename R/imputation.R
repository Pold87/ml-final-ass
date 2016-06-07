require(psych)
require(missForest)
require(Hmisc)
require(Amelia)
require(mice)

# Join train and test set for better data imputation

X.train <- read.csv("../data/train.csv", header = T,
              colClasses=c('numeric', 'factor', 'factor',
                           'numeric', 'factor', 'factor',
                           'factor', 'factor', 'factor',
                           'numeric', 'numeric', 'numeric',
                           'factor', 'factor'))

y <- X.train[, "rich"]

X.train$rich <- NULL

X.test <- read.csv("../data/test.csv", header = T,
              colClasses=c('numeric', 'factor', 'factor',
                           'numeric', 'factor', 'factor',
                           'factor', 'factor', 'factor',
                           'numeric', 'numeric', 'numeric',
                           'factor'))


X.all <- rbind(X.train, X.test)

X.imp <- missForest(X, verbose = TRUE, maxiter = 2)

pairs.panels(X, method = 'spearman')

X.imp.ame <- amelia(X.all, noms=c("workclass", "edu", "married", "occupation", "relationship", "race", "sex", "country"), amcheck=F, m = 1, boot.type = "none")

                                        #X.imp <- read.csv("../data/train_imp.csv", header = T)

write.amelia(X.imp.ame, file.stem = "data_set", orig.data = F)


tempData <- mice(X.all,m=1,maxit=5,meth='pmm',seed=500)
