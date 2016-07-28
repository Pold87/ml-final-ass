require(copula)
require(VineCopula)


features <- c("age", "yearsedu", "hours", "income", "loss")
X.train <- read.csv('../data/train_mice_hot.csv', header=T)
X.test <- read.csv('../data/test_mice_hot.csv', header=T)

X.train <- X.train[, c(features, "y")]
X.test <- X.test[, features]

X.train$win<- X.train$income - X.train$loss
X.test$win <- X.test$income - X.test$loss

X.test$income <- NULL
X.test$loss <- NULL
X.train$income <- NULL
X.train$loss <- NULL


## Rank data and divide by N + 1
X.train.pseudo <- pobs(as.matrix(X.train))
X.test.pseudo <- pobs(as.matrix(X.test))


## Convert to a copuladata object (as used by the Vine Copula package)
X.train.cop <- as.copuladata(X.train.pseudo)
X.test.cop <- as.copuladata(X.test.pseudo)

##samples <- 100
##X.train.sample <- X.train.cop[sample(nrow(X.train.cop), samples), ]
##X.train.sample <- as.copuladata(X.train.sample)
##pairs.copuladata(X.train.cop)

rvm <- RVineStructureSelect(X.train.cop, indeptest=TRUE)

zeros <- rep(0.3809637, nrow(X.test.cop))
ones <- rep(0.8809161, nrow(X.test.cop))

X.test.cop.0 <- as.copuladata(cbind(X.test.pseudo, zeros))
X.test.cop.1 <- as.copuladata(cbind(X.test.pseudo, ones))

targets = c()

val0 <- RVinePDF(X.test.cop.0, rvm)
val1 <- RVinePDF(X.test.cop.1, rvm)

for (i in 1:nrow(X.test.cop)) {
    if (val0[i] > val1[i]) {
        targets <- c(targets, 0)
        print("0")
    } else {
        targets <- c(targets, 1)
        print("1")
    }
}


python.sub <- read.csv("../Python/submissions/adaboost_mice_single_imputation.csv")
