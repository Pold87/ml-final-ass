require(mice)

# Read train data
X.train <- read.csv("../data/train.csv", header = T,
              colClasses=c('numeric', 'factor', 'factor',
                           'numeric', 'factor', 'factor',
                           'factor', 'factor', 'factor',
                           'numeric', 'numeric', 'numeric',
                           'factor', 'factor'),
              na.strings="NaN") # Important!!: NaN != NA

# Extract target labels
y <- X.train[, "rich"]
X.train$rich <- NULL

# Read test data
X.test <- read.csv("../data/test.csv", header = T,
              colClasses=c('numeric', 'factor', 'factor',
                           'numeric', 'factor', 'factor',
                           'factor', 'factor', 'factor',
                           'numeric', 'numeric', 'numeric',
                           'factor'),
              na.strings="NaN")


# Join train and test set for better data imputation
X.all <- rbind(X.train, X.test)

# Do the IMPUTATION
X.imp.mice <- mice(X.all, maxit = 5, MaxNWts = 20000, m = 5,
                   defaultMethod = c("norm.predict",
"logreg", "polyreg", "polr"),)

for (i in 1:5) {

    ## Extract imputed data set and write to CSV
    X.out <- complete(X.imp.mice, i)
    fn.all <- sprintf("../data/five_imps/all_mice_%d.csv", i)
    write.csv(X.out, fn.all, row.names = F, quote = F)

    ## Extract train set
    X.train.imp <- X.out[1:nrow(X.train), ]

    ## Add target labels
    X.train.imp <- cbind(X.train.imp, y)

    ## Write to CSV
    fn.train <- sprintf("../data/five_imps/train_mice_lin_%d.csv", i)
    write.csv(X.train.imp, fn.train, row.names = F, quote = F)

    ## Extract test set
    X.test.imp <- X.out[(nrow(X.train)+1):nrow(X.out), ]

    ## Write to CSV
    fn.test <-  sprintf("../data/five_imps/test_mice_lin_%d.csv", i)
    write.csv(X.test.imp, fn.test, row.names = F, quote = F)

}
