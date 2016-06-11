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
X.imp.mice <- mice(X.all, maxit = 1, MaxNWts = 20000, m = 1)



## Extract imputed data set and write to CSV
X.out <- complete(X.imp.mice, 1)
write.csv(X.out, "../data/all_mice.csv", row.names = F, quote = F)

## Extract train set
X.train.imp <- X.out[1:nrow(X.train), ]

## Add target labels
X.train.imp <- cbind(X.train.imp, y)

## Write to CSV
write.csv(X.train.imp, "../data/train_mice.csv", row.names = F, quote = F)

# Extract test set
X.test.imp <- X.out[(nrow(X.train)+1):nrow(X.out), ]

## Write to CSV
write.csv(X.test.imp, "../data/test_mice.csv", row.names = F, quote = F)

