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
                           'factor', 'factor'),
              na.strings="NaN") # Important!!: NaN != NA

y <- X.train[, "rich"]

X.train$rich <- NULL

X.test <- read.csv("../data/test.csv", header = T,
              colClasses=c('numeric', 'factor', 'factor',
                           'numeric', 'factor', 'factor',
                           'factor', 'factor', 'factor',
                           'numeric', 'numeric', 'numeric',
                           'factor'),
              na.strings="NaN")


X.all <- rbind(X.train, X.test)

#X.imp <- missForest(X.all, verbose = TRUE)

#pairs.panels(X, method = 'spearman')

#X.imp.ame <- amelia(X.all, noms=c("workclass", "edu", "married", "occupation", "relationship", "race", "sex", "country"), amcheck=F, m = 1, boot.type = "none")

##X.imp <- read.csv("../data/train_imp.csv", header = T)

##write.amelia(X.imp.ame, file.stem = "data_set", orig.data = F)

X.imp.mice <- mice(X.all, maxit = 1, MaxNWts = 20000, m = 1)


## Extract imputed data set
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



