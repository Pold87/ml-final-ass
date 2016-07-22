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


pMiss <- function(x){sum(is.na(x))/length(x)*100}
apply(X.train,2,pMiss)
apply(X.test,2,pMiss)
apply(X.all,2,pMiss)

library(VIM)
aggr_plot <- aggr(X.all, col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE, labels=names(X.train), cex.axis=.7, gap=3, ylab=c("Histogram of missing data","Pattern"))

marginplot(X.all[,c(2, 6)])


library(MissMech)
data(agingdata)


X.train.cont <- X.train[,c(1, 4, 10, 11, 12)]
