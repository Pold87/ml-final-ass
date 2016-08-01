require(psych)

X.train <- read.csv("../data/train.csv", header = T,
              colClasses=c('numeric', 'factor', 'factor',
                           'numeric', 'factor', 'factor',
                           'factor', 'factor', 'factor',
                           'numeric', 'numeric', 'numeric',
                           'factor', 'factor'),
              na.strings="NaN") # Important!!: NaN != NA

##y <- X.train[, "rich"]

##X.train$rich <- NULL

X.test <- read.csv("../data/test.csv", header = T,
              colClasses=c('numeric', 'factor', 'factor',
                           'numeric', 'factor', 'factor',
                           'factor', 'factor', 'factor',
                           'numeric', 'numeric', 'numeric',
                           'factor'),
              na.strings="NaN")

sub1 <- read.csv("../Python/submissions/sub_multi_self.csv")
