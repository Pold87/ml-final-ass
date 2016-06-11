# Read imputed train data
X.train <- read.csv("../data/train_mice.csv", header = T,
              colClasses=c('numeric', 'factor', 'factor',
                           'numeric', 'factor', 'factor',
                           'factor', 'factor', 'factor',
                           'numeric', 'numeric', 'numeric',
                           'factor', 'factor'),
              na.strings="NaN") # Important!!: NaN != NA

# Extract target labels
y <- X.train[, "y"]
X.train$y <- NULL
