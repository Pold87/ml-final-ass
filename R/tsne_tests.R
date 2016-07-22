require(tsne)
require(Rtsne)
require(ggplot2)


X <- read.csv('../data/train.csv')
#y <- X$y
#X$y <- NULL 

X.unique <- unique(X)

t <- Rtsne(X.unique, initial_dims = ncol(X), max_iter=3000, theta=0.2, perplexity=50)

plot(t$Y, col=y+1)


msk.neg <- which(X$y == 0)
msk.pos <- which(X$y == 1)
a <- density(X$yearsedu[msk.neg], width=1.5)
b <- density(X$yearsedu[msk.pos], width=1.5)
plot(b, col='green');
lines(a, col = 'red')
