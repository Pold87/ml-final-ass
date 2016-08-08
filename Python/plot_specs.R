X <- read.csv('specs.csv')

pdf("theta.pdf")
par(cex.lab=1.7, cex.axis = 1.5)
plot(X$theta, X$accuracy, xlab=expression(theta), ylab="",pch=1, col=1, type="b", ylim=c(0, 1),)
lines(X$theta,X$sensitivity, pch=2, col=2,type='b')
lines(X$theta,X$specificity, pch=3, col=3, type='b')
legend(9, 0.3,
       legend=c("Accuracy", "Sensitivity", "Specificity"),
       pch=1:3,
       col=1:3,
       lty=1,
       cex= 1.5)
dev.off()
