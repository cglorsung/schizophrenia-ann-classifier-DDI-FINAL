# Author : Conor Lorsung following David Selby's example from http://selbydavid.com/2018/01/09/neural-network/
# Purpose: This system is built for my final CSCI-332 project.
#          Ideally, it will be a neural network that acts as
#          a classifier for the schizophrenia data outlined
#          in the github repository this file is hosted in.

# Read data file
fileData <- read.csv("../DataFiles/ERPdata.csv", header<-TRUE)

# Sigmoid function (logistic)
sig <- function(x) {
    return (1 / (1 + exp(-x)))
}

# Feed forward
feedForward <- function(x, w1, w2) {
    z1 <- cbind(1, x) %*% w1
    h  <- sig(z1)
    z2 <- cbind(1, h) %*% w2
    list(output = sig(z2), h = h)
}

# Backpropagation
backProp <- function(x, y, y_hat, w1, w2, h, RATE) {
    dw2 <- t(cbind(1, h)) %*% (y_hat - y)
    dh  <- (y_hat - y) %*% t(w2[-1, , drop = FALSE])
    dw1 <- t(cbind(1, x)) %*% (h * (1 - h) * dh)

    w1 <- w1 - RATE * dw1
    w2 <- w2 - RATE * dw2

    list(w1 = w1, w2 = w2)
}

# Training function
train <- function(x, y, hidden=5, RATE = 1e-2, iter = 1e4) {
    d  <- ncol(x)+1
    w1 <- matrix(rnorm(d * hidden), d, hidden)
    w2 <- as.matrix(rnorm(hidden + 1))
    for(i in 1:iter) {
        ff <- feedForward(x, w1, w2)
        bp <- backProp(x, y, y_hat = ff$output, w1, w2, h = ff$h, RATE = RATE)
        w1 <- bp$w1; w2 <- bp$w2
    }
    list(output = ff$output, w1 = w1, w2 = w2)
}

# Training example
print(fileData[1,])
last <- 0
for(i in 1:nrow(fileData)) {
    if(fileData[i,]$class == 1) {
        print(fileData[i,]$subject)
    }
}