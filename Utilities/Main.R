# Author : Conor Lorsung following David Selby's example from http://selbydavid.com/2018/01/09/neural-network/
# Purpose: This system is built for my final CSCI-332 project.
#          Ideally, it will be a neural network that acts as
#          a classifier for the schizophrenia data outlined
#          in the github repository this file is hosted in.

# Read data file
fileData <- read.csv("../DataFiles/SampleData.csv", header<-TRUE)

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

# Training the neural network
print(fileData[1,])

x <- data.matrix(fileData[,3:11])
y <- fileData$class == '1'
nnet <- train(x, y, hidden=5, iter=1e2)

# Confusion Matrix
headings <- c('SAMPLES', 'PREDICT FALSE', 'PREDICT TRUE')

## Confusion values
cv <- {
    cv.PFAF <- 0
    cv.PTAF <- 0
    cv.PFAT <- 0
    cv.PTAT <- 0

    for(i in 1:length(fileData$class)) {
        if(nnet$output[i] <= .5 && fileData$class[i] == 0) {
            cv.PFAF <- cv.PFAF + 1
        }
        if(nnet$output[i] >  .5 && fileData$class[i] == 0) {
            cv.PTAF <- cv.PTAF + 1
        }
        if(nnet$output[i] <= .5 && fileData$class[i] == 1) {
            cv.PFAT <- cv.PFAT + 1
        }
        if(nnet$output[i] >  .5 && fileData$class[i] == 1) {
            cv.PTAT <- cv.PTAT + 1
        }
    }
    list(n = length(nnet$output), FF = cv.PFAF, TF = cv.PTAF, FT = cv.PFAT, TT = cv.PTAT)
}

## Set up Confusion matrix
matrixRows   <- c("ACTUAL FALSE", "ACTUAL TRUE", "TOTALS")
predictFalse <- c(cv$FF, cv$FT, (cv$FF + cv$FT))
predictTrue  <- c(cv$TF, cv$TT, (cv$TF + cv$TT))
predictTotal <- c((cv$FF + cv$TF), (cv$FT + cv$TT), cv$n)
confMat <- data.frame("PREDICTED FALSE"=predictFalse, "PREDICTED TRUE"=predictTrue, "TOTALS"=predictTotal, row.names=matrixRows)

# File output
write.csv(cbind(VALUES=c(nnet$output), CLASS=c(fileData$class)), file="Results/Results.csv", row.names=FALSE)
write.csv(confMat, file="Results/ConfusionMatrix.csv")