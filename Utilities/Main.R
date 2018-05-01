# Author : Conor Lorsung following David Selby's example from http://selbydavid.com/2018/01/09/neural-network/
# Purpose: This system is built for my final CSCI-332 project.
#          Ideally, it will be a neural network that acts as
#          a classifier for the schizophrenia data outlined
#          in the github repository this file is hosted in.

# Get arguments passed by user
args <- commandArgs(trailingOnly<-TRUE)

# Recompue the selected patients and files?
recompute <- TRUE

if(length(args) == 0) {
    cat(sprintf("Running all functions\n"))
} else {
    if(args[1] == 't') {
        recompute <<- TRUE
    } else if(args[1] == 'f'){
        recompute <<- FALSE
    }
}

pats <- 2

# Assign number of patients to use in training set
if(length(args) == 2) {
    pats <<- args[2]
}

# Time management
if(!file.exists("../DataFiles/TestPatientList.csv") || recompute == TRUE || length(args) > 1) {
    cat(sprintf("Calling LoadData.R\n"))
    source("LoadData.R")
    getTrainSet(numPatients=pats)
    getTestSet()
}

# Read data file
fileDir  <- "../DataFiles/"
fileName <- "SampleData"
cat(sprintf("Reading %s\n", fileName))
fileData <- read.csv(paste(fileDir, fileName, ".csv", sep=""), header<-TRUE)
cat(sprintf("Done reading %s\n", fileName))

# Read test file
cat(sprintf("Reading TestData\n"))
testData <- read.csv("../DataFiles/TestData.csv", header<-TRUE)
cat(sprintf("Done reading TestData\n"))

# How many iterations?
global.iter = 1e2
cat(sprintf("Training with %d iterations\n", global.iter))

# How many hidden layers?
global.layer = 10
cat(sprintf("Using %d hidden layers\n", global.layer))

# File output options
outDir  <- paste("../Results/", fileName, "/", "I", toString(global.iter), ".L", toString(global.layer), "/", sep="")
dir.create(outDir, showWarnings=FALSE)
resFile <- paste(outDir, fileName, toString(global.iter), "Results.csv", sep="")
matFile <- paste(outDir, fileName, toString(global.iter), "CMatrix.csv", sep="")
txtFile <- paste(outDir, "LastConfiguration.txt", sep="")
patFile <- paste(outDir, "TrainingSubjectIDs.csv")

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

d  <- NULL
w1 <- NULL
w2 <- NULL
ff <- NULL
bp <- NULL

# Training function
train <- function(x, y, hidden=5, RATE = 1e-2, iter = 1e4) {
    cat(sprintf("Training...\n"))
    d  <<- ncol(x)+1
    w1 <<- matrix(rnorm(d * hidden), d, hidden)
    w2 <<- as.matrix(rnorm(hidden + 1))
    for(i in 1:iter) {
        ff <<- feedForward(x, w1, w2)
        bp <<- backProp(x, y, y_hat = ff$output, w1, w2, h = ff$h, RATE = RATE)
        w1 <<- bp$w1
        w2 <<- bp$w2
    }
    list(output = ff$output, w1 = w1, w2 = w2)
}

#Testing function
testNet <- function(testData) {
    cat(sprintf("Testing...\n"))
    d  <- d
    w1 <- bp$w1
    w2 <- bp$w2
    #for(i in 1:nrow(testData)) {
        ff <<- feedForward(testData, w1, w2)
    #}
    list(output = ff$output)
}

# Training the neural network
x <- data.matrix(fileData[,3:11])
y <- fileData$class == '1'
runTime <- system.time({
    nnet <- train(x, y, hidden=global.layer, iter=global.iter)
})['elapsed']
cat(sprintf("Done training!\n"))

nnetTest <- testNet(testData=data.matrix(testData[,3:11]))
cat(sprintf("Done testing!\n"))

# Confusion Matrix
headings <- c('SAMPLES', 'PREDICT FALSE', 'PREDICT TRUE')

## Confusion values
cv <- {
    cv.PFAF <- 0
    cv.PTAF <- 0
    cv.PFAT <- 0
    cv.PTAT <- 0

    for(i in 1:length(fileData$class)) {
        if(nnet$output[i] <= .5 && fileData$class[i] == 0) {  # True negatives
            cv.PFAF <- cv.PFAF + 1
        }
        if(nnet$output[i] >  .5 && fileData$class[i] == 0) {  # False positives
            cv.PTAF <- cv.PTAF + 1
        }
        if(nnet$output[i] <= .5 && fileData$class[i] == 1) {  # False negatives
            cv.PFAT <- cv.PFAT + 1
        }
        if(nnet$output[i] >  .5 && fileData$class[i] == 1) {  # True positives
            cv.PTAT <- cv.PTAT + 1
        }
    }
    list(n = length(nnet$output), FF = cv.PFAF, TF = cv.PTAF, FT = cv.PFAT, TT = cv.PTAT)
}

## Set up Confusion matrix
matrixRows   <- c("ACTUAL.FALSE", "ACTUAL.TRUE", "TOTALS")
predictFalse <- c(cv$FF, cv$FT, (cv$FF + cv$FT))
predictTrue  <- c(cv$TF, cv$TT, (cv$TF + cv$TT))
predictTotal <- c((cv$FF + cv$TF), (cv$FT + cv$TT), cv$n)
confMat <- data.frame("PREDICTED FALSE"=predictFalse, "PREDICTED TRUE"=predictTrue, "TOTALS"=predictTotal, row.names=matrixRows)

# Configuration print file
sinfo <- Sys.info()
rinfo <- R.version
str <- sprintf(
"---PERFORMANCE---
RECORDS : %d
RUNTIME : %f3s
CORRECT : %f%%

---NN INFO---
NN ITERATIONS   : %d
NN HIDDEN LAYERS: %d

---SYSINFO---
SYSNAME : %s
RELEASE : %s
VERSION : %s
MACHINE : %s

---ENVINFO---
PLATFORM: %s
OS      : %s
VERSION : %s
V-TITLE : %s
",
cv$n, runTime, mean((nnet$output > .5) == y),                                 # Performance output
global.iter, global.layer,                                                    # Neural Net output
sinfo['sysname'], sinfo['release'], sinfo['version'], sinfo['machine'],       # System output
rinfo['platform'], rinfo['os'], rinfo['version.string'], rinfo['nickname'])   # R Environment output

# Print Configuration print file
cat(str)

# Write to the output files
write.csv(cbind(VALUES=c(nnet$output), CLASS=c(fileData$class)), file=resFile, row.names=FALSE)
cat(sprintf("Done writing result file\n"))
if(recompute) {
    write.csv(getPatSet(), file=patFile)
    cat(sprintf("Done writing patient file\n"))
}
write.csv(confMat, file=matFile)
cat(sprintf("Done writing confusion matrix for training data\n"))
write(str, file=txtFile)
cat(sprintf("Done writing configuration file\n"))
cat(sprintf("COMPLETE\n"))
