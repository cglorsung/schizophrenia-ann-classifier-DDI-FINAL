# Author : Conor Lorsung
# Purpose: This system is built for my final CSCI-332 project.
#          Ideally, it will be a neural network that acts as
#          a classifier for the schizophrenia data outlined
#          in the github repository this file is hosted in.

# Supervision
supervise <- TRUE

# Number of iterations
iterations <- 100

# Read data file
fileData <- read.csv("../DataFiles/SampleSet.csv", header<-FALSE)

# Scale file data to interval [-1, 1]
s.fileData <- t(apply(fileData, 1, function(x)2*((x-min(x))/(max(x)-min(x)))-1))

maxVals <- apply(fileData[,0:length(fileData)], 2, max)
minVals <- apply(fileData[,0:length(fileData)], 2, min)

# s.Data <- (as.data.frame(scale(fileData[,0:length(fileData)], center=minVals, scale=maxVals-minVals))) * 2 - 1

# Put data into matrix datatype
s.fileData <- matrix(s.fileData, nrow(s.fileData), ncol(s.fileData))

# print(s.fileData)
# Dimension values
sfdCols <- ncol(s.fileData)
sfdRows <- nrow(s.fileData)
sfdVals <- sfdCols * sfdRows

# Sigmoid function
sigmoid <- function(x, derive=FALSE) {
    if(derive) {
        return (x*(1-x))
    } else {
        return (1/(1+exp(-x)))
    }
}

# Build output reference array
if(supervise) {
    falArr <- matrix(c(0), sfdRows/2, 1)
    truArr <- matrix(c(1), sfdRows/2, 1)
    outArr <- rbind(falArr, truArr)
} else {
    outArr <- matrix(c(0), sfdRows, 1)
}

# Synapse 0
syn0 <- matrix(runif(sfdVals, -1.0, 1.0), sfdCols, sfdRows)
syn1 <- matrix(runif(length(outArr), -1.0, 1.0), nrow(outArr), ncol(outArr))
syn2 <- matrix(runif(length(syn1), -1.0, 1.0), ncol(syn1), nrow(syn1))
syn3 <- matrix(runif(length(outArr), -1.0, 1.0), nrow(outArr), ncol(outArr))

# Debugging print
# cat(sprintf("MAX SYN0: %s | MIN SYN0: %s\n", max(syn0), min(syn0)))
# cat(sprintf("MAX SYN1: %s | MIN SYN1: %s\n", max(syn1), min(syn1)))
# cat(sprintf("MAX SYN2: %s | MIN SYN2: %s\n", max(syn2), min(syn2)))
# print(sigmoid(s.fileData %*% syn0))
cat(sprintf("SYN3 DIMS: %s x %s \nOUTARR DIMS: %s x %s\n", nrow(syn3), ncol(syn3), nrow(outArr), ncol(outArr)))

for(i in 0:iterations) {
    # Evaluate layer 0
    lay0 <- s.fileData
    # Evaluate layer 1
    lay1 <- sigmoid(lay0 %*% syn0)
    # Evaluate layer 2
    lay2 <- sigmoid(lay1 %*% syn1)
    # Evaluate layer 3
    lay3 <- sigmoid(lay2 %*% syn2)
    # Evaluate layer 4
    lay4 <- sigmoid(lay3 %*% syn3)

    # Calculate errors and deltas
    error4 <- outArr - lay4
    delta4 <- error4 * sigmoid(lay4, derive=TRUE)

    error3 <- (delta4 %*% t(syn3))
    delta3 <- error3 * sigmoid(lay3, derive=TRUE)

    error2 <- (delta3 %*% t(syn2))
    delta2 <- error2 * sigmoid(lay2, derive=TRUE)

    error1 <- (delta2 %*% t(syn1))
    delta1 <- error1 * sigmoid(lay1, derive=TRUE)

    # Propagate errors back
    syn3 <- syn3 + (t(lay3) %*% delta4)
    syn2 <- syn2 + (t(lay2) %*% delta3)
    syn1 <- syn1 + (t(lay1) %*% delta2)
    syn0 <- syn0 + (t(lay0) %*% delta1)
}

# Print out final values
# cat(sprintf("LAY1: %s\n", lay1))
# NEED TO WORK ON LAYERING/ERROR PROPAGATION. CURRENTLY CANNOT TELL ONE GROUP FROM THE OTHER
print(lay4)
