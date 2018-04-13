# Author : Conor Lorsung
# Purpose: This system is built for my final CSCI-332 project.
#          Ideally, it will be a neural network that acts as
#          a classifier for the schizophrenia data outlined
#          in the github repository this file is hosted in.

# Supervision
supervise = TRUE

# Number of iterations
iterations = 10000

# Read data file
fileData <- read.csv("../DataFiles/SampleSet.csv", header=FALSE)

# Scale file data to interval [-1, 1]
s.fileData <- t(apply(fileData, 1, function(x)2*((x-min(x))/(max(x)-min(x)))-1))

# Put data into matrix datatype
s.fileData <- matrix(s.fileData, nrow(s.fileData), ncol(s.fileData))

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
    outArr <- matrix(c(1), sfdRows, 1)
} else {
    outArr <- matrix(c(0), sfdRows, 1)
}

# Synapse 0
syn0 <- matrix(runif(sfdVals, -1.0, 1.0), sfdCols, 1)

# Debugging print
# print(sigmoid(s.fileData %*% syn0))
# cat(sprintf("SYN0 DIMS: %s x %s \nOUTARR DIMS: %s x %s\n", nrow(syn0), ncol(syn0), nrow(outArr), ncol(outArr)))

for(i in 0:iterations) {
    # Evaluate layer 0
    lay0 = s.fileData
    # Evaluate layer 1
    lay1 = sigmoid((lay0 %*% syn0))

    # Calculate errors
    error = outArr - lay1

    # Calculate error deltas
    delta = error * sigmoid(lay1, TRUE)

    # Propagate errors back
    syn0 = syn0 + (t(lay0) %*% delta)
}

# Print out final values
cat(sprintf("LAY1: %s\n", lay1))
