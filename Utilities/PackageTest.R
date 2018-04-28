# Author : Conor Lorsung
# Purpose: This system is built for my final CSCI-332 project.
#          Ideally, it will be a neural network that acts as
#          a classifier for the schizophrenia data outlined
#          in the github repository this file is hosted in.

library(neuralnet)
library(caTools)
set.seed(10)

# Read data file
fileData <- read.csv("../DataFiles/ClassSampleSet.csv", header<-FALSE)

# Scale file data to interval [-1, 1]
# s.fileData <- apply(fileData, 1, function(x)2*((x-min(x))/(max(x)-min(x)))-1)

maxVals <- apply(fileData[,2:length(fileData)-1], 2, max)
minVals <- apply(fileData[,2:length(fileData)-1], 2, min)

s.Data <- (as.data.frame(scale(fileData[,2:length(fileData)-1], center=minVals, scale=maxVals-minVals))) * 2 - 1

split <- sample.split(s.Data$V18, SplitRatio=.60)
train <- subset(s.Data, split==TRUE)
test  <- subset(s.Data, split==FALSE)

features <- names(s.Data)

f <- paste(features, collapse=' | ')
f <- paste('V1 ~', f)

f <- as.formula(f)

nn <- neuralnet(f, train, hidden=c(10,10,10), linear.output=FALSE)

predicted <- compute(nn, test[2:10])