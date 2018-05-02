# Author : Conor Lorsung
# Purpose: This system is built for my final CSCI-332 project.
#          Ideally, it will be a neural network that acts as
#          a classifier for the schizophrenia data outlined
#          in the github repository this file is hosted in.

library(neuralnet)

print("Reading training data!")
trainFileData <- read.csv('../DataFiles/SampleData.csv', header<-TRUE)
trainData <- trainFileData[, 3:11]
print("Done reading training data!")

print("Reading testing data!")
testFileData <- read.csv('../DataFiles/TestData.csv', header<-TRUE)
testData <- testFileData[, 3:11]
print("Done reading testing data!")

trainScl = as.data.frame(rnorm(ncol(trainData)+1))

testScl = as.data.frame(rnorm(ncol(testData)+1))

trainNN <- trainScl
testNN  <- testScl

print("Beginning training!")
nnet <- neuralnet(trainData$class ~ colnames(trainNN), trainNN, hidden=5, linear.output = T)
print("Done training!")
