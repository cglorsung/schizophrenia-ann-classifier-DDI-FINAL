# Author : Conor Lorsung
# Purpose: This system is built for my final CSCI-332 project.
#          Ideally, it will be a neural network that acts as
#          a classifier for the schizophrenia data outlined
#          in the github repository this file is hosted in.

library(caret)
library(doParallel)
library(neuralnet)

# Direct errors to a log file
errFile <- file('error_log.RLOG', open='wt')
sink(errFile, type='message')

# Make the parallel cluster
print("Making cluster")
cl <- makeCluster(detectCores())
registerDoParallel(cl)
print("Done making cluster")

# print("Reading training data!")
# trainFileData <- read.csv('../DataFiles/SampleData.csv', header<-TRUE)
# trainData <- trainFileData[, 3:11]
# print("Done reading training data!")

# print("Reading testing data!")
# testFileData <- read.csv('../DataFiles/TestData.csv', header<-TRUE)
# testData <- testFileData[, 3:11]
# print("Done reading testing data!")

# print(colnames(testFileData)[3:11])
# trainNN <- trainData
# testNN  <- testData

print("Reading data")
trainData <- read.csv('../DataFiles/ERPdata.csv', header <- TRUE)
print("Done reading data")

y <- as.factor(make.names(trainData$class))

trainData$class <- y

# names <- setdiff(names(trainData), toRemove)

# data <- trainData[, names]

inputData <- createDataPartition(trainData$class, p <- .60)[[1]]

train <- trainData[inputData, ]
test  <- trainData[-inputData, ]

print("Beginning training!")
num <- trainControl(method = 'cv', number = 10, classProbs = TRUE, verboseIter = FALSE, summaryFunction = twoClassSummary)
fit <- train(class ~ Fz + FCz + Cz + FC3 + FC4 + C3 + C4 + CP3 + CP4, data = train, method = 'nnet', trControl = num, tuneGrid = expand.grid(size=c(10), decay=c(0.1)), linout = 0)
print("Done training!")

print("Predict train results!")
trainRslt <- predict(fit, newdata=train)
trainConf <- confusionMatrix(trainRslt, train$class)
print("Done!")

print("Predict test results!")
testRslt  <- predict(fit, newdata=test)
testConf  <- confusionMatrix(testRslt, test$class)
print("Done!")

probabilities <- predict(fit, newdata=test, type='prob')

out <- data.frame(SUBJECT=test$subject)
out <- cbind(out, CLASS=probs$X1)

write.csv(trainConf, file="../Results/caretOutput/TrainingConfMat.csv", row.names=FALSE)
write.csv(testConf, file="../Results/caretOutput/TestingConfMat.csv", row.names=FALSE)
write.csv(out, file="../Results/caretOutput/PackageOut.csv", row.names=FALSE)

#nnet <- neuralnet(trainFileData$class ~ trainNN$Fz + trainNN$FCz + trainNN$Cz + trainNN$FC3 + trainNN$FC4 + trainNN$C3 + trainNN$C4 + trainNN$CP3 + trainNN$CP4, trainNN, hidden=5, linear.output = T)
stopCluster(cl)
close(errFile)
print("Done!")
