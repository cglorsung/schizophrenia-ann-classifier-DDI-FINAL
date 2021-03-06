# Author : Conor Lorsung
# Purpose: This system is built for my final CSCI-332 project.
#          Ideally, it will be a neural network that acts as
#          a classifier for the schizophrenia data outlined
#          in the github repository this file is hosted in.

library(caret)
library(doParallel)
library(neuralnet)


# Reset all extra sink instances
sink.reset <- function() {
    for(i in seq_len(sink.number())) {
        sink(NULL)
    }
}

# Direct errors to a log file
errFile <- file('error_log.RLOG', open='wt')
sink(errFile, type='message')

# Method?
m <- "mlpML"

# Number of hidden layers?
hiddenNum <- 35
print(paste("Working with ", hiddenNum, " nodes in first hidden layer!"))

# Decay value
decayNum <- 1e-2
print(paste("Working with ", decayNum, " decay!"))

# Output directory
outDir <- paste('../Results/caretOutput/',m, hiddenNum, decayNum, '/', sep="")
dir.create(outDir, showWarnings=FALSE)
print(outDir)
# Make the parallel cluster
print("Making cluster")
cl <- makeCluster(detectCores())
print(cl)
registerDoParallel(cl)
print("Done making cluster!")

print("Reading data")
trainData <- read.csv('../DataFiles/ERPdata.csv', header <- TRUE)
print("Done reading data")

#cat('Continue?')
#a <- readLines("stdin", n=1)
#if(a == "n") {
#    print("Exiting..")
#    quit()
#}

set.seed(123)

trainData$class <- ifelse(trainData$class == '0', 'N', 'S')
y <- as.factor(as.character(trainData$class))

trainData$class <- y

trainData <- trainData[, names(trainData)]

inputData <- createDataPartition(trainData$class, p = .70, list = FALSE, times = 1)

print(nrow(inputData))
print(nrow(trainData[inputData,]))
print(nrow(trainData[-inputData,]))

trainSet <- trainData[inputData,]
testSet  <- trainData[-inputData,]

trainNA <- any(is.na(trainSet))
testNA <- any(is.na(testSet))

cat(sprintf("TRAIN NA?: %s\nTEST NA?: %s\n", trainNA, testNA))

print("Beginning training!")

mlpGrid = expand.grid(layer1 = 35, layer2 = 20, layer3 = 10)

num <- trainControl(method = 'cv',
                    number = 10,
                    classProbs = TRUE,
                    verboseIter = FALSE,
                    summaryFunction = twoClassSummary,
                    allowParallel = TRUE)

fit <- train(trainSet[, 3:11],
             trainSet$class,
             method = 'mlpML',
             trControl = num,
             tuneGrid = mlpGrid)

#fit <- train(trainSet[, 3:11],
#             trainSet$class,
#             method = 'nnet',
#             MaxNWts = 5000,
#             trControl = num,
#             tuneGrid = expand.grid(size = c(20, 15, 5), decay=c(decayNum, decayNum, decayNum)), linout = 0)

#fit <- train(class ~ Fz + FCz + Cz + FC3 + FC4 + C3 + C4 + CP3 + CP4,
 #            data = train,
  #           method = 'nnet',
   #          metric = 'Accuracy',
    #         trControl = num,
     #        tuneGrid = expand.grid(size = c(hiddenNum), decay=c(0.1)), linout = 0)

print("Done training!")

print("Predict train results!")
trainRslt <- predict(fit, newdata=trainSet)
trainConf <- confusionMatrix(trainRslt, trainSet$class, positive = "S")
print(trainConf, mode = "everything", digits = 4)
print("Done!")

print("Predict test results!")
testRslt  <- predict(fit, newdata=testSet)
testConf  <- confusionMatrix(testRslt, testSet$class, positive = "S")
print(testConf, mode = "everything", digits = 4)
print("Done!")

probabilities <- predict(fit, newdata=testSet, type='prob')

out <- data.frame(SUBJECT=testSet$subject)
out <- cbind(out, CLASS=probabilities$S)

print("Writing training confusion matrix!")
write.csv(as.matrix(trainConf), file=paste(outDir, 'TrainingConfMat.csv', sep=""))
write.csv(as.matrix(trainConf, what = "classes"), file=paste(outDir, 'TrainingConfMatClasses.csv', sep=""))
write.csv(as.matrix(trainConf, what = "classes"), file=paste(outDir, 'TrainingConfMatClasses.csv', sep=""))
write.csv(as.matrix(trainConf, what = "overall"), file=paste(outDir, 'TrainingConfMatOverall.csv', sep=""))
print("Done!")

print("Writing testing confusion matrix!")
write.csv(as.matrix(testConf), file=paste(outDir, 'TestingConfMat.csv', sep=""))
write.csv(as.matrix(testConf, what = "classes"), file=paste(outDir, 'TestingConfMatClasses.csv', sep=""))
write.csv(as.matrix(testConf, what = "overall"), file=paste(outDir, 'TestingConfMatOverall.csv', sep=""))
print("Done!")

write.csv(out, file=paste(outDir, "PackageOut.csv", sep=""), row.names=FALSE)

#nnet <- neuralnet(trainFileData$class ~ trainNN$Fz + trainNN$FCz + trainNN$Cz + trainNN$FC3 + trainNN$FC4 + trainNN$C3 + trainNN$C4 + trainNN$CP3 + trainNN$CP4, trainNN, hidden=5, linear.output = T)
stopCluster(cl)

