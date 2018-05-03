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

# Number of hidden layers?
hiddenNum <- 8

# Output directory
outDir <- paste('../Results/caretOutput/',hiddenNum,'/', sep="")
dir.create(outDir, showWarnings=FALSE)

# Make the parallel cluster
print("Making cluster")
cl <- makeCluster(detectCores())
registerDoParallel(cl)
print("Done making cluster!")

print("Reading data")
trainData <- read.csv('../DataFiles/ERPdata.csv', header <- TRUE)
print("Done reading data")

cat('Continue?')
a <- readLines("stdin", n=1)
if(a == "n") {
    print("Exiting..")
    quit()
}
y <- as.factor(make.names(trainData$class))

trainData$class <- y

# names <- setdiff(names(trainData), toRemove)

# data <- trainData[, names]

inputData <- createDataPartition(trainData$class, p <- .65)[[1]]

train <- trainData[inputData, ]
test  <- trainData[-inputData, ]

print("Beginning training!")
num <- trainControl(method = 'cv',
                    number = 8,
                    classProbs = TRUE,
                    verboseIter = FALSE,
                    allowParallel = TRUE)

fit <- train(class ~ Fz + FCz + Cz + FC3 + FC4 + C3 + C4 + CP3 + CP4,
             data = train,
             method = 'nnet',
             metric = 'Accuracy',
             trControl = num,
             tuneGrid = expand.grid(size = c(hiddenNum), decay=c(0.1)), linout = 0)

print("Done training!")

print("Predict train results!")
trainRslt <- predict(fit, newdata=train)
trainConf <- confusionMatrix(trainRslt, train$class)
print(trainConf, mode = "everything", digits = 4)
print("Done!")

print("Predict test results!")
testRslt  <- predict(fit, newdata=test)
testConf  <- confusionMatrix(testRslt, test$class)
print(testConf, mode = "everything", digits = 4)
print("Done!")

probabilities <- predict(fit, newdata=test, type='prob')

out <- data.frame(SUBJECT=test$subject)
out <- cbind(out, CLASS=probabilities$X1)

print("Writing training confusion matrix!")
write.csv(as.matrix(trainConf), file=paste(outDir, 'TrainingConfMat.csv', sep=""))
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

print("Done!")
