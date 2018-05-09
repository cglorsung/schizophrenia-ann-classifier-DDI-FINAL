# Author : Conor Lorsung
# Purpose: This system is built for my final CSCI-332 project.
#          Ideally, it will be a neural network that acts as
#          a classifier for the schizophrenia data outlined
#          in the github repository this file is hosted in.

library(tfestimators)
library(dplyr)

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
hiddenNum <- 5
print(paste("Working with ", hiddenNum, " layers!"))

# Output directory
outDir <- paste('./Results/tensorOutput/',hiddenNum,'/', sep="")
dir.create(outDir, showWarnings=FALSE)

print("Reading data")
trainData <- read.csv('./DataFiles/ERPdata.csv', header <- TRUE)
print("Done reading data")

predictors <- c(names(trainData)[3:11])

fileData <- mutate_at(trainData, .vars = predictors, .funs = as.factor)

fileData$class <- ifelse(fileData$class == "0", "N", "S")

inData <- sample(1:nrow(fileData), size = 0.7 * nrow(fileData))

data.train <- fileData[inData,  ]
data.test  <- fileData[-inData, ]

cols <- feature_columns(column_numeric(names(fileData[3:11])))

predict.fun <- function(data) {
    input_fn(data,
                features = c(names(fileData[3:11])),
                response = "class")
}

model <- dnn_classifier(feature_columns = cols, hidden_units = c(10), n_classes = 2, label_vocabulary = c('N', 'S'))

model %>% train(input_fn = predict.fun(data.train))

prediction <- predict(model, input_fn = predict.fun(data.test))
evaluation <- evaluate(model, input_fn = predict.fun(data.test))