# Author : Conor Lorsung
# Purpose: This system is built for my final CSCI-332 project.
#          Ideally, it will be a neural network that acts as
#          a classifier for the schizophrenia data outlined
#          in the github repository this file is hosted in.

# Read demographics file
demoDir  <- "../DataFiles/button-tone-sz/"
demoName <- "demographic"
demoData <- read.csv(paste(demoDir, demoName, ".csv", sep=""), header<-TRUE)

# Get lists based on class value
sList <- demoData$subject[(demoData$group == 1)]
nList <- demoData$subject[(demoData$group == 0)]

# Build dataframe with subject numbers according to class values
# sList = patients with schizophrenia
# nList = patients without schizophrenia
demoFrame <- data.frame(subset(cbind(mVal=
{
    n <- max(length(sList), length(nList)) # Max vector length
    length(sList) <- n                     # Assign length = max length
    length(nList) <- n                     # ''
},
 sList=sList, nList=nList), select=-mVal)) # Dataframe with columns sList & nList

# Read data file
fileDir  <- "../DataFiles/button-tone-sz/"
fileName <- "ERPdata"
fileData <- read.csv(paste(fileDir, fileName, ".csv", sep=""), header<-TRUE)