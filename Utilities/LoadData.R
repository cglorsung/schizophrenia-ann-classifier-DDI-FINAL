# Author : Conor Lorsung
# Purpose: This system is built for my final CSCI-332 project.
#          Ideally, it will be a neural network that acts as
#          a classifier for the schizophrenia data outlined
#          in the github repository this file is hosted in.

# Global variables
global.totSet <- data.frame()
global.patSet <- data.frame()

# Read demographics file
demoDir  <- "DataFiles/"
demoName <- "demographic"
cat(sprintf("Reading demographics file\n"))
demoData <- read.csv(paste(demoDir, demoName, ".csv", sep=""), header<-TRUE)
cat(sprintf("Done reading demographics file\n"))

# Get lists based on class value
sList <- demoData$subject[(demoData$group == 1)] # Schizophrenia list
nList <- demoData$subject[(demoData$group == 0)] # Non-schizophrenia list

# Build dataframe with subject numbers according to class values
demoFrame <- data.frame(subset(cbind(mVal=
{
    n <- max(length(sList), length(nList)) # Max vector length
    length(sList) <- n                     # Assign length = max length
    length(nList) <- n                     # ''
},
 sList=sList, nList=nList), select=-mVal)) # Dataframe with columns sList & nList

# Read data file
fileDir  <- "DataFiles/"
fileName <- "ERPdata"
cat(sprintf("Reading ERPdata file\n"))
fileData <- read.csv(paste(fileDir, fileName, ".csv", sep=""), header<-TRUE)
cat(sprintf("Done reading ERPdata file\n"))

# Write a CSV with all generated training records
getTrainSet <- function(frame=demoFrame, numPatients=2) {
    if("nList" %in% colnames(frame) == FALSE && "sList" %in% colnames(frame) == FALSE) {
        stop("No \"nList\" or \"sList\" in evaluated frame.\nDataframe must contain only two columns: nList and sList")
    } else if("sList" %in% colnames(frame) == FALSE) {
        stop("No \"sList\" in evaluated frame.\nDataframe must contain only two columns: nList and sList")
    } else if("nList" %in% colnames(frame) == FALSE) {
        stop("No \"nList\" in evaluated frame.\nDataframe must contain only two columns: nList and sList")
    } else {
        adjnList <- as.data.frame(na.omit(frame$nList))             # Omit NA values from nList
        nPatient <- adjnList[sample(nrow(adjnList), numPatients), ] # Randomly get <numPatients> values from adjnList

        adjsList <- as.data.frame(na.omit(frame$sList))             # Omit NA values from sList
        sPatient <- adjsList[sample(nrow(adjsList), numPatients), ] # Randomly get <numPatients> values from adjsList

        global.patSet <<- as.data.frame(cbind(nPatient=nPatient, sPatient=sPatient)) # Dataframe with subject numbers

        nSet <- as.data.frame(na.omit(fileData[(fileData$subject %in% global.patSet$nPatient), ])) # Get records for nList
        sSet <- as.data.frame(na.omit(fileData[(fileData$subject %in% global.patSet$sPatient), ])) # Get records for sList

        global.totSet <<- rbind(nSet, sSet)

        cat(sprintf("Writing TestPatientList.csv\n"))
        write.csv(global.patSet, file="DataFiles/TestPatientList.csv", row.names=FALSE)
	cat(sprintf("Done writing TestPatientList.csv\nWriting SampleData.csv\n"))
        write.csv(global.totSet, file="DataFiles/SampleData.csv", row.names=FALSE)
        cat(sprintf("Done writing SampleData.csv\n"))
    }
}

getTestSet <- function(div=TRUE) {
    if(div) {
        # Remove training records from test set
        cat(sprintf("Writing TestData.csv\n"))
        write.csv(fileData[!(fileData$subject %in% global.totSet$subject), ], file="DataFiles/TestData.csv", row.names=FALSE)
        cat(sprintf("Done writing TestData.csv without training set\n"))
    } else {
        # Test set will be all records
        cat(sprintf("Writing TestData.csv\n"))
        write.csv(fileData, file="DataFiles/TestData.csv", row.names=FALSE)
        cat(sprintf("Done writing TestData.csv with all records\n"))
    }
}

getPatSet <- function() {
    return (global.patSet)
}
