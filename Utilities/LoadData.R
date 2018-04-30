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
fileDir  <- "../DataFiles/button-tone-sz/"
fileName <- "ERPdata"
fileData <- read.csv(paste(fileDir, fileName, ".csv", sep=""), header<-TRUE)

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

        newDatFrame <- as.data.frame(cbind(nPatient=nPatient, sPatient=sPatient))
        print(newDatFrame)

        nSet <- as.data.frame(na.omit(fileData[(fileData$subject %in% newDatFrame$nPatient), ]))
        sSet <- as.data.frame(na.omit(fileData[(fileData$subject %in% newDatFrame$sPatient), ]))
        
        return(rbind(nSet, sSet))
    }
}