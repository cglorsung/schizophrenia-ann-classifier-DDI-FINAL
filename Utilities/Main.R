# Author : Conor Lorsung
# Purpose: This system is built for my final CSCI-332 project.
#          Ideally, it will be a neural network that acts as
#          a classifier for the schizophrenia data outlined
#          in the github repository this file is hosted in.

# Supervision
supervise = TRUE

# Number of iterations
iterations = 10000

# Read data file **TO BE COMPLETED**

# Scale file data

# Put data into matrix datatype

# Sigmoid function
sigmoid <- function(x, derive=FALSE) {
    if(derive) {
        return (x*(1-x))
    } else {
        return (1/(1+exp(-x)))
    }
}