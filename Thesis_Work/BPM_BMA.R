library(readr)
library(BMS)
library(BMA)
library(BAS)
library(psych)
library(caret)
library(tidyverse)

# DATA with Continuous TurnOver Intentions as DV
io<-read.csv("IO_Level1_Data.csv")
io<-io[,-c(1,3)]
View(io)

# BMS 
m1<-bms(io)

# DATA with Binary TO Intentions as DV
io_dichot <- read_csv("IO_Level1_Data.csv")
io_dichot<-io_dichot[,-c(1,3)]
io_dichot$TI2<-ifelse(io_dichot$TI > 3, "Quit", "NoQuit")
io_dichot$TI2<-as.factor(io_dichot$TI2)
io_dichot<-io_dichot[-1]
io_dichot<-io_dichot %>% relocate(TI2, .before=age)
View(io_dichot)

#BAS
m2<-bas.glm(TI2 ~ ., family = "binomial", data=io_dichot)


#Splitting Dichot data as training and test set.
set.seed(300)
indxTrain <- createDataPartition(y = io_dichot$TI2,p = 0.7,list = FALSE)
training <- io_dichot[indxTrain,]
testing <- io_dichot[-indxTrain,]

# Descriptives
#Checking distibution in original data and partitioned data
prop.table(table(training$TI2)) * 100
prop.table(table(testing$TI2)) * 100
prop.table(table(io_dichot$TI2)) * 100

## Descriptives by Group
describeBy(io_use[2:10], io_use$TI2, digits = 1)
