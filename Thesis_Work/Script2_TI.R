library(ISLR)
library(caret)
library(ROCR)
library(tidyverse)
library(cvms)

io1_filt <- read_csv("io1_filt2.csv")
io1_filt<-io1_filt[,-c(1,3:4)]
io1_filt$TI2<-as.factor(io1_filt$TI2)
## Loading required package: lattice
## Loading required package: ggplot2
set.seed(300)
#Spliting data as training and test set. Using createDataPartition() function from caret
indxTrain <- createDataPartition(y = io1_filt$TI2,p = 0.75,list = FALSE)
training <- io1_filt[indxTrain,]
testing <- io1_filt[-indxTrain,]

#Checking distibution in orignal data and partitioned data
prop.table(table(training$TI2)) * 100
 

prop.table(table(testing$TI2)) * 100

prop.table(table(io1$TI2)) * 100
## 



#Preprocessing
#kNN requires variables to be normalized or scaled. caret provides facility to preprocess data. I am going to choose centring and scaling

trainX <- training[,names(training) != "TI2"]
preProcValues <- preProcess(x = trainX,method = c("center", "scale"))
preProcValues

set.seed(400)

# KNN Based off of Overall Accuracy
ctrl <- trainControl(method="repeatedcv",repeats = 9) #,classProbs=TRUE,summaryFunction = twoClassSummary)
knnFit <- train(TI2 ~ ., data = training, method = "knn", trControl = ctrl, preProcess = c("center","scale"), tuneLength = 20)

#Output of kNN fit
knnFit
plot(knnFit)

# Predictions of KNN where k was tuned based off overall accuracy
knnPredict <- predict(knnFit, newdata = testing, type="prob")
fitted.results<-predict(knnFit, newdata=testing)

# ROC/AUC for KNN with Accuracy used for Hyper parameters


pr <- prediction(knnPredict[,2], testing$TI2)
prf <- performance(pr, measure = "sens", x.measure = "fpr")
plot(prf)
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc

#Confusion Matrix for KNN where k selected for overall accuracy

fitted.results<-as.factor(fitted.results)
misClasificError <- mean(fitted.results != testing$TI2)
print(paste('Accuracy',1-misClasificError))

d <- tibble("target" = testing$TI2, "prediction" = fitted.results)
basic_d<-table(d)
cfm <- as_tibble(basic_d)

conf_mat <- confusion_matrix(targets = d$target,predictions = d$prediction)

plot_confusion_matrix(
  conf_mat$`Confusion Matrix`[[1]],
  font_counts = font(
    size = 3,
    angle = 360,
    color = "red"
  ),
  add_sums = TRUE,
  sums_settings = sum_tile_settings(
    palette = "Oranges",
    label = "Total",
    tc_tile_border_color = "black"
  )
)

mean(knnPredict == testing$TI2)

##### KNN Fit based off of ROC metric
ctrl <- trainControl(method="repeatedcv",repeats = 9,classProbs=TRUE,summaryFunction = twoClassSummary)
knnFit <- train(TI2 ~ ., data = training, method = "knn", trControl = ctrl, preProcess = c("center","scale"), tuneLength = 20)
knnFit

plot(knnFit)

knnPredict <- predict(knnFit,newdata = testing, type = "prob")
fitted.results<-predict(knnFit, newdata=testing)

# ROC/AUC for KNN with ROC/AUC used for Hyper parameters


pr <- prediction(knnPredict[,2], testing$TI2)
prf <- performance(pr, measure = "sens", x.measure = "fpr")
plot(prf)
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc


# KNN Confusion Matrix for k selected based off ROC/AUC
fitted.results<-as.factor(fitted.results)
misClasificError <- mean(fitted.results != testing$TI2)
print(paste('Accuracy',1-misClasificError))

d <- tibble("target" = testing$TI2, "prediction" = fitted.results)
basic_d<-table(d)
cfm <- as_tibble(basic_d)

conf_mat <- confusion_matrix(targets = d$target,predictions = d$prediction)
plot_confusion_matrix(
  conf_mat$`Confusion Matrix`[[1]],
  font_counts = font(
    size = 3,
    angle = 360,
    color = "red"
  ),
  add_sums = TRUE,
  sums_settings = sum_tile_settings(
    palette = "Oranges",
    label = "Total",
    tc_tile_border_color = "black"
  )
)







### Random Forest
set.seed(400)

# RF Fit based on Accuracy 
ctrl <- trainControl(method="repeatedcv",repeats = 3)
#,classProbs=TRUE,summaryFunction = twoClassSummary)


rfFit <- train(TI2 ~ ., data = training, method = "rf", trControl = ctrl, preProcess = c("center","scale"), tuneLength = 20)

rfFit
plot(rfFit)

# ROC/AUC for RF with Accuracy used for Hyper parameters

rfPredict <- predict(rfFit,newdata = testing, type="prob")
pr <- prediction(rfPredict[,2], testing$TI2)
prf <- performance(pr, measure = "sens", x.measure = "fpr")
plot(prf)
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc

# Confusion Matrix
fitted.results <- ifelse(rfPredict[,2] > 0.5,"Quit","NoQuit")
fitted.results<-as.factor(fitted.results)
misClasificError <- mean(fitted.results != testing$TI2)
print(paste('Accuracy',1-misClasificError))

d <- tibble("target" = testing$TI2, "prediction" = fitted.results)
basic_d<-table(d)
cfm <- as_tibble(basic_d)

conf_mat <- confusion_matrix(targets = d$target,predictions = d$prediction)
plot_confusion_matrix(
  conf_mat$`Confusion Matrix`[[1]],
  font_counts = font(
    size = 3,
    angle = 360,
    color = "red"
  ),
  add_sums = TRUE,
  sums_settings = sum_tile_settings(
    palette = "Oranges",
    label = "Total",
    tc_tile_border_color = "black"
  )
)


# RF With ROC metric used
ctrl <- trainControl(method="repeatedcv",repeats = 9,classProbs=TRUE,summaryFunction = twoClassSummary)

rfFit <- train(TI2 ~ ., data = training, method = "rf", trControl = ctrl, preProcess = c("center","scale"), tuneLength = 20)
rfFit
plot(rfFit)


# ROC/AUC for RF with ROC Metric used for Hyper parameters

rfPredict <- predict(rfFit,newdata = testing, type="prob")
pr <- prediction(rfPredict[,2], testing$TI2)
prf <- performance(pr, measure = "sens", x.measure = "fpr")
plot(prf)
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc

# Confusion Matrix
fitted.results <- predict(rfFit, newdata=testing)
fitted.results<-as.factor(fitted.results)
misClasificError <- mean(fitted.results != testing$TI2)
print(paste('Accuracy',1-misClasificError))

d <- tibble("target" = testing$TI2, "prediction" = fitted.results)
basic_d<-table(d)
cfm <- as_tibble(basic_d)

conf_mat <- confusion_matrix(targets = d$target,predictions = d$prediction)
plot_confusion_matrix(
  conf_mat$`Confusion Matrix`[[1]],
  font_counts = font(
    size = 3,
    angle = 360,
    color = "red"
  ),
  add_sums = TRUE,
  sums_settings = sum_tile_settings(
    palette = "Oranges",
    label = "Total",
    tc_tile_border_color = "black"
  )
)
