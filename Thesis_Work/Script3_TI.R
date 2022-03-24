library(ISLR)
library(caret)
library(ROCR)
library(tidyverse)
library(cvms)
library(smotefamily)

io1_filt <- read_csv("io1_filt2.csv")
io1_filt<-io1_filt[,-c(1,3:4)]
io1_filt$TI2<-as.factor(io1_filt$TI2)

set.seed(300)

#Spliting data as training and test set
indxTrain <- createDataPartition(y = io1_filt$TI2,p = 0.70,list = FALSE)
training <- io1_filt[indxTrain,]
testing <- io1_filt[-indxTrain,]

# Checking Class Balance/Imbalance in baseline (non-SMOTE) datasets
prop.table(table(io1_filt$TI2))
prop.table(table(training$TI2))
prop.table(table(testing$TI2))

# SMOTE
train.smote <- SMOTE(training[,-1],training$TI2, K = 5)
train.smote <- train.smote$data # extract only the balanced dataset
names(train.smote)[10]<-"TI2"
train.smote$TI2 <- as.factor(train.smote$TI2)

# Checking class balance/imbalance for smote datasets
prop.table(table(train.smote$TI2))

### Evaluate logreg performance with SMOTE dataset

# Fit model (no tuning parameters for GLM)
ctrl <- trainControl(method="repeatedcv",repeats = 9)
logregFit_smote <- train(TI2 ~ ., data = train.smote, method = "glm", family="binomial",
                   trControl = ctrl, preProcess = c("center","scale"), tuneLength = 20)

#Output of logreg fit
summary(logregFit_smote)

# Predictions of LogReg
logregPredict_smote <- predict(logregFit_smote, newdata = testing, type="prob")
fitted.results_smote<-predict(logregFit_smote, newdata = testing) # Quit/NoQuit

# ROC/AUC for LogReg
pr <- prediction(logregPredict_smote[,2], testing$TI2)
prf <- performance(pr, measure = "sens", x.measure = "fpr")
plot(prf)
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc

#Confusion Matrix for LogReg

misClasificError <- mean(fitted.results_smote != testing$TI2)
print(paste('Accuracy',1-misClasificError))

d <- tibble("target" = testing$TI2, "prediction" = fitted.results_smote)
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

# Visual ProbPred
fitted.results = predict(logregFit, newdata = testing, type = "prob")
head(fitted.results)
bob<-data.frame(trueobs = testing$TI2, pred_prob_Q = fitted.results)
bob$pred_prob_NoQ = (1-fitted.results)
head(bob)

ggplot(bob, aes(x = pred_prob_Q.Quit)) + 
  geom_histogram(binwidth = .05) + 
  facet_wrap(~trueobs) + 
  xlab("Probability of Quit")

# Accuracy 
misClasificError <- mean(fitted.results != testing$TI2)
print(paste('Accuracy',1-misClasificError))



#### ADASYN

train.adas <- ADAS(training[,-1],training$TI2, K = 5)
train.adas <- train.adas$data # extract only the balanced dataset
names(train.adas)[10]<-"TI2"
train.adas$TI2 <- as.factor(train.adas$TI2)

# Checking class balance/imbalance for smote datasets
prop.table(table(train.adas$TI2))

### Evaluate logreg performance with ADAS dataset

# Fit model (no tuning parameters for GLM)
ctrl <- trainControl(method="repeatedcv",repeats = 3)
logregFit_adas <- train(TI2 ~ ., data = train.adas, method = "glm", family="binomial",
                         trControl = ctrl, preProcess = c("center","scale"), tuneLength = 20)

#Output of logreg fitADAS
summary(logregFit_adas)

# Predictions of LogRegADAS
logregPredict_adas <- predict(logregFit_adas, newdata = testing, type="prob")
fitted.results_adas<-predict(logregFit_adas, newdata = testing) # Quit/NoQuit

# ROC/AUC for LogRegADAS
pr <- prediction(logregPredict_adas[,2], testing$TI2)
prf <- performance(pr, measure = "sens", x.measure = "fpr")
plot(prf)
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc

#Confusion Matrix for LogRegADAS

misClasificError <- mean(fitted.results_adas != testing$TI2)
print(paste('Accuracy',1-misClasificError))

d <- tibble("target" = testing$TI2, "prediction" = fitted.results_adas)
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


# Accuracy ADAS LogReg
misClasificError <- mean(fitted.results != testing$TI2)
print(paste('Accuracy',1-misClasificError))




### KNN with SMOTE 
#Preprocessing
#kNN requires variables to be normalized or scaled. caret provides facility to preprocess data. I am going to choose centring and scaling
set.seed(400)
trainX <- train.smote[,names(train.smote) != "TI2"]
preProcValues <- preProcess(x = trainX, method = c("center", "scale"))
preProcValues
## 


# SMOTE KNN trained Based off of Overall Accuracy
ctrl <- trainControl(method="repeatedcv",repeats = 9) #,classProbs=TRUE,summaryFunction = twoClassSummary)
knnFit_smote <- train(TI2 ~ ., data = train.smote, method = "knn", trControl = ctrl, preProcess = c("center","scale"), tuneLength = 20)

#Output of kNN fit
knnFit_smote
plot(knnFit_smote)

# Predictions of SMOTE KNN where k was tuned based off overall accuracy
knnPredict_smote <- predict(knnFit_smote, newdata = testing, type="prob")
fitted.results_smote<-predict(knnFit_smote, newdata=testing)

# ROC/AUC for SMOTE-based KNN with Accuracy used for Hyper parameters

pr <- prediction(knnPredict_smote[,2], testing$TI2)
prf <- performance(pr, measure = "sens", x.measure = "fpr")
plot(prf)
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc

#Confusion Matrix for SMOTE KNN where k selected for overall accuracy

fitted.results_smote<-as.factor(fitted.results_smote)
misClasificError <- mean(fitted.results_smote != testing$TI2)
print(paste('Accuracy',1-misClasificError))

d <- tibble("target" = testing$TI2, "prediction" = fitted.results_smote)
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

mean(fitted.results_smote == testing$TI2)

##### SMOTE-based KNN Fit based off of ROC metric
ctrl <- trainControl(method="repeatedcv",repeats = 9,classProbs=TRUE,summaryFunction = twoClassSummary)
knnFit_smote <- train(TI2 ~ ., data = train.smote, method = "knn", trControl = ctrl, preProcess = c("center","scale"), tuneLength = 20)
knnFit_smote

plot(knnFit_smote)

knnPredict_smote <- predict(knnFit_smote, newdata = testing, type = "prob")
fitted.results_smote<-predict(knnFit_smote, newdata = testing)

# ROC/AUC for KNN with ROC/AUC used for Hyper parameters


pr <- prediction(knnPredict_smote[,2], testing$TI2)
prf <- performance(pr, measure = "sens", x.measure = "fpr")
plot(prf)
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc


# KNN Confusion Matrix for k selected based off ROC/AUC
fitted.results_smote<-as.factor(fitted.results_smote)
misClasificError <- mean(fitted.results_smote != testing$TI2)
print(paste('Accuracy',1-misClasificError))

d <- tibble("target" = testing$TI2, "prediction" = fitted.results_smote)
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




#### Random Forest SMOTE

### Random Forest
set.seed(400)

# RF SMOTE Fit based on Accuracy 
ctrl <- trainControl(method="repeatedcv",repeats = 9)
#,classProbs=TRUE,summaryFunction = twoClassSummary)


rfFit_smote <- train(TI2 ~ ., data = train.smote, method = "rf", trControl = ctrl, 
               preProcess = c("center","scale"), tuneLength = 20)

rfFit_smote
plot(rfFit_smote)

# ROC/AUC for RF with Accuracy used for Hyper parameters

rfPredict_smote <- predict(rfFit_smote,newdata = testing, type="prob")
pr <- prediction(rfPredict_smote[,2], testing$TI2)
prf <- performance(pr, measure = "sens", x.measure = "fpr")
plot(prf)
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc

# Confusion Matrix
fitted.results_smote <- ifelse(rfPredict_smote[,2] > 0.5,"Quit","NoQuit")
fitted.results_smote<-as.factor(fitted.results_smote)
misClasificError <- mean(fitted.results_smote != testing$TI2)
print(paste('Accuracy',1-misClasificError))

d <- tibble("target" = testing$TI2, "prediction" = fitted.results_smote)
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


# RF SMOTE With ROC metric used
set.seed(400)
ctrl <- trainControl(method="repeatedcv",repeats = 9,classProbs=TRUE,summaryFunction = twoClassSummary)

rfFit_smote <- train(TI2 ~ ., data = train.smote, method = "rf", trControl = ctrl, preProcess = c("center","scale"), tuneLength = 20)
rfFit_smote
plot(rfFit_smote)


# ROC/AUC for RF SMOTE with ROC Metric used for Hyper parameters

rfPredict_smote <- predict(rfFit_smote,newdata = testing, type="prob")
pr <- prediction(rfPredict_smote[,2], testing$TI2)
prf <- performance(pr, measure = "sens", x.measure = "fpr")
plot(prf)
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc

# Confusion Matrix
fitted.results_smote <- ifelse(rfPredict_smote[,2] > 0.5,"Quit","NoQuit")
fitted.results_smote<-as.factor(fitted.results_smote)
misClasificError <- mean(fitted.results_smote != testing$TI2)
print(paste('Accuracy',1-misClasificError))

d <- tibble("target" = testing$TI2, "prediction" = fitted.results_smote)
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
