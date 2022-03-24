library(tidyverse)
library(readr)
library(ROCR)
library(caret)
library(cvms)
library(DescTools)
library(class)
library(ISLR)

io1_filt <- read_csv("IO_BothLevels_Data.csv")
io1_filt<-io1_filt[,1:34]
io1_filt$TI2<-ifelse(io1_filt$TI>3,1,0)
io1_filt$TI2<-as.factor(io1_filt$TI2)
indxTrain<-createDataPartition(y=io1_filt$TI2,p=0.70,list=FALSE)

set.seed(300)
#Splitting data as training and test set. Using createDataPartition() function from caret
indxTrain <- createDataPartition(y = io1_filt$TI2,p = 0.70,list = FALSE)
training <- io1_filt[indxTrain,]
testing <- io1_filt[-indxTrain,]

#Checking distibution in orignal data and partitioned data
prop.table(table(training$TI2)) * 100
prop.table(table(testing$TI2)) * 100
prop.table(table(io1_filt$TI2)) * 100

###LOGREG USING CARET
# Fit model (no tuning parameters for GLM)
ctrl <- trainControl(method="repeatedcv",repeats = 9)
logregFit <- train(TI2 ~ ., data = training, method = "glm", family="binomial",
                trControl = ctrl, preProcess = c("center","scale"), tuneLength = 20)

#Output of logreg fit
summary(logregFit)

# Predictions of LogReg
logregPredict <- predict(logregFit, newdata = testing, type="prob")
fitted.results<-predict(logregFit, newdata = testing)

# ROC/AUC for LogReg
pr <- prediction(logregPredict[,2], testing$TI2)
prf <- performance(pr, measure = "sens", x.measure = "fpr")
plot(prf)
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc

#Confusion Matrix for LogReg

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



