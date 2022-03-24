library(tidyverse)
library(caret)
library(readr)
library(stringr)
library(psych)
library(BAS)
library(BMS)
library(boot)
library(ROCR)
library(cvms)
library(DescTools)

##############################################################################
# Data Load/Clean
##############################################################################

io1<-read.csv("IO_Level1_clean.csv") # All continuous predictors mean-centered
io1cont<-io1[,-1] # Turnover treated as continuous
io1dich<-io1[,-2] # Turnover dichotomized (TI > 3)
io1dich$TI2<-factor(io1dich$TI2, labels = c("LowRisk", "HiRisk"))

#############################################################################
# Initial Exploration
#############################################################################
corPlot(io1dich[,2:14])
describeBy(io1dich, group=io1dich$TI2)
# Problem Areas
ggplot(io1dich, aes(TI2, conhour)) + geom_boxplot(aes(fill=TI2)) # SKEWED!!!!

io1dich<-io1dich[,-6] # Drop conhour
io1cont<-io1cont[,-6]
#############################################################################

##############################################################################
# Classification Modeling - Preliminary
##############################################################################

# Train/Test Split

set.seed(300)

# Get row numbers for the training data with 75% split
index <- createDataPartition(io1dich$TI2, p=0.75, list=FALSE)

# Create the training and testing dataset
train <- io1dich[index,]
test <- io1dich[-index,]

# Check Class Balance
table(train$TI2)/nrow(train)*100
table(test$TI2)/nrow(test)*100

# Set aside vector of test set responses as 0/1 for Brier Score
test_y<-ifelse(test$TI2=="LowRisk",0,1)

###############################################################################
# Classification Modeling - BMA using BAS Package vs LogReg
###############################################################################

# BMA with BAS package 

m1bma<-bas.glm(TI2 ~ ., family = binomial, data=train, laplace=TRUE, 
            method="deterministic", force.heredity = TRUE, modelprior=beta.binomial(1,2))

# Prediction using BAS
bmaPredict <- predict(m1bma, newdata = test, type="link") # Prediction vector
logpred<-bmaPredict$Ybma                            # Only gives logodds
probpred<-inv.logit(logpred)                        # Convert to probabalistic
actualpred<-ifelse(probpred>0.4999,"HiRisk","LowRisk") # Convert to class pred.
table(actualpred)


###############################################################################

# LogReg with Single Model

m2<-glm(TI2 ~ ., family = "binomial", data = train)

# Prediction using LogReg
m2pred<-predict(m2, newdata = test, type = "response") # LogReg Prediction vector
head(m2pred)                                           # Verify probabilistic
pred<-ifelse(m2pred>0.4999, "HiRisk", "LowRisk")       # Convert to class pred.
table(pred)

###############################################################################
# Classification Performance -> Model Fit and Summary -> BMA vs Log Reg
###############################################################################
# Summary BMA
summary(m1bma)

# Performance BMA

## AUC BMA
pr <- prediction(probpred, test$TI2)
prf <- performance(pr, measure = "sens", x.measure = "fpr")
plot(prf)
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc

## Overall Accuracy BMA
fitted.results<-as.factor(actualpred)
misClasificError <- mean(fitted.results != test$TI2)
print(paste('Accuracy',1-misClasificError))

## Confusion Matrix BMA
d <- tibble("target" = test$TI2, "prediction" = fitted.results)
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

## Brier Score BMA

BrierScore(test_y,probpred)

###############################################################################
# Summary LogReg
summary(m2)

# Performance LogReg

## AUC Log Reg
pr <- prediction(m2pred, test$TI2)
prf <- performance(pr, measure = "sens", x.measure = "fpr")
plot(prf)
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc

## Overall Accuracy Log Reg
fitted.results<-as.factor(pred)
misClasificError <- mean(fitted.results != test$TI2)
print(paste('Accuracy',1-misClasificError))

## Confusion Matrix Log Reg
d <- tibble("target" = test$TI2, "prediction" = fitted.results)
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

## Brier Score LogReg

BrierScore(test_y,m2pred)

###############################################################################
# Regression Models with BMA using BMS vs OLS Single Model
###############################################################################

# Create the training and testing dataset
train <- io1cont[index,]
test <- io1cont[-index,]
###############################################################################

# BMA Model Fit and Predictions

m1bma<-bms(train, g="hyper=2.001",burn=20000, iter=50000, mprior = "random", 
           mcmc="bd", user.int=FALSE)
bmaPredict<-pred.density(m1bma, newdata=test)
head(bmaPredict$fit)

# BMA summary
m1bma

###############################################################################

# LinReg Model Fit and Predictions

m2<-lm(TI ~ ., train)
olsPredict<-predict(m2, newdata = test)
head(olsPredict)

# OLS Summary
summary(m2)

##############################################################################

# Performance OLS vs BMA -> MSE

MSE<-(sum(((test$TI - olsPredict)^2)))/nrow(test)
MSE

MSE<-(sum(((test$TI - bmaPredict$fit)^2)))/nrow(test)
MSE

##############################################################################