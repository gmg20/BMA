#########################################################################################
## R code for the article "Building Prediction Models with Grouped Data:
## A Case Study on the Prediction of Turnover Intention"
## Part 2: build prediction models
## Note that this part of code is standalone, and, to avoid repetition (same procedure applies to predict an observation 
## from a new group versus from an existing group), we illustrate the code with an example data sets (only include data sets 
## in which groups of test sets are different from those of training sets) 
#########################################################################################
## the code serves as an example for implementing the 20 models as described in the submission
## because of the huge computational demand, some models are implemented in parallel computation
#########################################################################################

## A total of 20 models have been examined that combine 5 modeling approaches and 4 prediction models
## these models are indexed as follows
##############################################################################################################################
###                                                  ### regression ### regression trees ### lasso regression ### bagged trees
##############################################################################################################################
### the single level approach                        ###      1     ###       6          ###        11        ###       16
##############################################################################################################################
### the random effects approach                      ###      2     ###       7          ###        12        ###       17
##############################################################################################################################
### the fixed effects approach with target encoders  ###      3     ###       8          ###        13        ###       18
##############################################################################################################################
### the fixed effects approach with one-hot encoders ###      4     ###       9          ###        14        ###       19
##############################################################################################################################
### the correlated random effects approach           ###      5     ###       10         ###        15        ###       20
##############################################################################################################################

########## packaging loading
library(dplyr)
library(glmmLasso) ## for lasso regression with and without random effects
library(lme4) ## for linear regression with random effects
library(Metrics) ## functions to calculate RMSE
library(caret) ## a platform for model training and testing
library(rpart) ## for regression trees
library(Matrix) ## a package for matrix operation
library(doParallel) ## for parallel computation
library(REEMtree) ## for regression trees with random effects
library(Hmisc) ## calculate correlations

#########################################################################################

########### description of data sets
## each type of data is a list consists of 200 elements. Each element is either the training set or the test set for one replication (i.e., a total of 200 replications)
## e.g., train_set.o denotes a list of training sets with one-hot encoders, test_set.o denotes a list of test sets with one-hot encoders. The data sets for the first replication is train_set.o[[1]] and test_set.o[[1]]
## train_set.s and test_set.s: data sets created for the single level approach
## train_set.r and test_set.r: data sets created for the random effects approach
## train_set.t and test_set.t: data sets created for the fixed effects approach with target encoders
## train_set.c and test_set.c: data sets created for the correlated random effects approach
## fold.partition: indices of fold assignment in each partition
## each data set includes a grouping variable "firm", a variable ".folds" indicating random folds, and an outcome "TI".
load("example.RData")
train_set.o <- partition.data.processed[[9]]
test_set.o <- partition.data.processed[[10]]
train_set.s <- partition.data.processed[[3]]
test_set.s <- partition.data.processed[[4]]
train_set.r <- partition.data.processed[[1]]
test_set.r <- partition.data.processed[[2]]
train_set.t <- partition.data.processed[[7]]
test_set.t <- partition.data.processed[[8]]
train_set.c <- partition.data.processed[[5]]
test_set.c <- partition.data.processed[[6]]
folds <- partition.data.processed[[11]]

######################################################################################
############### illustration of the total of 20 prediction models
######################################################################################
## Note: The two types of prediction problems should be estimated separately.
## Here we illustrate a general routine that could be applied to either procedure
###############
## model 1: linear regression with the single level approach
###############
set.seed(1)
rep <- 200
## a vector to record the r squares of all 200 replications
model1.r2 <- rep(NA, rep)
for (i in 1:rep){
  train_set.i <- train_set.s[[i]]
  ## exclude the grouping variable as well as the indicator for the folding assignment
  train_set.i <- within(train_set.i, rm(code, .folds))
  test_set.i <- test_set.s[[i]]
  test_set.i <- within(test_set.i ,rm(code))
  # training of model 1
  model1 <- lm(TI ~ ., data = train_set.i)
  # test on model 1
  predict.1 <- predict(model1, test_set.i, allow.new.levels = TRUE)
  model1.r2[i] <- rcorr(predict.1, test_set.i$TI)$r[2,1] ^ 2
}

###############
## model 2: linear regression with the random effects approach
###############
set.seed(1)
rep <- 200
## a vector to record the r squares of all 200 replications
model2.r2 <- rep(NA, rep)
for (i in 1:rep){
  train_set.i <- train_set.r[[i]]
  train_set.i <- within(train_set.i, rm(.folds))
  test_set.i <- test_set.r[[i]]
  n <- names(train_set.i)
  f <- as.formula(paste("TI ~", paste(n[!n %in% c("TI", "code")], collapse = " + "), "+(1|code)"))
  # train model 2 with the whole data set
  model2 <- lmer(f,  data = train_set.i)
  # test on model 2
  predict.2 <- predict(model2, test_set.i, allow.new.levels = TRUE)
  model2.r2[i] <- rcorr(predict.2, test_set.i$TI)$r[2,1] ^ 2
}

###############
## model 3: linear regression with the fixed effects approach (with target encoders)
###############
set.seed(1)
rep <- 200
## a vector to record the r squares of all 200 replications
model3.r2 <- rep(NA, rep)
for (i in 1:rep){
  train_set.i <- train_set.t[[i]]
  train_set.i <- within(train_set.i, rm(code, .folds))
  test_set.i <- test_set.t[[i]]
  test_set.i <- within(test_set.i ,rm(code))
  # train model 3 with the whole data set
  model3 <- lm(TI ~ ., data = train_set.i)
  # test on model 3
  predict.3 <- predict(model3, test_set.i, allow.new.levels = TRUE)
  model3.r2[i] <- rcorr(predict.3, test_set.i$TI)$r[2,1] ^ 2
}

###############
## model 4: linear regression with the fixed effects approach (with one-hot encoders)
###############
set.seed(1)
rep <- 200
## a vector to record the r squares of all 200 replications
model4.r2 <- rep(NA, rep)
for (i in 1:rep){
  train_set.i <- train_set.o[[i]]
  train_set.i <- within(train_set.i, rm(code, .folds))
  test_set.i <- test_set.o[[i]]
  test_set.i <- within(test_set.i ,rm(code))
  ## only retain the predictors with non-zero variances
  del <- which(apply(train_set.i,2,function(x) sum(x!=0))==0)
  if(length(del)!=0){
    train_set_del <- train_set.i[,-del]
  }
  if(length(del)==0){
    train_set_del <- train_set.i
  }
  # train model 4 with the whole data set
  model4 <- lm(TI ~ ., data = train_set_del)
  # test on model 4
  predict.4 <- predict(model4, test_set.i, allow.new.levels = TRUE)
  model4.r2[i] <- rcorr(predict.4, test_set.i$TI)$r[2,1] ^ 2
}

###############
## model 5: linear regression with the correlated random effects approach
###############
set.seed(1)
rep <- 200
## a vector to record the r squares of all 200 replications
model5.r2 <- rep(NA, rep)
for (i in 1:rep){
  train_set.i <- train_set.c[[i]]
  train_set.i <- within(train_set.i, rm(.folds))
  test_set.i <- test_set.c[[i]]
  n <- names(train_set.i)
  f <- as.formula(paste("TI ~", paste(n[!n %in% c("TI", "code")], collapse = " + "), "+(1|code)"))
  ## train model 5 with the whole training set
  model5 <- lmer(f,  data = train_set.i)
  # test on model 5
  predict.5 <- predict(model5, test_set.i, allow.new.levels = TRUE)
  model5.r2[i] <- rcorr(predict.5, test_set.i$TI)$r[2,1] ^ 2
}

###############
## model 6: regression trees with the single level approach
###############
set.seed(1)
rep <- 200
## a vector to record the r squares of all 200 replications
model6.r2 <- rep(NA, rep)
for (i in 1:rep){
  train_set.i <- train_set.s[[i]]
  train_set.i <- within(train_set.i, rm(code, .folds))
  test_set.i <- test_set.s[[i]]
  test_set.i <- within(test_set.i ,rm(code))
  ## folds could be either folds.exist (for predicting observations of an existing group)
  ## or folds.new (for predicting observations of a new group)
  part.index <- folds[[i]]

  # set a number of parameters for model training
  my_control <- trainControl(method = "cv",
                             number = 10,
                             # the index indicating training folds is specified
                             index = part.index,
                             allowParallel = FALSE)
  # train model 6 (use a try-error configuration to avoid brutal interruption)
  model6 <- NULL
  try_results <- try(model6 <- caret::train(TI ~ ., data = train_set.i,
                                            method = "rpart",
                                            trControl = my_control,
                                            ## cp is the tuning parameter in tree models (see the main text for a detailed explanation)
                                            tuneGrid = expand.grid(cp = seq(.001, .03, .002)),
                                            metric = "RMSE",
                                            # the control argument here is used to control other (not important) paramters in training the trees
                                            # (the most important parameter is "cp", which is optimized in the training process)
                                            control = rpart.control(minsplit = 10, maxdepth = 30)), silent = T)
  # test on model 6
  if(!is.null(model6)){
    predict.6 <- predict(model6, test_set.i, allow.new.levels = TRUE)
    model6.r2[i] <- rcorr(predict.6, test_set.i$TI)$r[2,1] ^ 2
  }
}

###############
## model 7: regression trees with the random effects approach
###############
set.seed(1)
rep <- 200
n.fold <- 10
## cp is the tuning parameter in tree models (see the main text for a detailed explanation)
cp.grid <- seq(.031, .001, by = -.002)
## store results in the training process
rmse.mltree <- matrix(0, ncol = n.fold, nrow = length(cp.grid))
## a vector to record the r squares of all 200 replications
model7.r2 <- rep(NA, rep)

for (i in 1:rep){
  train_set.i <- as.data.frame(train_set.r[[i]])
  test_set.i <- as.data.frame(test_set.r[[i]])

  ## since RE-EM tree has yet been integrated to the framework of caret,
  ## it is required to manually process the cross-validation procedure
  for (l in 1:n.fold){
    test.obs <- which(train_set.i$.folds == l)
    train.set <- train_set.i[-test.obs,]
    test.set <- train_set.i[test.obs,]
    train.set<- within(train.set ,rm(.folds))

    for (j in 1:length(cp.grid)){
      rpart.ctr <-rpart.control(minsplit=10, cp=cp.grid[j], maxcompete=4, maxsurrogate=5, xval=10, maxdepth=30)
      n <- names(train.set)
      # constitute the formula with regard to the datasets under consideration
      f <- as.formula(paste("TI ~", paste(n[!n %in% c("TI", "code")], collapse = " + ")))
      model7 <- REEMtree(f, data = train.set, random = ~1|code, tree.control = rpart.ctr)
      # compute the prediction performance for each dataset
      a <- predict(model7, newdata = test.set, id=test.set$code)
      rmse.mltree[j,l] <- Metrics::rmse(a, test.set$TI)
    }
  }

  # compute the average prediction performance with regard to each potential value of the tuning parameter
  rmse.sum <- apply(rmse.mltree, 1, mean)
  # select the optimal value of the tuning parameter cp
  opt<-which.min(rmse.sum)
  # set the parameters before training
  rpart.ctr <-rpart.control(minsplit=10, cp=cp.grid[opt], maxcompete=4, maxsurrogate=5, xval=10, maxdepth=30)
  ## obtain the initial random effects. The initial random effects formulate a rational start for more advanced calculations
  n <- names(train.set)
  f1 <- as.formula(paste("TI ~", paste(n[!n %in% c("TI", "code")], collapse = " + "), "+(1|code)"))
  model7.app <- lmer(f1,  data = train_set.i)
  r <- ranef(model7.app)$code[,1]

  # conditional on the optimal value of cp, train the model with the whole training datasets
  train_set.i<- within(train_set.i ,rm(.folds))
  model7 <- REEMtree(f, data = train_set.i, random = ~1|code, tree.control = rpart.ctr,
                           initialRandomEffects = r)
  # test on model 7
  predict.7 <- predict(model7, test_set.i, id = test_set.i$code)
  model7.r2[i] <- rcorr(predict.7, test_set.i$TI)$r[2,1] ^ 2
}

###############
## model 8: regression trees with the fixed effects approach (with target encoders)
###############
set.seed(1)
rep <- 200
## a vector to record the r squares of all 200 replications
model8.r2 <- rep(NA, rep)
for (i in 1:rep){
  train_set.i <- train_set.t[[i]]
  train_set.i <- within(train_set.i, rm(code, .folds))
  test_set.i <- test_set.t[[i]]
  test_set.i <- within(test_set.i ,rm(code))

  ## folds could be either folds.exist (for predicting observations of an existing group)
  ## or folds.new (for predicting observations of a new group)
  part.index <- folds[[i]]

  # training of model 8
  my_control <- trainControl(method = "cv",
                             number = 10,
                             # the index indicating training folds is specified
                             index = part.index,
                             allowParallel = FALSE)

  model8 <- NULL
  try_results <- try(model8 <- caret::train(TI ~ ., data = train_set.i,
                                            method = "rpart",
                                            trControl = my_control,
                                            tuneGrid = expand.grid(cp = seq(.001, .03, .002)),
                                            metric = "RMSE",
                                            # the control argument here is used to control some less important paramters in training the trees
                                            # (the most important parameter is "cp", which is optimized in the training process)
                                            control = rpart.control(minsplit = 10, maxdepth = 30)), silent = T)
  # test on model 8
  if(!is.null(model8)){
    predict.8 <- predict(model8, test_set.i, allow.new.levels = TRUE)
    model8.r2[i] <- rcorr(predict.8, test_set.i$TI)$r[2,1] ^ 2
  }
}

###############
## model 9: regression trees with the fixed effects approach (with one-hot encoders)
###############
set.seed(1)
rep <- 200
## a vector to record the r squares of all 200 replications
model9.r2 <- rep(NA, rep)
for (i in 1:rep){
  train_set.i <- train_set.o[[i]]
  train_set.i <- within(train_set.i, rm(code, .folds))
  test_set.i <- test_set.o[[i]]
  test_set.i <- within(test_set.i ,rm(code))

  ## folds could be either folds.exist (for predicting observations of an existing group)
  ## or folds.new (for predicting observations of a new group)
  part.index <- folds[[i]]

  # training of model 9
  my_control <- trainControl(method = "cv",
                             number = 10,
                             # the index indicating training folds is specified
                             index = part.index,
                             allowParallel = FALSE)

  ## only retain predictors with non-zero variance
  del <- which(apply(train_set.i,2,function(x) sum(x!=0))==0)
  if(length(del)!=0){
    train_set_del <- train_set.i[,-del]
  }
  if(length(del)==0){
    train_set_del <- train_set.i
  }
  ### solve the issue that some of the variable names are not legit
  colnames(train_set_del) <- make.names(colnames(train_set_del))
  model9 <- NULL
  try_results <- try(model9 <- caret::train(TI ~ ., data = train_set_del,
                                            method = "rpart",
                                            trControl = my_control,
                                            ## cp is the tuning parameter in tree models.
                                            ## the grid of cp is kept constant in two (single) tree models
                                            tuneGrid = expand.grid(cp = seq(.001, .02, .002)),
                                            metric = "RMSE",
                                            # the control argument here is used to control some less important paramters in training the trees
                                            # (the most important parameter is "cp", which is optimized in the training process)
                                            control = rpart.control(minsplit = 10, maxdepth = 30)), silent = T)
  # test on model 9
  colnames(test_set.i) <- make.names(colnames(test_set.i))
  if(!is.null(model9)){
    predict.9 <- predict(model9, test_set.i, allow.new.levels = TRUE)
    model9.r2[i] <- rcorr(predict.9, test_set.i$TI)$r[2,1] ^ 2
  }
}

###############
## model 10: regression trees with the correlated random effects approach
################
set.seed(1)
rep <- 200
## cp is the tuning parameter in tree models (see the main text for a detailed explanation)
cp.grid <- seq(.031, .001, by = -.002)
## store results in the training process
rmse.mltree <- matrix(0, ncol = n.fold, nrow = length(cp.grid))
## a vector to record the r squares of all 200 replications
model10.r2 <- rep(NA, rep)
## the number of random folds
n.folds <- 10
for (i in 1:rep){
  train_set.i <- as.data.frame(train_set.c[[i]])
  test_set.i <- as.data.frame(test_set.c[[i]])
  ## manual cross validation
  for (l in 1:n.fold){
    test.obs <- which(train_set.i$.folds == l)
    train.set <- train_set.i[-test.obs,]
    test.set <- train_set.i[test.obs,]
    train.set<- within(train.set ,rm(.folds))

    for (j in 1:length(cp.grid)){
      rpart.ctr <-rpart.control(minsplit=10, cp=cp.grid[j], maxcompete=4, maxsurrogate=5, xval=10, maxdepth=30)
      n <- names(train.set)
      # constitute the formula with regard to the datasets under consideration
      f <- as.formula(paste("TI ~", paste(n[!n %in% c("TI", "code")], collapse = " + ")))
      model10 <- REEMtree(f, data = train.set, random = ~1|code, tree.control = rpart.ctr)
      # compute the prediction performance for each dataset
      a <- predict(model10, newdata = test.set, id=test.set$code)
      rmse.mltree[j,l] <- Metrics::rmse(a, test.set$TI)
    }
  }

  # compute the average prediction performance with regard to each potential value of the tuning parameter
  rmse.sum <- apply(rmse.mltree, 1, mean)
  # select the optimal value of the tuning parameter cp
  opt<-which.min(rmse.sum)
  rpart.ctr <-rpart.control(minsplit=10, cp=cp.grid[opt], maxcompete=4, maxsurrogate=5, xval=10, maxdepth=30)

  ## obtain the initial random effects. The initial random effects formulate a rational start for more advanced calculations
  n <- names(train.set)
  f1 <- as.formula(paste("TI ~", paste(n[!n %in% c("TI", "code")], collapse = " + "), "+(1|code)"))
  model10.app <- lmer(f1,  data = train_set.i)
  r <- ranef(model10.app)$code[,1]

  # conditional on the optimal value of cp, train the model with the whole training datasets
  train_set.i<- within(train_set.i ,rm(.folds))
  # train model 10 on the whole training set
  model10.final <- REEMtree(f, data = train_set.i, random = ~1|code, tree.control = rpart.ctr,
                            initialRandomEffects = r)
  # test model 10 on the test set
  predict.10 <- predict(model10.final, newdata = test_set.i, id=test_set.i$code)
  model10.r2[i] <- rcorr(predict.10, test_set.i$TI)$r[2,1] ^ 2
}

###############
## model 11: lasso regression with the single level approach
################
set.seed(1)
## the grid of the tuning parameter
lambda <- seq(14,0,by=-1)
## number of folders
n.fold <- 10
rep <- 200
## a matrix to record the results of each training replication
rmse.lasso <- matrix(0, ncol = n.fold, nrow = length(lambda))
## a vector to record the r squares of all 200 replications
model11.r2 <- rep(NA, rep)
for(i in 1:rep){
  train_set.i <- train_set.s[[i]]
  train_set.i <- within(train_set.i, rm(code))
  test_set.i <- test_set.s[[i]]
  test_set.i <- within(test_set.i ,rm(code))

  ## training procedure with the cross validation procedure
  for (l in 1:n.fold){
    test.obs <- which(train_set.i$.folds == l)
    train.set <- train_set.i[-test.obs,]
    test.set <- train_set.i[test.obs,]
    train.set<- within(train.set ,rm(.folds))
    ## starting values of the training parameters
    delta.start <- as.matrix(t(rep(0, ncol(train.set))))
    q.start <- .1

    for (j in 1:length(lambda)){
      n <- names(train.set)
      ## construct the formula that is specific to the datasets under consideration
      f <- as.formula(paste("TI ~", paste(n[!n %in% c("TI")], collapse = " + ")))
      model11 <- glmmLasso::glmmLasso(f, rnd = NULL, data = train.set,
                                     lambda = lambda[j], switch.NR = T, final.re = TRUE,
                                     control = list(start=delta.start[j,], q_start=q.start))
      ## record the prediction performance of the current fold and update the starting values of the parameters
      rmse.lasso[j,l] <- rmse(predict(model11, newdata = test.set), test.set$TI)
      delta.start<-rbind(delta.start,model11$Deltamatrix[model11$conv.step,])
    }
  }
  rmse.mean <- apply(rmse.lasso, 1, mean)
  ## select the optimal value of the tuning parameter
  opt<- which.min(rmse.mean)
  train_set.i<- within(train_set.i ,rm(.folds))
  ## train model 11 on the whole training set
  model11.final <- glmmLasso(f, rnd = NULL, data = train_set.i,
                             lambda = lambda[opt], switch.NR = T, final.re = TRUE,
                             control = list(start=delta.start[opt,], q_start=q.start))
  predict.11 <- predict(model11.final, newdata = test_set.i)
  model11.r2[i] <- rcorr(predict.11, test_set.i$TI)$r[2,1] ^ 2
}

###############
## model 12: lasso regression with the random effects approach
################
set.seed(1)
## the grid of the tuning parameter
lambda <- seq(14,0,by=-1)
## number of folders
n.fold <- 10
rep <- 200
## a matrix to record the results of each training replication
rmse.lasso <- matrix(0, ncol = n.fold, nrow = length(lambda))
## a vector to record the r squares of all 200 replications
model12.r2 <- rep(NA, rep)
for(i in 1:rep){
  train_set.i <- as.data.frame(train_set.r[[i]])
  test_set.i <- as.data.frame(test_set.r[[i]])
  for (l in 1:n.fold){
    test.obs <- which(train_set.i$.folds == l)
    train.set <- train_set.i[-test.obs,]
    test.set <- train_set.i[test.obs,]
    train.set<- within(train.set ,rm(.folds))
    delta.start <- as.matrix(t(rep(0, ncol(train.set) - 1 + length(levels(train.set$code)))))
    q.start <- .1
    n <- names(train.set)
    f <- as.formula(paste("TI ~", paste(n[!n %in% c("TI", "code")], collapse = " + ")))
    for (j in 1:length(lambda)){
      model12 <- glmmLasso(f, rnd = list(code =~ 1), data = train.set,
                          lambda = lambda[j], switch.NR = F, final.re = TRUE,
                          control = list(start=delta.start[j,], q_start=q.start[j]))
      rmse.lasso[j,l] <- rmse(predict(model12, newdata = test.set), test.set$TI)
      delta.start<-rbind(delta.start,model12$Deltamatrix[model12$conv.step,])
      q.start<-c(q.start,model12$Q_long[[model12$conv.step+1]])
    }
  }

  rmse.mean <- apply(rmse.lasso, 1, mean)
  ## select the value of the tuning parameter that optimizes the preidction
  opt <- which.min(rmse.mean)
  train_set.i <- within(train_set.i ,rm(.folds))
  ## train the model on the full training set
  model12.final <- glmmLasso(f, rnd = list(code =~ 1), data = train_set.i,
                             lambda = lambda[opt], switch.NR = F, final.re = TRUE,
                             control = list(start=delta.start[opt,], q_start=q.start[opt]))
  ## test the model on the test set
  predict.12 <- predict(model12.final, newdata = test_set.i)
  model12.r2[i] <- rcorr(predict.12, test_set.i$TI)$r[2,1] ^ 2
}

###############
## model 13: lasso regression with the fixed effects approach (with target encoders)
################
set.seed(1)
## the grid of the tuning parameter
lambda <- seq(14,0,by=-1)
## number of folders
n.fold <- 10
rep <- 200
## a matrix to record the results of each training replication
rmse.lasso <- matrix(0, ncol = n.fold, nrow = length(lambda))
## a vector to record the r squares of all 200 replications
model13.r2 <- rep(NA, rep)
for(i in 1:rep){
  train_set.i <- train_set.t[[i]]
  train_set.i <- within(train_set.i, rm(code))
  test_set.i <- test_set.t[[i]]
  test_set.i <- within(test_set.i ,rm(code))
  for (l in 1:n.fold){
    test.obs <- which(train_set.i$.folds == l)
    train.set <- train_set.i[-test.obs,]
    test.set <- train_set.i[test.obs,]
    train.set<- within(train.set ,rm(.folds))
    ## starting values of the training parameters
    delta.start <- as.matrix(t(rep(0, ncol(train.set))))
    q.start <- .1
    for (j in 1:length(lambda)){
      n <- names(train.set)
      ## construct the formula that is specific to the datasets under consideration
      f <- as.formula(paste("TI ~", paste(n[!n %in% c("TI")], collapse = " + ")))
      model13 <- glmmLasso::glmmLasso(f, rnd = NULL, data = train.set,
                                     lambda = lambda[j], switch.NR = T, final.re = TRUE,
                                     control = list(start=delta.start[j,], q_start=q.start))
      ## record the prediction performance of the current fold and update the starting values of the parameters
      rmse.lasso[j,l] <- rmse(predict(model13, newdata = test.set), test.set$TI)
      delta.start<-rbind(delta.start,model13$Deltamatrix[model13$conv.step,])
    }
  }
  rmse.mean <- apply(rmse.lasso, 1, mean)
  ## select the optimal value of the tuning parameter
  opt<- which.min(rmse.mean)
  train_set.i<- within(train_set.i ,rm(.folds))

  # train the model with the entire training set
  model13.final <- glmmLasso(f, rnd = NULL, data = train_set.i,
                             lambda = lambda[opt], switch.NR = T, final.re = TRUE,
                            control = list(start=delta.start[opt,], q_start=q.start))
  ## test on the test set
  predict.13 <- predict(model13.final, newdata = test_set.i)
  model13.r2[i] <- rcorr(predict.13, test_set.i$TI)$r[2,1] ^ 2
}

###############
## model 14: lasso regression with the fixed effects approach (with one-hot encoders)
################
set.seed(1)
## the grid of the tuning parameter
lambda <- seq(14,0,by=-1)
## number of folders
n.fold <- 10
rep <- 200
## a matrix to record the results of each training replication
rmse.lasso <- matrix(0, ncol = n.fold, nrow = length(lambda))
## a vector to record the r squares of all 200 replications
model14.r2 <- rep(NA, rep)
for(i in 1:rep){
  train_set.i <- train_set.o[[i]]
  train_set.i <- within(train_set.i, rm(code))
  test_set.i <- test_set.o[[i]]
  test_set.i <- within(test_set.i,rm(code))

  ## training the model with cross validation
  for (l in 1:n.fold){
    test.obs <- which(train_set.i$.folds == l)
    train.set <- train_set.i[-test.obs,]
    test.set <- train_set.i[test.obs,]
    train.set<- within(train.set ,rm(.folds))
    ## only retain variables with non-zero variance
    if(length(which(apply(train.set,2,function(x) sum(x!=0))==0)) != 0){
      train.set.del <- train.set[,-which(apply(train.set,2,function(x) sum(x!=0))==0)]
    }
    if(length(which(apply(train.set,2,function(x) sum(x!=0))==0)) == 0){
      train.set.del <- train.set
    }
    colnames(train.set.del) <- make.names(colnames(train.set.del))
    ## starting values of the training parameters
    delta.start <- as.matrix(t(rep(0, ncol(train.set.del))))
    q.start <- .1

    for (j in 1:length(lambda)){
      n <- names(train.set.del)
      f <- as.formula(paste("TI ~", paste(n[!n %in% c("TI")], collapse = " + ")))
      model14 <- glmmLasso::glmmLasso(f, rnd = NULL, data = train.set.del,
                                     lambda = lambda[j], switch.NR = FALSE, final.re = FALSE,
                                     control = list(start=delta.start[j,], q_start=q.start))
      ## record the prediction performance of the current fold and update the starting values of the parameters
      colnames(test.set) <- make.names(colnames(test.set))
      rmse.lasso[j,l] <- rmse(predict(model14, newdata = test.set), test.set$TI)
      delta.start<-rbind(delta.start,model14$Deltamatrix[model14$conv.step,])
    }
  }
  rmse.mean <- apply(rmse.lasso, 1, mean)
  ## select the optimal value of the tuning parameter
  opt<- which.min(rmse.mean)
  train_set.i<- within(train_set.i ,rm(.folds))
  ## only retain variables with non-zero variance
  del <- which(apply(train_set.i,2,function(x) sum(x!=0))==0)
  if(length(del)!=0){
    train_set_del <- train_set.i[,-del]
  }
  if(length(del)==0){
    train_set_del <- train_set.i
  }
  colnames(train_set_del) <- make.names(colnames(train_set_del))
  # train the model with the whole training set
  n <- names(train_set_del)
  f <- as.formula(paste("TI ~", paste(n[!n %in% c("TI")], collapse = " + ")))
  # conditional on the optimal value of cp, train the model with the whole training datasets
  model14.final <- glmmLasso(f, rnd = NULL, data = train_set_del,
                             lambda = lambda[opt], switch.NR = T, final.re = TRUE,
                             control = list(q_start=q.start))
  colnames(test_set.i) <- make.names(colnames(test_set.i))
  predict.14 <- predict(model14.final, newdata = test_set.i)
  model14.r2[i] <- rcorr(predict.14, test_set.i$TI)$r[2,1] ^ 2
}

###############
## model 15: lasso regression with the correlated random effects approach
################
set.seed(1)
## the grid of the tuning parameter
lambda <- seq(14,0,by=-1)
## number of folders
n.fold <- 10
rep <- 200
## a matrix to record the results of each training replication
rmse.lasso <- matrix(0, ncol = n.fold, nrow = length(lambda))
## a vector to record the r squares of all 200 replications
model15.r2 <- rep(NA, rep)
for(i in 1:rep){
  train_set.i <- as.data.frame(train_set.c[[i]])
  test_set.i <- as.data.frame(test_set.c[[i]])
  ## tune the hyper-parameters with cross-validation
  for (l in 1:n.fold){
    test.obs <- which(train_set.i$.folds == l)
    train.set <- train_set.i[-test.obs,]
    test.set <- train_set.i[test.obs,]
    train.set<- within(train.set ,rm(.folds))
    delta.start <- as.matrix(t(rep(0, ncol(train.set) - 1 + length(levels(train.set$code)))))
    q.start <- .1
    n <- names(train.set)
    f <- as.formula(paste("TI ~", paste(n[!n %in% c("TI", "code")], collapse = " + ")))
    for (j in 1:length(lambda)){
      model15 <- glmmLasso(f, rnd = list(code =~ 1), data = train.set,
                          lambda = lambda[j], switch.NR = F, final.re = TRUE,
                          control = list(start=delta.start[j,], q_start=q.start[j]))
      rmse.lasso[j,l] <- rmse(predict(model15, newdata = test.set), test.set$TI)
      delta.start<-rbind(delta.start,model15$Deltamatrix[model15$conv.step,])
      q.start<-c(q.start,model15$Q_long[[model15$conv.step+1]])
    }
  }

  rmse.mean <- apply(rmse.lasso, 1, mean)
  ## select the tuning parameter that fits the data set best
  opt <- which.min(rmse.mean)
  train_set.i <- within(train_set.i ,rm(.folds))
  ## train the data set on the whole training set
  model15.final <- glmmLasso(f, rnd = list(code =~ 1), data = train_set.i,
                             lambda = lambda[opt], switch.NR = F, final.re = TRUE,
                             control = list(start=delta.start[opt,], q_start=q.start[opt]))
  # test the trained model on the test data set
  predict.15 <- predict(model15.final, newdata = test_set.i)
  model15.r2[i] <- rcorr(predict.15, test_set.i$TI)$r[2,1] ^ 2
}

###############
## model 16: bagged trees with the single level approach
################
set.seed(1)
## Each tree in bagged trees is supposed to grow as large as possible (i.e. without active pruning)
## therefore, the value of the tuning parameter cp is fixed at a minimal value .01
## no cross validation procedure is involved in bagged trees in the current analysis
cp.grid <- 0.01
## values of other parameters
n.fold <- 10
## the number of bootstrapping replications
n.boot <- 500
rep <- 200
## the list to store the estimation results of each replication
bag.pred <- list()
## a vector to record the r squares of all 200 replications
model16.r2 <- rep(NA, rep)
for (i in 1:rep){

  train_set.i <- within(train_set.s[[i]], rm(code))
  test_set.i <- within(test_set.s[[i]], rm(code))

  # note that the training procedure is without cross validation
  my_control <- trainControl(method = "none",
                             number = 10,
                             verboseIter = FALSE,
                             allowParallel = FALSE)

  train_set.i <- within(train_set.i, rm(.folds))

  # to make each bootsrtapped tree, we first create a bootstrap sample by randomly sampling (with replacement)
  # from the original sample so that the size of the bootstrap sample equals the size of the original sample. Then,
  # we create none-pruned trees for each bootstraped sample.
  for (x in 1:n.boot){
    train_set.x <- train_set.i[base::sample(1:nrow(train_set.i), size = nrow(train_set.i), replace = TRUE),]
    model16 <- caret::train(TI ~ ., data = train_set.x,
                            method = "rpart",
                            trControl = my_control,
                            tuneGrid = expand.grid(cp = cp.grid),
                            metric = "RMSE",
                            control = rpart.control(minsplit = 10, minbucket = 3, maxdepth = 30))

    # the estimation of each bootstrapped tree
    bag.pred[[x]] <- predict(model16, newdata = test_set.i)
  }

  # the final prediction equals the average of all predictions derived from all bootstrapped trees
  predict.16 <- Reduce("+", bag.pred)/n.boot
  model16.r2[i] <- rcorr(predict.16, test_set.i$TI)$r[2,1] ^ 2
}
###############
## model 17: bagged trees with the random effects approach
################
set.seed(1)

cp.grid <- 0.01
## values of other parameters
n.fold <- 10
## the number of bootstrapping replications
n.boot <- 500
rep <- 200
## the list to store the estimation results of each replication
bag.pred <- list()
## a vector to record the r squares of all 200 replications
model17.r2 <- rep(NA, rep)
for (i in 1:rep){
  train_set.i <- as.data.frame(train_set.r[[i]])
  test_set.i <- as.data.frame(test_set.r[[i]])
  train_set.i <- within(train_set.i, rm(.folds))
  for (x in 1:n.boot){
    train_set.x <- train_set.i[base::sample(1:nrow(train_set.i), size = nrow(train_set.i), replace = TRUE),]
    rpart.ctr <-rpart.control(minsplit = 10, minbucket = 3, maxdepth = 30, cp = cp.grid)

    train_set.x$code <- factor(train_set.x$code)
    n <- names(train_set.x)
    f2 <- as.formula(paste("TI ~", paste(n[!n %in% c("TI", "code")], collapse = " + ")))

    model17.final <- REEMtree(f2, data = train_set.x, random = ~1|code, tree.control = rpart.ctr)
    bag.pred[[x]] <- predict(model17.final, newdata = test_set.i, id = test_set.i$code)
  }

  predict.17 <- Reduce("+", bag.pred)/n.boot
  model17.r2[i] <- rcorr(predict.17, test_set.i$TI)$r[2,1] ^ 2
}
###############
## model 18: bagged trees with the fixed effects approach with target encoders
################
set.seed(1)

cp.grid <- 0.01
## values of other parameters
n.fold <- 10
## the number of bootstrapping replications
n.boot <- 500
rep <- 200
## the list to store the estimation results of each replication
bag.pred <- list()
## a vector to record the r squares of all 200 replications
model18.r2 <- rep(NA, rep)
for (i in 1:rep){
  train_set.i <- within(train_set.t[[i]], rm(code))
  test_set.i <- within(test_set.t[[i]], rm(code))
  model18.r2[i] <- rcorr(predict.18, test_set.i$TI)$r[2,1] ^ 2

  my_control <- trainControl(method = "none",
                             number = 10,
                             verboseIter = FALSE,
                             allowParallel = FALSE)

  bag.pred <- list()
  train_set.i <- within(train_set.i, rm(.folds))

  for (x in 1:n.boot){
    train_set.x <- train_set.i[base::sample(1:nrow(train_set.i), size = nrow(train_set.i), replace = TRUE),]
    model18 <- caret::train(TI ~ ., data = train_set.x,
                            method = "rpart",
                            trControl = my_control,
                            tuneGrid = expand.grid(cp = cp.grid),
                            metric = "RMSE",
                            control = rpart.control(minsplit = 10, minbucket = 3, maxdepth = 30))

    # predict for each bootstrapped tree
    bag.pred[[x]] <- predict(model18, newdata = test_set.i)
  }

  # the final prediction equals the average of all predictions derived from all bootstrapped trees
  predict.18 <- Reduce("+", bag.pred)/n.boot
  model18.r2[i] <- rcorr(predict.18, test_set.i$TI)$r[2,1] ^ 2
}

###############
## model 19: bagged trees with the fixed effects approach with one-hot encoders
################
set.seed(1)
cp.grid <- 0.01
## values of other parameters
n.fold <- 10
## the number of bootstrapping replications
n.boot <- 500
rep <- 200
## the list to store the estimation results of each replication
bag.pred <- list()
## a vector to record the r squares of all 200 replications
model19.r2 <- rep(NA, rep)
for (i in 1:rep){
  train_set.i <- train_set.o[[i]]
  train_set.i <- within(train_set.i, rm(code))
  test_set.i <- test_set.o[[i]]
  test_set.i <- within(test_set.i ,rm(code))

  my_control <- trainControl(method = "none",
                             number = 10,
                             verboseIter = FALSE,
                             allowParallel = FALSE)

  train_set.i <- within(train_set.i, rm(.folds))

  for (x in 1:n.boot){
    train_set.x <- train_set.i[base::sample(1:nrow(train_set.i), size = nrow(train_set.i), replace = TRUE),]
    ## only retain variables with non-zero variance
    del <- which(apply(train_set.x,2,function(x) sum(x!=0))==0)
    if(length(del)!=0){
      train_set_del <- train_set.x[,-del]
    }
    if(length(del)==0){
      train_set_del <- train_set.x
    }
    colnames(train_set_del) <- make.names(colnames(train_set_del))
    n <- names(train_set_del)
    f <- as.formula(paste("TI ~", paste(n[!n %in% c("TI")], collapse = " + ")))
    ## train the model on the retained data set
    model19 <- caret::train(f, data = train_set_del,
                            method = "rpart",
                            trControl = my_control,
                            tuneGrid = expand.grid(cp = cp.grid),
                            metric = "RMSE",
                            control = rpart.control(minsplit = 10, minbucket = 3, maxdepth = 30))

    # generate the prediction (for the validation set) based on each bootstrapped tree
    colnames(test_set.i) <- make.names(colnames(test_set.i))
    bag.pred[[x]] <- predict(model19, newdata = test_set.i)
  }

  # the final prediction equals the average of all predictions derived from all bootstrapped trees
  predict.19 <- Reduce("+", bag.pred)/n.boot
  model19.r2[i] <- rcorr(predict.19, test_set.i$TI)$r[2,1] ^ 2
}
###############
## model 20: bagged trees with the correlated random effects approach
################
set.seed(1)
cp.grid <- 0.01
## values of other parameters
n.fold <- 10
## the number of bootstrapping replications
n.boot <- 500
rep <- 200
## the list to store the estimation results of each replication
bag.pred <- list()
## a vector to record the r squares of all 200 replications
model20.r2 <- rep(NA, rep)
for (i in 1:rep){
  train_set.i <- as.data.frame(train_set.c[[i]])
  test_set.i <- as.data.frame(test_set.c[[i]])
  train_set.i <- within(train_set.i, rm(.folds))

  for (x in 1:n.boot){
    train_set.x <- train_set.i[base::sample(1:nrow(train_set.i), size = nrow(train_set.i), replace = TRUE),]
    rpart.ctr <-rpart.control(minsplit = 10, minbucket = 3, maxdepth = 30, cp = cp.grid)

    train_set.x$code <- factor(train_set.x$code)
    n <- names(train_set.x)
    f2 <- as.formula(paste("TI ~", paste(n[!n %in% c("TI", "code")], collapse = " + ")))

    model20.final <- REEMtree(f2, data = train_set.x, random = ~1|code, tree.control = rpart.ctr)
    bag.pred[[x]] <- predict(model20.final, newdata = test_set.i, id = test_set.i$code)
  }

  predict.20 <- Reduce("+", bag.pred)/n.boot
  model20.r2[i] <- rcorr(predict.20, test_set.i$TI)$r[2,1] ^ 2
}

