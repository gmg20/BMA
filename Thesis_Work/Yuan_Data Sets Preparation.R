#########################################################################################
## R code for the article "Building Prediction Models with Grouped Data:
## A Case Study on the Prediction of Turnover Intention"
## Part 2: Prepare Data Sets for Prediction Models
#########################################################################################

#########################################################################################
# designated packages should be uploaded before analyses
#########################################################################################
library(dplyr)  ## for basic data oprations
library(fastDummies) ## packages for creating a set of dummy variables

####################################################################
### In this section, we pre-process each data sets so that they can be 
### fed into the 20 prediction models presented in the next part of the code
### We note that how to pre-process data is still a controversial issue; 
### therefore, the following code only serves as a demonstration and researchers are encouraged
### to choose the way of data-preprocessing based on theores/data charecteristics
###################################################################
### this step can be neglected (if starting from the beginning of this script)
load("partition.RData")
train_set.b.s <- partition.data$train_set.b.s
test_set.b.s <- partition.data$test_set.b.s
train_set.f.s <- partition.data$train_set.f.s
test_set.f.s <- partition.data$test_set.f.s
train_set.b.r <- partition.data$train_set.b.r
test_set.b.r <- partition.data$test_set.b.r
train_set.f.r <- partition.data$train_set.f.r
test_set.f.r <- partition.data$test_set.f.r
folds.r <- partition.data$fold.r
folds.s <- partition.data$fold.s

## with one-hot encoding (i.e., dummy variables)
train_set.d.r <- list()
train_set.d.s <- list()
test_set.d.r <- list()
test_set.d.s <- list()
## with target encoding (i.e., one continuous variables)
train_set.e.r <- list()
test_set.e.r <- list()
train_set.e.s <- list()
test_set.e.s <- list()
## with group-mean centering
train_set.g.r <- list()
train_set.g.s <- list()
test_set.g.r <- list()
test_set.g.s <- list()


for (i in 1:rep){
  #############################################
  ### create data sets with one-hot encoding
  train_set.d.s[[i]] <- dummy_cols(train_set.f.s[[i]], select_columns = "code")
  test_set.d.s[[i]] <- dummy_cols(test_set.f.s[[i]], select_columns = "code")
  train_set.d.r[[i]] <- dummy_cols(train_set.f.r[[i]], select_columns = "code")
  test_set.d.r[[i]] <- dummy_cols(test_set.f.r[[i]], select_columns = "code")
  #############################################
  ### create data sets with target encoding
  ### step 1: obtain group means
  group_means <- train_set.b.s[[i]] %>%
    group_by(code) %>%
    summarise(ti.code = mean(TI))
  ### step 2: transform the grouping variable in training sets
  train_set.e.s[[i]] <- dplyr::left_join(train_set.b.s[[i]], group_means, by = "code")
  ### step 3: transform the grouping variable in test sets
  ### when groups from the test sets are different from those from the training sets
  ### the grand mean was used to impute the continuous grouping variable
  ti.mean <- mean(train_set.b.s[[i]]$TI)
  test_set.e.s[[i]] <- test_set.b.s[[i]]
  test_set.e.s[[i]]$ti.code <- rep(ti.mean, nrow(test_set.b.s[[i]]))
  ### step 1: obtain group means
  group_means <- train_set.b.r[[i]] %>%
    group_by(code) %>%
    summarise(ti.code = mean(TI))
  ### step 2: transform the grouping variable in training sets
  train_set.e.r[[i]] <- dplyr::left_join(train_set.b.r[[i]], group_means, by = "code")
  ### step 3: transform the grouping variable in test sets
  ### when groups from the test sets are among those from the training sets
  ### the group means computed from the training sets were used to impute the continuous grouping variable  
  test_set.e.r[[i]] <- dplyr::left_join(test_set.b.r[[i]], group_means, by = "code")
  #################################################
  ### create data sets with group mean centering and include group means as level-2 predeictors
  variables.group.mean <- train_set.f.s[[i]][,2:(ncol(train_set.f.s[[i]])-1)] %>%
    group_by(code) %>%
    dplyr::summarise_at(vars(age:caropp), mean)
  names(variables.group.mean)[2:14] <- paste0(names(variables.group.mean)[2:14], "_gm")
  train_set.g.s[[i]] <- left_join(train_set.b.s[[i]], variables.group.mean, by = "code")
  train_set.g.s[[i]][,3:15] <- train_set.g.s[[i]][,3:15] - train_set.g.s[[i]][,40:52]
  #######################################################3
  ## test sets that include different groups should use the group means computed from the test sample  
  variables.group.mean <- test_set.f.s[[i]][,2:(ncol(test_set.f.s[[i]]))] %>%
    group_by(code) %>%
    dplyr::summarise_at(vars(age:caropp), mean)
  names(variables.group.mean)[2:14] <- paste0(names(variables.group.mean)[2:14], "_gm")
  test_set.g.s[[i]] <- dplyr::left_join(test_set.b.s[[i]], variables.group.mean, by = "code")
  test_set.g.s[[i]][,3:15] <- test_set.g.s[[i]][,3:15] - test_set.g.s[[i]][,39:51]
  ####
  variables.group.mean <- train_set.f.r[[i]][,2:(ncol(train_set.f.r[[i]])-1)] %>%
    group_by(code) %>%
    dplyr::summarise_at(vars(age:caropp), mean)
  names(variables.group.mean)[2:14] <- paste0(names(variables.group.mean)[2:14], "_gm")
  train_set.g.r[[i]] <- dplyr::left_join(train_set.b.r[[i]], variables.group.mean, by = "code")
  train_set.g.r[[i]][,3:15] <- train_set.g.r[[i]][,3:15] - train_set.g.r[[i]][,40:52]
  ###
  ## test sets that include different groups should use the group means computed from the entire sample 
  variables.group.mean <- variables.level1.gc[,2:(ncol(variables.level1.gc))] %>%
    group_by(code) %>%
    dplyr::summarise_at(vars(age:caropp), mean)
  names(variables.group.mean)[2:14] <- paste0(names(variables.group.mean)[2:14], "_gm")
  test_set.g.r[[i]] <- dplyr::left_join(test_set.b.r[[i]], variables.group.mean, by = "code")
  test_set.g.r[[i]][,3:15] <- test_set.g.r[[i]][,3:15] - test_set.g.r[[i]][,39:51]
}

