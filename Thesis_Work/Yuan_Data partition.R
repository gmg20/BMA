#########################################################################################
## R code for the article "Building Prediction Models with Grouped Data:
## A Case Study on the Prediction of Turnover Intention"
## Part 1: Data Partition
#########################################################################################

#########################################################################################
# designated packages should be uploaded before analyses
#########################################################################################
library(dplyr)  ## for basic data oprations
library(groupdata2) ## for creating folds that are randomized at the group level
####################################################################
### In this section, we create data partitions for 200 repetitions
### for both types of data sets (i.e., data sets that include both levels of predictors
### and data sets that include only level-1 predictors)
###################################################################
##################################################################
load("complete.RData")
### data sets with only level 1 variables
data_fl <- complete.data[[1]]
### data sets with level 1 and level 2 variables
data_bl <- complete.data[[2]]
### lists of data sets to store data partitions
## the following lists store data sets in which the groups of test sets 
## differ from the groups of training sets
train_set.b.s <- list()
test_set.b.s <- list()
train_set.f.s <- list()
test_set.f.s <- list()
## the list to store indicies for each partition
folds.s <- list()
##################################
## the following lists store data sets in which the groups of test sets 
## are among the groups of training sets
train_set.b.r <- list()
test_set.b.r <- list()
train_set.f.r <- list()
test_set.f.r <- list()
## the list to store indicies for each partition
folds.r <- list()
####################################################################
## create the data partitions
## the value of the seed can be specified by the user
set.seed(1)
## total number of repetitions
rep <- 5
## the number of folds
n.folds <- 10
for (i in 1:rep){
  ## create the partitions in such a way that the groups from the test sets are different from those from the training sets
  ### note that "code" should be replaced by a grouping variable
  all_data <- groupdata2::partition(data_bl, p = 1/n.folds, id_col = "code", list_out = FALSE)
  partition.i <- all_data$.partitions
  nrow.test <- sum(partition.i == 1)
  nrow.train <- sum(partition.i == 2)
  train_set <- data_bl[partition.i == 2,]
  test_set <-  data_bl[partition.i == 1,]
  ## from the training set, create the 10 folds used in 10-fold cross validation
  random_train_set <- fold(train_set, k = n.folds, id_col = "code")
  folds.i <-random_train_set$.folds
  test_set.b.s[[i]] <- test_set
  train_set.b.s[[i]] <- random_train_set
  ## adapt the same partition to single-level data sets
  train_set <- data_fl[partition.i == 2,]
  test_set <- data_fl[partition.i == 1,]
  ## adapt the same partition to folds
  train_set$.folds <- folds.i
  ##
  test_set.f.s[[i]] <- test_set
  train_set.f.s[[i]] <- train_set

############################################################
  ## create the partitions in such a way that the groups from the test sets are among those from the training sets
  conv <- 1
  index.i <- sample(1:nrow(data_bl), size = nrow.test)
  while(conv){
    index.code <- data_bl[index.i, ]$code
    rest.code <- data_bl[-index.i, ]$code
    rest.index <- setdiff(1:nrow(data_bl), index.i)
    new.data <- index.i[!index.code %in% rest.code]
    if(length(new.data) == 0){
      conv <- 0
    }
    if(length(new.data) != 0){
      shift.index <- sample(rest.index, size = length(new.data))
      index.int <- setdiff(index.i, new.data)
      index.i <- c(index.int, shift.index)
    }
  }
  test_set <- data_bl[index.i,]
  train_set <- data_bl[-index.i,]
  random_train_set <- fold(train_set, k = n.folds, id_col = "code")
  folds.i <-random_train_set$.folds
  test_set.b.r[[i]] <- test_set
  train_set.b.r[[i]] <- random_train_set
  
  ## adapt the same partition to single-level data sets
  train_set <- data_fl[-index.i,]
  test_set <- data_fl[index.i,]
  ## adapt the same partition to folds
  train_set$.folds <- folds.i
  test_set.f.r[[i]] <- test_set
  train_set.f.r[[i]] <- train_set
}

# create the list to store the indices for fold assignment (i.e., folds.r and folds.s)
n.folds <- 10
for (i in 1:rep){
  data.set <- train_set.b.r[[i]]
  folds.r[[i]] <- list()
  for(j in 1:n.folds){
    folds.r[[i]][[j]] <- which(data.set$.folds != j)
  }
  data.set <- train_set.b.s[[i]]
  folds.s[[i]] <- list()
  for(j in 1:n.folds){
    folds.s[[i]][[j]] <- which(data.set$.folds != j)
  }
}

## data storage
partition.data <- list(train_set.b.r = train_set.b.r, test_set.b.r = test_set.b.r,
                       train_set.b.s = train_set.b.s, test_set.b.s = test_set.b.s,
                       train_set.f.r = train_set.f.r, test_set.f.r = test_set.f.r,
                       train_set.f.s = train_set.f.s, test_set.f.s = test_set.f.s,
                       fold.r = folds.r, fold.s = folds.s)

save(partition.data, file = "partition.RData")
#########################################################################################

