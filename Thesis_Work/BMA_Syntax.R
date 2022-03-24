# Syntax for Using BMA for Classification and/or Regression

################################################################################
# Housekeeping & Data Prep
################################################################################

# Packages

library(psych)

library(BAS)

library(tidyverse)

library(caret)

library(randomForest)

##################################################################################
# Data Prep (All variables except T/O intent mean-centered already)

dat1 <- read.csv("turnover_data.csv") # Load original dataset

# Remove company code and protected status variables (Age & Gender)

dat1 <- dat1[, -c(2, 3, 4)]

# Convert categorical/indicators to factors

dat1$manager <- factor(dat1$manager) # Manager Status (N/Y)
levels(dat1$manager) <- c("No", "Yes")

dat1$contype <- factor(dat1$contype) # Contract Type (Perm/Temp)
levels(dat1$contype) <- c("Temp", "Perm")

dat1$employee_council <- factor(dat1$employee_council) # Employee council (N/Y)
levels(dat1$employee_council) <- c("No", "Yes")

dat1$hr_rep <- factor(dat1$hr_rep) # HRM on-site (N/Y)
levels(dat1$hr_rep) <- c("No", "Yes")

dat1$familybiz <- factor(dat1$familybiz) # Family Business (N/Y)
levels(dat1$familybiz) <- c("No", "Yes")

dat1$superboard <- factor(dat1$superboard) # Supervisory Board (N/Y)
levels(dat1$superboard) <- c("No", "Yes")

dat1$adv_board <- factor(dat1$adv_board) # Advisory Board (N/Y)
levels(dat1$adv_board) <- c("No", "Yes")

#################################################################################

# Because the train/test partitions are stratified with respect to the outcome measures,
# we need to make 2 different full datasets prior to train/test partitioning.

# The first full dataset will have a continuous outcome measure, and the second full
# dataset will have a dichotomous outcome measure. 

# This way we can ensure that the train/test partitions will be stratified correctly 
# according to their respective outcome measure


dat2 <- dat1 # Make 2nd full dataset

dat2$TI_risk <- ifelse(dat2$TI > 3, 1, 0) # Dichotomous outcome measure

dat2$TI_risk <- factor(dat2$TI_risk) # Make factor

dat2 <- dat2 %>% relocate(TI_risk, .before = TI) # Relocate

dat2 <- dat2[-2] # Delete continuous outcome from this 2nd  dataset

# Double-checking datasets 1 & 2 are correct

View(dat1) # View dataset 1

View(dat2) # View dataset 2

str(dat1) # Check Variable descriptions/structure for datasets 1 & 2

str(dat2)

################################################################################

# Listwise Deletion for both datasets

dat1<-na.omit(dat1)

dat2<-na.omit(dat2)

nrow(dat1) # Check 1

nrow(dat2) # Check 2

##################################################################################

# Partitions full datasets 1 & 2 into training and testing sets. 

# We will end up with a training and test set for dataset1 (continuous outcome), 
# and a training and test set for dataset2 (classification outcome)

set.seed(300)

# Partitions via 70/30 train/test split

# First train/test set

indxTrain1 <- createDataPartition(y = dat1$TI, p = 0.7, list = FALSE) # index

training1 <- dat1[indxTrain1, ] # training set 1

testing1 <- dat1[-indxTrain1, ] # testing set 1

# Second train/test set

indxTrain2 <- createDataPartition(y = dat2$TI_risk, p = 0.7, list = FALSE) # index

training2 <- dat2[indxTrain2, ] # training set 2

testing2 <- dat2[-indxTrain2, ] # testing set 2

################################################################################

# Check outcome distributions are correctly stratified 


# Continuous outcome distributions 

summary(training1$TI) # training1

summary(testing1$TI) # testing1

summary(dat1$TI) # overall


# Classification outcome distributions 

prop.table(table(training2$TI_risk)) * 100 # Outcome balance for training2

prop.table(table(testing2$TI_risk)) * 100 # Outcome balance for testing2

prop.table(table(dat2$TI_risk)) * 100 # full/non-partioned dataset2

#################################################################################
##################################################################################

# BMA Case Study 1 (Continuous Outcome Measure via Normal Linear Regression)

# 'prior' sets the regression coefficient prior to a hyper-g coefficient prior 
# with Laplace approximation for the integration over 'g'. 

# 'method' is set to "MCMC" to specify a MCMC sampling routine should be used

# "MCMC.iterations" specifies the number of iterations. 

# "Force.heredity" requires the factor levels to be kept together 

#  "na.action=omit" is the default for missing data  

#  "model prior" defaults to Beta-Binomial (1, 1) on predictor inclusion prob.

# "Renormalization" is specified in order to calculate posterior model probabilities 
# exactly after model space has been sampled, rather than used frequency-based approximations 
# (done when data is noisy and convergence of MCMC routine is poor).



# Model Specification

BMA1 <- bas.lm(TI ~ edu + manager + conhour + contype + lmx + infshare + voice +
                 paysatis + fair + proact + caropp + N + FTEN + FTELY + employee_council +
                 hr_rep + n_levels + n_deps + n_magers + familybiz + HR1 + HR2 +
                 HR3 + HR4 + HR5 + HR6 + HR7 + HR8 + HR9 + superboard + adv_board +
                 corp_entr + stratplan + entr_orient,
  data = training1, prior = "hyper-g-laplace", alpha = 3, method = "MCMC",
  MCMC.iterations = 50000000, force.heredity = TRUE, renormalize = TRUE
)


##################################################################################

# MCMC Convergence Diagnostics (for use if "renormalized=FALSE" was used)



# MCMC approximated predictor inclusion probs vs calculated predictor inclusion probs

diagnostics(BMA1, type = "pip", pch = 16)



# MCMC approximated posterior model probabilities vs calculated posterior model probs

diagnostics(BMA1, type = "model", pch = 16)



# Number of Unique Models Sampled/Explored

BMA1$n.Unique


################################################################################

# Results & Diagnostics


# initial results summary

summary(BMA1) 



# Plots -> (1) Resids vs Fitted, (2) Cumulative Model Probs,
# (3) Marginal Likelihood vs complexity, Marginal Predictor inclusion probs

plot(BMA1, ask = F) # Plots



# Visualization of Posterior Model Probabilities and Variable Inclusion.
# Models with indistinguishable log posterior odds have the same color.

image(BMA1, rotate = F)


# Regression Coefficients

BMA1coefs <- coef(BMA1)

BMA1coefs


# 95% Credible Interval (Highest Posterior Density) for Coefficients

confint(BMA1coefs)


# Plot of 95% Credible Intervals (HPD) from above

# "parm" argument can specify specific predictors (helpful with large #)


plot(confint(BMA1coefs, parm = 1:8))
plot(confint(BMA1coefs, parm = 9:15))
plot(confint(BMA1coefs, parm = 16:22))
plot(confint(BMA1coefs, parm = 23:29))
plot(confint(BMA1coefs, parm = 30:35))


# Marginal Posterior Distributions for coefficients.

# The vertical bar is the posterior probability that the coefficient is 0

# bell shaped curve represents the density of plausible values from all the models 
# where the coefficient is non-zero. 

# This is scaled so that the density height for non-zero values is the probability 
# that the coefficient is non-zero

plot(BMA1coefs, ask = F) # subset argument can be used to specify specific predictors


#####################################################################################

# Fitted Values for current data and Prediction of new data


# fitted values for current data

fitted_BMA1 <- fitted(BMA1, estimator = "BMA")


# prediction for new data (testing set 1)

predicted_BMA1 <- predict(BMA1, newdata = testing1, estimator = "BMA")


# View Available attributes

names(predicted_BMA1) 


# Predicted Outcome Measure for newdata

predY_BMA1 <- predicted_BMA1$fit 

head(predY_BMA1)


# Predictive intervals for predictions (can't get this to work due to vector size)

BMA1.pred <- predict(BMA1, estimator = "BMA", predict = FALSE, se.fit = TRUE)

confint(BMA1.pred)


##################################################################################


# Evaluation of BMA1 Predictive Performance (RMSE, R^2, MAE)


postResample(predY_BMA1, obs = testing1$TI)


###################################################################################
####################################################################################


# Full OLS model Case Study 1 (Continuous Turnover Intent Outcome Measure)


# Fit OLS model using training dataset 1

olsm1 <- lm(TI ~ edu + manager + conhour + contype + lmx + infshare + voice +
              paysatis + fair + proact + caropp + N + FTEN + FTELY + employee_council +
              hr_rep + n_levels + n_deps + n_magers + familybiz + HR1 + HR2 +
              HR3 + HR4 + HR5 + HR6 + HR7 + HR8 + HR9 + superboard + adv_board +
              corp_entr + stratplan + entr_orient, data = training1) 


# Model Summary

summary(olsm1) 

# 95% CI for Regression Coefficients (Unstandardized)

confint(olsm1) 



# Predictions for new data (testing set 1)

predY_OLS1 <- predict(olsm1, newdata = testing1)

head(predY_OLS1)



# Evaluation of OLS Predictive Performance (BMA wins!)

postResample(predY_OLS1, obs = testing1$TI)



######################################################################################
###################################################################################


# Random Forest Regression Case Study 1 (Continuous Outcome Measure of T/O Intent)


set.seed(300)


### Random Forest

set.seed(400)

# Set training control to use 10-fold Cross Validation

ctrl1 <- trainControl(method = "repeatedcv", repeats = 10)


# Train/Fit Random Forest Regression model to training dataset 1

rf1 <- train(TI ~ edu + manager + conhour + contype + lmx + infshare + voice +
               paysatis + fair + proact + caropp + N + FTEN + FTELY + employee_council +
               hr_rep + n_levels + n_deps + n_magers + familybiz + HR1 + HR2 +
               HR3 + HR4 + HR5 + HR6 + HR7 + HR8 + HR9 + superboard + adv_board +
               corp_entr + stratplan + entr_orient, 
             data = training1, trControl = ctrl1, method = "rf", na.action = na.omit)



# Predictions for new data (testing set 1)

predY_RF1 <- predict(rf1, newdata = testing1)

head(predY_RF1)


# Evaluation of RF Regression Predictive Performance (RMSE, R^2, MAE)  #BMA does better!

postResample(predY_RF1, obs = testing1$TI)


# Variable Importance for RF if desired to compare to PIPs

varImp(rf1)

####################################################################################
#################################################################################
##################################################################################


# BMA Case Study 2 (Binary Turnover Risk Outcome via Logistic Regression)


# "family" sets binomial logistic regression likelihood family

# 'betaprior' sets the regression coefficient prior to a CCH(1, 2, 0) coefficient prior 
#  equivalent to the hyper-g (alpha = 3) prior used for regression contexts. 

# "modelprior" sets beta.binomial(1, 1) prior over predictor inclusion prob.

# 'method' is set to "MCMC" to specify a MCMC sampling routine should be used

# "MCMC.iterations" specifies the number of iterations. 

# "Force.heredity" requires the factor levels to be kept together 

#  "na.action=omit" is the default for missing data  

# "laplace" = TRUE specifies Laplace integration for marginal likelihood 

# "Renormalization" is specified in order to calculate posterior model probabilities 
# exactly after model space has been sampled, rather than used frequency-based approximations 
# (done when data is noisy and convergence of MCMC routine is poor).



# Model Specification

BMA2 <- bas.glm(TI_risk ~ edu + manager + conhour + contype + lmx + infshare + voice +
                  paysatis + fair + proact + caropp + N + FTEN + FTELY + employee_council +
                  hr_rep + n_levels + n_deps + n_magers + familybiz + HR1 + HR2 +
                  HR3 + HR4 + HR5 + HR6 + HR7 + HR8 + HR9 + superboard + adv_board +
                  corp_entr + stratplan + entr_orient, data = training2, 
                family = binomial(link = "logit"), betaprior = CCH(1, 2, 0), 
                modelprior = beta.binomial(1,1),method = "MCMC", 
                MCMC.iterations = 5000000, force.heredity = TRUE, 
                renormalize = TRUE
)

#### DEFAULT BAS PRIOR

BMA2b <- bas.glm(TI_risk ~ edu + manager + conhour + contype + lmx + infshare + voice +
                  paysatis + fair + proact + caropp + N + FTEN + FTELY + employee_council +
                  hr_rep + n_levels + n_deps + n_magers + familybiz + HR1 + HR2 +
                  HR3 + HR4 + HR5 + HR6 + HR7 + HR8 + HR9 + superboard + adv_board +
                  corp_entr + stratplan + entr_orient, data = training2, 
                family = binomial(link = "logit"), 
                modelprior = beta.binomial(1,1),method = "MCMC", 
                MCMC.iterations = 5000000, force.heredity = TRUE, 
                renormalize = TRUE
)

##################################################################################

# MCMC Convergence Diagnostics (for use if "renormalized=FALSE" was used)



# MCMC approximated predictor inclusion probs vs calculated predictor inclusion probs

diagnostics(BMA2, type = "pip", pch = 16)



# MCMC approximated posterior model probabilities vs calculated posterior model probs

diagnostics(BMA2, type = "model", pch = 16)



# Number of Unique Models Sampled/Explored

BMA2$n.Unique


################################################################################

# Results & Diagnostics


# initial results summary

summary(BMA2) 



# Plots -> (1) Resids vs Fitted, (2) Cumulative Model Probs,
# (3) Marginal Likelihood vs complexity, Marginal Predictor inclusion probs

plot(BMA2, ask = F) # Plots



# Visualization of Posterior Model Probabilities and Variable Inclusion.
# Models with indistinguishable log posterior odds have the same color.

image(BMA2, rotate = F)


# Regression Coefficients

BMA2coefs <- coef(BMA2)

BMA2coefs


# 95% Credible Interval (Highest Posterior Density) for Coefficients

confint(BMA2coefs)


# Plot of 95% Credible Intervals (HPD) from above

# "parm" argument can specify specific predictors (helpful with large #)


plot(confint(BMA2coefs, parm = 1:8))
plot(confint(BMA2coefs, parm = 9:15))
plot(confint(BMA2coefs, parm = 16:22))
plot(confint(BMA2coefs, parm = 23:29))
plot(confint(BMA2coefs, parm = 30:35))


# Marginal Posterior Distributions for coefficients.

# The vertical bar is the posterior probability that the coefficient is 0

# bell shaped curve represents the density of plausible values from all the models 
# where the coefficient is non-zero. 

# This is scaled so that the density height for non-zero values is the probability 
# that the coefficient is non-zero

plot(BMA2coefs, ask = F) # subset argument can be used to specify specific predictors


#####################################################################################

# Fitted Values for current data and Prediction of new data


# fitted values for current data

fitted_BMA2 <- fitted(BMA2, estimator = "BMA")


# prediction for new data (testing set 2)

predicted_BMA2 <- predict(BMA2, newdata = testing2, estimator = "BMA", type="response")


# View Available attributes

names(predicted_BMA2) 


# Predicted Outcome Measure for newdata

predY_BMA2 <- predicted_BMA2$fit 

head(predY_BMA2)


# Predictive intervals for predictions (can't get this to work due to vector size)

BMA2.pred <- predict(BMA2, estimator = "BMA", predict = FALSE, se.fit = TRUE)

confint(BMA2.pred)


##################################################################################


# Evaluation of BMA2 Predictive Performance 


postResample(predY_BMA2, obs = testing2$TI_risk)


###################################################################################
####################################################################################


# Full Logistic Regression model Case Study 2 (Binary Turnover Risk Outcome)


# Fit Logistic Regression model using training dataset 2

logregm2 <- lm(TI_risk ~ edu + manager + conhour + contype + lmx + infshare + voice +
                 paysatis + fair + proact + caropp + N + FTEN + FTELY + employee_council +
                 hr_rep + n_levels + n_deps + n_magers + familybiz + HR1 + HR2 +
                 HR3 + HR4 + HR5 + HR6 + HR7 + HR8 + HR9 + superboard + adv_board +
                 corp_entr + stratplan + entr_orient, data = training2) 


# Model Summary

summary(logregm2) 

# 95% CI for Regression Coefficients (Unstandardized)

confint(logregm2) 



# Predictions for new data (testing set 1)

predY_logreg2 <- predict(logregm2, newdata = testing2)

head(predY_logreg2)



# Evaluation of OLS Predictive Performance (BMA wins!)

postResample(predY_logreg2, obs = testing2$TI_risk)



######################################################################################
###################################################################################


# Random Forest Classifier Case Study 2 (Binary Turnover Risk Outcome)


set.seed(300)


# Set training control to use 10-fold Cross Validation

ctrl1 <- trainControl(method = "repeatedcv", repeats = 10)


# Train/Fit Random Forest Regression model to training dataset 1

rf2 <- train(TI_risk ~ edu + manager + conhour + contype + lmx + infshare + voice +
               paysatis + fair + proact + caropp + N + FTEN + FTELY + employee_council +
               hr_rep + n_levels + n_deps + n_magers + familybiz + HR1 + HR2 +
               HR3 + HR4 + HR5 + HR6 + HR7 + HR8 + HR9 + superboard + adv_board +
               corp_entr + stratplan + entr_orient
             , data = training2, trControl = ctrl1, method = "rf", na.action = na.omit)



# Predictions for new data (testing set 1)

predY_RF2 <- predict(rf2, newdata = testing2)

head(predY_RF2)


# Evaluation of RF Regression Predictive Performance (RMSE)  #BMA does better!

postResample(predY_RF2, obs = testing2$TI_risk)

####################################################################################
#################################################################################