# LIBRARY
library(BAS)
library(tidyverse)

# DATA PREP
hr$dep<-factor(hr$dep)
hr$educ<-factor(hr$educ)
hr$gender<-factor(hr$gender)
hr$married<-factor(hr$married)
hr<-hr[-c(26:27)]
hr<-hr %>% relocate(attritition,.before = annpay)
colnames(hr)[1]<-"quit"

# DO WORK
# Number of iterations based on number of unique models desired (10*n.models)
m1 = bas.glm( quit ~ ., data=hr, n.models = 2^20,
                      method="MCMC",
                      betaprior=robust(), family=binomial(), force.heredity = T, 
                      modelprior=beta.binomial(1,1))

# Number of iterations fixed
m1 = bas.glm( quit ~ ., data=hr, 
              method="MCMC", MCMC.iterations = 100000, force.heredity = T,
              betaprior=robust(), family=binomial(), # auto link = logit
              modelprior=beta.binomial(1,1))

m2 = bas.glm( quit ~ ., data=hr, 
              method="MCMC", MCMC.iterations = 100000, force.heredity = T,
               family=binomial(), # auto link = logit
              modelprior=uniform())


m1$n.models  # Number of unique models sampled
m1$n.Unique  # Number of unique models sampled
sort(m1$freq, decreasing = TRUE) # returns vector # visits for each unique model
sum(m1$freq) # returns number of total iterations

# Examine Results
summary(m1)
plot(m1)
image(m1)
coefficients(m1)
fitted(m1)
predict(m1)
diagnostics(m1)
coefficients(m1)

