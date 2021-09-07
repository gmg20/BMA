library(MASS)
library(BAS)
data(Pima.tr)
# enumeration with default method="BAS"
pima.cch = bas.glm(type ~ ., data=Pima.tr, n.models= 2^7,
                   method="BAS",
                   betaprior=CCH(a=1, b=532/2, s=0), family=binomial(),
                   modelprior=beta.binomial(1,1))
summary(pima.cch)
image(pima.cch)

pima.robust = bas.glm(type ~ ., data=Pima.tr, n.models= 2^7,
                      method="MCMC", MCMC.iterations=10000,
                      betaprior=robust(), family=binomial(),
                      modelprior=beta.binomial(1,1))
summary(pima.robust)

pumapred<-predict(pima.robust, newdata=Pima.tr, type = 'response')
fitted(pima.robust, newdata=Pima.tr, type = 'response')
library(BAS)

