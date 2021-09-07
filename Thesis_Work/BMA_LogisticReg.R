hrbma<-bms(hr, mprior = 'uniform', burn = 20000, iter = 50000, user.int = FALSE)
coef(hrbma)

table(hr$attritition)

## Not run:
### logistic regression

y<- hrlog$attritition
x<- data.frame(hrlog[,-1])
x$dep<- as.factor(x$dep)
x$gender <- as.factor(x$gender)
x$educ<- as.factor(x$educ)
x$ot <- as.factor(x$ot)
x$married <- as.factor(x$married)

glm.out.att <- bic.glm(x, y, strict = FALSE, OR = 20,
                      glm.family="binomial", factor.type=TRUE)

summary(glm.out.att)
imageplot.bma(glm.out.att)

## Not run:
### logistic regression
library("MASS")
data(birthwt)
y<- birthwt$lo
x<- data.frame(birthwt[,-1])
x$race<- as.factor(x$race)
x$ht<- (x$ht>=1)+0
x<- x[,-9]
x$smoke <- as.factor(x$smoke)
x$ptl<- as.factor(x$ptl)
x$ht <- as.factor(x$ht)
x$ui <- as.factor(x$ui)
glm.out.FT <- bic.glm(x, y, strict = FALSE, OR = 20,
                      glm.family="binomial", factor.type=TRUE)
summary(glm.out.FT)
imageplot.bma(glm.out.FT)

bic.glm.bwT <- bic.glm(x, y, strict = FALSE, OR = 20,
                       glm.family="binomial",
                       factor.type=TRUE)
summary(bic.glm.bwT, conditional = T)
predict( bic.glm.bwT, newdata = x)




## GLIB

## Not run:
### Finney data
data(vaso)
x<- vaso[,1:2]
y<- vaso[,3]
n<- rep(1,times=length(y))
finney.models<- rbind(
  c(1, 0),
  c(0, 1),
  c(1, 1))
finney.glib <- glib (x,y,n, error="binomial", link="logit",
                     models=finney.models, glimvar=TRUE,
                     output.priorvar=TRUE, output.postvar=TRUE)
summary(finney.glib)

