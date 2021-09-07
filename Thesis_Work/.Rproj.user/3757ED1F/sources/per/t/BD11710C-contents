## Example of Comparing BMA with GLM
## BMA
att_bma<-bms(attitude[1:29,],burn=20000,iter=50000)
coef(att_bma,std.coefs = TRUE)

att_pdens<-pred.density(att_bma,newdata = attitude[30,])
plot(att_pdens)

plot(att_pdens, realized.y = attitude[30, 1])

quantile.pred.density(att_pdens,probs=c(.025,.975))

att_glm<-lm(rating~complaints,attitude) ###GLM
summary(att_glm)

predict.glm(att_glm,newdata=attitude[30,])
predict.lm(att_glm,newdata=attitude[30,],type="response", interval = "prediction")
