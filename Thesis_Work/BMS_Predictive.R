## Prediction with BMS

## Creating a BMA object based on 70 out of 72 data points/subjects
fcstbma = bms(datafls[1:70, ], mprior = "uniform", burn = 20000,
              iter = 50000, user.int = FALSE)

## Creating a predictive density for 2 new data points based on BMA
pdens = pred.density(fcstbma, newdata = datafls[71:72, ])

## 90% PI for both new data points based on previous data and predictor scores
quantile(pdens, c(0.025, 0.975))

## Plot of posterior predictive density
plot(pdens)

## Point estimate and standard errors

pdens$fit
pdens$std.err

## Forecast Error
pdens$fit - datafls[71:72,1]

## Predictive Density for ACTUAL response value
pdens$dyf(datafls[71:72, 1])

## Visualize Actual Observation in Relation to Predictive Density
plot(pdens, realized.y = datafls[71:72, 1])

## Compare a BMA result to the single-best model
att_best = as.zlm(att, model = 1) ## Best BMA Ranked Model
summary(att_best)


## Pull single-best (or any rank) model from BMA and convert to OLS
att_bestlm = lm(model.frame(as.zlm(att,model=1)))
