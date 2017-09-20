setwd("~/GitHub/kaggle-Titanic/R/")
# read data
titanic = read.csv("~/GitHub/kaggle-Titanic/data/train.csv")
titanic = read.csv("~/GitHub/kaggle-Titanic/data/train - processed.csv")

# check data
# Age has 177 missing values, Embarked has 2 blank values
# target class: Survived
summary(titanic)

# del unnecessary cols: 
# PassengerId(auto_increment), Name(specific), Ticket(specific), Cabin(too few)
titanic_o = titanic
titanic = titanic_o[,c(-1,-4,-9,-11)]

# fill in Age and Embarked using mean/major of the same class
s_0 = mean(subset(titanic, Survived==0)$Age, na.rm=TRUE)
s_1 = mean(subset(titanic, Survived==1)$Age, na.rm=TRUE)
titanic[is.na(titanic$Age),]$Age = s_0

# correlation analysis
# Pclass:Fare: -0.55, SibSp:Parch: 0.41
# Sex? Embarked? <- interaction plot: nearly parallel
cor(titanic[,c(-3,-8)])
interaction.plot(titanic$Sex, titanic$Embarked, titanic$Survived)

# transformation
# transform predictor variables---all log
corrs = c(rep(0,6))
best = c(rep(0,5))
for(n in c(2,4,5:7)){
  corrs[1] = glm(Survived ~ titanic[,n], family=binomial, data=titanic)$aic # not change
  corrs[2] = glm(Survived ~ titanic[,n]^2, family=binomial, data=titanic)$aic # square
  corrs[3] = glm(Survived ~ titanic[,n]^3, family=binomial, data=titanic)$aic# cube
  corrs[4] = glm(Survived ~ sqrt(titanic[,n]), family=binomial, data=titanic)$aic# square root
  corrs[5] = glm(Survived ~ log(titanic[,n]+1), family=binomial, data=titanic)$aic # log
  corrs[6] = glm(Survived ~ 1/(titanic[,n]), family=binomial, data=titanic)$aic # reciprocal
  cand = 100000000
  for( i in 1:6 ){
    if( corrs[i] < cand ){
      cand = corrs[i]
      best[n] = i
    }  
  }
}
test = cbind(Survived=0,test)
# square root
titanic[which(best %in% 4)] = sqrt(titanic[which(best %in% 4)])
test[which(best %in% 4)] = sqrt(test[which(best %in% 4)])
# log
titanic[which(best %in% 5)] = log(titanic[which(best %in% 5)] + 1)
test[which(best %in% 5)] = log(test[which(best %in% 5)] + 1)
# reciprocal
titanic[which(best %in% 6)] = 1 / (titanic[which(best %in% 6)] + 1)
test[which(best %in% 6)] = 1 / (test[which(best %in% 6)] + 1)
test$Survived = NULL

# logistic model
# Signif.: Pclass, Sex, Age, SibSp
logis = glm(Survived ~ ., data=titanic, family=binomial)
summary(logis)

# logistic with stepwise selection (full interactions)
logis = glm(Survived ~ .*., data=titanic, family=binomial)
logis = step(logis)
drop1(logis, test='F')

# random forest with importance plot
# importance: Sex >> Age = Fare >> others
library(randomForest)
rf = randomForest(factor(Survived) ~ ., data=titanic, ntree=500)
rf = randomForest(factor(Survived) ~ .+Pclass:Sex+Sex:Age+Age:Parch+Age:Fare+Parch:Embarked,
                  data=titanic, ntree=1000, mtry=3)
varImpPlot(rf)

# lasso with 10CV
library(glmnet)
data = sparse.model.matrix(Survived ~ .*., titanic)
lasso_cv = cv.glmnet(data, titanic$Survived, alpha=1, nfolds=10, family='binomial')
lasso = glmnet(data, titanic$Survived, alpha=1, family='binomial')

# knn
data = titanic
data$Sex = as.numeric(data$Sex)
data$Embarked = as.numeric(data$Embarked)
nn = knn.cv(data[,-1], data$Survived, k=1)

# predict and create submission
test = read.csv('~/GitHub/kaggle-Titanic/data/test - processed.csv')
yhat = round(predict(logis, newdata=test, type='response'))
yhat = predict(rf, newdata=test, type='response')
test = cbind(Survived=0,test)
yhat = round(predict(lasso, s=lasso_cv$lambda.min, newx=sparse.model.matrix(Survived ~ .*., test), type="response"))
yhat = predict(nn, newdata=test, type='response')
sub = data.frame(PassengerId=892:1309, Survived=yhat)
write.csv(sub,'~/GitHub/kaggle-Titanic/submission/submission_rf_tf.csv',row.names = F)
