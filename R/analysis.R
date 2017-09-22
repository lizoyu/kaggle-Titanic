# set working directory
setwd("~/GitHub/kaggle-Titanic/R/")

# read data and merge, get the index of traning and test data
titanic = read.csv("~/GitHub/kaggle-Titanic/data/train.csv", row.names='PassengerId', na.strings=c('NA',' '))
test = read.csv("~/GitHub/kaggle-Titanic/data/test.csv", row.names='PassengerId')
test = cbind(Survived=NA, test)
test$Survived = NA
titanic = rbind(titanic, test)
train = !is.na(titanic$Survived)
test = is.na(titanic$Survived)

# check data
#   Age has 177 missing values, Embarked has 2 blank values
#   target class: Survived
summary(titanic)

# del unnecessary columns: 
#   Ticket(specific), Cabin(too few)
titanic_o = titanic
titanic = titanic[,c(-8,-10)]

# fill in Age and Embarked using MICE imputation
library('mice')
titanic[train,] = complete(mice(titanic[train,]))
titanic[test,-1] = complete(mice(titanic[test,-1]))

# create new variable 'familysize' and remove SibSp and Parch
#   familysize = SibSp + Parch + 1
attach(titanic)
titanic$familysize = SibSp + Parch + 1
detach(titanic)
titanic = titanic[,c(-6,-7)]

# create new variable 'title' using name, put unusual title into 'other', then remove name
name = sub('.*?, (.*?) .*', '\\1', titanic$Name)
name[which(!name %in% c('Mr.','Miss.','Mrs.','Master.'))] = 'other'
titanic$title = factor(name)
titanic$Name = NULL
  
# correlation analysis
# Pclass:Fare: -0.55
# Sex? Embarked? <- interaction plot: nearly parallel
cor(titanic[,c(4,5,7)])
interaction.plot(titanic[train,]$Sex, titanic[train,]$Embarked, titanic[train,]$Survived)

# transformation
#   for numeric: Age, Fare, familysize
corrs = c(rep(0,6))
best = c(rep(0,3))
for(n in c(4,5,7)){
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

# square root
titanic[which(best %in% 4)] = sqrt(titanic[which(best %in% 4)])
# log
titanic[which(best %in% 5)] = log(titanic[which(best %in% 5)] + 1)
# reciprocal
titanic[which(best %in% 6)] = 1 / (titanic[which(best %in% 6)] + 1)

##########################################################################
#
# Models
#
##########################################################################
# logistic model
# Signif.: Pclass, Sex, Age, SibSp
logis = glm(Survived ~ ., data=titanic[train,], family=binomial)
summary(logis)

# logistic with stepwise selection (full interactions)
logis = glm(Survived ~ .*., data=titanic[train,], family=binomial)
logis = step(logis)
drop1(logis, test='F')

# random forest with importance plot
# importance: Sex >> Age = Fare >> others
library(randomForest)
rf = randomForest(factor(Survived) ~ ., data=titanic[train,], cutoff=c(0.47,0.53))
rf = randomForest(factor(Survived) ~ .+Pclass:Sex+Sex:Age+Age:Parch+Age:Fare+Parch:Embarked,
                  data=titanic[train,], ntree=1000, mtry=3)
varImpPlot(rf)

# lasso with 10CV
library(glmnet)
data = sparse.model.matrix(Survived ~ .*., titanic[train,])
lasso_cv = cv.glmnet(data, titanic[train,]$Survived, alpha=1, nfolds=10, family='binomial')
lasso = glmnet(data, titanic[train,]$Survived, alpha=1, family='binomial')

# knn
data = titanic[train,]
data$Sex = as.numeric(data$Sex)
data$Embarked = as.numeric(data$Embarked)
nn = knn.cv(data[,-1], data$Survived, k=1)

# predict and create submission
yhat = round(predict(logis, newdata=titanic[test,-1], type='response'))
yhat = predict(rf, newdata=titanic[test,-1], type='response')
yhat = round(predict(lasso, s=lasso_cv$lambda.min, newx=sparse.model.matrix(Survived ~ .*., titanic[test,]), type="response"))
yhat = predict(nn, newdata=titanic[test,-1], type='response')
sub = data.frame(PassengerId=892:1309, Survived=yhat)
write.csv(sub, '~/GitHub/kaggle-Titanic/submission/submission_rf_new_tf_biased.csv', row.names = F)
