############# PROJECT #########################

library(caret)
library(ggplot2)
library(randomForest)

swd="C:/Users/11620943/Desktop/machinelearningPROJECT"

testing= read.csv("C:/Users/11620943/Desktop/machinelearningPROJECT/pml-testing.csv")
training= read.csv("C:/Users/11620943/Desktop/machinelearningPROJECT/pml-training.csv")

#training = read.csv("C:/Users/Ospite/Desktop/Coursera/Project/pml-training.csv")
#testing = read.csv("C:/Users/Ospite/Desktop/Coursera/Project/pml-testing.csv")

# Remove rows with new_window=yes because it is not relevant, it contains the summary and average of above rows (single replication)
preprocessing <- subset(training, new_window !="yes")

# Remove columns with no sense data, like NA or empty values 
preprocessing[preprocessing==""] <- NA
preprocessing <- preprocessing[, colSums(is.na(preprocessing)) == 0] 

# Delete variables with nearZeroVar=TRUE because it doesn't have any useful data
nzv=nearZeroVar(preprocessing)
print(nzv)
preprocessing <- preprocessing[, -nzv]

# looking data frame, delete variables with not relevant information, like name of user and so on
preprocessing <- preprocessing[,-c(1:6)]

# Delete variables with highly-unbalanced data
prenzv=nearZeroVar(preprocessing,saveMetrics=TRUE)
subprenzv <- subset(prenzv, freqRatio > 20)
mycols <- c("roll_arm", "pitch_arm", "yaw_arm", "pitch_forearm")
which(names(preprocessing) %in% mycols)
preprocessing <- preprocessing[,-c(14,15,16,41)]


M <- abs(cor(preprocessing[,-49]))
diag(M) <- 0
CM <- which(M > 0.99, arr.ind=T)

set.seed(31355)

#make sub test and sub training set
inTrain <- createDataPartition(y=preprocessing$classe, p=0.75, list=FALSE) 
subtrain <- preprocessing[inTrain,]
subtest <- preprocessing[-inTrain,]

#apply random forest with hold-out cross validation
mod_rf <- train(subtrain$classe ~.,method="rf", data=subtrain, ntree=10)
mod_rf
#prediction
pred_rf <- predict(mod_rf, subtest)
#confusion matrix
confusionMatrix(subtest$classe,pred_rf)



#cross validation with k-fold, k large
ctrl <- trainControl(method = "cv", number = 20)
#apply random forest
mod_rf_cv <- train(subtrain$classe ~.,method="rf", trainControl=ctrl, data=subtrain, ntree=10)
mod_rf_cv
#prediction
pred_rf_cv <- predict(mod_rf_cv, subtest)
#confusion matrix
confusionMatrix(subtest$classe,pred_rf_cv)


#cross validation with k-fol, k small
ctrl <- trainControl(method = "cv", number = 3)
#apply random forest
mod_rf_cv <- train(subtrain$classe ~.,method="rf", trainControl=ctrl, data=subtrain, ntree=10)
mod_rf_cv
#prediction
pred_rf_cv <- predict(mod_rf_cv, subtest)
#confusion matrix
confusionMatrix(subtest$classe,pred_rf_cv)

mod_lda <- train(subtrain$classe ~.,method="lda", data=subtrain) 
mod_lda
pred_lda <- predict(mod_lda, subtest)
conf_lda=confusionMatrix(subtest$classe,pred_lda)
conf_lda


mod_rpart <- train(subtrain$classe ~.,method="rpart", data=subtrain)
mod_rpart
pred_rpart <- predict(mod_rpart, subtest)
conf_rpart=confusionMatrix(subtest$classe,pred_rpart)
conf_rpart


mod_gbm <- train(subtrain$classe ~.,method="gbm", n.trees=10, data=subtrain, verbose=FALSE)
mod_gbm
pred_gbm <- predict(mod_gbm, subtest)
conf_gbm=confusionMatrix(subtest$classe,pred_gbm)



#final prediction
final_pred_rf <- predict(mod_rf, testing)
final_pred_rf
