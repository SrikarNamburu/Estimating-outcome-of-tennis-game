
###########Best accuracy####################

#XGBoost

sampling_strategy <- trainControl(method = "cv", number = 5, verboseIter = T, allowParallel = T)

param_grid <- expand.grid(.nrounds = 50, .max_depth = c(2, 4, 6), .eta = c(0.1,0.3,0.5,0.7),
                          .gamma = c(0.6, 0.3), .colsample_bytree = c(0.6, 0.8),
                          .min_child_weight=1,.subsample = c(0.8, 0.9))

xgb_tuned_model <- train(x = x.train, 
                         y = y.train, 
                         method = "xgbTree",
                         trControl = sampling_strategy,
                         metrics="Accuracy",
                         objective="multi:softprob",
                         tuneGrid = param_grid)

xgb_tuned_model$bestTun
View(xgb_tuned_model$results)

tuned_params_train_preds <- predict(xgb_tuned_model, x.train)

tuned_params_val_preds <- predict(xgb_tuned_model,x.val)

confusionMatrix(tuned_params_train_preds, y.train)

confusionMatrix(tuned_params_val_preds, y.val)

tuned_params_test_preds <- predict(xgb_tuned_model,x.test)

sub_preds$outcome = tuned_params_test_preds
write.csv(sub_preds, "xg_boost6.csv", row.names = FALSE)

metrics_xg_boost = data.frame(model = "xg boost" ,train_accuracy =94.84 , val_accuracy = 87.74, test_accuracy = 87.19 )
metrics = rbind(metrics,metrics_xg_boost)

metrics_xg_boost1 = data.frame(model = "xg boost1" ,train_accuracy =94.75 , val_accuracy = 87.99, test_accuracy = 88.09 )
metrics = rbind(metrics,metrics_xg_boost1)




rm(list=ls(all=TRUE))

tennis =read.csv("train-1542197608821.csv", header=TRUE, sep=",")
test = read.csv("test-1542197608821.csv", header=TRUE, sep=",")
sub_preds = read.csv("sample_submission-1542197608821 - Copy.csv", header=TRUE, sep=",")

#Response variable is "outcome" 
# Understanding the data 
summary(tennis)
str(tennis)
head(tennis)

sum(is.na(tennis))
# Do necessary type conversions
factors =c("serve","outside.sideline","outside.baseline","same.side","server.is.impact.player")
tennis[,factors]=data.frame(sapply(tennis[,factors],as.character))
tennis[,factors]=data.frame(sapply(tennis[,factors],as.factor))
test[,factors]=data.frame(sapply(test[,factors],as.character))
test[,factors]=data.frame(sapply(test[,factors],as.factor))

# Removing unnecessary column ID 
tennis$ID = NULL
test_id = test$ID 
test$ID = NULL
# Do Train-Val Split
library(caret)
set.seed(1234)
rows=createDataPartition(tennis$outcome,p = 0.7,list = FALSE)
train=tennis[rows,]
val=tennis[-rows,]

# PreProcess the data to standadize the numeric attributes
preProc<-preProcess(train[,setdiff(names(train),c("outcome","serve","outside.sideline","outside.baseline","same.side","server.is.impact.player"))],method = c("center", "scale"))

train<-predict(preProc,train)
val<-predict(preProc,val)
test_st = predict(preProc, test)
tennis_st = predict(preProc, tennis)
str(train)
sum(is.na(train))
#checking for class imbalances
prop.table(table(train$outcome))
prop.table(table(val$outcome))

#visualization

library(plotly)
p <- plot_ly(z = tennis, type = "heatmap")




#model 1 Logistic

library(nnet)
logm1=multinom(outcome~.,train)
summary(logm1)

train_pred=predict(logm1,train,type="class")
confusionMatrix(train_pred,train$outcome)

val_pred=predict(logm1,val,type="class")
confusionMatrix(val_pred, val$outcome)

test_pred=predict(logm1,test_st,type="class")
sub_preds$outcome = test_pred
write.csv(sub_preds, "logistic.csv", row.names = FALSE)

metrics = data.frame(model = "logistic" ,train_accuracy = 82.18, val_accuracy = 81.53, test_accuracy = 81.89 )

#decision trees

library(rpart)
model_dt <- rpart(outcome ~ . , train)

preds_train_dt <- predict(model_dt,train, type = "class")
confusionMatrix(preds_train_dt, train$outcome)

preds_val_dt <- predict(model_dt, val, type="class")
confusionMatrix(preds_val_dt, val$outcome)


pred_test_dt =predict(model_dt,test_st,type="class")
sub_preds$outcome = pred_test_dt
write.csv(sub_preds, "dt.csv", row.names = FALSE)

metricsdt = data.frame(model = "decison trees" ,train_accuracy = 79.36, val_accuracy = 79.53, test_accuracy = 78.39 )

metrics = rbind(metrics,metricsdt)

#c5.0
library(DMwR)
library(C50)
DT_C50 <- C5.0(outcome~.,data=train, Rules = TRUE )
summary(DT_C50)


##predict on train and validation
pred_Train_c5.0 = predict(DT_C50,newdata=train, type="class")
pred_val_c5.0 = predict(DT_C50, newdata=val, type="class")

#Error Metrics on train and test
confusionMatrix(train$outcome,pred_Train_c5.0)
confusionMatrix(val$outcome,pred_val_c5.0)

#Check variable importance
C5imp(DT_C50, pct=TRUE)
# Store the rules in a notepad
write(capture.output(summary(DT_C50)), "c50model.txt")

pred_test_dt_c5.0 =predict(DT_C50,test_st,type="class")
sub_preds$outcome = pred_test_dt_c5.0
write.csv(sub_preds, "dt_c5.0.csv", row.names = FALSE)

metricsdt_5.0 = data.frame(model = "decison trees_c.5.0" ,train_accuracy = 92.29, val_accuracy = 83.53, test_accuracy = 84.09 )
metrics = rbind(metrics,metricsdt_5.0)


#knn.1
library(caret)
model_knn <- knn3(outcome ~ . , train, k = 5)

preds_train_k <- predict(model_knn, train, type = "class")
confusionMatrix(preds_train_k, train$outcome)

preds_k_val <- predict(model_knn, val, type= "class")
confusionMatrix(preds_k_val, val$outcome)

preds_k_test <- predict(model_knn, test_st, type = "class")
sub_preds$outcome = preds_k_test
write.csv(sub_preds, "k.1.csv", row.names = FALSE)

metrics_k.1 = data.frame(model = "k.1" ,train_accuracy = 81.72, val_accuracy = 73.53, test_accuracy = NA )
metrics = rbind(metrics,metrics_k.1)

#knn.2
test_st$outcome = NA
dummies=dummyVars(~.,data=test_st)
dumvar=dummyVars(outcome~.,data=train)
class(train$outcome)
train_target=train$outcome
val_target=val$outcome
train1=predict(dumvar,train)
val1=predict(dumvar,val)
test1 = predict(dumvar,test_st)
library(class)

#Deciding k value for k-NN
#Experiment with various odd values of k; k={1,3,5,7,..}

# k = 1
noOfNeigh <- 1
pred_knn.2 = knn(train1, val1 ,train_target, k = noOfNeigh)
a = confusionMatrix(pred_knn.2,val_target)
a

#k = 3
noOfNeigh <- 3
pred = knn(train1, val1 ,train_target, k = noOfNeigh)
a = confusionMatrix(pred,val_target)
a

# k = 5
noOfNeigh <- 5
pred = knn(train1, val1, train_target, k = noOfNeigh)
a = confusionMatrix(pred,val_target)
a

# Speeding up knn
### Condensing the data 
#condensing the number of records to compute distances from a test record 

keep = condense(train1, train_target)


#Run the model on condensed data
pred  = knn(train1[keep,], val1, 
           train_target[keep],k=5)
a = confusionMatrix(pred,val_target)
a

#Knn with cross validation and Grid search from library caret

trctrl <- trainControl(method = "cv", number = 5)
set.seed(3333)
grid <- expand.grid(k=c(5,7,9,13,15))
knn_fit <- train(train1,train_target, method = "knn",
                 trControl=trctrl,
                 tuneGrid=grid)
plot(knn_fit)

pred = predict(knn_fit,newdata = val1)
a = confusionMatrix(pred,val_target)
a

metrics_knn_2.0 = data.frame(model = "knn_2.0" ,train_accuracy = NA, val_accuracy = 78.28, test_accuracy = NA )
metrics = rbind(metrics,metrics_knn_2.0)

#svm
x.test = predict(dummies, newdata = test_st)

x.train=predict(dumvar, newdata = train)
y.train=train$outcome
x.val = predict(dumvar, newdata = val)
y.val = val$outcome

####Classification using "e1071"####

library(e1071)

# Building the model on train data
svm_model  =  svm(x = x.train, y = y.train, type = "C-classification", kernel = "linear", cost = 10)
summary(svm_model)

#The "cost" parameter balances the trade-off between having a large margin and classifying
#all points correctly. It is important to choose it well to have good
#generalization.

# Predict on train and val using the model
pred_train  =  predict(svm_model, x.train) # x is all the input variables
pred_val=predict(svm_model,x.val)

# Build Confusion matrix
confusionMatrix(pred_train,y.train)
confusionMatrix(pred_val,y.val)

pred_test_svm =predict(svm_model,x.test)
sub_preds$outcome = pred_test_svm
write.csv(sub_preds, "svm.csv", row.names = FALSE)

metrics_svm = data.frame(model = "svm_linear" ,train_accuracy = 82.92, val_accuracy = 82.58, test_accuracy = 82.14 )
metrics = rbind(metrics,metrics_svm)

trctrl <- trainControl(method = "cv", number = 8)
set.seed(3333)
grid <- expand.grid(cost =c(0.1,0.5,0.7,1,3,5,7,9,11,13,15,17))
svm_fit <- train(train1,train_target, method = "svmLinear2",
                 trControl=trctrl,
                 tuneGrid=grid)
plot(svm_fit)

#Build SVM model with RBF kernel 
model_RBF = svm(x.train,y.train, method = "C-classification", kernel = "radial", cost = 10,
                gamma = 0.1)
summary(model_RBF)

# Predict on train and test using the model
pred_train  =  predict(model_RBF, x.train) # x is all the input variables
pred_val=predict(model_RBF,x.val)

# Build Confusion matrix
confusionMatrix(pred_train,y.train)
confusionMatrix(pred_val,y.val)

#Grid Search/Hyper-parameter tuning
prop.table(table(train$outcome))
tuneResult <- tune(svm, train.x = x.train, train.y = y.train, 
                   ranges = list(gamma = 10^(-4:0), cost = 2^(1:4)),class.weights= c("UE" = 2.32, "W" = 4.54,"FE" =3.03 ),tunecontrol=tune.control(cross=5))
print(tuneResult) 
summary(tuneResult)

#Predict model and calculate errors
tunedModel <- tuneResult$best.model;tunedModel

# Predict on train and test using the model
pred_train  =  predict(tunedModel, x.train) # x is all the input variables
pred_val=predict(tunedModel,x.val)
# Build Confusion matrix
confusionMatrix(pred_train,y.train)
confusionMatrix(pred_val,y.val)

pred_test_svm_rbf =predict(tunedModel,x.test)
sub_preds$outcome = pred_test_svm_rbf
write.csv(sub_preds, "svm_rbf.csv", row.names = FALSE)

metrics_svm_rbf = data.frame(model = "svm_rbf" ,train_accuracy = 88.52, val_accuracy = 84.24, test_accuracy = 82.54 )
metrics = rbind(metrics,metrics_svm_rbf)

## Random Forest

library(randomForest)

model_rf <- randomForest(outcome ~ . , train,ntree = 100,mtry = 9)

importance(model_rf)
varImpPlot(model_rf)

# Predict on the train data
preds_train_rf <- predict(model_rf)
confusionMatrix(preds_train_rf, train$outcome)


# Store predictions from the model
preds_val_rf <- predict(model_rf, val)
confusionMatrix(preds_val_rf, val$outcome)

preds_test_rf =predict(model_rf,test_st)
sub_preds$outcome = preds_test_rf
write.csv(sub_preds, "rf_without_grid.csv", row.names = FALSE)

metrics_rf_without_grid = data.frame(model = "rf_without_grid" ,train_accuracy = 85.95, val_accuracy = 87.62, test_accuracy = 86.79 )
metrics = rbind(metrics,metrics_rf_without_grid)

####Building  randomforest using caret

control <- trainControl(method="cv", number=10)
set.seed(1235869)
tunegrid <- expand.grid(mtry=c(1:25))
rf_gridsearch <- train(outcome ~ ., data=train, method = "rf",
                       trControl=control,
                       tuneGrid = tunegrid)


# Predict on the train data
preds_train_rf1 <- predict(rf_gridsearch)
confusionMatrix(preds_train_rf1, train$outcome)


# Store predictions from the model
preds_rf1 <- predict(rf_gridsearch, val)

confusionMatrix(preds_rf1, val$outcome)

pred_test_rf =predict(rf_gridsearch,test_st)
sub_preds$outcome = pred_test_rf
write.csv(sub_preds, "rf.csv", row.names = FALSE)

metrics_rf = data.frame(model = "rf" ,train_accuracy = 100, val_accuracy = 87.41, test_accuracy = 87.29 )
metrics = rbind(metrics,metrics_rf)

#bagged d trees

library(ipred)
set.seed(1234)
model_tree_bag <- bagging(outcome ~ . , data=train, nbagg = 10,control = rpart.control(cp = 0.01, xval = 6))

preds_train_tree_bag <- predict(model_tree_bag,train)
confusionMatrix(preds_train_tree_bag, train$outcome)

preds_tree_bag <- predict(model_tree_bag, val)
confusionMatrix(preds_tree_bag, val$outcome)

pred_test_tree_bag =predict(model_tree_bag,test_st)
sub_preds$outcome = pred_test_tree_bag
write.csv(sub_preds, "tree_bag.csv", row.names = FALSE)

metrics_tree_bag = data.frame(model = "tree_bag" ,train_accuracy = 81.95, val_accuracy = 82.28, test_accuracy = 81.34 )
metrics = rbind(metrics,metrics_tree_bag)

#stacked ensemble

train_preds_df <- data.frame(rf = preds_train_rf1, knn = preds_train_k, svm = pred_train,
                             tree = pred_Train_c5.0, tree_bag = preds_train_tree_bag, logistic = train_pred,
                             Outcome = train$outcome)
val_preds_df =  data.frame(rf = preds_rf1, knn = preds_k_val, svm = pred_val,
                           tree = pred_val_c5.0, tree_bag = preds_tree_bag, logistic = val_pred,
                           Outcome = val$outcome)
test_preds_df =  data.frame(rf = pred_test_rf, knn = preds_k_test, svm = pred_test_svm_rbf,
                           tree = pred_test_dt_c5.0, tree_bag = pred_test_tree_bag, logistic = test_pred)

stacked_model <- multinom(Outcome ~ . , data = train_preds_df)

preds_train_stack <- predict(stacked_model,train_preds_df)
confusionMatrix(preds_train_stack, train_preds_df$Outcome)

preds_stack_val <- predict(stacked_model,val_preds_df)
confusionMatrix(preds_stack_val, val_preds_df$Outcome)

preds_test_stack <- predict(stacked_model,test_preds_df)
sub_preds$outcome = preds_test_stack
write.csv(sub_preds, "stack2.csv", row.names = FALSE)
metrics_stack = data.frame(model = "stack" ,train_accuracy = 89.64, val_accuracy = 87.41, test_accuracy = 87.29 )
metrics = rbind(metrics,metrics_stack)
train_preds_df <- data.frame(knn = preds_train_k, svm = pred_train,
                             tree = pred_Train_c5.0, tree_bag = preds_train_tree_bag, logistic = train_pred,
                             Outcome = train$outcome)
val_preds_df =  data.frame(knn = preds_k_val, svm = pred_val,
                           tree = pred_val_c5.0, tree_bag = preds_tree_bag, logistic = val_pred,
                           Outcome = val$outcome)
test_preds_df =  data.frame(knn = preds_k_test, svm = pred_test_svm_rbf,
                            tree = pred_test_dt_c5.0, tree_bag = pred_test_tree_bag, logistic = test_pred)

stacked_model <- multinom(Outcome ~ . , data = train_preds_df)

preds_train_stack <- predict(stacked_model,train_preds_df)
confusionMatrix(preds_stack, train_preds_df$Outcome)

preds_stack_val <- predict(stacked_model,val_preds_df)
confusionMatrix(preds_stack_val, val_preds_df$Outcome)

preds_stack_test <- predict(stacked_model,test_preds_df)
sub_preds$outcome = preds_stack_test
write.csv(sub_preds, "stack.csv", row.names = FALSE)

metrics_stack_without_rf = data.frame(model = "stack_without_rf" ,train_accuracy = 100, val_accuracy = 84.95, test_accuracy = 84.04 )
metrics = rbind(metrics,metrics_stack_without_rf)

#ada boost

#using fastAdaboost
library(fastAdaboost)
library(adabag)
ada = boosting(outcome~.,data=train,  mfinal = 100, coeflearn = 'Freund')

ypred_train_ada = predict.boosting(ada, train)
#yhat_train_ada 
levels(train$outcome)
ypred_train_ada$class = as.factor(ypred_train_ada$class)
confusionMatrix(train$outcome,ypred_train_ada$class)

ypred_val_ada = predict.boosting(ada,val)
#yhat_test_ada
ypred_val_ada$class = as.factor(ypred_val_ada$class)
confusionMatrix(val$outcome,ypred_val_ada$class)

metrics_adaboost = data.frame(model = "ada boost" ,train_accuracy = 88.54, val_accuracy = 86.54, test_accuracy = 85.79 )
metrics = rbind(metrics,metrics_adaboost)

#grid search 

trctrl <- trainControl(method = "cv", number = 8)
library(plyr)
grid <- expand.grid(mfinal = c(100,150,200),maxdepth=7:12,coeflearn = c("Freund","Breiman","Zhu"))
set.seed(3233)
Ada_Model <- train(outcome~., data=train, method = "AdaBoost.M1",
                   trControl=trctrl,
                   tuneGrid = grid)


Ada_Model$bestTune
ada_preds <- predict(Ada_Model, val)
confusionMatrix(val$outcome, ada_preds)

ypred_test_ada = predict.boosting(ada,test_st)
ypred_test_ada$class = as.factor(ypred_test_ada$class)
sub_preds$outcome = ypred_test_ada$class
write.csv(sub_preds, "ada_boost.csv", row.names = FALSE)

metrics_adaboost_grid = data.frame(model = "ada boost_grid" ,train_accuracy = 100, val_accuracy = 88.08, test_accuracy = 86.64 )
metrics = rbind(metrics,metrics_adaboost_grid)

#xgboost

#Convert data into an object of the class "xgb.Dmatrix", which works well with the xgboost model

library(xgboost)


# Tuning an XGBoost Model with the caret package

modelLookup("xgbTree")

sampling_strategy <- trainControl(method = "cv", number = 5, verboseIter = T, allowParallel = T)

#2
param_grid <- expand.grid(.nrounds = 50, .max_depth = c(2, 4, 6), .eta = c(0.1,0.3,0.5,0.7),
                          .gamma = c(0.6, 0.3), .colsample_bytree = c(0.6, 0.8),
                          .min_child_weight=1,.subsample = c(0.8, 0.9))

xgb_tuned_model <- train(x = x.train, 
                         y = y.train, 
                         method = "xgbTree",
                         trControl = sampling_strategy,
                         metrics="Accuracy",
                         objective="multi:softprob",
                         tuneGrid = param_grid)

xgb_tuned_model$bestTun
View(xgb_tuned_model$results)

tuned_params_train_preds <- predict(xgb_tuned_model, x.train)

tuned_params_val_preds <- predict(xgb_tuned_model,x.val)

confusionMatrix(tuned_params_train_preds, y.train)

confusionMatrix(tuned_params_val_preds, y.val)

tuned_params_test_preds <- predict(xgb_tuned_model,x.test)

sub_preds$outcome = tuned_params_test_preds
write.csv(sub_preds, "xg_boost6.csv", row.names = FALSE)

metrics_xg_boost = data.frame(model = "xg boost" ,train_accuracy =94.84 , val_accuracy = 87.74, test_accuracy = 87.19 )
metrics = rbind(metrics,metrics_xg_boost)

metrics_xg_boost1 = data.frame(model = "xg boost1" ,train_accuracy =94.75 , val_accuracy = 87.99, test_accuracy = 88.09 )
metrics = rbind(metrics,metrics_xg_boost1)

write.csv(metrics,"metrics.csv", row.names= FALSE)

