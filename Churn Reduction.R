#clearing the environment
rm(list = ls())
#set working directory
setwd("E:/DS/Project/Churn Reduction")
#loading Libraries
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "e1071",
      "DataCombine", "pROC", "doSNOW", "class", "readxl","ROSE")


#install.packages if not
#lapply(x, install.packages)

#load Packages
lapply(x, require, character.only = TRUE)



#Input Train & Test Data Source
train = read.csv('Train_data.csv',header = T,na.strings = c(""," ","NA"))
test = read.csv('Test_data.csv',header = T,na.strings = c(""," ","NA"))


#############################
#EXPLORING DATA
############################

dim(train)

#structure of data or data types
str(train)  

#Summary of data 
summary(train)

#unique value of each count
apply(train, 2,function(x) length(table(x)))

#Convertiung area code as factor
train$area.code <- as.factor(train$area.code)
test$area.code <- as.factor(test$area.code)

#Considering the business needs we are removing certain variables
#Removing phone number
train$phone.number <- NULL
test$phone.number <- NULL

#Calculating distribution of dependent variable
round(prop.table(table(train$Churn))*100,2)
# False.   True. 
# 85.51   14.49 

#Our target Class is suffering from target imbalance


##############################################3
#Checking Missing data
#########################################################

sum(is.na(train))
sum(is.na(test))

#No Missing data found in Test and Train datasets

#########################################################################
#          Visualizing the data
#########################################################################

#Target class distribution 
ggplot(data = train,aes(x = Churn))+geom_bar() +  labs(y='Churn Count', title = 'Customer Churn or Not')

#Churning of customer with respect to State
ggplot(train, aes(fill=Churn, x=state)) + geom_bar(position="dodge")  + labs(title="Churning ~ State")

#Churning of customer according to Voice Mail Plan
ggplot(train, aes(fill=Churn, x=voice.mail.plan)) +  geom_bar(position="dodge") + labs(title="Churning ~ Voice Mail Plan")

#Churning of customer according to international.plan
ggplot(train, aes(fill=Churn, x=international.plan)) +  geom_bar(position="dodge") + 
  labs(title="Churning ~ international.plan")# Churning of customer according to area.code + ggplot(train, aes(fill=Churn, x=area.code)) +
  geom_bar(position="dodge") + labs(title="Churning ~ Area Code")

#Churning of customer according to area.code by international.plan
ggplot(train, aes(fill=Churn, x=area.code)) +  geom_bar(position="dodge") + facet_wrap(~international.plan)+
  labs(title="Churning ~ Area Code by International Plan")

#Churning of customer according to area.code by voicemail_plan
ggplot(train, aes(fill=Churn, x=area.code)) +  geom_bar(position="dodge") + facet_wrap(~voice.mail.plan)+
  labs(title="Churning ~ Area Code by Voice Mail Plan")

#Churn of international.plan by voice.mail.plan and Area Code
ggplot(train, aes(fill=Churn, x=international.plan)) +  geom_bar(position="dodge") + facet_wrap(area.code~voice.mail.plan)+
  labs(title="Churn of international.plan by voice.mail.plan and Area Code")

#Churn ~ international.plan by voice.mail.plan
ggplot(train, aes(fill=Churn, x=international.plan)) +  geom_bar(position="dodge") + facet_wrap(~voice.mail.plan)+
  labs(title="Churn ~ international.plan by voice.mail.plan")

#########################################################################
#                   Exploratory Data Analysis
#########################################################################

#Function for Assigning factors of var to levels
cat_to_num <- function(df){
  for(i in 1:ncol(df)){
    if(class(df[,i]) == 'factor'){
      df[,i] = factor(df[,i],labels = (1:length(levels(factor(df[,i])))))
    }
  }
  return(df)
}

#Converting Categorical to level -> factors
train = cat_to_num(train)
test = cat_to_num(test)

#all numeric var
num_index = sapply(train, is.numeric)#Fetching all the numeric index & later data
num_data = train[,num_index]#contains all numeric variable data
num_col = colnames(num_data)

#Checking for categorical features
cat_index = sapply(train,is.factor) #Fetching all the categorical index & later data
cat_data = train[,cat_index]#contains all categoric variable data including Churn
cat_col = colnames(cat_data)

################################################################
#               Outlier Analysis
################################################################

for (i in 1:length(num_col))
{
  assign(paste0("gn",i),
         ggplot(aes_string(y = (num_col[i]), x = 'Churn'),data = train) +
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="blue", fill = "skyblue",
                        outlier.shape=18,outlier.size=1, notch=FALSE) +
           labs(y=num_col[i],x="Churn")+
           ggtitle(paste("Box plot of responded for",num_col[i])))
}

# Plotting plots together
gridExtra::grid.arrange(gn1,gn2,gn3,ncol=3)
gridExtra::grid.arrange(gn4,gn5,gn6,ncol=3)
gridExtra::grid.arrange(gn7,gn8,gn9,ncol=3)
gridExtra::grid.arrange(gn10,gn11,gn12,ncol=3)
gridExtra::grid.arrange(gn13,gn14,gn15,ncol=3)

#Removing oulier by replacing with NA and then impute
for(i in num_col){
  print(i)
  outv = train[,i][train[,i] %in% boxplot.stats(train[,i])$out]
  print(length(outv))
  train[,i][train[,i] %in% outv] = NA
}
 #imputation with Knn imputation
 train = knnImputation(train, k = 3)

#Checking all the missing values
sum(is.na(train))


################################################################
#               Feacture Selection
################################################################

#Here we will use corrgram library to find corelation

##Correlation plot
library(corrgram)

corrgram(train[,num_index],
         order = F,  #we don't want to reorder
         upper.panel=panel.pie,
         lower.panel=panel.shade,
         text.panel=panel.txt,
         main = 'CORRELATION PLOT')
#We can see var the highly corrrelated var in plot marked dark blue.
#Dark blue color means highly positive cor related

##---------Chi Square Analysis----------------------------------

for(i in cat_col){
  print(names(cat_data[i]))
  print((chisq.test(table(cat_data$Churn,cat_data[,i])))[3])  #printing only pvalue
}

##-----------------Removing Highly Corelated and Independent var----------------------
train = subset(train,select= -c(state,total.day.charge,total.eve.charge,
                                    total.night.charge,total.intl.charge))

test = subset(test,select= -c(state,total.day.charge,total.eve.charge,
                                total.night.charge,total.intl.charge))


################################################################
#               Feature Scaling
################################################################

#all numeric var
num_index = sapply(train, is.numeric)
num_data = train[,num_index]
num_col = colnames(num_data) #storing all the column name


#Checking Data of Continuous Variable

################  Histogram   ##################
hist(train$total.day.calls)
hist(train$total.day.minutes)
hist(train$account.length)
#Most of the data is uniformally distributed


#Using data Standardization
for(i in num_col){
   print(i)
   train[,i] = (train[,i] - mean(train[,i]))/sd(train[,i])
   test[,i] = (test[,i] - mean(test[,i]))/sd(test[,i])
}

################################################################
#               Sampling of Data
################################################################

# #Divide data into train and test using stratified sampling method

# library(caret)
set.seed(101)
split_index = createDataPartition(train$Churn, p = 0.66, list = FALSE)
trainset = train[split_index,]
validation_set  = train[-split_index,]

#Checking Train Set Target Class
table(trainset$Churn)
# 1    2 
# 1881  319 


# #Our class is Imbalanced 
# Synthetic Over Sampling the minority class & Under Sampling Majority Class to have a good Training Set



trainset <- ROSE(Churn~.,data = trainset,p = 0.5,seed = 101)$data     

table(trainset$Churn) # 1 = 1101  2 = 1099


#Removing All the custom variable from memory
rmExcept(c("test","train","train","test","trainset","validation_set"))

###########################################################################################
#                           Model Development
###########################################################################################
# #function for calculating the FNR,FPR,Accuracy
calc <- function(cm){
  TN = cm[1,1]
  FP = cm[1,2]
  FN = cm[2,1]
  TP = cm[2,2]
  # #calculations
  print(paste0('Accuracy :- ',((TN+TP)/(TN+TP+FN+FP))*100))
  print(paste0('FNR :- ',((FN)/(TP+FN))*100))
  print(paste0('FPR :- ',((FP)/(TN+FP))*100))
  print(paste0('precision :-  ',((TP)/(TP+FP))*100)) 
  print(paste0('recall//TPR :-  ',((TP)/(TP+FP))*100))
  print(paste0('Sensitivity :-  ',((TP)/(TP+FN))*100))
  print(paste0('Specificity :-  ',((TN)/(TN+FP))*100))
  plot(cm)
}

### ##----------------------- Random Forest ----------------------- ## ###
# library(randomForest)

set.seed(101)
RF_model = randomForest(Churn ~ ., trainset,ntree= 500,importance=T,type='class')

#Predict test data using random forest model
RF_Predictions = predict(RF_model, validation_set[,-15])

##Evaluate the performance of classification model
cm_RF = table(validation_set$Churn,RF_Predictions)
confusionMatrix(cm_RF)
calc(cm_RF)
plot(RF_model)

# [1] "Accuracy :- 81.8181818181818"
# [1] "FNR :- 34.1463414634146"
# [1] "FPR :- 15.4798761609907"
# [1] "FPR :- 15.4798761609907"
# [1] "precision :-  41.8604651162791"
# [1] "recall//TPR :-  41.8604651162791"
# [1] "Sensitivity :-  65.8536585365854"
# [1] "Specificity :-  84.5201238390093"

### ##----------------------- LOGISTIC REGRESSION ----------------------- ## ###
set.seed(101)
logit_model = glm(Churn ~., data = trainset, family =binomial(link="logit")) 
summary(logit_model)
#Prediction
logit_pred = predict(logit_model,newdata = validation_set[,-15],type = 'response')

#Converting Prob to number or class
logit_pred = ifelse(logit_pred > 0.5, 2,1)
##Evaluate the performance of classification model
cm_logit = table(validation_set$Churn, logit_pred)
confusionMatrix(cm_logit)
calc(cm_logit)
plot(logit_model)


# Result on validaton set
# [1] "Accuracy :- 78.1994704324801"
# [1] "FNR :- 28.6585365853659"
# [1] "FPR :- 20.6398348813209"
# [1] "precision :-  36.9085173501577"
# [1] "recall//TPR :-  36.9085173501577"
# [1] "Sensitivity :-  71.3414634146341"
# [1] "Specificity :-  79.360165118679"


### ##----------------------- KNN----------------------- ## ###

##Predicting Test data

knn_Pred = knn(train = trainset[,1:14],test = validation_set[,1:14],cl = trainset$Churn, k = 5,prob = T)
#Confusion matrix
cm_knn = table(validation_set$Churn,knn_Pred)
confusionMatrix(cm_knn)
calc(cm_knn)

# Result on validaton set
# [1] "Accuracy :- 78.8172992056487"
# [1] "FNR :- 46.3414634146341"
# [1] "FPR :- 16.9246646026832"
# [1] "FPR :- 16.9246646026832"
# [1] "precision :-  34.9206349206349"
# [1] "recall//TPR :-  34.9206349206349"
# [1] "Sensitivity :-  53.6585365853659"
# [1] "Specificity :-  83.0753353973168"

### ##----------------------- Naive Bayes ----------------------- ## ###

# library(e1071) #lib for Naive bayes
set.seed(101)
#Model Development and Training
naive_model = naiveBayes(Churn ~., data = trainset, type = 'class')
#prediction
naive_pred = predict(naive_model,validation_set[,1:14])

#Confusion matrix
cm_naive = table(validation_set[,15],naive_pred)
confusionMatrix(cm_naive)
calc(cm_naive)

# Result on validaton set
 
# [1] "Accuracy :- 76.081200353045"
# [1] "FNR :- 35.9756097560976"
# [1] "FPR :- 21.8782249742002"
# [1] "precision :-  33.1230283911672"
# [1] "recall//TPR :-  33.1230283911672"
# [1] "Sensitivity :-  64.0243902439024"
# [1] "Specificity :-  78.1217750257998"
####################################################################################################

#Reduction customer churn is important because cost of acquiring a new customer is higher then retaining the older one."

## From above statement it's clear that Cost matters alot. 
## We are using default threshold cutoff here for Churning and Not Churn  

# # COnsidering the accuracy, FNR, FPR etc Random Forest is the best model
# Hence we chose Random FOrest as our final model.

rmExcept(c("train","test","train","test","trainset","validation_set","calc"))

###########################################################################################
#  #             Model Selection - Random Forest
###########################################################################################

set.seed(101)
final_model = randomForest(Churn ~ ., trainset,ntree= 500,importance=T,type='class')
final_validation_pred = predict(final_model,validation_set[,-15])
cm_final_valid = table(validation_set[,15],final_validation_pred)
confusionMatrix(cm_final_valid)
calc(cm_final_valid)
#Result on validation set after parameter tuning

# [1] "Accuracy :- 81.8181818181818"
# [1] "FNR :- 34.1463414634146"
# [1] "FPR :- 15.4798761609907"
# [1] "precision :-  41.8604651162791"
# [1] "recall//TPR :-  41.8604651162791"
# [1] "Sensitivity :-  65.8536585365854"
# [1] "Specificity :-  84.5201238390093"



###################################################################################
# #       Final Prediction On test Data set 
###################################################################################

rmExcept(c("final_model","train","test","train","test","calc"))

set.seed(101)
final_test_pred = predict(final_model,test[,-15])
cm_final_test = table(test[,15],final_test_pred)
confusionMatrix(cm_final_test)
calc(cm_final_test)

# #Final Test Prediction
# [1] "Accuracy :- 85.9028194361128"
# [1] "FNR :- 15.625"
# [1] "FPR :- 13.8600138600139"
# [1] "FPR :- 13.8600138600139"
# [1] "precision :-  48.586118251928"
# [1] "recall//TPR :-  48.586118251928"
# [1] "Sensitivity :-  84.375"
# [1] "Specificity :-  86.1399861399861"

#Plotting ROC curve and Calculate AUC metric
# library(pROC)
finalPredictionwithProb <-predict(final_model,test[,-15],type = 'prob')
auc <- auc(test$Churn,finalPredictionwithProb[,2])
auc
# # AUC = 91.74
plot(roc(test$Churn,finalPredictionwithProb[,2]))

################################################################################################################################
#  #            Saving output to file ( For re run uncomment the code (ctrl+shift+c))
################################################################################################################################

test$predicted_output <- final_test_pred
test$predicted_output <- gsub(1,"False",test$predicted_output)
test$predicted_output <- gsub(2,"True",test$predicted_output)

#Phonenumber and Churning class and probab
submit <- data.frame(test$state,
                     test$area.code,
                     test$international.plan,
                     test$voice.mail.plan,
                     test$phone.number,
                     test$predicted_output,
                     finalPredictionwithProb[,1],
                     finalPredictionwithProb[,2])

colnames(submit) <- c("State","Area Code","International Plan","Voice Mail Plan","phone.number",
                      "Predicted_Output","Probability_of_False","Probability_of_True")

write.csv(submit,file = 'Churn_Class_Prob.csv',row.names = F)
rm(list = ls())
