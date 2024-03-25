#Import library
library(dplyr)
library(rpart)
library(rpart.plot)
library(Metrics)
library(mlr)
library(ggplot2)
library(plotly)

triageData <- read.csv('triage.csv')
View(triageData)
#Remove duplicated or unimportant column/attribute
triageData = subset(triageData, select = -c(X))

#Correct the datatype of the attribute from numerical/character to categorical/factor
triageData$gender <- as.factor(triageData$gender)
triageData$chest.pain.type <- as.factor(triageData$chest.pain.type)
triageData$exercise.angina <- as.factor(triageData$exercise.angina)
triageData$hypertension <- as.factor(triageData$hypertension)
triageData$heart_disease <- as.factor(triageData$heart_disease)
triageData$Residence_type <- as.factor(triageData$Residence_type)
triageData$smoking_status <- as.factor(triageData$smoking_status)
triageData$triage <- as.factor(triageData$triage)

#Handling Missing Values and Outliers
triageData$gender = ifelse(triageData$gender==0, "Female", "Male")
triageData$gender <- as.factor(triageData$gender)
summary(triageData[ triageData$smoking_status != "Unknown", , drop=FALSE])
triageData <- triageData[ triageData$smoking_status != "Unknown", , drop=FALSE]; triageData$smoking_status <- factor(triageData$smoking_status); summary(triageData)
triageData <- triageData%>%
  mutate(triage=case_when(
    .$triage=="green" ~ 1,
    .$triage=="blue" ~ 2,
    .$triage=="yellow" ~ 3,
    .$triage=="orange" ~ 4,
    .$triage=="red" ~ 5,
  ))



#Triage is a multi-classification dataset with inscript 5 factors, thus decision tree ML is deployed. 
model <- rpart(triage~., cp=0.001,maxdepth=5,minbucket=5,method='class',data=triageData)

#Multi-param tuning
model_tune <- rpart(triage~., cp=0.001,maxdepth=9,minsplit=1,method='class',data=triageData)

# cp - complexity parameter
# maxdepth - max tree depth
# minbucket - min number of obs in leaf nodes
# method - return classification 
options(repr.plot.width = 6, repr.plot.height = 6)
prp(model, space=4,split.cex=1.2,nn.border.col=0)

options(repr.plot.width = 6, repr.plot.height = 6)
prp(model_tune, space=4,split.cex=1.2,nn.border.col=0)

#Make prediction onto the dataset 
train_preds <- predict(model, newdata=triageData, type="class")
train_preds_tune <- predict(model_tune, newdata=triageData, type="class")

#Evaluation metric 
confusionMatrix(factor(train_preds), factor(triageData$triage))
confusionMatrix(factor(train_preds_tune), factor(triageData$triage))

multiclass.roc(triageData$triage, train_preds)

#Cross-validation [k-fold CV vs LOOCV]
#k-fold CV=10
train_control_cv<- trainControl(method="cv",number=10)
#LOOCV
train_control_loocv <- trainControl(method = "LOOCV")

#set required parametes for the model type 
tune_grid = expand.grid(cp=c(0.001))

# Use the train() function to create the model
validated_tree_cv <- train(triage~.,method='rpart',data=triageData,maxdepth=5,minbucket=5, trControl=train_control_cv,tuneGrid=tune_grid)
validated_tree_loocv <- train(triage~.,method='rpart',data=triageData,maxdepth=5,minbucket=5, trControl=train_control_loocv,tuneGrid=tune_grid)

#summary of the model
validated_tree_cv
validated_tree_loocv

#hyperparameter tuning for Decision Tree 
set.seed(123)
train_data <- triageData %>% sample_frac(0.8)
test_data <- triageData %>% anti_join(train_data)

# Hyperparameter Tuning training with mlr
getParamSet("classif.rpart")
d.tree.mlr <- makeClassifTask(data=train_data, target="triage")

# Tweaking multiple hyperparameters
param_grid_multi <- makeParamSet( makeDiscreteParam("maxdepth", values=1:30),makeNumericParam("cp", lower = 0.001, upper = 0.01),makeDiscreteParam("minsplit", values=1:10)
)
dt_tuneparam_multi <- tuneParams(learner='classif.rpart', 
                                 task=d.tree.mlr, 
                                 resampling = resample,
                                 measures = measure,
                                 par.set=param_grid_multi, 
                                 control=control_grid, 
                                 show.info = TRUE)

# Extracting best Parameters from Multi Search
best_parameters_multi = setHyperPars(
  makeLearner("classif.rpart", predict.type = "prob"), 
  par.vals = dt_tuneparam_multi$x
)

best_model_multi = train(best_parameters_multi, d.tree.mlr)


# Predicting the best Model
results <- predict(best_model_multi, task = d.tree.mlr.test)$data

accuracy(results$truth, results$response)
confusionMatrix(factor(results), factor(triageData$triage))

# Extracting results from multigrid
result_hyperparam.multi <- generateHyperParsEffectData(dt_tuneparam_multi, partial.dep = TRUE)

# Sampling just for visualization
result_sample <- result_hyperparam.multi$data %>%
  sample_n(300)


hyperparam.plot <- plot_ly(result_sample, 
                           x = ~cp, 
                           y = ~maxdepth, 
                           z = ~minsplit,
                           marker = list(color = ~acc.test.mean,  colorscale = list(c(0, 1), c("darkred", "darkgreen")), showscale = TRUE))
hyperparam.plot <- hyperparam.plot %>% add_markers()
hyperparam.plot


## KNN

#import library
library(dplyr)

triageData <- read.csv('triage.csv')
View(triageData)
#Remove duplicated or unimportant column/attribute
triageData = subset(triageData, select = -c(X))
summary(triageData)

#Handling Missing Values and Outliers
triageData <- na.omit(triageData)
triageData <- triageData[ triageData$smoking_status != "Unknown", , drop=FALSE]; triageData$smoking_status <- factor(triageData$smoking_status)

#Convert all categorical data to numerical data 
#as KNN works best with numerical values
triageData <- triageData%>%
  mutate(Residence_type=case_when(
    .$Residence_type=="Urban" ~ 1,
    .$Residence_type=="Rural" ~ 2,
  ))
triageData <- triageData%>%
  mutate(smoking_status=case_when(
    .$smoking_status=="never smoked" ~ 1,
    .$smoking_status=="formerly smoked" ~ 2,
    .$smoking_status=="smokes" ~ 3,
  ))

triageData <- triageData%>%
  mutate(triage=case_when(
    .$triage=="green" ~ 1,
    .$triage=="blue" ~ 2,
    .$triage=="yellow" ~ 3,
    .$triage=="orange" ~ 4,
    .$triage=="red" ~ 5,
  ))

#Data normalization 
#define Min-Max normalization function
normalization <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

#normalize all data except the target attrib --> Triage
triageData_norm <- as.data.frame(lapply(triageData[1:16], normalization))

#add back Triage column
triageData_norm$triage <- triageData$triage

#ready for KNN model training
set.seed(123)
library(class)
# Split the data into a training set and a test set
train_split <- sample(1:nrow(triageData_norm),size=nrow(triageData_norm)*0.7,replace = FALSE)
train_data <- triageData[train_split, ]
test_data <- triageData[-train_split, ]

#finalize label value for the knn model training
train_label <- triageData[train_split,17]
test_label  <- triageData[-train_split,17]

#find a suitable "K" value to kick start!
#lets get a random value of k=20 before we proceed to the list of k values

knn_model <- knn(train=train_data, test=test_data, cl=train_label, k=20)

#install.packages('caret')
library(caret)
#confusion matrix of KNN
confusionMatrix(table(knn_model ,test_label))

#elbow rule
i=1
y=1
for (i in 1:30){
  x <- knn(train=train_data, test=test_data, cl=train_label, k=i)
  y[i] <- 100 * sum(test_label == x)/NROW(test_label)
  k=i
  cat(k,'=',y[i],'
')
}
#Accuracy plot
plot(y, type="b", xlab="Value of K",ylab="Accuracy (%)")

## SVM 

#import library
library(dplyr)

triageData <- read.csv('triage.csv')
#Remove duplicated or unimportant column/attribute
triageData = subset(triageData, select = -c(X))
summary(triageData)

triageData <- triageData[ triageData$smoking_status != "Unknown", , drop=FALSE]; triageData$smoking_status <- factor(triageData$smoking_status)

#Convert all categorical data to numerical data 
triageData <- triageData%>%
  mutate(Residence_type=case_when(
    .$Residence_type=="Urban" ~ 1,
    .$Residence_type=="Rural" ~ 2,
  ))
triageData <- triageData%>%
  mutate(smoking_status=case_when(
    .$smoking_status=="never smoked" ~ 1,
    .$smoking_status=="formerly smoked" ~ 2,
    .$smoking_status=="smokes" ~ 3,
  ))

#convert the target attributes into categorical data for classification
triageData$triage <- as.factor(triageData$triage)

#Handling Missing Values and Outliers
triageData <- na.omit(triageData)

#Data normalization 
#define Min-Max normalization function
normalization <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

triageData_norm <- as.data.frame(lapply(triageData[1:16], normalization))

#add back Triage column
triageData_norm$triage <- triageData$triage


#split test and train data with ratio 8:2
trainIndex <- createDataPartition(y = triageData_norm$triage, p = .8, list = FALSE)
train_data <- triageData_norm[trainIndex,0:16]
train_label <- triageData_norm[trainIndex,17]
test_data <- triageData_norm[-trainIndex,0:16]
test_label <- triageData_norm[-trainIndex,17]


#SVM 
library(caret)
model <- train(x = train_data, y = train_label, method = "svmLinear")
predictions <- predict(model, test_data)
confusionMatrix(predictions, test_label)










