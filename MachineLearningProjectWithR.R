#R project version 2
df <- read.csv("/Users/yarenuyaroglu/Desktop/R Programming/heart.csv")
head(df)
summary(df)

set.seed(155)#To shuffle the dataset
df <- df[sample(nrow(df)),]
head(df)
#Replacing null values with the mean value.
sum(is.na(df))
df <- sapply(df, function(x) ifelse(is.na(x), mean(x, na.rm = TRUE), x))
#If-else statement: if the test is true, return 'yes' (value to be returned in this condition), otherwise return 'no' (value to be returned in the other condition)."
sum(is.na(df))


#Data visualization
if (!require(corrplot)) install.packages("corrplot")
library(corrplot)
correlation_matrix <- cor(df)
corrplot(
  correlation_matrix,
  method = "color",    
  type = "lower",      
  order = "hclust",   
  tl.col = "black",   
  tl.srt = 45,         
  addCoef.col = "black", 
  number.cex = 0.7)    

library(ggplot2) 
df <- as.data.frame(df) 
#The error occurred due to either the object not being a matrix or not being convertible to a matrix. To resolve this error:
ggplot(df, aes(x = target)) +
  geom_bar(fill= "skyblue") +
  labs(title = "Count of Target", x = "Target", y = "Count")

#--------------New Feature extraction ------- 
df$cholesterol_hdl_ratio <- df$chol / df$thalach
df$smoking_status <- ifelse(df$trestbps > 130 &
                            df$chol > 200 &
                            df$thalach > 80 & df$target == 1,
                            1 , 0) #1 current smoker, 0  non-smoking)
#To make the target variable the last column.
duplicated_target <- df[14]
df$target <- NULL
df$target <- duplicated_target
head(df)
correlation_matrix <- cor(df)
corrplot(
  correlation_matrix,
  method = "color",    
  type = "lower",      
  order = "hclust",   
  tl.col = "black",   
  tl.srt = 45,         
  addCoef.col = "black", 
  number.cex = 0.7) 

#----------------Normalization -----
#MIN_MAX Scaler:
min_max_scaler <- function(x){
  return((x-min(x))/(max(x)-min(x)))
}
df<- as.data.frame(lapply(df, min_max_scaler))
min_max_normalized_df <- df #duplicated with dataFrame
head(df)

#other normalization method

#MaxAbs Scaler:
maxAbs_scaler<- function(x){
  return(x / max(abs(x)))
}
#df<- as.data.frame(lapply(df, min_max_scaler))
maxAbs_normalized_df <- df
maxAbs_normalized_df[,1:15] <- sapply(df[,1:15],maxAbs_scaler)
head(maxAbs_normalized_df)


#VISUALIZATION FOR THE DETECTION OF OUTLIER DATA
boxplot(df)
par(mfrow = c(1, 2))  
hist(df$chol, col = "red", main = "Histogram")
boxplot(df$chol, main = "Boxplot", col = "orange")

hist(df$trestbps, col = "yellow" , main = "Histogram")
boxplot(df$trestbps, main = "Boxplot", col = "pink")

hist(df$thalach, col = "purple", main = "Histogram")
boxplot(df$thalach, main = "Boxplot", col = "blue")
par(mfrow = c(1,1))
#-------------------------------------------

#THE FUNCTION THAT FINDS THE BOUNDARIES
replace_outliers <- function(df,column_name){
  column_data <- df[[column_name]]
  Q1<- quantile(column_data,0.25)
  Q3<- quantile(column_data,0.75)
  IQR_val<- IQR(column_data)
  lower_boundry <- Q1 - 1.5* IQR_val
  upper_boundry <- Q3 + 1.5* IQR_val
  outliers <- column_data > upper_boundry | column_data < lower_boundry
  max_non_outlier <- max(column_data[!outliers], na.rm = TRUE)
  column_data[outliers] <- max_non_outlier
  df[[column_name]] <- column_data
  return(df)
}
#to get rid of contrary data by applying this function to all columns
df <- replace_outliers(df, 'chol')
df <- replace_outliers(df, 'trestbps')
df <- replace_outliers(df, 'thalach')
df <- replace_outliers(df, 'fbs')
df <- replace_outliers(df, 'oldpeak')
df <- replace_outliers(df, 'ca')
df <- replace_outliers(df, 'thal')
df <- replace_outliers(df, 'cholesterol_hdl_ratio')
df <- replace_outliers(df, 'smoking_status')
head(df)

boxplot(df$chol, main = "Boxplot", col = "orange")
boxplot(df$trestbps, main = "Boxplot", col = "pink")
boxplot(df$thalach, main = "Boxplot", col = "blue")
boxplot(df)
head(df)


#------------------TEST TRAIN VALIDATION partition----------------------

if (!require(caret)) install.packages("caret")
library(caret)

# Splitting the data into 70% train,15% test, 15% validation
set.seed(100)  # Seed for reproducibility

splitIndex_train <- createDataPartition(df$target, p = 0.7, list = FALSE)
splitIndex_test <- createDataPartition(df$target[-splitIndex_train], p = 0.5, list = FALSE)
# Create Train, Test and Validation sets
train_data <- df[splitIndex_train, ]
test_validation_data <- df[-splitIndex_train, ]
test_data <- test_validation_data[splitIndex_test, ]
validation_data <- test_validation_data[-splitIndex_test, ]

# Converting a target variable to a factor type
train_data$target <- as.factor(train_data$target)
test_data$target <- as.factor(test_data$target)
validation_data$target <- as.factor(validation_data$target)


#to import all necessary packages
if (!require(caret)) install.packages("caret")
if (!require(randomForest)) install.packages("randomForest")
if (!require(keras)) install.packages("keras")
if (!require(xgboost)) install.packages("xgboost")
if (!require(class)) install.packages("class")
if (!require(e1071)) install.packages("e1071")
if (!require(magrittr)) install.packages("magrittr")

library(ggplot2)
library(magrittr)
library(caret)
library(randomForest)
library(keras)
library(xgboost)
library(class)
library(e1071)

#CLASSIFICATION 


# Random Forest 
rf_model <- randomForest(target ~ ., data = train_data, ntree = 100)
# Perform the Test Set
predicted_labels_rf_test <- predict(rf_model, newdata = test_data)
confusion_matrix_rf_test <- table(predicted_labels_rf_test, test_data$target)
confusionMatrix(predicted_labels_rf_test, test_data$target)
accuracy_rf_test <- sum(diag(confusion_matrix_rf_test)) / sum(confusion_matrix_rf_test)

# Perform the Validation Set
predicted_labels_rf_validation <- predict(rf_model, newdata = validation_data)
confusion_matrix_rf_validation <- table(predicted_labels_rf_validation, validation_data$target)
confusionMatrix(predicted_labels_rf_validation, validation_data$target)
accuracy_rf_validation <- sum(diag(confusion_matrix_rf_validation)) / sum(confusion_matrix_rf_validation)


#Deep Learning Method
#multiple Layer with keras
#an R package that provides an interface to Python
reticulate::use_python("/Library/Frameworks/Python.framework/Versions/3.11/bin/python3") #actual path of the Python application
#defines a sequential neural network model using Keras
model_keras <- keras_model_sequential() %>%
  layer_dense(units = 128, activation = 'relu', input_shape = ncol(df[,16])) %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 1, activation = 'sigmoid')

#compile the model
model_keras %>% compile(
  optimizer = 'adam',
  loss = 'binary_crossentropy',
  metrics = c('accuracy')
)

# fit the model
history <- model_keras %>% fit(
  x = as.matrix(train_data[, !(names(train_data) %in% c("target"))]),
  y = as.numeric(train_data$target) - 1,  # 0 ve 1 sınıfları için
  epochs = 10,
  batch_size = 32,
  validation_split = 0.2
)

#eveluate the model in test set
evaluate_result <- model_keras %>% evaluate(
  x = as.matrix(test_data[, !(names(test_data) %in% c("target"))]),
  y = as.numeric(test_data$target) - 1
)
accuracy_keras_test <- evaluate_result["accuracy"]
print(paste("Keras Model : Accuracy on Test Set:", accuracy_keras_test))

#evaluate the model in validation set
evaluate_result_validation <- model_keras %>% evaluate(
  x = as.matrix(validation_data[, !(names(validation_data) %in% c("target"))]),
  y = as.numeric(validation_data$target) - 1
)
accuracy_keras_validation <- evaluate_result_validation["accuracy"]
print(paste("Keras Model : Accuracy on Validation Set:", accuracy_keras_validation))

#------XGBOOST
# Train to xgboost model
xgb_model <- xgboost(
  data = as.matrix(train_data[, !(names(train_data) %in% c("target"))]),
  label = as.numeric(train_data$target) - 1,
  nrounds = 50, #train 50 laps
  objective = "binary:logistic",  #binary classification is performed
  eval_metric = "error",
  verbose = 0
)
# Predictions
predicted_labels_xgb_test <- predict(xgb_model, as.matrix(test_data[, !(names(test_data) %in% c("target"))]), iteration_range = c(0, xgb_model$best_iteration))
predicted_labels_xgb_test <- ifelse(predicted_labels_xgb_test > 0.5, 1, 0)
confusion_matrix_xgb_test <- table(predicted_labels_xgb_test, test_data$target)
confusionMatrix(factor(predicted_labels_xgb_test), test_data$target)

accuracy_xgb_test <- sum(diag(confusion_matrix_xgb_test)) / sum(confusion_matrix_xgb_test)
print(paste("XGBoost MODEL : Test Accuracy:", accuracy_xgb_test))

predicted_labels_xgb_validation <- predict(xgb_model, as.matrix(validation_data[, !(names(validation_data) %in% c("target"))]), iteration_range = c(0, xgb_model$best_iteration))
predicted_labels_xgb_validation <- ifelse(predicted_labels_xgb_validation > 0.5, 1, 0)
confusion_matrix_xgb_validation <- table(predicted_labels_xgb_validation, validation_data$target)
confusionMatrix(factor(predicted_labels_xgb_validation), validation_data$target)

accuracy_xgb_validation <- sum(diag(confusion_matrix_xgb_validation)) / sum(confusion_matrix_xgb_validation)
print(paste("XGBoost MODEL : Validation Accuracy:", accuracy_xgb_validation))

#KNN
# create a grid for k values
k_values <- seq(1, 50, by = 2)

# Create variables to store the best k and its truth value
best_k_test <- NULL
best_accuracy_test <- 0
best_k_validation <- NULL
best_accuracy_validation <- 0

#Experimenting with different k values with the for loop
for (k in k_values) {
# Evaluation on KNN model training and test set
  knn_model_test <- knn(train = as.matrix(train_data[, !(names(train_data) %in% c("target"))]),
                        test = as.matrix(test_data[, !(names(test_data) %in% c("target"))]),
                        cl = train_data$target,
                        k = k)
  confusion_matrix_knn_test <- table(knn_model_test, test_data$target)
  accuracy_knn_test <- sum(diag(confusion_matrix_knn_test)) / sum(confusion_matrix_knn_test)
  #Update the best k value and accuracy on the test set
  if (accuracy_knn_test > best_accuracy_test) {
    best_k_test <- k
    best_accuracy_test <- accuracy_knn_test
  }
  
#Evaluation on KNN model training and validation set
  knn_model_validation <- knn(train = as.matrix(train_data[, !(names(train_data) %in% c("target"))]),
                              test = as.matrix(validation_data[, !(names(validation_data) %in% c("target"))]),
                              cl = train_data$target,
                              k = k)
  
  confusion_matrix_knn_validation <- table(knn_model_validation, validation_data$target)
  accuracy_knn_validation <- sum(diag(confusion_matrix_knn_validation)) / sum(confusion_matrix_knn_validation)
  #Update the best k value and accuracy on the validation set
  if (accuracy_knn_validation > best_accuracy_validation) {
    best_k_validation <- k
    best_accuracy_validation <- accuracy_knn_validation
  }
}

print("For Test Set:")
print(paste("knn model test: best k value:", best_k_test))
print(paste("knn model test: best accuracy:", best_accuracy_test))

print("For Validation Set:")
print(paste("knn model val: best k value:", best_k_validation))
print(paste("knn model val: best accuracy:", best_accuracy_validation))


#LOGISTIC REGRESYON 
logistic_model <- glm(target ~ ., data = train_data, family = "binomial")

predicted_probabilities_test <- predict(logistic_model, newdata = test_data, type = "response")
predicted_labels_logistic_test <- ifelse(predicted_probabilities_test > 0.5, 1, 0)
confusion_matrix_logistic_test <- table(predicted_labels_logistic_test, test_data$target)

print("Logistic Regression Test Confusion Matrix:")
print(confusion_matrix_logistic_test)

accuracy_logistic_test <- sum(diag(confusion_matrix_logistic_test)) / sum(confusion_matrix_logistic_test)
print(paste("Logistic Regression Test Accuracy:", accuracy_logistic_test))

predicted_probabilities_validation <- predict(logistic_model, newdata = validation_data, type = "response")
predicted_labels_logistic_validation <- ifelse(predicted_probabilities_validation > 0.5, 1, 0)
confusion_matrix_logistic_validation <- table(predicted_labels_logistic_validation, validation_data$target)

print("nLogistic Regression Validation Confusion Matrix:")
print(confusion_matrix_logistic_validation)

accuracy_logistic_validation <- sum(diag(confusion_matrix_logistic_validation)) / sum(confusion_matrix_logistic_validation)
print(paste("Logistic Regression Validation Accuracy:", accuracy_logistic_validation))

#SVM 

svm_model <- svm(target ~ . - fbs - smoking_status, data = train_data, kernel = "radial", probability = TRUE)

tuned_svm_model <- tune(svm, train.x = train_data[, -ncol(train_data)], train.y = train_data$target, 
                        kernel = "radial", ranges = list(cost = c(0.1, 1, 10), gamma = c(0.01, 0.1, 1)))

#Choosing the best model
best_svm_model <- tuned_svm_model$best.model

predictions_test <- predict(best_svm_model, newdata = test_data[, -ncol(test_data)])
conf_matrix_test <- confusionMatrix(predictions_test, test_data$target)
print("SVM Test Set Confusion Matrix:")
print(conf_matrix_test)

accuracy_svm_test <- conf_matrix_test$overall["Accuracy"]
print(paste("SVM Test Seti Accuracy:", accuracy_svm_test))

# Performance Evaluation For the Validation Set
predictions_validation <- predict(best_svm_model, newdata = validation_data[, -ncol(validation_data)])
conf_matrix_validation <- confusionMatrix(predictions_validation, validation_data$target)
print("SVM Validation Seti Confusion Matrix:")
print(conf_matrix_validation)

accuracy_svm_validation <- conf_matrix_validation$overall["Accuracy"]
print(paste("SVM Validation Seti Accuracy:", accuracy_svm_validation))

# NAIVE BAYES
nb_model <- naiveBayes(target ~ ., data = train_data)

predictions_nb_test <- predict(nb_model, newdata = test_data[, -ncol(test_data)])
conf_matrix_nb_test <- confusionMatrix(predictions_nb_test, test_data$target)
accuracy_nb_test <- conf_matrix_nb_test$overall["Accuracy"]
print(paste("Naive Bayes Test Seti Accuracy:", accuracy_nb_test))

predictions_nb_validation <- predict(nb_model, newdata = validation_data[, -ncol(validation_data)])
conf_matrix_nb_validation <- confusionMatrix(predictions_nb_validation, validation_data$target)
print("Naive Bayes Validation Seti Confusion Matrix:")
print(conf_matrix_nb_validation)

accuracy_nb_validation <- conf_matrix_nb_validation$overall["Accuracy"]
print(paste("Naive Bayes Validation Seti Accuracy:", accuracy_nb_validation))



# Create a data frame with model names and accuracies
model_names <- c("Random Forest", "Multiple Layer", "XGBoost", "k-Nearest Neighbors", "Logistic Regression", "SVM", "Naive Bayes")
accuracies_test <- c(accuracy_rf_test, accuracy_keras_test, accuracy_xgb_test, accuracy_knn_test, accuracy_logistic_test, accuracy_svm_test, accuracy_nb_test)
accuracies_validation <- c(accuracy_rf_validation, accuracy_keras_validation, accuracy_xgb_validation, accuracy_knn_validation, accuracy_logistic_validation, accuracy_svm_validation, accuracy_nb_validation)

accuracy_table_test <- data.frame(Model = model_names, Accuracy = accuracies_test)

# Print the table
print(accuracy_table_test)

# Oluşturulan accuracy_table'a göre çubuk grafiği oluştur
ggplot(accuracy_table_test, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_minimal() +
  labs(title = "Model Accuracy TEST Comparison",
       x = "Model",
       y = "Accuracy")


accuracy_table_val <- data.frame(Model = model_names, Accuracy = accuracies_validation)

# Print the table
print(accuracy_table_val)
# Create a bar graph according to the generated accuracy_table
ggplot(accuracy_table_val, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_minimal() +
  labs(title = "Model Accuracy VALIDATION Comparison",
       x = "Model",
       y = "Accuracy")

# Combining test and validation accuracy_tables
accuracy_table_combined <- rbind(accuracy_table_test, accuracy_table_val)
accuracy_table_combined$Data <- rep(c("Test", "Validation"), each = length(model_names))

ggplot(accuracy_table_combined, aes(x = Model, y = Accuracy, fill = Data)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.8) +
  theme_minimal() +
  labs(title = "Model Accuracy Comparison - Test vs Validation",
       x = "Model",
       y = "Accuracy",
       fill = "Data") +
  scale_fill_manual(values = c("Test" = "pink", "Validation" = "purple"))

best_model_name <- model_names[which.max(accuracies_validation)]
print(paste("Best Model:", best_model_name))



