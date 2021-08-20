library(plyr)
library(readr)
library(dplyr)
library(caret)
library(ggplot2)
library(repr)
library(readxl)
library(Amelia)
library(mFilter)
library(corrplot)

# function to calculate model Rsquared, RMSE
eval_results <- function(true, predicted, df) {
  SSE <- sum((predicted - true)^2)
  SST <- sum((true - mean(true))^2)
  R_square <- 1 - SSE / SST
  RMSE = sqrt(SSE/nrow(df))
  
  
  # Model performance metrics
  data.frame(
    RMSE = RMSE,
    Rsquare = R_square
  )
  
}

# import data
project6769_data.raw <- read_excel("group_project_data.xlsx", skip=2)
glimpse(project6769_data.raw)

# visualization of data sparsity (could make the analysis more impressive if included)
missmap(project6769_data.raw, main = "Missing vs observed") # visualization of missing values
data = subset(project6769_data.raw, select=-c(1, 2, 4, 16)) # pick subset according to some rules

# replace null values with mean 
for (i in colnames(data)) {
  data[[i]][is.na(data[[i]])] = mean(data[[i]],na.rm=T)
}

# correlation matrix
correlation = cor(data)
corrplot.mixed(correlation, tl.cex = 0.01, tl.col = "black", number.cex = 0.4, lower.col = 'black')

# no more null values
missmap(data, main = "Missing values vs observed") 

# randomly separates data into train and test datasets (70 / 30 split)
set.seed(100) 
index = sample(1:nrow(data), 0.7*nrow(data)) 
train = data[index,] # Create the training data 
test = data[-index,] # Create the test data
dim(train)
dim(test)

cols = c(colnames(data))[-c(1)]

# preprocessing of data (standardization)
pre_proc_val <- preProcess(train[,cols], method = c("center", "scale"))
train[,cols] = predict(pre_proc_val, train[,cols])
test[,cols] = predict(pre_proc_val, test[,cols])
summary(train)

# print(pre_proc_val$mean[[1]])
# print(pre_proc_val$std[[1]])

# reguralization (necessary for lasso and elastic net regression)
dummies <- dummyVars(`Real interest rate (%)` ~ . , data = data)
train_dummies = predict(dummies, newdata = train)
test_dummies = predict(dummies, newdata = test)
print(dim(train_dummies)); print(dim(test_dummies))


library(glmnet)

# more preprocessing necessary for glmnet
x = as.matrix(train_dummies)
y_train = train$`Real interest rate (%)`
x_test = as.matrix(test_dummies)
y_test = test$`Real interest rate (%)`

# lasso regression

# hyperparameter tuning (lambda)
# find optimal lambda using cross validation
lambdas <- 10^seq(2, -3, by = -.1)
lasso_cv <- cv.glmnet(x, y_train, alpha = 1, lambda = lambdas, standardize = TRUE, nfolds = 5)
lambda_best <- lasso_cv$lambda.min 
lambda_best

lasso_coef = coef(lasso_cv)
lasso_coef_out = capture.output(print(lasso_coef))
cat("Lasso Model", lasso_coef_out, file="lasso_model.txt", sep="\n")

# fit the lasso regression model
lasso_model <- glmnet(x, y_train, alpha = 1, lambda = lambda_best, standardize = TRUE)
summary(lasso_model)
lasso_model$beta

# performance of model on train set
predictions_train_lasso <- predict(lasso_cv, s = lambda_best, newx = x)
eval_results(y_train, predictions_train_lasso, train)

# performance of model on test set (important result)
predictions_test_lasso <- predict(lasso_cv, s = lambda_best, newx = x_test)
eval_results(y_test, predictions_test_lasso, test)


# elastic net regression
# tune hyperparameters (5 iterations of 10-fold cross-validation), and train model
train_cont <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 5,
                           search = "random",
                           verboseIter = TRUE)


elastic_reg <- train(y_train ~ .,
                     data = cbind(x, y_train),
                     method = "glmnet",
                     preProcess = c("center", "scale"),
                     tuneLength = 10,
                     trControl = train_cont)

elastic_coef = coef(elastic_reg$finalModel, elastic_reg$bestTune$alpha)
elastic_coef_out = capture.output(print(elastic_coef))
cat("Elastic Net Model", elastic_coef_out, file="elastic_net_model.txt")

# Best tuning parameter
elastic_reg$bestTune


# performance on training set
predictions_train_net <- predict(elastic_reg, x)
eval_results(y_train, predictions_train_net, train) 

# performance on test set
predictions_test_net <- predict(elastic_reg, x_test)
eval_results(y_test, predictions_test_net, test)

plot(project6769_data.raw$Year, project6769_data.raw$`10 Year Tresury ex post Bond Yield`, type='l', col = 'red',
     main='Long Term vs Short Term Interest Rates: USA', xlab = 'Year', ylab = 'rate %')
lines(project6769_data.raw$Year, project6769_data.raw$`Short-Term interest rates (%)`, col='blue')
legend('topleft', c("Long-term", "Short-Term"), fill=c("red", "blue"))

plot(project6769_data.raw$Year, project6769_data.raw$`Net Operating Surplus, non financial corporations, percentage of net value added`, 
     type='l', col = 'blue', main='USA Saving Trend (non finance sector)', xlab = 'Year', ylab = 'Savings')
#lines(project6769_data.raw$Year, project6769_data.raw$`Net Operating Surplus, financial corporations, percentage of net value added`, col='blue')
#legend('topleft', c("Non Finance Sector", "Finance Sector"), fill=c("red", "blue"))

plot(project6769_data.raw$Year, project6769_data.raw$`Net Operating Surplus, financial corporations, percentage of net value added`, 
     type='l', col='blue', main='USA Saving Trend (finance sector)', xlab = 'Year', ylab = 'Savings')

plot(project6769_data.raw$Year, project6769_data.raw$`Nominal interest rate (%)`, type='l', col="red",
     main='USA Nominal vs Real Interest Rates', xlab = 'Year', ylab = 'rate')
lines(project6769_data.raw$Year, project6769_data.raw$`Real interest rate (%)`, col='blue')
legend('topleft', c("Nominal", "Real"), fill=c("red", "blue"))

plot(project6769_data.raw$Year[-index], y_test, type='l', col="red", 
     xlab = 'Year', ylab = 'Rate', main="Actual vs Predicted Low Frequency Long Term Interest Rates (Test Set)")
lines(project6769_data.raw$Year[-index], predictions_test_net, col='blue')
legend('topleft', c("Real", "Predicted"), fill=c("red", "blue"))

train_predictions = cbind(project6769_data.raw$Year[index], predictions_train_net)
test_predictions = cbind(project6769_data.raw$Year[-index], predictions_test_net)
preds = rbind(train_predictions, test_predictions)
class(preds) = 'numeric'
colnames(preds) = c("year", "rate")
preds.df = as.data.frame(preds)
preds.df = preds.df[order(preds.df$year),]

plot(preds.df$year, preds.df$rate, type='l', col="red", 
     xlab = 'Year', ylab = 'Rate', main="Actual vs Predicted Low Frequency Long Term Interest Rates")
lines(project6769_data.raw$Year, project6769_data.raw$`Real interest rate (%)`, col='blue')
legend('topleft', c("Real", "Predicted"), fill=c("red", "blue"))

# just change year to compare real vs predicted
indices = which(preds.df$year == 1978)
preds.df[indices,]$year
preds.df[indices,]$rate 
project6769_data.raw[indices,]$`Real interest rate (%)`


# removing nominal interest rates does not affect model significance


