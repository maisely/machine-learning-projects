#############################################
# Created on December 12 2017
# @author: Thy Khue Ly
#############################################

set.seed(10)

# LOAD DATA
raw <- read.csv('train.csv', header=T) %>% na.omit()
test <- read.csv('test.csv', header=T) %>% na.omit()

# convert to ts object
data <- ts(raw[,-1], start=1987, frequency=12)

# extract time as indepednet variables
t <- time(data)

# TRAIN TEST SPLIT
train_end <- 2005
train <- window(data, start=c(1987,1), end=c(train_end,12))
val <- window(data, start=c(train_end+1,1), end=c(2010,12))

trn_len <- (train_end+1-1987)*12
val_len <- (2010-train_end)*12
pred_len <- 24
features <- colnames(train)[!colnames(train) %in% "Bankruptcy_Rate"]

# SCALING
bkrpt_scale <- 100
bkrpt <- data[, "Bankruptcy_Rate"]*bkrpt_scale
bkrpt_trn <- train[, "Bankruptcy_Rate"]*bkrpt_scale
bkrpt_val <- val[, "Bankruptcy_Rate"]*bkrpt_scale


#-------------------------------------------------
# VAR Model
#-------------------------------------------------
# helper function to create seasonal dummy matrix
create_seas_dummy <- function(df) {
  df.seas <- data.frame(feb = (df$Month==2)*1, mar = (df$Month==3)*1, apr = (df$Month==4)*1,
                        may = (df$Month==5)*1, jun = (df$Month==6)*1, jul = (df$Month==7)*1,
                        aug = (df$Month==8)*1, sep = (df$Month==9)*1, nov = (df$Month==11)*1,
                        dec = (df$Month==12)*1)
  return(df.seas)
}

# create seasonal dummy matrix
train_seas <- create_seas_dummy(raw_df[1:trn_len,])
val_seas <- create_seas_dummy(raw_df[(trn_len+1):nrow(raw),])
raw_seas <- create_seas_dummy(raw_df)
test_seas <- create_seas_dummy(test_df)

# check if stationary
par(mfrow = c(2,2), cex.main=0.9, mar = c(5,5,5,5))
for(i in 2:dim(raw)[2]){
  ts_obj <- ts(data = raw[,i], start=1987, frequency=12)
  acf(diff(ts_obj), main = colnames(raw)[i], xlab = "lag", lag.max=48)
}

# selected lambda for transformation based on BoxCox
selected_lambda <- BoxCox.lambda(bkrpt)


# helper function to retrieve metrics AIC, loglike and RMSE
get_metrics <- function(p, df, lmbd=1) {
  bcx_bkrpt <- BoxCox(bkrpt_trn, lambda=selected_lambda)
  df <- cbind(data.frame(bcx_bkrpt), df)
  colnames(df)[1] <- "Bankruptcy_Rate"
  
  m <- VAR(y=df, p=p, type=c("both"), exogen=train_seas)
  residuals <- m$varresult$Bankruptcy_Rate$residuals
  inv_fitted <- InvBoxCox(bcx_bkrpt - residuals, lambda=lmbd)
  rmse_train <- sqrt(mean((inv_fitted - bkrpt_trn)**2))
  
  pred_df <- predict(m, n.ahead=val_len, dumvar=val_seas, season=12)
  pred <- pred_df$fcst$Bankruptcy_Rate[,1]
  inv_pred <- InvBoxCox(pred, lambda=lmbd)
  rmse_test <- sqrt(mean((inv_pred - bkrpt_val)**2))
  
  return(list("rmse_train"=rmse_train,"rmse_test"=rmse_test))
} 

# grid search function for var models with the best p values
eval_VAR <- function(p_range, df, lmbd=1){
  res_df <- data.frame(matrix(ncol=6, nrow=0))
  for (p in p_range) {
    metrics <- get_metrics(p, df, lmbd)
    reg_name <- paste(colnames(df), collapse="-")
    res_df <- rbind(data.frame(p = p,
                               variables=reg_name,
                               rmse_train = metrics[1],
                               rmse_test = metrics[2],
                               row.names = NULL), res_df)
  }
  res_df <- res_df %>% arrange(rmse_test)
  return(res_df[1:3,]) 
}

# DEFINE PARAMETERS
p_range <- seq(1,12)


# MODEL WITH ONE REGRESSOR
one_var_df <- data.frame(matrix(ncol=6, nrow=0))
selected_var <- combn(features, m=1)

for (j in 1:ncol(selected_var)) {
  which.feature <- colnames(train) %in% selected_var[,j]
  edogen_df <- data.frame(train)[which.feature]
  metrics_df <- eval_VAR(p_range, edogen_df, lmbd=selected_lambda)
  one_var_df <- rbind(metrics_df, one_var_df) %>% arrange(p)
}


# MODEL WITH TWO REGRESSORS
two_var_df <- data.frame(matrix(ncol=6, nrow=0))
selected_var <- combn(features, m=2)

for (j in 1:ncol(selected_var)) {
  which.feature <- colnames(train) %in% selected_var[,j]
  edogen_df <- data.frame(train)[which.feature]
  metrics_df <- eval_VAR(p_range, edogen_df,lmbd=selected_lambda)
  two_var_df <- rbind(metrics_df, two_var_df) %>% arrange(p)
}


# OPTIMAL MODEL SEARCH
final_df <- rbind(one_var_df, two_var_df, three_var_df)  %>% arrange(rmse_test)
colnames(final_df)[1] <- c('regressors')
final_df %>% knitr::kable(format.args = list(big.mark = ",")
                          , caption = "VAR Models by Testing RMSE"
                          , col.names = c('Parameter', 'Variables', 'Training RMSE', 'Testing RMSE')
                          , align=c('l','l','c', 'c', 'c','c','c'), digits=3)


# BEST MODEL DIAGNOSIS
# boxcox transform
df <- cbind(data.frame(BoxCox(bkrpt, lambda=selected_lambda)), data.frame(data[,features]))
colnames(df)[1] <- "Bankruptcy_Rate"

# predict validation with best model
m <- VAR(y=df, p=7, type=c("both"), exogen=raw_seas)
pred_df <- predict(m, n.ahead=24, dumvar=test_seas)
pred <- pred_df$fcst$Bankruptcy_Rate[,1]
inv_pred <- InvBoxCox(pred, lambda=selected_lambda)

# plot prediction
plot(ts(inv_pred, start=2011, frequency=12), xlim=c(1987,2012), ylim=c(0, max(bkrpt))
     , main="112x212 Unemployment-Population-HPI", xlab="month", ylab="bankruptcy %", col='blue')
lines(bkrpt, col='black')
abline(v=2011, col='black',lty=2) # add a line where prediction starts
legend("topleft", legend = c("Observed", "Predicted"), lty = 1, col = c("black", "blue"), cex = 0.4)
