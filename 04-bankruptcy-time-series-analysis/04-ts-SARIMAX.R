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
# SARIMAX Model
#-------------------------------------------------

# helper function to retrieve metrics AIC, loglike and RMSE
get_metrics <- function(p,d,q,P,D,Q, lmbd, xreg, newxreg) {
  # boxcox transform
  bcx_bkrpt <- BoxCox(bkrpt_trn, lambda=lmbd)
  # build model
  m <- arima(bcx_bkrpt, order=c(p,d,q), seasonal=list(order=c(P,D,Q), period=12), xreg)
  inv_fitted <- InvBoxCox(bcx_bkrpt - m$residuals, lambda=lmbd)
  rmse_train <- sqrt(mean((inv_fitted - bkrpt_trn)**2))
  
  if(is.null(xreg)) {
    pred <- forecast::forecast(m, h=val_len, level=0.95)
  }else{
    pred <- forecast::forecast(m, h=val_len, level=0.95, xreg=newxreg)
  }
  inv_pred <- InvBoxCox(pred$mean, lambda=lmbd)
  rmse_test <- sqrt(mean((inv_pred - bkrpt_val)**2))
  
  return(list("loglik"=m$loglik,"AIC"=m$aic, 
              "rmse_train"=rmse_train,"rmse_test"=rmse_test))
}

# helper function to return top 3 by RMSE for a set of models with given parameters
eval_sarima <- function(params, lmbd=1, xreg=NULL, newxreg=NULL) {
  # all possible combinations
  res_df <- data.frame(matrix(ncol=6, nrow=0))
  for (i in 1:nrow(params)) {
    tryCatch({
      p <- params[i,][[1]]
      d <- params[i,][[2]]
      q <- params[i,][[3]]
      P <- params[i,][[4]]
      D <- params[i,][[5]]
      Q <- params[i,][[6]]
      
      param_set <- paste("p=", p,", d=", d,", q=",q,", P=",P,", D=",D,", Q=",Q, sep="")
      df <- p+q+P+Q
      metrics <- get_metrics(p,d,q,P,D,Q, lmbd=lmbd, xreg, newxreg)
      res_df <- rbind(data.frame(param_set = param_set,
                                 df = df, 
                                 loglik = metrics[1],
                                 aic = metrics[2],
                                 rmse_train = metrics[3],
                                 rmse_test = metrics[4],
                                 row.names = NULL), res_df)
    }, error=function(e){})
  }
  top3 <- arrange(res_df, rmse_test)[1:3,]  
  return(top3) 
}


# DEFINE PARAMETERS
selected_lambda <- BoxCox.lambda(bkrpt)
p_range <- seq(0,2)
d_range <- 1
q_range <- seq(0,2)
P_range <- seq(0,2)
D_range <- seq(0,1)
Q_range <- seq(0,2)
params <- expand.grid(p_range, d_range, q_range, P_range, D_range, Q_range)

# MODEL WITH ONE REGRESSOR
selected_var <- combn(features, m=1)
one_reg_df <- data.frame(matrix(ncol=6, nrow=0))

for (j in 1:ncol(selected_var)) {
  which.feature <- colnames(train) %in% selected_var[,j]
  xreg_trn <- data.frame(train)[which.feature]
  xreg_val <- data.frame(val)[which.feature]
  
  metrics_df <- eval_sarima(params, lmbd=selected_lambda, xreg=xreg_trn, newxreg=xreg_val)
  
  reg_name <- paste(colnames(train)[which.feature], collapse="-")
  regressors <- rep(reg_name, ncol(selected_var))
  metrics_df <- cbind(regressors, metrics_df)
  one_reg_df <- rbind(metrics_df, one_reg_df) %>% arrange(rmse_test)
}

# MODEL WITH TWO REGRESSORS
selected_var <- combn(features, m=2)
two_reg_df <- data.frame(matrix(ncol=5, nrow=0))

for (j in 1:ncol(selected_var)) {
  which.feature <- colnames(train) %in% selected_var[,j]
  xreg_trn <- data.frame(train)[which.feature]
  xreg_val <- data.frame(val)[which.feature]
  
  metrics_df <- eval_sarima(params, lmbd=selected_lambda,
                            xreg=xreg_trn, newxreg=xreg_val)
  
  reg_name <- paste(colnames(train)[which.feature], collapse="-")
  regressors <- rep(reg_name, ncol(selected_var))
  metrics_df <- cbind(regressors, metrics_df)
  two_reg_df <- rbind(metrics_df, two_reg_df)
}

# OPTIMAL MODEL SEARCH
final_df <- rbind(one_reg_df, two_reg_df)  %>% arrange(rmse_test)
colnames(final_df)[1] <- c('regressors')
final_df %>% knitr::kable(format.args = list(big.mark = ",")
                          , caption = "SARIMAX Models by Log-likelihood"
                          , col.names = c('Regressors', 'Parameters','Degree of Freedom'
                                          , 'loglik', 'AIC', 'Training RMSE', 'Testing RMSE')
                          , align=c('l','l','c', 'c', 'c','c','c'), digits=3)


# BEST MODEL DIAGNOSIS
row1 <- final_df[1,]
reg1 <- strsplit(as.vector(row1[[1]]), split="-")[[1]]
params1 <- strsplit(as.vector(row1[[2]]), split=",") %>% 
  lapply(function(x) as.numeric(substr(x, nchar(x), nchar(x))))
params1 <- params1[[1]]

# boxcox transform
bcx_bkrpt <- BoxCox(bkrpt_trn, lambda=selected_lambda)

m1 <- arima(bcx_bkrpt, order=c(1,1,2), seasonal=list(order=c(2,1,2), period=12),
            xreg=train[,reg1])

inv_fitted1 <- InvBoxCox(bcx_bkrpt - m1$residuals, lambda=selected_lambda)
rmse_train1 <- sqrt(mean((inv_fitted1 - bkrpt_trn)**2))
pred1 <- forecast::forecast(m1, h=val_len, level=0.95, xreg=val[,reg1])
inv_pred1 <- InvBoxCox(pred1$mean, lambda=selected_lambda)
rmse_test1 <- sqrt(mean((inv_pred1 - bkrpt_val)**2))


# PREDICTION with BEST MODEL
# train for the whole dataset
bcx_bkrpt_all <- BoxCox(bkrpt, lambda=selected_lambda)
m_best <- arima(bcx_bkrpt_all, order=c(0,1,0), seasonal=list(order=c(2,1,1), period=12), method='CSS',
                xreg=data[,reg1])

pred <- forecast::forecast(m_best, h=48, level=0.95, xreg=test[,reg1])
inv_pred <- InvBoxCox(pred$mean, lambda=selected_lambda)

# plot
plot(inv_pred, xlim=c(1987, 2012), ylim=c(0,max(bkrpt)),
     main="112x212 Unemployment-Population-HPI", xlab="month", ylab="bankruptcy %", col='blue')
lines(bkrpt, col='black')
abline(v=2011, col='black',lty=2) # add a line where prediction starts
legend("topleft", legend = c("Observed", "Predicted"), lty = 1, col = c("black", "blue"), cex = 0.4)