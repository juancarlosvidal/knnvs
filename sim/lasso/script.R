# Load the necessary library
library(gamlss)

gamlss_data <- function(x, y) {
  df <- data.frame(x)
  df$y <- y
  return (df)
}

gamlss_train <- function(df) {
  model <- gamlss(y ~ ., sigma.formula = ~ ., family = NO, data = df)
  print(model)
  return (model)
}

gamlss_pred <- function(x, model, df) {
  dfn <- data.frame(x)
  print(dfn)
  pre <- predictAll(model, newdata = dfn, data = df)
  print(pre)
  return (pre)
}

gamlss_cdf <- function(x, model, df, x_values) {
  dfn <- data.frame(x)
  pre <- predictAll(model, newdata = dfn, data = df)
  cdf <- pNO(x_values, mu = pre$mu, sigma = pre$sigma)
  #pNO(0, mu = fitted(model, "mu")[1], sigma = fitted(model, "sigma")[1])
  return (cdf)
}
