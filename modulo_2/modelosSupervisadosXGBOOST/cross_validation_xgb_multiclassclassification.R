
makeXGBMatrix <- function(xvars, yvar, df){
  
  XGBmatrix <-
    df %>% 
    select(all_of(xvars)) %>% 
    as.matrix %>% 
    xgb.DMatrix(., 
                label=df[[yvar]])
  
  return(XGBmatrix)
  
}

fitXGBMulticlass <- function(xgbTrain, xgbTest, iterations){
  
  # iniciar el valor de la métrica del cual partimos (alto para métricas que queremos reducir
  # y viceversa)
  
  logloss = 0
  params = list()
  
  for (i in 1:iterations){
    
    # muestrear números de forma aleatoria (dentro de un rango) para los hiperparámetros 
    # cada iteración 
      
    hparams = list(
      eta = runif(1, 0.01, 0.3),
      lambda = runif(1, 0.01, 0.2),
      alpha = runif(1, 0.01, 0.2),
      gamma = runif(1, 0, 20),
      max_depth = sample(5:14,1), 
      subsample = runif(1,0.5,1),
      colsample_bytree = runif(1, 0.5, 1)
  )  


    # ajustar un modelo utilizando validación cruzada para probar los hiperparámetros en
    # todas las regiones de los datos de entrenamiento
    
  xgbCV <- xgb.cv(
    booster = 'gbtree',
    objective = 'multi:softmax', 
    eval_metric = 'mlogloss',
    params = hparams, 
    nfold = 10,
    nrounds = 10000,
    early_stopping_rounds = 2,
    maximize = F,
    data = xgbTrain,
    verbose = 2,
    num_class = 9
  )
  
  # registrar la métrica
  
  logloss2 <- xgbCV$evaluation_log[xgbCV$best_iteration]$test_mlogloss_mean
  
  # si la métrica alcanzada fuera mejor que la actual, reemplazar la actual  y guardar los
  # hiperparámetros del modelo que llegó a ella.
  # Nota : la métrica debe ser menor si se esta reduciendo (ej rmse) o mayor si se esta aumentando (ej auc)
  
  
  if(logloss2 < logloss){
    logloss2 = logloss
    params = hparams
  }
  }
  
  # Ajustar un modelo final con los mejores hiperparámetros, probándolo ahora en el test set
  
 finalmodel <-  xgb.train(
   booster = 'gbtree',
   objective = 'multi:softmax', 
   eval_metric = 'mlogloss',
    params= params,
    data=xgbTrain,
    nrounds=50,
    early_stopping_rounds=3,
    num_class = 9,
    watchlist = list(training = xgbTrain,
                     testing = xgbTest),
    maximize=F
  )
  return(finalmodel)
}




      
