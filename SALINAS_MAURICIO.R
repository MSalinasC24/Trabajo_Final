rm(list = ls())
graphics.off()
cat("\014")
options(scipen = 999)
options(digits = 8)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()

library(pacman)

p_load(Boruta, DataExplorer, foreign, gmodels, partykit, rpart, 
       rpart.plot, gbm, adabag, caTools, caret, ggplot2, recipes,
       MLmetrics, randomForest, ISLR, manipulate, xgboost, 
       dplyr, tictoc,smotefamily,tidyr, 
       forcats,VIM,imbalance,performanceEstimation)


datos <- read.csv("dataset_fuga_telefonia.csv",sep = ",",stringsAsFactors = T)

semilla <- 123456789

target <- "fuga"


str(datos)

datos$pagos_atrasados = as.factor(datos$pagos_atrasados)


library(caret)
nearZeroVar(datos, saveMetrics = TRUE)

datos$codigo_canal_venta <- NULL
datos$ingreso_normalizado <- NULL

###### VARIABLE GENERO ############

levels(datos$genero)
levels(datos$genero) <- c("Masc", "Fem")

levels(datos$fuga) # CAMBIAR
pos <- levels(datos$fuga)[2] ; pos
neg <- levels(datos$fuga)[1] ; neg

##########################################################################
# Selección muestra de entrenamiento (75%) y de evaluación (25%) ----
##########################################################################                         

library(caret)

set.seed(semilla) 

index      <- createDataPartition(datos[[target]], 
                                  p = 0.80, 
                                  list = FALSE)

data.train <- datos[ index, ]                      
data.test  <- datos[-index, ]          



round(prop.table(table(datos[[target]])), 3)
round(prop.table(table(data.train[[target]])), 3)
round(prop.table(table(data.test[[target]])), 3)



##########################################################################
##### BALANCEO ####################################
##########################################################################


library(performanceEstimation)        
set.seed(semilla)
smote_train <- smote(fuga ~ ., ## CAMBIAR SEGUN EL TARGET
                     data = data.train, 
                     perc.over  = 2,  #2  # SMOTE     
                     perc.under = 1.5)  #1.5

round(prop.table(table(smote_train[[target]])), 3)


##########################################################################
# TRANSFORMACIONES CON RECIPES #####################################
##########################################################################

library(recipes)
set.seed(semilla)
trained_recipe <- recipe(fuga ~  .,
                         data =  smote_train) %>%
  step_nzv(all_predictors()) %>%
  step_range(all_numeric()) %>%   # Min-Max [0,1]
  step_dummy(all_nominal(), -all_outcomes()) %>% 
  prep()

trained_recipe

training <- bake(trained_recipe, new_data = NULL)
test     <- bake(trained_recipe, new_data = data.test)

training <- as.data.frame(training)
test     <- as.data.frame(test)

colnames(training)
colnames(test)
colnames(datos)

str(training)
str(test)


predictores <- setdiff(names(training), target)
predictores


##########################################################################
#### METODO DE SELECCIO DE VARIABLES ##########
##########################################################################

library(Boruta)
set.seed(semilla)
boruta.smote <- Boruta(fuga ~ ., #CAMBIAR
                      data = training, 
                      doTrace = 2)

print(boruta.smote)

plot(boruta.smote, las = 2, cex.axis = 0.6)

plotImpHistory(boruta.smote, lty = 1)


final.boruta <- TentativeRoughFix(boruta.smote)
print(final.boruta)

getSelectedAttributes(final.boruta, withTentative = F)

boruta.df <- attStats(final.boruta)
print(boruta.df)
boruta.df[order(-boruta.df$meanImp), ]


predictores <- setdiff(names(training), target)
predictores

##########################################################################
############################################################################################
##########################################################################



library(caretEnsemble)

# 6.1 Definiendo el trainControl para todos los modelos -------
fitControl2   <- trainControl(method = "cv",
                              number = 10,
                              savePredictions = 'final',
                              classProbs = T)



modelos = list(
  #treebag  =  caretModelSpec(method = "treebag",
                         #   tuneLength = 3),
  nb       =  caretModelSpec(method = "nb", 
                             tuneLength = 3),
  #glm      =  caretModelSpec(method = "glm",
   #                          family = "quasibinomial",
    #                        tuneLength = 3),
  knn      =  caretModelSpec(method = "knn", 
                              tuneLength = 3),
  
  rf       =  caretModelSpec(method = "rf", preProcess = "range",
                             tuneGrid = data.frame( mtry = c(2,3,4)), ntree = 200),
  
  rpart   = caretModelSpec(method = "rpart", tuneLength = 3),
  
  svm     = caretModelSpec(method = "svmLinear", tuneLength = 3)
  
)
# mtry = raiz(nro_variables) -- clasificacion
# mtry = nro_variables/3 -- regresion 


set.seed(semilla) 
modelo_ensamble <- caretList(training[, predictores],
                             training[, target],
                             trControl = fitControl2,
                             tuneList = modelos,
                             metric = "Accuracy")

# Vista global de los resultados de todos los modelos
modelo_ensamble

# Vista de los resultados de cada modelo individualmente
modelo_ensamble$treebag
modelo_ensamble$nb
modelo_ensamble$knn
modelo_ensamble$rf
modelo_ensamble$svm
modelo_ensamble$rpart
#############################################################



########################################################################

resultados <- resamples(modelo_ensamble)

summary(resultados)


densityplot(resultados, 
            metric = "Accuracy", 
            auto.key = TRUE)

descrCor2 <- modelCor(resultados)
descrCor2

# Seleccionar los algoritmos no correlacionados
descrCor2
summary(descrCor2[upper.tri(descrCor2)])
altaCorr2 <- findCorrelation(descrCor2, cutoff = 0.5, names = TRUE)
altaCorr2



##########################################################
# Ensamble de modelos -------------------------------------
##########################################################


# caretEnsemble uses a glm to create a simple linear blend 
# of models

ensamble <- caretEnsemble(modelo_ensamble,
                          trControl = fitControl2)

class(ensamble)

ensamble

ensamble$models

summary(ensamble)

test$proba_ensam  <- predict(object = ensamble, test[,predictores])
test$proba_ensam <- predict(ensamble, newdata = test[,predictores])
head(test$proba_ensam)

test$proba_ensam <- test$proba_ensam$Si_Fuga #CAMBIAR

test$clase_ensam <- predict(object = ensamble,
                            test[,predictores],  
                            return_class_only = T)

head(test$clase_ensam)


cm_ensamble <- caret::confusionMatrix(test$clase_ensam,
                                      test[, target],
                                      positive = pos)

cm_ensamble$byClass["Sensitivity"] 
cm_ensamble$byClass["Specificity"] 
cm_ensamble$overall["Accuracy"]
cm_ensamble$byClass["Balanced Accuracy"]

