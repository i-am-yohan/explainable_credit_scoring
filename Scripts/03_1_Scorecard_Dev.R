list.of.packages <- c('scorecard','DBI','RPostgres','ROCR','performanceEstimation','ggplot2','glmnet','caret')
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

library('scorecard')
library('DBI')
library('RPostgres')
library('ROCR')
library('performanceEstimation')
library('ggplot2')
library('glmnet')
library('caret')

args <- commandArgs(trailingOnly = TRUE)
usr = as.character(args[1])
pwd = as.character(args[2])


db <- 'hm_crdt'  #provide the name of your db
host_db <- 'localhost' #i.e. # i.e. 'ec2-54-83-201-96.compute-1.amazonaws.com'  
db_port <- '5432'  # or any other port specified by the DBA
db_user <- as.character(usr)
db_password <- as.character(pwd)
con <- dbConnect(RPostgres::Postgres(), dbname = db, host=host_db, port=db_port, user=db_user, password=db_password)

print('Creating credit scorecard model')
#extract Data
extract_tbl <- function(table_name){
  out_tbl <- dbGetQuery(con, sprintf("select * from %s",table_name))
  row.names(out_tbl) <- out_tbl$sk_id_curr
  out_tbl <- out_tbl[,!(names(out_tbl) %in% 'sk_id_curr')]
}
ABT_Train <- extract_tbl('abt.abt_train')
ABT_Test <- extract_tbl('abt.abt_test')
ABT_Kagl <- extract_tbl('abt.abt_kaggle_submission')

ABT_Train[['amt_annuity_bur_ratio']] <- NULL
ABT_Test[['amt_annuity_bur_ratio']] <- NULL
ABT_Kagl[['amt_annuity_bur_ratio']] <- NULL

ABT_Train[,'target'] <- as.factor(ABT_Train[,'target'])
ABT_Test[,'target'] <- as.factor(ABT_Test[,'target'])
ABT_Kagl[,'target'] <- as.factor(ABT_Kagl[,'target'])


WOE_Bin <- woebin(ABT_Train, 'target')


#Drop Shit IVs
IV_List <- c(NA)
Col_List <- c('target')
for (col in names(ABT_Train)[(!names(ABT_Train) %in% 'target')]){
  if (WOE_Bin[[col]][1]$total_iv >= 0.05){
    Col_List <- append(Col_List , col)
    IV_List <- append(IV_List , WOE_Bin[[col]][1]$total_iv)
  }
}


IV_DF <- cbind(data.frame(Col_List), data.frame(IV_List))
IV_DF[,'Col_List'] <- paste(IV_DF[,'Col_List'], '_woe', sep = '')
colnames(IV_DF) <- c('feature','IV')

ABT_Train <- ABT_Train[,Col_List]
ABT_Test <- ABT_Test[,Col_List]
ABT_Kagl <- ABT_Kagl[,Col_List]

ABT_Kagl_WOE <- data.frame(woebin_ply(ABT_Kagl, bins=WOE_Bin))
ABT_Train_WOE <- data.frame(woebin_ply(ABT_Train, bins=WOE_Bin))
ABT_Test_WOE <- data.frame(woebin_ply(ABT_Test, bins=WOE_Bin))



#Columns to drop
Drop_Col <- c(#Dropped due to multicollinearity
              'ext_source_2_woe'
              ,'ext_source_3_woe'
              ,'ext_source_1_woe'
              ,'ext_source_2x3_woe'
              ,'ltv_t0_x_ext_source_2_woe'
              ,'credit_income_ratio_x_ext_source_2_woe'
              ,'ext_source_3xage_woe'
              ,'ext_source_1x2x3_woe'
              ,'ext_source_2xage_woe'
              ,'min_days_enddate_fact_woe'
              ,'ext_source_1xage_woe'
              ,'ltv_t0_x_ext_source_3_woe'
              ,'ext_source_1x2_woe'
              ,'credit_income_ratio_x_ext_source_3_woe'
              ,'amt_goods_price_woe'
              ,'ext_source_min_woe'
              
              #Dropped due to insignificance
              ,'amt_credit_woe'
              ,'credit_annuity_ratio_woe'
              ,'credit_annuity_ratio_x_ext_source_3_woe'
              ,'ltv_t0_x_ext_source_1_woe'
              ,'goods_price_annu_bur_ratio_woe'
              ,'payment_income_ratio_x_ext_source_3_woe'
              ,'min_days_credit_enddate_bur_woe'
              ,'amt_credit_bur_ratio_woe'
              ,'credit_annuity_ratio_x_ext_source_1_woe'
              ,'credit_annuity_ratio_x_ext_source_2_woe'
              ,'ext_source_1x3_woe'
              ,'num_refused_prev_woe'
              ,'prop_approved_prev_woe'
              
              #Dropped due to wrong sign
              ,'payment_income_ratio_x_ext_source_2_woe'
              ,'days_birth_woe'
              )

ABT_Train_WOE_sub <- ABT_Train_WOE
ABT_Train_WOE_sub[Drop_Col] <- NULL

ABT_Train_WOE_sub[ABT_Train_WOE_sub['target'] == 0 ,'weight_vec'] <- nrow(ABT_Train_WOE_sub)/(2*nrow(ABT_Train_WOE_sub[ABT_Train_WOE_sub['target'] == 0,]))
ABT_Train_WOE_sub[ABT_Train_WOE_sub['target'] == 1 ,'weight_vec'] <- nrow(ABT_Train_WOE_sub)/(2*nrow(ABT_Train_WOE_sub[ABT_Train_WOE_sub['target'] == 1,]))

ABT_Test_WOE[ABT_Test_WOE['target'] == 0 ,'weight_vec'] <- nrow(ABT_Train_WOE_sub)/(2*nrow(ABT_Train_WOE_sub[ABT_Train_WOE_sub['target'] == 0,]))
ABT_Test_WOE[ABT_Test_WOE['target'] == 1 ,'weight_vec'] <- nrow(ABT_Train_WOE_sub)/(2*nrow(ABT_Train_WOE_sub[ABT_Train_WOE_sub['target'] == 1,]))


#Train the model
LR_Model <- glm(target ~ . -weight_vec , weights=weight_vec, data=ABT_Train_WOE_sub, family=binomial())
summary(LR_Model)

Test_SC <- function(In_Model , In_Data , Prob = 0.5){
  
  #The confusion matrix Prob is the decision boundary
  CM <- table(In_Data$target , predict(In_Model, newdata = In_Data, type = "response") > Prob)
  TP <- CM[2,2]
  TN <- CM[1,1]
  FP <- CM[1,2]
  FN <- CM[2,1]
  
  p <- predict(In_Model, newdata = In_Data, type = "response")
  pr <- prediction(p, In_Data$target)
  
  prf <- ROCR::performance(pr, measure = "tpr", x.measure = "fpr")
  
  auc <- ROCR::performance(pr, measure = "auc")
  
  Output <- NULL
  Output$Confusion_Matrix <- CM
  Output$Accuracy <- (TP + TN)/(TP + TN + FP + FN)
  Output$Precision <- TP/(TP + FP)
  Output$Recall <- TP/(TP + FN)
  Output$False_Positive_Rate <- FP/(TN + FP) 
  Output$AUC <- auc@y.values[[1]]
  Output$F1 <- 2*(Output$Precision*Output$Recall)/(Output$Precision+Output$Recall)
  print(plot(prf , main = "ROC Curve"))
  return(Output)
}

#Cross-Validation
folds <- createFolds(ABT_Train_WOE_sub$target, k = 5)
for (i in 1:5){
  #print(folds[i])
  fold_i <- folds[[i]]
  train_kf <- ABT_Train_WOE_sub[-fold_i,]
  CV_kf <- ABT_Train_WOE_sub[fold_i,]
  
  LR_Model_CV <- glm(target ~ . -weight_vec , weights=weight_vec, data=train_kf, family=binomial())
  
  print(paste('Evaluation for fold',i))
  print('Train')
  print(Test_SC(LR_Model_CV , train_kf , 0.5))
  print('CV')
  print(Test_SC(LR_Model_CV , CV_kf , 0.5))
}


Test_SC(LR_Model , ABT_Train_WOE_sub , 0.5)
Test_SC(LR_Model , ABT_Test_WOE , 0.5)


Kagl_sub <- ABT_Kagl_WOE
ABT_Kagl_WOE['weight_vec'] <- 0

Kagl_sub['target'] <- predict(LR_Model, newdata = ABT_Kagl_WOE, type = "response")
Kagl_sub['sk_id_curr'] <- row.names(ABT_Kagl)
Kagl_sub <- Kagl_sub[c('target','sk_id_curr')]

write.csv(Kagl_sub , 'Kaggle_Submissions/01_Scorecard.csv', row.names = FALSE)
