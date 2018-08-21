if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, irlba, lightgbm, magrittr, tictoc, fastknn, stringr, zoo, caret, moments,
               DT, data.table, viridis, skimr, GGally, umapr, naniar, softImpute)

set.seed(0)

#---------------------------
cat("Loading data...\n")

bbalance <- read_csv("bureau_balance.csv") 
bureau <- read_csv("bureau.csv")
cc_balance <- read_csv("credit_card_balance.csv")
payments <- read_csv("installments_payments.csv") 
pc_balance <- read_csv("POS_CASH_balance.csv")
prev <- read_csv("previous_application.csv")
tr <- read_csv("application_train.csv") 
te <- read_csv("application_test.csv")

tr <- tr[tr$CODE_GENDER != "XNA", ]

tr <- tr %>%
  mutate_if(is.character, funs(factor(.) %>% as.integer))


#---------------------------
cat("Preprocessing...\n")

fn <- funs(mean, sd, min, max, sum, n_distinct, .args = list(na.rm = TRUE))

temp1 <- bbalance
temp1 <- temp1 %>% mutate_if(is.character, as.factor)

dummies = dummyVars(~ STATUS, data = temp1)
temp1 <- as.data.frame(predict(dummies, newdata = temp1))

bbalance <- data.frame(bbalance, temp1)

sum_bbalance <- bbalance %>%
  mutate_if(is.character, funs(factor(.) %>% as.integer)) %>% 
  group_by(SK_ID_BUREAU) %>% 
  summarise_all(fn) 

rm(bbalance); gc(); gc()


temp1 <- bureau[, c("CREDIT_ACTIVE", "CREDIT_CURRENCY", "CREDIT_TYPE")]
temp1 <- temp1 %>% mutate_if(is.character, as.factor)

dummies = dummyVars(~., data = temp1)
temp1 <- as.data.frame(predict(dummies, newdata = temp1))

bureau <- data.frame(bureau, temp1)

rm(dummies, temp1); gc(); gc()
#---------------------------
sum_bureau <- bureau %>% 
  left_join(sum_bbalance, by = "SK_ID_BUREAU") %>% 
  select(-SK_ID_BUREAU) %>% 
  mutate_if(is.character, funs(factor(.) %>% as.integer))

# softimpute
fits <- softImpute(sum_bureau, rank.max=3, lambda=1.9, trace=TRUE, type="svd")
sum_bureau2 <- complete(sum_bureau, fits)
write_csv(sum_bureau2, "sum_bureau2_imp.csv")


sum_bureau <- sum_bureau2 %>% 
  group_by(SK_ID_CURR) %>% 
  summarise_all(fn)
rm(bureau, sum_bbalance, sum_bureau2, fits); gc(); gc()

nzv_cols <- nearZeroVar(sum_bureau)
sum_bureau <- sum_bureau[, -nzv_cols]

sum_bureau_agg <- na.aggregate(sum_bureau)
df_corr = cor(na.omit(sum_bureau_agg))
df_corr[is.na(df_corr)] <- 0

hc = findCorrelation(df_corr, cutoff=0.7)
hc = sort(hc)
dt1_num3 = as.data.frame(sum_bureau_agg)[,-c(hc)]
rm_col_hc <- setdiff(colnames(sum_bureau_agg), colnames(dt1_num3))

sum_prev <- as.data.frame(sum_bureau)[, !(colnames(sum_bureau) %in% rm_col_hc[-1])]
rm(sum_bureau_agg, dt1_num3, hc, nzv_cols, rm_col_hc, df_corr); gc(); gc()

gc(); gc()
#---------------------------

temp1 <-  as.data.frame(cc_balance$NAME_CONTRACT_STATUS)

temp1 <- temp1 %>% mutate_if(is.character, as.factor)

dummies = dummyVars(~., data = temp1)

temp1 <- as.data.frame(predict(dummies, newdata = temp1))

cc_balance <- data.frame(cc_balance, temp1)

rm(dummies, temp1); gc(); gc()

sum_cc_balance <- cc_balance %>% 
  select(-SK_ID_PREV) %>% 
  mutate_if(is.character, funs(factor(.) %>% as.integer)) 
  
# softimpute
fits <- softImpute(sum_cc_balance, rank.max=3, lambda=1.9, trace=TRUE, type="svd")
sum_cc_balance <- complete(sum_cc_balance, fits)

sum_cc_balance <- sum_cc_balance %>%   
  group_by(SK_ID_CURR) %>% 
  summarise_all(fn)

rm(cc_balance); gc(); gc()
#---------------------------
fits <- softImpute(payments, rank.max=3, lambda=1.9, trace=TRUE, type="svd")
payments <- complete(payments, fits)

sum_payments <- payments %>% 
  select(-SK_ID_PREV) %>% 
  mutate(PAYMENT_PERC = AMT_PAYMENT / AMT_INSTALMENT,
         PAYMENT_DIFF = AMT_INSTALMENT - AMT_PAYMENT,
         DPD = DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT,
         DBD = DAYS_INSTALMENT - DAYS_ENTRY_PAYMENT,
         DPD = ifelse(DPD > 0, DPD, 0),
         DBD = ifelse(DBD > 0, DBD, 0)) %>% 
  group_by(SK_ID_CURR) %>% 
  summarise_all(fn) 
rm(payments); gc()
#---------------------------

temp1 <-  as.data.frame(pc_balance$NAME_CONTRACT_STATUS)
temp1 <- temp1 %>% mutate_if(is.character, as.factor)

dummies = dummyVars(~., data = temp1)

temp1 <- as.data.frame(predict(dummies, newdata = temp1))

pc_balance <- data.frame(pc_balance, temp1)

rm(dummies, temp1); gc(); gc()


sum_pc_balance <- pc_balance %>% 
  select(-SK_ID_PREV) %>% 
  mutate_if(is.character, funs(factor(.) %>% as.integer)) 
  
fits <- softImpute(sum_pc_balance, rank.max=3, lambda=1.9, trace=TRUE, type="svd")
sum_pc_balance <- complete(sum_pc_balance, fits)

sum_pc_balance <- sum_pc_balance %>% 
  group_by(SK_ID_CURR) %>% 
  summarise_all(fn)

rm(pc_balance); gc(); gc()
#---------------------------


temp1 <- prev[, c("NAME_CONTRACT_TYPE", 
                  "FLAG_LAST_APPL_PER_CONTRACT", "WEEKDAY_APPR_PROCESS_START",
                  "NAME_CASH_LOAN_PURPOSE", "NAME_CONTRACT_STATUS",
                  "NAME_PAYMENT_TYPE", "CODE_REJECT_REASON",
                  "NAME_TYPE_SUITE", "NAME_CLIENT_TYPE",
                  "NAME_GOODS_CATEGORY", "NAME_PORTFOLIO", 
                  "NAME_PRODUCT_TYPE", "CHANNEL_TYPE",
                  "NAME_SELLER_INDUSTRY", "NAME_YIELD_GROUP",
                  "PRODUCT_COMBINATION")]

temp1[is.na(temp1)] <- "nadata"

temp1 <- temp1 %>% mutate_if(is.character, as.factor)

dummies = dummyVars(~., data = temp1)
temp1 <- as.data.frame(predict(dummies, newdata = temp1))


prev <- data.frame(prev, temp1)

rm(dummies, temp1); gc(); gc()


sum_prev <- prev %>%
  select(-SK_ID_PREV) %>% 
  mutate_if(is.character, funs(factor(.) %>% as.integer)) %>% 
  mutate(DAYS_FIRST_DRAWING = ifelse(DAYS_FIRST_DRAWING == 365243, NA, DAYS_FIRST_DRAWING),
         DAYS_FIRST_DUE = ifelse(DAYS_FIRST_DUE == 365243, NA, DAYS_FIRST_DUE),
         DAYS_LAST_DUE_1ST_VERSION = ifelse(DAYS_LAST_DUE_1ST_VERSION == 365243, NA, DAYS_LAST_DUE_1ST_VERSION),
         DAYS_LAST_DUE = ifelse(DAYS_LAST_DUE == 365243, NA, DAYS_LAST_DUE),
         DAYS_TERMINATION = ifelse(DAYS_TERMINATION == 365243, NA, DAYS_TERMINATION),
         APP_CREDIT_PERC = AMT_APPLICATION / AMT_CREDIT) %>% 
  group_by(SK_ID_CURR) %>% 
  summarise_all(fn) 

rm(prev); gc(); gc()


nzv_cols <- nearZeroVar(sum_prev)
sum_prev <- sum_prev[, -nzv_cols]

gc(); gc()

sum_prev[which(sum_prev == -Inf, TRUE)] <- NA
sum_prev[which(sum_prev == -NaN, TRUE)] <- NA

sum_prev_agg <- na.aggregate(sum_prev)
df_corr = cor(na.omit(sum_prev_agg))
df_corr[is.na(df_corr)] <- 0

hc = findCorrelation(df_corr, cutoff=0.7)
hc = sort(hc)
dt1_num3 = as.data.frame(sum_prev_agg)[,-c(hc)]
rm_col_hc <- setdiff(colnames(sum_prev_agg), colnames(dt1_num3))

sum_prev <- as.data.frame(sum_prev)[, !(colnames(sum_prev) %in% rm_col_hc[-1])]
rm(sum_prev_agg, dt1_num3, hc, nzv_cols, rm_col_hc, df_corr); gc(); gc()
#---------------------------

tri <- 1:nrow(tr)
y <- tr$TARGET

tr <- tr %>%
  mutate_if(is.character, funs(factor(.) %>% as.integer))
fits <- softImpute(tr, rank.max=3, lambda=1.9, trace=TRUE, type="svd")
tr <- complete(tr, fits)

te <- te %>%
  mutate_if(is.character, funs(factor(.) %>% as.integer))
fits <- softImpute(te, rank.max=3, lambda=1.9, trace=TRUE, type="svd")
te <- complete(te, fits)


tr_te <- tr %>% 
  select(-TARGET) %>% 
  bind_rows(te) %>%
  left_join(sum_bureau, by = "SK_ID_CURR") %>% 
  left_join(sum_cc_balance, by = "SK_ID_CURR") %>% 
  left_join(sum_payments, by = "SK_ID_CURR") %>% 
  left_join(sum_pc_balance, by = "SK_ID_CURR") %>% 
  left_join(sum_prev, by = "SK_ID_CURR") %>% 
  select(-SK_ID_CURR) %>% 
  mutate_if(is.character, funs(factor(.) %>% as.integer)) %>% 
  mutate(na = apply(., 1, function(x) sum(is.na(x))),
         NEW_CREDIT_TO_ANNUITY_RATIO = AMT_CREDIT / AMT_ANNUITY,
         NEW_CREDIT_TO_GOODS_RATIO = AMT_CREDIT / AMT_GOODS_PRICE,
         NEW_INC_PER_CHLD = AMT_INCOME_TOTAL / (1 + CNT_CHILDREN),
         DAYS_EMPLOYED_PERC = sqrt(DAYS_EMPLOYED / DAYS_BIRTH),
         NEW_DAYS_EMPLOYED = ifelse(DAYS_EMPLOYED == 365243, NA, DAYS_EMPLOYED),
         ANNUITY_INCOME_PERC = sqrt(AMT_ANNUITY / (1 + AMT_INCOME_TOTAL)),
         NEW_SOURCES_PROD = EXT_SOURCE_1 * EXT_SOURCE_2 * EXT_SOURCE_3,
         INCOME_CREDIT_PERC = AMT_INCOME_TOTAL / AMT_CREDIT,
         INCOME_PER_PERSON = log1p(AMT_INCOME_TOTAL / CNT_FAM_MEMBERS),
         NEW_CREDIT_TO_INCOME_RATIO = AMT_CREDIT / AMT_INCOME_TOTAL,
         LOAN_INCOME_RATIO = AMT_CREDIT / AMT_INCOME_TOTAL,
         CHILDREN_RATIO = CNT_CHILDREN / CNT_FAM_MEMBERS, 
         CAR_TO_BIRTH_RATIO = OWN_CAR_AGE / DAYS_BIRTH,
         CAR_TO_EMPLOY_RATIO = OWN_CAR_AGE / DAYS_EMPLOYED,
         PHONE_TO_BIRTH_RATIO = DAYS_LAST_PHONE_CHANGE / DAYS_BIRTH,
         PHONE_TO_BIRTH_EMPLOYED = DAYS_LAST_PHONE_CHANGE / DAYS_EMPLOYED) 

docs <- str_subset(names(tr), "FLAG_DOC")
live <- str_subset(names(tr), "(?!NFLAG_)(?!FLAG_DOC)(?!_FLAG_)FLAG_")
inc_by_org <- tr_te %>% 
  group_by(ORGANIZATION_TYPE) %>% 
  summarise(m = median(AMT_INCOME_TOTAL)) %$% 
  setNames(as.list(m), ORGANIZATION_TYPE)

rm(fn, sum_bureau, sum_cc_balance, 
   sum_payments, sum_pc_balance, sum_prev); gc()

#---------------------------

tr_te %<>% 
  mutate(DOC_IND_KURT = apply(tr_te[, docs], 1, moments::kurtosis),
         LIVE_IND_SUM = apply(tr_te[, live], 1, sum),
         NEW_INC_BY_ORG = recode(tr_te$ORGANIZATION_TYPE, !!!inc_by_org),
         NEW_EXT_SOURCES_MEAN = apply(tr_te[, c("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3")], 1, mean),
         NEW_SCORES_STD = apply(tr_te[, c("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3")], 1, sd))%>%
  mutate_all(funs(ifelse(is.nan(.), NA, .))) %>% 
  mutate_all(funs(ifelse(is.infinite(.), NA, .))) 

nzv_cols <- nearZeroVar(tr_te)
tr_te <- tr_te[, -nzv_cols]


live <- str_subset(names(tr), "(?!NFLAG_)(?!FLAG_DOC)(?!_FLAG_)FLAG_")
tr_te <- as.data.frame(tr_te)[, !(colnames(tr_te) %in% live)]
gc(); gc()

df_corr = cor(na.omit(na.aggregate(tr_te)))
hc = findCorrelation(df_corr, cutoff=0.7)
hc = sort(hc)
dt1_num3 = as.data.frame(tr_te)[,-c(hc)]
rm_col_hc <- setdiff(colnames(tr_te), colnames(dt1_num3))

tr_te <- as.data.frame(tr_te)[, !(colnames(tr_te) %in% rm_col_hc[-1])]
rm(tr, te, dt1_num3); gc(); gc()


tr <- read_csv("application_train.csv") 
te <- read_csv("application_test.csv")

tr <- tr[tr$CODE_GENDER != "XNA", ]

docs <- str_subset(names(tr), "FLAG_DOC")
name <- str_subset(names(tr), "NAME_")

live2 <- c("FONDKAPREMONT_MODE", "HOUSETYPE_MODE", "WALLSMATERIAL_MODE",
           "EMERGENCYSTATE_MODE", "EMERGENCYSTATE_MODE",
           "ORGANIZATION_TYPE", "WEEKDAY_APPR_PROCESS_START",
           "OCCUPATION_TYPE", "CODE_GENDER")

tr_te3 <- tr %>% 
  select(-TARGET) %>% 
  bind_rows(te) %>%
  select(-SK_ID_CURR) 

fac_trte <- data.frame(tr_te3[, docs], tr_te3[, live2], tr_te3[, name]) 
fac_trte[is.na(fac_trte)] <- "nadata"

fac_trte <- fac_trte %>% 
  mutate_if(is.integer, as.factor) %>% 
  mutate_if(is.character, as.factor)

rm(tr, te, tr_te3); gc(); gc()
#---------------------------


set.seed(42)
tri2 <- caret::createDataPartition(y, p = 0.85, list = F) %>% c()

fac_data <- fac_trte[tri, ]
fac_train <-  fac_data[tri2, ]
fac_val <-  fac_data[-tri2, ]
fac_test <- fac_trte[-tri, ]

dummies1 = dummyVars(~., data = fac_train)
dummies2 = dummyVars(~., data = fac_val)
dummies3 = dummyVars(~., data = fac_test)

fac_train <- as.data.frame(predict(dummies1, newdata = fac_train))
fac_val <- as.data.frame(predict(dummies2, newdata = fac_val))
fac_test <- as.data.frame(predict(dummies3, newdata = fac_test))

pcaObject1 <- prcomp(na.aggregate(fac_train), scale = F, center = F)
pcaObject2 <- prcomp(na.aggregate(fac_val), scale = F, center = F)
pcaObject3 <- prcomp(na.aggregate(fac_test), scale = F, center = F)

rm(fac_train, fac_val, fac_test); gc(); gc()

fac_train <- as.data.frame(pcaObject1$x[,  1:5])
fac_val <-  as.data.frame(pcaObject2$x[,  1:5])
fac_test <- as.data.frame(pcaObject3$x[,  1:5])

rm(fac_trte, fac_data, dummies1, dummies2, dummies3, 
   pcaObject1, pcaObject2, pcaObject3); gc(); gc()

tri2 <- caret::createDataPartition(y, p = 0.85, list = F) %>% c()

dtest <- tr_te[-tri, ]
work <-tr_te[tri, ]

work[which(work == -Inf, TRUE)] <- NA
work[which(work == -NaN, TRUE)] <- NA


dtrain <- data.frame(work[tri2, ], fac_train)
dval <- data.frame(work[-tri2, ], fac_val)
dtest <- data.frame(dtest, fac_test)

rm(fac_train, fac_val, fac_test, df_corr); gc(); gc()
rm(tr_te, work); gc(); gc()

set.seed(71)
new.data1 <- knnExtract(xtr = as.matrix(na.aggregate(dtrain)), ytr = as.factor(y[tri2]), xte = as.matrix(na.aggregate(dval)), k = 5, folds = 3, nthread = 4)
new.data2 <- knnExtract(xtr = as.matrix(na.aggregate(dtrain)), ytr = as.factor(y[tri2]), xte = as.matrix(na.aggregate(dtest)), k = 5, folds = 3, nthread = 4)

dtrain <- data.frame(dtrain, new.data1$new.tr)
dval <- data.frame(dval, new.data1$new.te)
dtest <- data.frame(dtest, new.data2$new.te)
rm(new.data1, new.data2, inc_by_org); gc(); gc()

lgb.train = lgb.Dataset(data.matrix(dtrain), label = y[tri2])
lgb.valid = lgb.Dataset(data.matrix(dval), label = y[-tri2])


set.seed(0)
params = list(objective = "binary", 
              metric = "auc", 
              learning_rate= 0.02, 
              min_sum_hessian_in_leaf = 100, 
              num_leaves= 256,
              subsample= 0.85,
              subsample_freq= 1,
              colsample_bytree= 0.95,
              colsample_bylevel = 0.7,
              min_split_gain=0.05,
              min_child_weight= 30,
              scale_pos_weight=11) # calculated for this dataset


set.seed(0)
lgb.model <- lgb.train(
  params = params
  , data = lgb.train
  , valids = list(val = lgb.valid)
  , nrounds = 10000
  , early_stopping_rounds = 300
  , eval_freq = 50
)              

read_csv("sample_submission.csv") %>%  
  mutate(SK_ID_CURR = as.integer(SK_ID_CURR),
         TARGET = predict(lgb.model, data = data.matrix(dtest), n = lgb.model$best_iter)) %>%
  write_csv("lightgbm_7.csv")