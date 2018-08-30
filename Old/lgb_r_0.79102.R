# Forked from https://www.kaggle.com/kailex/tidy-xgb-all-tables-0-796
# Changed to LGBM model
# Changed to one-hot encoding
# https://www.kaggle.com/returnofsputnik/good-transformations-to-continuous-variables

pacman::p_load(tidyverse, irlba, lightgbm, magrittr, tictoc, fastknn, stringr, zoo, caret, moments,
               DT, data.table, viridis, skimr, GGally, onehot)

set.seed(0)

#---------------------------

cat("Loading data...\n")

bbalance <- read_csv("bureau_balance.csv") 
bureau <- read_csv("bureau.csv")
cc_balance <- read_csv("CCB.csv")
payments <- read_csv("installments_payments.csv") 
pc_balance <- read_csv("POS_CASH_balance.csv")
prev <- read_csv("previous_application.csv")
tr <- read_csv("application_train.csv") 
te <- read_csv("application_test.csv")


tr <- tr[tr$CODE_GENDER != "XNA", ]
tr <- tr[tr$NAME_FAMILY_STATUS != "Unknown", ]
tr <- tr[tr$NAME_INCOME_TYPE != "Maternity leave", ]


tr$DAYS_LAST_PHONE_CHANGE[tr$DAYS_LAST_PHONE_CHANGE== 0] <- NA
te$DAYS_LAST_PHONE_CHANGE[te$DAYS_LAST_PHONE_CHANGE== 0] <- NA

bureau$DAYS_CREDIT_ENDDATE[bureau$DAYS_CREDIT_ENDDATE < -40000] <- NA
bureau$DAYS_CREDIT_UPDATE[bureau$DAYS_CREDIT_UPDATE < -40000] <- NA
bureau$DAYS_ENDDATE_FACT[bureau$DAYS_ENDDATE_FACT < -40000] <- NA

cc_balance$AMT_DRAWINGS_ATM_CURRENT[cc_balance$AMT_DRAWINGS_ATM_CURRENT < 0] <- NA
cc_balance$AMT_DRAWINGS_CURRENT[cc_balance$AMT_DRAWINGS_ATM_CURRENT < 0] <- NA

#tr$AMT_INCOME_TOTAL <- log1p(tr$AMT_INCOME_TOTAL)
#tr$AMT_CREDIT <- log1p(tr$AMT_CREDIT)
#tr$AMT_ANNUITY <- log1p(tr$AMT_ANNUITY)
#tr$AMT_GOODS_PRICE <- log1p(tr$AMT_GOODS_PRICE)
tr$REGION_POPULATION_RELATIVE <- sqrt(tr$REGION_POPULATION_RELATIVE)
#tr$DAYS_BIRTH <- sqrt(abs(tr$DAYS_BIRTH))
#tr$DAYS_EMPLOYED <- sqrt(abs(tr$DAYS_EMPLOYED))
tr$DAYS_REGISTRATION <- sqrt(abs(tr$DAYS_REGISTRATION))
#tr$OWN_CAR_AGE <- sqrt(abs(tr$OWN_CAR_AGE))
tr$APARTMENTS_AVG <- log1p(50*tr$APARTMENTS_AVG)
tr$YEARS_BEGINEXPLUATATION_AVG <- (tr$YEARS_BEGINEXPLUATATION_AVG)^30
tr$YEARS_BUILD_AVG <- (tr$YEARS_BUILD_AVG)^3
tr$COMMONAREA_AVG <- (tr$COMMONAREA_AVG)^(-1/200)
tr$ELEVATORS_AVG <- (tr$ELEVATORS_AVG)^(1/40)
tr$ENTRANCES_AVG <- (tr$ENTRANCES_AVG)^(1/3)
tr$FLOORSMAX_AVG <- (tr$FLOORSMAX_AVG)^(1/2.5)
tr$FLOORSMIN_AVG <- (tr$FLOORSMIN_AVG)^(1/2.2)
tr$LANDAREA_VG <- (tr$LANDAREA_AVG)^(1/5)
tr$LIVINGAPRTMENTS_AVG <- (tr$LIVINGAPARTMENTS_AVG)^(1/3)
tr$LIVINGAREA_AVG <- (tr$LIVINGAREA_AVG)^(1/3.5)
tr$NONLIVINGAPARTMENTS_AVG <- (tr$NONLIVINGAPARTMENTS_AVG)^(1/7)
tr$NONLIVINGAREA_AVG <- (tr$NONLIVINGAREA_AVG)^(1/5)
tr$TOTALAREA_MODE <- (tr$TOTALAREA_MODE)^(1/3)
tr$OBS_30_CNT_SOCIAL_CIRCLE <- (tr$OBS_30_CNT_SOCIAL_CIRCLE)^(1/7)
tr$DEF_30_CNT_SOCIAL_CIRCLE <- (tr$DEF_30_CNT_SOCIAL_CIRCLE)^(1/7)
tr$OBS_60_CNT_SOCIAL_CIRCLE <- (tr$OBS_60_CNT_SOCIAL_CIRCLE)^(1/7)
tr$DEF_60_CNT_SOCIAL_CIRCLE <- (tr$DEF_60_CNT_SOCIAL_CIRCLE)^(1/7)
#tr$DAYS_LAST_PHONE_CHANGE <- (abs(tr$DAYS_LAST_PHONE_CHANGE))^(1/2)

#te$AMT_INCOME_TOTAL <- log1p(te$AMT_INCOME_TOTAL)
#te$AMT_CREDIT <- log1p(te$AMT_CREDIT)
#te$AMT_ANNUITY <- log1p(te$AMT_ANNUITY)
#te$AMT_GOODS_PRICE <- log1p(te$AMT_GOODS_PRICE)
te$REGION_POPULATION_RELATIVE <- sqrt(te$REGION_POPULATION_RELATIVE)
#te$DAYS_BIRTH <- sqrt(abs(te$DAYS_BIRTH))
#te$DAYS_EMPLOYED <- sqrt(abs(te$DAYS_EMPLOYED))
te$DAYS_REGISTRATION <- sqrt(abs(te$DAYS_REGISTRATION))
#te$OWN_CAR_AGE <- sqrt(abs(te$OWN_CAR_AGE))
te$APARTMENTS_AVG <- log1p(50*te$APARTMENTS_AVG)
te$YEARS_BEGINEXPLUATATION_AVG <- (te$YEARS_BEGINEXPLUATATION_AVG)^30
te$YEARS_BUILD_AVG <- (te$YEARS_BUILD_AVG)^3
te$COMMONAREA_AVG <- (te$COMMONAREA_AVG)^(-1/200)
te$ELEVATORS_AVG <- (te$ELEVATORS_AVG)^(1/40)
te$ENTRANCES_AVG <- (te$ENTRANCES_AVG)^(1/3)
te$FLOORSMAX_AVG <- (te$FLOORSMAX_AVG)^(1/2.5)
te$FLOORSMIN_AVG <- (te$FLOORSMIN_AVG)^(1/2.2)
te$LANDAREA_VG <- (te$LANDAREA_AVG)^(1/5)
te$LIVINGAPRTMENTS_AVG <- (te$LIVINGAPARTMENTS_AVG)^(1/3)
te$LIVINGAREA_AVG <- (te$LIVINGAREA_AVG)^(1/3.5)
te$NONLIVINGAPARTMENTS_AVG <- (te$NONLIVINGAPARTMENTS_AVG)^(1/7)
te$NONLIVINGAREA_AVG <- (te$NONLIVINGAREA_AVG)^(1/5)
te$TOTALAREA_MODE <- (te$TOTALAREA_MODE)^(1/3)
te$OBS_30_CNT_SOCIAL_CIRCLE <- (te$OBS_30_CNT_SOCIAL_CIRCLE)^(1/7)
te$DEF_30_CNT_SOCIAL_CIRCLE <- (te$DEF_30_CNT_SOCIAL_CIRCLE)^(1/7)
te$OBS_60_CNT_SOCIAL_CIRCLE <- (te$OBS_60_CNT_SOCIAL_CIRCLE)^(1/7)
te$DEF_60_CNT_SOCIAL_CIRCLE <- (te$DEF_60_CNT_SOCIAL_CIRCLE)^(1/7)
#te$DAYS_LAST_PHONE_CHANGE <- (abs(te$DAYS_LAST_PHONE_CHANGE))^(1/2)

#---------------------------
cat("Preprocessing...\n")

# Set up a function to automatically get aggregated features 
# Note that funs() is from the dplyr package. 
# .args is a named list of additional arguments to be added to the functions
fn <- funs(mean, var, min, max, sum, n_distinct, (na.rm = TRUE))


# To get all aggregated features for the bureau balance file
encoder<- onehot(bbalance, stringsAsFactors=TRUE, addNA=FALSE)
bbalance<- predict(encoder, bbalance)

sum_bbalance <- bbalance %>%
  as_tibble() %>%
  mutate_if(is.character, funs(factor(.) %>% as.integer)) %>%  #change all categorical variables to numerical variables
  group_by(SK_ID_BUREAU) %>% # Aggregate by SK_ID_BUREAU
  summarise_all(fn) 
rm(bbalance); gc()

#---- bureau file
bureau2 <- read_csv("bureau.csv")

bureau$credit_active_binary <- as.integer(bureau$CREDIT_ACTIVE != 'Closed')
bureau$credit_enddate_binary <- as.integer(bureau$DAYS_CREDIT_ENDDATE > 40)

bureau2$credit_active_binary <- as.integer(bureau$CREDIT_ACTIVE != 'Closed')
bureau2$credit_enddate_binary <- as.integer(bureau$DAYS_CREDIT_ENDDATE > 40)

sum_bureau2 <- bureau2 %>% 
  as.tibble() %>% 
  select(-c(SK_ID_BUREAU, SK_ID_CURR, CREDIT_ACTIVE, CREDIT_CURRENCY)) %>% 
  group_by_("CREDIT_TYPE") %>%
  summarise_all(mean)

colnames(sum_bureau2) <- paste0("TYPE_",colnames(sum_bureau2), sep="")

colnames(sum_bureau2)[1] <- "CREDIT_TYPE"

bureau <- left_join(bureau , sum_bureau2, by= "CREDIT_TYPE")

sum_bureau3 <- bureau2 %>% 
  as.tibble() %>% 
  select(-c(SK_ID_BUREAU, SK_ID_CURR, CREDIT_ACTIVE, CREDIT_TYPE)) %>% 
  group_by_("CREDIT_CURRENCY") %>%
  summarise_all(mean)

colnames(sum_bureau3) <- paste0("CURRENCY_", colnames(sum_bureau3), sep="")

colnames(sum_bureau3)[1] <- "CREDIT_CURRENCY"

bureau <- left_join(bureau , sum_bureau3, by= "CREDIT_CURRENCY")


sum_bureau4 <- bureau2 %>% 
  as.tibble() %>% 
  select(-c(SK_ID_BUREAU, SK_ID_CURR, CREDIT_ACTIVE)) %>% 
  group_by_(.dots=c("CREDIT_CURRENCY", "CREDIT_TYPE")) %>%
  summarise_all(mean) 

colnames(sum_bureau4) <- paste0("CURRENCY_TYPE_", colnames(sum_bureau4), sep="")
colnames(sum_bureau4)[1] <- "CREDIT_CURRENCY"
colnames(sum_bureau4)[2] <- "CREDIT_TYPE"

bureau <- left_join(bureau, sum_bureau4, by= c("CREDIT_CURRENCY", "CREDIT_TYPE"))

encoder<- onehot(bureau, stringsAsFactors=TRUE, addNA=FALSE, max_levels=5)
bureau<- predict(encoder, bureau)

sum_bureau <- bureau %>% 
  as.tibble() %>%
  left_join(sum_bbalance, by = "SK_ID_BUREAU") %>% 
  select(-SK_ID_BUREAU) %>% 
  mutate(NEW_AMT_CREDIT_SUM_DEBT_DIVIDE_AMT_CREDIT_SUM = AMT_CREDIT_SUM_DEBT / AMT_CREDIT_SUM,
         NEW_AMT_CREDIT_SUM_LIMIT_DIVIDE_AMT_CREDIT_SUM = AMT_CREDIT_SUM_LIMIT/ AMT_CREDIT_SUM,
         NEW_AMT_CREDIT_SUM_OVERDUE_DIVIDE_AMT_CREDIT_SUM = AMT_CREDIT_SUM_OVERDUE / AMT_CREDIT_SUM,
         NEW_AMT_CREDIT_SUM_DEBT_DIVIDE_AMT_ANNUITY = AMT_CREDIT_SUM_DEBT / AMT_ANNUITY,
         NEW_AMT_CREDIT_SUM_LIMIT_DIVIDE_AMT_ANNUITY = AMT_CREDIT_SUM_LIMIT / AMT_ANNUITY,
         NEW_AMT_CREDIT_SUM_OVERDUE_DIVIDE_AMT_ANNUITY = AMT_CREDIT_SUM_OVERDUE / AMT_ANNUITY,
         NEW_AMT_CREDIT_SUM_OVERDUE_DIVIDE_AMT_CREDIT_SUM_DEBT = AMT_CREDIT_SUM_OVERDUE / AMT_CREDIT_SUM_DEBT,
         NEW_AMT_CREDIT_MAX_OVERDUE_DIVIDE_AMT_CREDIT_SUM_OVERDUE = AMT_CREDIT_MAX_OVERDUE / AMT_CREDIT_SUM_DEBT) %>% 
  mutate_if(is.character, funs(factor(.) %>% as.integer)) %>% 
  group_by(SK_ID_CURR) %>% # Aggregate by SK_ID_CURR
  summarise_all(fn)

rm(bureau, sum_bbalance, bureau2, sum_bureau2, sum_bureau3, sum_bureau4)


gc(); gc()

sum_bureau[which(sum_bureau == -Inf, TRUE)] <- NA
sum_bureau[which(sum_bureau == -NaN, TRUE)] <- NA

nzv_cols <- nearZeroVar(sum_bureau)
sum_bureau <- sum_bureau[, -nzv_cols];gc(); gc()

#----credit card balance file
encoder<- onehot(cc_balance, stringsAsFactors=TRUE, addNA=FALSE, max_levels=5)
cc_balance<- predict(encoder, cc_balance)

sum_cc_balance <- cc_balance %>% 
  as.tibble() %>%
  select(-SK_ID_PREV) %>% 
  mutate_if(is.character, funs(factor(.) %>% as.integer)) %>% 
  group_by(SK_ID_CURR) %>% 
  summarise_all(fn)
rm(cc_balance); gc()

#-----payments file
payments2 <- payments %>% 
  as.tibble() %>%
  select(-SK_ID_PREV) %>% 
  mutate(PAYMENT_PERC = AMT_PAYMENT / AMT_INSTALMENT,
         PAYMENT_DIFF = AMT_INSTALMENT - AMT_PAYMENT,
         DPD = DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT,
         DBD = DAYS_INSTALMENT - DAYS_ENTRY_PAYMENT,
         DPD = ifelse(DPD > 0, DPD, 0),
         DBD = ifelse(DBD > 0, DBD, 0))

payments2$DPD_cut <- as.factor(cut(payments2$DPD, 
                                   breaks = c(-1, 0, 1, 5, 10, 30, 100, 300, 1000, 3000)))

payments2$DBD_cut <- as.factor(cut(payments2$DBD, 
                                   breaks = c(-1, 0, 1, 5, 10, 30, 100, 300, 3500)))

payments <- payments %>% 
  as.tibble() %>%
  select(-SK_ID_PREV) %>% 
  mutate(PAYMENT_PERC = AMT_PAYMENT / AMT_INSTALMENT,
         PAYMENT_DIFF = AMT_INSTALMENT - AMT_PAYMENT,
         DPD = DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT,
         DBD = DAYS_INSTALMENT - DAYS_ENTRY_PAYMENT,
         DPD = ifelse(DPD > 0, DPD, 0),
         DBD = ifelse(DBD > 0, DBD, 0)) 

payments$DPD_cut <- as.factor(cut(payments$DPD, 
                                  breaks = c(-1, 0, 1, 5, 10, 30, 100, 300, 1000, 3000)))

payments$DBD_cut <- as.factor(cut(payments$DBD, 
                                  breaks = c(-1, 0, 1, 5, 10, 30, 100, 300, 3500)))

sum_payments2 <- payments2 %>% 
  select(-DPD_cut) %>% 
  group_by(DBD_cut) %>% 
  summarise_all(mean) 

colnames(sum_payments2) <- paste0("DBD_cut_", colnames(sum_payments2), sep="")
colnames(sum_payments2)[1] <- "DBD_cut"


payments <- left_join(payments, sum_payments2, by = "DBD_cut")

sum_payments3 <- payments2 %>% 
  select(-DBD_cut) %>% 
  group_by(DPD_cut) %>% 
  summarise_all(mean) 

colnames(sum_payments3) <- paste0("DPD_cut_", colnames(sum_payments3), sep="")
colnames(sum_payments3)[1] <- "DPD_cut"


payments <- left_join(payments, sum_payments3, by = "DPD_cut")

payments$DBD_cut_SK_ID_CURR <- NULL
payments$DBD_cut_DPD <- NULL
payments$DBD_cut_DBD <- NULL
payments$DPD_cut_SK_ID_CURR <- NULL
payments$DPD_cut_DPD <- NULL
payments$DPD_cut_DBD <- NULL

sum_payments <- payments %>% 
  as.tibble() %>%
  mutate_if(is.factor, funs(factor(.) %>% as.integer)) %>%
  group_by(SK_ID_CURR) %>% 
  summarise_all(fn) 

rm(payments, payments2, sum_payments2, sum_payments3); gc(); gc()


#-----pc_balance file
pc_balance$pos_cash_paid_late <- as.integer(pc_balance$SK_DPD > 0)
pc_balance$pos_cash_paid_late_with_tolerance <- as.integer(pc_balance$SK_DPD_DEF > 0)

encoder<- onehot(pc_balance, stringsAsFactors=TRUE, addNA=FALSE, max_levels=5)
pc_balance<- predict(encoder, pc_balance)

sum_pc_balance <- pc_balance %>% 
  as.tibble() %>%
  select(-SK_ID_PREV) %>% 
  mutate_if(is.character, funs(factor(.) %>% as.integer)) %>% 
  group_by(SK_ID_CURR) %>% 
  summarise_all(fn)
rm(pc_balance); gc()

#----- prev file
encoder<- onehot(prev, stringsAsFactors=TRUE, addNA=FALSE, max_levels=5)
prev<- predict(encoder, prev)

sum_prev <- prev %>%
  as.tibble() %>%
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

rm(prev); gc()

nzv_cols <- nearZeroVar(sum_prev)
sum_prev <- sum_prev[, -nzv_cols];gc(); gc()

#---- Merge files
tri <- 1:nrow(tr)
y <- tr$TARGET

tr_te <- tr %>% 
  select(-TARGET) %>% 
  bind_rows(te)

tr_te$DAYS_BIRTH2 <- (tr_te$DAYS_BIRTH/365)*-1
tr_te$age_bucket <- as.factor(cut(tr_te$DAYS_BIRTH2, 
                                  breaks = c(20, 24, 29, 34, 39, 44, 49, 54, 59, 64, 70)))

tr_te$DAYS_BIRTH2 <- NULL

tr_te <- tr_te  %>% 
  left_join(sum_bureau, by = "SK_ID_CURR") %>% 
  left_join(sum_cc_balance, by = "SK_ID_CURR") %>% 
  left_join(sum_payments, by = "SK_ID_CURR") %>% 
  left_join(sum_pc_balance, by = "SK_ID_CURR") %>% 
  left_join(sum_prev, by = "SK_ID_CURR") %>% 
  select(-SK_ID_CURR) %>% 
  mutate_if(is.character, funs(factor(.) %>% as.integer)) %>% 
  mutate(na = apply(., 1, function(x) sum(is.na(x))),
         DAYS_EMPLOYED = ifelse(DAYS_EMPLOYED == 365243, NA, DAYS_EMPLOYED),
         DAYS_EMPLOYED_PERC = sqrt(DAYS_EMPLOYED / DAYS_BIRTH),
         INCOME_CREDIT_PERC = AMT_INCOME_TOTAL / AMT_CREDIT,
         INCOME_PER_PERSON = log1p(AMT_INCOME_TOTAL / CNT_FAM_MEMBERS),
         ANNUITY_INCOME_PERC = sqrt(AMT_ANNUITY / (1 + AMT_INCOME_TOTAL)),
         LOAN_INCOME_RATIO = AMT_CREDIT / AMT_INCOME_TOTAL,
         PAYMENT_RATE = AMT_ANNUITY / AMT_CREDIT ,
         CHILDREN_RATIO = CNT_CHILDREN / CNT_FAM_MEMBERS, 
         cnt_non_child = CNT_FAM_MEMBERS - CNT_CHILDREN,
         CREDIT_TO_GOODS_RATIO = AMT_CREDIT / AMT_GOODS_PRICE,
         credit_per_person = AMT_CREDIT / CNT_FAM_MEMBERS,
         credit_per_child = AMT_CREDIT / (1 + CNT_CHILDREN),        
         INC_PER_CHLD = credit_per_child / (1 + CNT_CHILDREN),
         SOURCES_PROD = EXT_SOURCE_1 * EXT_SOURCE_2 * EXT_SOURCE_3,
         CAR_TO_BIRTH_RATIO = OWN_CAR_AGE / DAYS_BIRTH,
         CAR_TO_EMPLOY_RATIO = OWN_CAR_AGE / DAYS_EMPLOYED,
         PHONE_TO_BIRTH_RATIO = DAYS_LAST_PHONE_CHANGE / DAYS_BIRTH,
         PHONE_TO_BIRTH_RATIO = DAYS_LAST_PHONE_CHANGE / DAYS_EMPLOYED) 

tr_te$child_to_non_child_ratio <- tr_te$CNT_CHILDREN / tr_te$cnt_non_child
tr_te$credit_per_non_child <- tr_te$AMT_CREDIT / tr_te$cnt_non_child
tr_te$retirement_age <- as.integer(tr_te$DAYS_BIRTH < -14000)
tr_te$long_employment <- as.integer(tr_te$DAYS_EMPLOYED < -2000)


docs <- str_subset(names(tr), "FLAG_DOC")
live <- str_subset(names(tr), "(?!NFLAG_)(?!FLAG_DOC)(?!_FLAG_)FLAG_")
inc_by_org <- tr_te %>% 
  group_by(ORGANIZATION_TYPE) %>% 
  summarise(m = median(AMT_INCOME_TOTAL)) %$% 
  setNames(as.list(m), ORGANIZATION_TYPE)

tr_te %<>% 
  mutate(DOC_IND_KURT = apply(tr_te[, docs], 1, moments::kurtosis),
         LIVE_IND_SUM = apply(tr_te[, live], 1, sum),
         NEW_INC_BY_ORG = recode(tr_te$ORGANIZATION_TYPE, !!!inc_by_org),
         NEW_EXT_SOURCES_MEAN = apply(tr_te[, c("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3")], 1, mean),
         NEW_SCORES_STD = apply(tr_te[, c("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3")], 1, sd)) %>%
  mutate_all(funs(ifelse(is.nan(.), NA, .))) %>% 
  mutate_all(funs(ifelse(is.infinite(.), NA, .))) 

live <- str_subset(names(tr_te), "(?!NFLAG_)(?!FLAG_DOC)(?!_FLAG_)FLAG_")
tr_te <- as.data.frame(tr_te)[, !(colnames(tr_te) %in% live)]
gc(); gc()

rm(tr, te, fn, sum_bureau, sum_cc_balance, 
   sum_payments, sum_pc_balance, sum_prev); gc(); gc()

tr_te2 <- tr_te %>%
  group_by_(.dots=c("age_bucket", "CODE_GENDER")) %>%
  summarise(age_sex_credit = mean(AMT_CREDIT),
            age_sex_INCOME_TOTAL = mean(AMT_INCOME_TOTAL),
            age_sex_credit_per_person = mean(credit_per_person),
            age_sex_credit_per_child = mean(credit_per_child),
            age_sex_PAYMENT_RATE = mean(PAYMENT_RATE),
            age_sex_NEW_EXT_SOURCES_MEAN = mean(NEW_EXT_SOURCES_MEAN),
            age_sex_EXT_SOURCE_1_MEAN = mean(EXT_SOURCE_1),
            age_sex_EXT_SOURCE_2_MEAN = mean(EXT_SOURCE_2),
            age_sex_EXT_SOURCE_3_MEAN = mean(EXT_SOURCE_3),
            age_sex_OWN_CAR_AGE = mean(credit_per_child)
  )

tr_te <- left_join(tr_te , tr_te2, by= c("age_bucket", "CODE_GENDER"))


tr_te3 <- tr_te %>%
  group_by_(.dots=c("NAME_EDUCATION_TYPE", "CODE_GENDER")) %>%
  summarise(edu_sex_credit = mean(AMT_CREDIT),
            edu_sex_INCOME_TOTAL = mean(AMT_INCOME_TOTAL),
            edu_sex_credit_per_person = mean(credit_per_person),
            edu_sex_credit_per_child = mean(credit_per_child),
            edu_sex_PAYMENT_RATE = mean(PAYMENT_RATE),
            edu_sex_NEW_EXT_SOURCES_MEAN = mean(NEW_EXT_SOURCES_MEAN),
            edu_sex_EXT_SOURCE_1_MEAN = mean(EXT_SOURCE_1),
            edu_sex_EXT_SOURCE_2_MEAN = mean(EXT_SOURCE_2),
            edu_sex_EXT_SOURCE_3_MEAN = mean(EXT_SOURCE_3),
            edu_sex_OWN_CAR_AGE = mean(credit_per_child)
  )

tr_te <- left_join(tr_te , tr_te3, by= c("NAME_EDUCATION_TYPE", "CODE_GENDER"))

tr_te4 <- tr_te %>%
  group_by_(.dots=c("ORGANIZATION_TYPE", "CODE_GENDER")) %>%
  summarise(org_sex_credit = mean(AMT_CREDIT),
            org_sex_INCOME_TOTAL = mean(AMT_INCOME_TOTAL),
            org_sex_credit_per_person = mean(credit_per_person),
            org_sex_credit_per_child = mean(credit_per_child),
            org_sex_PAYMENT_RATE = mean(PAYMENT_RATE),
            org_sex_NEW_EXT_SOURCES_MEAN = mean(NEW_EXT_SOURCES_MEAN),
            org_sex_EXT_SOURCE_1_MEAN = mean(EXT_SOURCE_1),
            org_sex_EXT_SOURCE_2_MEAN = mean(EXT_SOURCE_2),
            org_sex_EXT_SOURCE_3_MEAN = mean(EXT_SOURCE_3),
            org_sex_OWN_CAR_AGE = mean(credit_per_child)
  )

tr_te <- left_join(tr_te , tr_te4, by= c("ORGANIZATION_TYPE", "CODE_GENDER"))

tr_te5 <- tr_te %>%
  group_by_(.dots=c("ORGANIZATION_TYPE", "NAME_EDUCATION_TYPE")) %>%
  summarise(org_edu_credit = mean(AMT_CREDIT),
            org_edu_INCOME_TOTAL = mean(AMT_INCOME_TOTAL),
            org_edu_credit_per_person = mean(credit_per_person),
            org_edu_credit_per_child = mean(credit_per_child),
            org_edu_PAYMENT_RATE = mean(PAYMENT_RATE),
            org_edu_NEW_EXT_SOURCES_MEAN = mean(NEW_EXT_SOURCES_MEAN),
            org_edu_EXT_SOURCE_1_MEAN = mean(EXT_SOURCE_1),
            org_edu_EXT_SOURCE_2_MEAN = mean(EXT_SOURCE_2),
            org_edu_EXT_SOURCE_3_MEAN = mean(EXT_SOURCE_3),
            org_edu_OWN_CAR_AGE = mean(credit_per_child)
  )


tr_te <- left_join(tr_te , tr_te5, by= c("ORGANIZATION_TYPE", "NAME_EDUCATION_TYPE"))


tr_te6 <- tr_te %>%
  group_by_("OCCUPATION_TYPE") %>%
  summarise(occ_credit = mean(AMT_CREDIT),
            occ_INCOME_TOTAL = mean(AMT_INCOME_TOTAL),
            occ_credit_per_person = mean(credit_per_person),
            occ_credit_per_child = mean(credit_per_child),
            occ_PAYMENT_RATE = mean(PAYMENT_RATE),
            occ_NEW_EXT_SOURCES_MEAN = mean(NEW_EXT_SOURCES_MEAN),
            occ_EXT_SOURCE_1_MEAN = mean(EXT_SOURCE_1),
            occ_EXT_SOURCE_2_MEAN = mean(EXT_SOURCE_2),
            occ_EXT_SOURCE_3_MEAN = mean(EXT_SOURCE_3),
            occ_OWN_CAR_AGE = mean(credit_per_child)
  )


tr_te <- left_join(tr_te , tr_te6, by= "OCCUPATION_TYPE") 

tr_te <- tr_te %>% 
  mutate_if(is.character, funs(factor(.) %>% as.integer)) %>%
  mutate_all(funs(ifelse(is.nan(.), NA, .))) %>% 
  mutate_all(funs(ifelse(is.infinite(.), NA, .))) 

nzv_cols <- nearZeroVar(tr_te)
tr_te <- tr_te[, -nzv_cols];gc(); gc()

rm(tr_te2, tr_te3, tr_te4, tr_te5, tr_te6);gc(); gc()

tr_te <- data.matrix(tr_te)

tr_te7 <- na.aggregate(tr_te)
tr_te7[is.na(tr_te7)] <- 0

df_corr = cor(tr_te7)
hc = findCorrelation(df_corr, cutoff=0.9)
hc = sort(hc)
dt1_num3 = as.data.frame(tr_te7)[,-c(hc)]

rm_col_hc <- setdiff(colnames(tr_te7), colnames(dt1_num3))
tr_te <- as.data.frame(tr_te)[, !(colnames(tr_te) %in% rm_col_hc)]

pcaObject <- prcomp(tr_te7,  scale = TRUE, center = TRUE)
tr_te <- data.frame(tr_te, pcaObject$x[,  1:30])

tr_te <- tr_te %>% 
  mutate_if(is.character, funs(factor(.) %>% as.integer)) %>%
  mutate_all(funs(ifelse(is.nan(.), NA, .))) %>% 
  mutate_all(funs(ifelse(is.infinite(.), NA, .))) 

rm(df_corr, dt1_num3, rm_col_hc, tr_te7, pcaObject);gc(); gc()

#---------------------------
cat("Preparing data...\n")

work <- tr_te[tri, ]
tri2 <- caret::createDataPartition(y, p = 0.85, list = F) %>% c()

dtrain <- data.frame(work[tri2, ])
dval <- data.frame(work[-tri2, ])
dtest <- data.frame(tr_te[-tri, ])

rm(work, new.data1, new.data2); gc(); gc()

rm(encoder); gc(); gc()

lgb.train <- lgb.Dataset(data.matrix(dtrain), label = y[tri2])
lgb.valid <- lgb.Dataset(data.matrix(dval), label = y[-tri2])
cols <- colnames(tr_te)
rm(tr_te, y, tri, dtrain, dval); gc(); gc()

#------------------------------
cat("Training LGBM....\n")

lgb.params <- list(objective = "binary",
                   metric = "auc",
                   max_depth = 8,
                   num_leaves = 32,
                   num_threads = 4,
                   max_bin = 255,
                   min_data_in_leaf = 10,
                   feature_fraction = 0.95,
                   bagging_fraction = 0.87,
                   lambda_l1 = 0.04, 
                   lambda_l2 = 0.075,
                   min_gain_to_split = 0.02,
                   min_child_weight = 39
)

lgb.model <- lgb.train(params = lgb.params,
                       data = lgb.train,
                       valids = list(val = lgb.valid),
                       learning_rate = 0.02,
                       nrounds = 5000,
                       early_stopping_rounds = 100,
                       eval_freq = 50
)

# Importance Plot
# lgb.importance(lgb.model, percentage = TRUE) %>% head(20) %>% kable()
tree_imp <- lgb.importance(lgb.model, percentage = TRUE) %>% head(30)
lgb.plot.importance(tree_imp, measure = "Gain")

# Make prediction and submission
lgb_pred <- predict(lgb.model, data = data.matrix(dtest), n = lgb.model$best_iter)

read_csv("sample_submission.csv") %>%  
  mutate(SK_ID_CURR = as.integer(SK_ID_CURR),
         TARGET = lgb_pred) %>%
  write_csv(paste0("lgb_r_", round(lgb.model$best_score, 5), ".csv"))