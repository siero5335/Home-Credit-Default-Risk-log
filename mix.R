# Forked from https://www.kaggle.com/kailex/tidy-xgb-all-tables-0-796
# Changed to LGBM model
# Changed to one-hot encoding
# https://www.kaggle.com/returnofsputnik/good-transformations-to-continuous-variables

pacman::p_load(tidyverse, irlba, lightgbm, magrittr, tictoc, fastknn, stringr, zoo, caret, moments,
               DT, data.table, viridis, skimr, GGally, onehot)

test1 <- read_csv("lgb_r_0.79179.csv") 
test2 <- read_csv("JYLF_submission.csv") 
test3 <- read_csv("submission_kernel02_cee847.csv") 
test4 <- read_csv("sub_nn_ihara.csv") 
test5 <- read_csv("sub_ridge_10fold_ihara.csv") 
test6 <- read_csv("tidy_xgb_0.78889.csv") 
test7 <- read_csv("pure_submission_scirpus.csv") 

test_sum  <- data.frame(test1$TARGET,
                        test2$TARGET,
                        test3$TARGET,
                        test4$TARGET,
                        test5$TARGET,
                        test6$TARGET,
                        test7$TARGET)

ggpairs(test_sum)


test_sum$TARGET <- (0.8 * (0.3 * test1$TARGET + 0.3 * test2$TARGET + 0.3 * test3$TARGET + 0.1 * test6$TARGET)) + 
                   (0.2 * (0.5 * test7$TARGET + 0.4 * test4$TARGET + 0.1 * test5$TARGET))

sub_mix1 <- data.frame(test1$SK_ID_CURR, test_sum$TARGET)
colnames(sub_mix1) <- c("SK_ID_CURR", "TARGET")

write_csv(sub_mix1, "sub_mix1.csv")


test1 <- read_csv("lgb_r_0.79179.csv") 
test2 <- read_csv("JYLF_submission.csv") 
test3 <- read_csv("submission_kernel02_cee847.csv") 
test4 <- read_csv("sub_nn_ihara.csv") 
test5 <- read_csv("sub_ridge_10fold_ihara.csv") 
test6 <- read_csv("tidy_xgb_0.78889.csv") 
test7 <- read_csv("pure_submission_scirpus.csv") 

test_sum  <- data.frame(test1$TARGET,
                        test2$TARGET,
                        test3$TARGET,
                        test4$TARGET,
                        test5$TARGET,
                        test6$TARGET,
                        test7$TARGET)

test_sum$TARGET <- apply(test_sum, 1, median)
  
sub_mix2 <- data.frame(test1$SK_ID_CURR, test_sum$TARGET)
colnames(sub_mix2) <- c("SK_ID_CURR", "TARGET")

write_csv(sub_mix2, "sub_mix2.csv")


test1 <- read_csv("lgb_r_0.79179.csv") 
test2 <- read_csv("JYLF_submission.csv") 
test3 <- read_csv("submission_kernel02_cee847.csv") 
test4 <- read_csv("sub_nn_ihara.csv") 
test5 <- read_csv("sub_ridge_10fold_ihara.csv") 
test6 <- read_csv("tidy_xgb_0.78889.csv") 
test7 <- read_csv("pure_submission_scirpus.csv") 

test_sum  <- data.frame(test1$TARGET,
                        test2$TARGET,
                        test3$TARGET,
                        test4$TARGET,
                        test5$TARGET,
                        test6$TARGET,
                        test7$TARGET)


test_sum$TARGET <- (0.9 * (0.3 * test1$TARGET + 0.3 * test2$TARGET + 0.3 * test3$TARGET + 0.1 * test6$TARGET)) + 
  (0.1 * (0.5 * test7$TARGET + 0.4 * test4$TARGET + 0.1 * test5$TARGET))

sub_mix3 <- data.frame(test1$SK_ID_CURR, test_sum$TARGET)
colnames(sub_mix3) <- c("SK_ID_CURR", "TARGET")

write_csv(sub_mix3, "sub_mix3.csv")