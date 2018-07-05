if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, skimr, GGally, plotly, viridis, caret, DT, data.table, lightGBM)

bureau <-fread('bureau.csv', stringsAsFactors = FALSE, showProgress=F,
               data.table = F, na.strings=c("NA","NaN","?", ""))

bureau_balance <-fread('bureau_balance.csv', stringsAsFactors = FALSE, showProgress=F,
                       data.table = F, na.strings=c("NA","NaN","?", ""))

str(bureau)

str(bureau_balance)
