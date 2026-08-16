library(dplyr)
library(tidyverse)
library(stringr)
library(xgboost)
library(caret)


##################################################################

# Pitching Data Preparation

# Import and Combine Pitch data into signular dataframe
CS2B <- read.csv("Model_Training/PitchCS2B.csv") %>% mutate(target_base = '2B', SBA_Outcome = 'CS')
SB2B <- read.csv("Model_Training/PitchSB2B.csv") %>% mutate(target_base = '2B', SBA_Outcome = 'SB')
CS3B <- read.csv("Model_Training/PitchCS3B.csv") %>% mutate(target_base = '3B', SBA_Outcome = 'CS')
SB3B <- read.csv("Model_Training/PitchSB3B.csv") %>% mutate(target_base = '3B', SBA_Outcome = 'SB')
PitchesPbP <- rbind(CS2B, SB2B, CS3B, SB3B)


# Defining Pitch Groups
Primary <- c("FF", "SI", "FC")
Breaking <- c("SL", "ST", "CU", "KC", "SV", "SC")
OffSpeed <- c("CH", "FS", "FO")

# Create Pitch Groupings and Flight time 
PitchesPbP <- PitchesPbP %>% mutate(
  pitch_classification = case_when(
    pitch_type %in% Primary ~ "Fast",
    pitch_type %in% Breaking ~ "Break",
    pitch_type %in% OffSpeed ~ "Off",
    .default = 'Other'
  )
) %>% mutate(flight_time = (60.5 - release_extension) / (release_speed * 1.46667))


####################################################################


# Import Catcher throwdown data and Remove Instances where throws werent tracked
CatcherThrows <- read.csv("Model_Training/CatcherThrows_pbp.csv") %>% filter(Throw.Type != "Untracked")

# Import Base Stealers Play by Play. Only use plays where result was SB or CS and change "--" to "No Throw"
BaseStealers <- read.csv("Model_Training/StealAttempts_pbp.csv") %>% 
  filter(Result %in% c("SB", "CS"), Lead_Distance_Gained != "--") %>% mutate(
  Fielder_Name = if_else(Fielder_Name == "--", "No Throw", Fielder_Name)
)

# NOTE: Catcher and Base Stealer data do not include a Game, at bat, or inning identifier. Instances with 
# identical join keys will be removed for ease of model training. 
keys <- c("Date","Catcher_Name","Pitcher_Name","Fielder_Name","Runner_Name",  "Target.Base", "Result")
BaseStealers <- BaseStealers %>% group_by(across(all_of(keys))) %>% filter(n() == 1) %>% ungroup()

keys2 <- c("Date","Catcher_Name", "Pitcher_Name",  "Fielder_Name", "Runner_Name",  "Base", "Result")
CatcherThrows <- CatcherThrows %>% group_by(across(all_of(keys2))) %>% filter(n() == 1) %>% ungroup()


# Joining Catching and Base Stealing data on its matching events using the following columns. Date is formatted
# into Year-Month-Day
Catchers_BaseStealers <- CatcherThrows %>% 
  inner_join(BaseStealers, by = c("Date" = "Date", "Catcher_Name" = "Catcher_Name", 
                                  "Pitcher_Name" = "Pitcher_Name", "Fielder_Name" = "Fielder_Name",
                                   "Runner_Name" = "Runner_Name", "Base" = "Target.Base",
                                  "Result" = "Result")) %>%
  mutate(Date = as.Date(Date, format = "%m/%d/%Y"))


####################################################################


# Abbreviating Pitcher Name into format used in Catchers_BaseStealers
PitchesPbP <- PitchesPbP %>% 
  mutate(Pitcher_Name_abbr = paste0(word(player_name, 1, sep = ","),", ",
                                    str_sub(str_trim(word(player_name, 2, sep = ",")), 1, 1),"."))

# Instances with idneitical keys being removed
keys3 <- c("game_date","Pitcher_Name_abbr","fielder_2","on_1b","target_base","SBA_Outcome")
PitchesPbP <- PitchesPbP %>% group_by(across(all_of(keys3))) %>%filter(n() == 1) %>% ungroup()
         

# Merge Catcher_BaseStealers on the Seperated Pitching Data for stolen base attempts to 2nd and 3rd base. The Two
# resulting DataFrames are combined into one. 
SBA_2B <- PitchesPbP %>% filter(target_base == "2B") %>% mutate(game_date = as.Date(game_date)) %>%
  inner_join(Catchers_BaseStealers %>%  filter(Base == "2B"), by = c("game_date" = "Date", 
                                                                    "Pitcher_Name_abbr" = "Pitcher_Name", 
                                                                    "fielder_2" = "Catcher_ID", "on_1b" = "Runner_ID", 
                                                                    "target_base" = "Base", "SBA_Outcome" = "Result"))

SBA_3B <- PitchesPbP %>% filter(target_base == "3B") %>% mutate(game_date = as.Date(game_date)) %>%
  inner_join(Catchers_BaseStealers %>%  filter(Base == "3B"), by = c("game_date" = "Date", 
                                                                     "Pitcher_Name_abbr" = "Pitcher_Name", 
                                                                     "fielder_2" = "Catcher_ID", "on_2b" = "Runner_ID", 
                                                                     "target_base" = "Base", "SBA_Outcome" = "Result"))

StolenBaseData <- rbind(SBA_2B, SBA_3B)

###################################################################

# Model Preparation

# Creating Data Frame with the variabls to be used in the model's training
StolenBaseData_Shortened <- StolenBaseData %>% select(
  game_date, Pitcher_Name_abbr, Catcher_Name, Runner_Name, p_throws, release_speed, release_extension, flight_time, pitch_classification,
  Throw_Speed, Exchange, Accuracy, Teamwork, Lead_Distance_Gained, Pitcher_First_Move, At_Pitch_Release, target_base,
  SBA_Outcome
) %>% mutate(Stole = ifelse(SBA_Outcome == "SB", 1, 0))


# Create Predictor matrix for xgBoost.
x_data <- StolenBaseData_Shortened %>% select(
  p_throws, release_speed, release_extension, flight_time, pitch_classification,
  Throw_Speed, Exchange, Lead_Distance_Gained, Pitcher_First_Move, At_Pitch_Release, target_base
)
cols <- c("Lead_Distance_Gained", "Pitcher_First_Move", "At_Pitch_Release")
x_data[cols] <- lapply(x_data[cols], as.numeric)

# Create X and Y matrix variables for final model
y <- StolenBaseData_Shortened$Stole
dummy <- dummyVars(~ ., data = x_data)
x_matrix <- predict(dummy, newdata = x_data)
x_matrix <- as.matrix(x_matrix)



train_index <- createDataPartition(y, p = 0.8, list = FALSE)
x_train <- x_matrix[train_index, ]
x_test  <- x_matrix[-train_index, ]
y_train <- y[train_index]
y_test  <- y[-train_index]



# Model Creation
dtrain <- xgb.DMatrix(
  data = x_train,
  label = y_train
)
dtest <- xgb.DMatrix(
  data = x_test,
  label = y_test
)
params <- list(
  objective = "binary:logistic",
  eval_metric = "logloss",
  max_depth = 3,
  eta = 0.05,
  subsample = 0.8,
  colsample_bytree = 0.8
)
model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 300,
  watchlist = list(
    train = dtrain,
    test = dtest
  ),
  early_stopping_rounds = 20
)

pred_prob <- predict(model, dtest)


predictions <- StolenBaseData_Shortened[-train_index, ] %>%
  select(game_date, Runner_Name, Pitcher_Name_abbr, Catcher_Name, target_base) %>%
  mutate(
    Actual = y_test,
    Predicted_Probability = pred_prob
  )

Calibration_Steps <- predictions %>%
  mutate(
    Bin = cut(
      Predicted_Probability,
      breaks = seq(0.5, 1, by = 0.1),
      include.lowest = TRUE
    )
  ) %>%
  group_by(Bin) %>%
  summarise(
    Attempts = n(),
    Predicted = mean(Predicted_Probability),
    Actual = mean(Actual)
  )


mean((pred_prob - y_test)^2)
