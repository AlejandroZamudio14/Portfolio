library(dplyr)
library(tidyverse)


############################


# The following script is used to create the player profiles for the Shiny App. Note that the current intent of the
# projects is to only use Pitchers and Catchers from the 2023 Texas Rangers. All Runners from the 2023 season with
# data in the statcast base stealing leaderboard will be included. 


############################

# List of Rangers Pitchers used for application
pitcherNames <- c('Chapman', 'Dunning', 'Eovaldi', 'Gray', 'Heaney', 'Leclerc', 'Montgomery',
                  'Perez', 'Sborz', 'Smith')

# Defining the pitch groupings
Primary <- c("FF", "SI", "FC")
Breaking <- c("SL", "ST", "CU", "KC", "SV", "SC")
OffSpeed <- c("CH", "FS", "FO")


# Read in the first pitchers data and then combine with the remaining
FullPitcherData <- read.csv(paste("Players/", pitcherNames[1], "_2023.csv", sep = ""))
for (i in 2:length(pitcherNames)){
  file <- paste("Players/", pitcherNames[i], "_2023.csv", sep = "")
  FullPitcherData <- rbind(FullPitcherData, read.csv(file))
}

############################


# Create the pitch groupings in the data frame
FullPitcherData <- FullPitcherData %>% mutate(
  pitch_classification = case_when(
    pitch_type %in% Primary ~ "Fast",
    pitch_type %in% Breaking ~ "Break",
    pitch_type %in% OffSpeed ~ "Off",
    .default = 'Other'
  )
)


# Create the Profiles for the Rangers Pitchers
PitchProfiles_ROB <- FullPitcherData %>%
  mutate(flight_time = (60.5 - release_extension) / (release_speed * 1.46667)) %>% 
  group_by(player_name, pitch_classification) %>%
  summarise(
    Count = n(),
    avg_speed = mean(release_speed, na.rm = TRUE),
    avg_extension = mean(release_extension, na.rm = TRUE),
    avg_flight_time = mean(flight_time, na.rm = TRUE),
    .groups = "drop"
  ) 


############################ 


# Import in the full catching play by play data
FullCatcherData <- read.csv("Model_Training/CatcherThrows_pbp.csv")


# Filter for the three rangers catchers in 2023, filter for instances where a throw happened, and 
# create average profiles
CatcherProfiles <- FullCatcherData %>% 
  filter(Catcher_Full_Name %in% c("Heim, Jonah", "Garver, Mitch", "Hedges, Austin"),
         Throw.Type %in% c("Ground", "On Fly")) %>%
  group_by(Catcher_Full_Name, Base) %>% summarise(
    avg_TS = mean(Throw_Speed, na.rm = TRUE),
    avg_Exchange = mean(Exchange, na.rm = TRUE),
    avg_Accuracy = mean(Accuracy, na.rM = TRUE),
    avg_Teamwork = mean(Teamwork, na.rm = TRUE)
  )


############################ 

# Import the full base stealers steal attempts data and filter out instances with no tracked data
FullRunnerData <- read.csv("Model_Training/StealAttempts_pbp.csv") %>% filter(Lead_Distance_Gained != "--")

# Transform needed columns into numeric columns
cols <- c("Lead_Distance_Gained", "Pitcher_First_Move", "At_Pitch_Release")
FullRunnerData[cols] <- lapply(FullRunnerData[cols], as.numeric)

# For the application, the model will only include players who had at least 7 attempts to second
Qualified_Players <- FullRunnerData %>% group_by(Runner_Full_Name, Target.Base) %>%
  summarise(Count = n()) %>% filter(Target.Base == '2B', Count >= 7)
RunnerList <- Qualified_Players$Runner_Full_Name

# Grouo by Runner Name and Base being stolen and create the average runner profile for the 2023 season. 
RunnerProfiles <- FullRunnerData  %>%
  group_by(Runner_Full_Name, Target.Base) %>% summarise(
    Count = n(),
    avg_LeadGained = mean(Lead_Distance_Gained, na.rm = TRUE),
    avg_FirstMove = mean(Pitcher_First_Move, na.rm = TRUE),
    avg_atRelease = mean(At_Pitch_Release, na.rm = TRUE)
  ) %>% filter(Runner_Full_Name %in% RunnerList)


##########################


# Export the Data
write.csv(PitchProfiles_ROB, "Rangers2023_PitcherProfiles.csv")
write.csv(CatcherProfiles, "Rangers2023_CatcherProfiles.csv")
write.csv(RunnerProfiles, "MLB2023_RunnerProfiles.csv")
