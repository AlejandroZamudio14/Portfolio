library(ggplot2)
library(tidyverse)



## Project Work ##


# import Data
schedule <- read.csv('TexSoftbal_Schedules - FINAL.csv')
hitting <- read.csv('TexSoftbal_Hitting_FINAL.csv')
pitching <- read.csv('TexSoftbal_Pitching_FINAL.csv')

# Schedule Filters #

HomeGms <- schedule %>% filter(
  home_team == 'Texas'
)

AwayGms <- schedule %>% filter(
  away_team == 'Texas'
)

RealHomeGms <- schedule %>% filter(
  home_team == 'Texas',
  Neutral.Site == 0
)

TournamentGms <- schedule %>% filter(
  Tournament == 1
)

NonHome_Distance <- schedule %>% filter(
  Distance..Miles. > 0
) %>% arrange(
  Distance..Miles.
)


## Working ##

### Average distance traveled per season; Games away from Austin 
NonHome_Distance %>% group_by(
  Season
) %>% summarise(
  Games = n(),
  Avg_distance = mean(Distance..Miles.)
)


### Average RPI Rank of Opponents for each season
schedule %>% group_by(
  Season
) %>% summarise(
  Games = n(),
  Avg_Opp_RPI = mean(Opponent.RPI.Rank)
)

ggplot(schedule, aes(x = Distance..Miles., y = Opponent.RPI.Rank))+
  geom_point()

ggplot(schedule, aes(x = Texas.Win, y = Opponent.RPI.Rank))+
  geom_point()



################################################################

### Summarize by each game ###

series_data <- schedule %>% select(
  game_id, Series, Distance..Miles.
)

series_hitting <- hitting %>% inner_join(
  series_data, by = "game_id"
)

series_hitting_IndGames <- series_hitting %>% group_by(
  game_id
) %>% summarize(
  ab = sum(ab),
  rbi = sum(rbi),
  runs = sum(r),
  hits = sum(h),
  doubles = sum(x2b),
  triples = sum(x3b),
  home_runs = sum(hr),
  total_bases = sum(tb),
  walks = sum(bb),
  intentional_walks = sum(ibb),
  hit_by_pitch = sum(hbp),
  sac_fly = sum(sf),
  sac_hit = sum(sh),
  strikeout = sum(k),
  strikeout_looking = sum(kl),
  double_plays = sum(dp),
  gidp = sum(gdp),
  triple_plays = sum(tp),
  stolen_bases = sum(sb),
  caught_stealing = sum(cs),
  picked = sum(picked),
  groundout = sum(go),
  flyout = sum(fo),
  Distance = mean(Distance..Miles.),
  Season = mean(season)
) %>% arrange(
  Distance
)

series_pitching <- pitching %>% inner_join(
  series_data, by = "game_id"
)


series_pitching_IndGames <- series_pitching %>% group_by(
  game_id
) %>% summarize(
  avg_ip = mean(ip),
  hits_allowed = sum(ha),
  earned_runs = sum(er),
  walks = sum(bb),
  hit_batters = sum(hb),
  strikeouts = sum(so),
  batters_faced = sum(bf),
  hr_allowed = sum(hr_a),
  grondouts = sum(go),
  flyouts = sum(fo),
  Distance = mean(Distance..Miles.),
  Season = mean(season)
) %>% arrange(
  Distance
)


# Hitting Measurable #

cor(series_hitting_IndGames$Distance, series_hitting_IndGames)


ggplot(series_hitting_IndGames, aes(x = Distance, y = strikeout, color = Season)) +
  geom_point(size = 1.5) +
  facet_wrap(~ Season, ncol = 1) +
  theme_minimal()

ggplot(series_hitting_IndGames, aes(x = Distance, y = walks, color = Season)) +
  geom_point(size = 1.5) +
  facet_wrap(~ Season, ncol = 1) +
  theme_minimal()

ggplot(series_hitting_IndGames, aes(x = Distance, y = strikeout_looking, color = Season)) +
  geom_point(size = 1.5) +
  facet_wrap(~ Season, ncol = 1) +
  theme_minimal()

ggplot(series_hitting_IndGames, aes(x = Distance, y = runs, color = Season)) +
  geom_point(size = 1.5) +
  facet_wrap(~ Season, ncol = 1) +
  theme_minimal()


# Pitching Measurable #
cor(series_pitching_IndGames$Distance, series_pitching_IndGames)

ggplot(series_pitching_IndGames, aes(x = Distance, y = avg_ip, color = Season)) +
  geom_point(size = 1.5) +
  facet_wrap(~ Season, ncol = 1) +
  theme_minimal()

ggplot(series_pitching_IndGames, aes(x = Distance, y = earned_runs, color = Season)) +
  geom_point(size = 1.5) +
  facet_wrap(~ Season, ncol = 1) +
  theme_minimal()

ggplot(series_pitching_IndGames, aes(x = Distance, y =strikeouts, color = Season)) +
  geom_point(size = 1.5) +
  facet_wrap(~ Season, ncol = 1) +
  theme_minimal()

ggplot(series_pitching_IndGames, aes(x = Distance, y = walks, color = Season)) +
  geom_point(size = 1.5) +
  facet_wrap(~ Season, ncol = 1) +
  theme_minimal()


### Summarize by each series ###

series_hitting_summary <- series_hitting %>% group_by(
  Series
) %>% summarize(
  games = length(unique(game_id)),
  ab = sum(ab),
  rbi = sum(rbi),
  runs = sum(r),
  hits = sum(h),
  doubles = sum(x2b),
  triples = sum(x3b),
  home_runs = sum(hr),
  total_bases = sum(tb),
  walks = sum(bb),
  strikeout = sum(k),
  strikeout_looking = sum(kl),
  double_plays = sum(dp),
  gidp = sum(gdp),
  ba = sum(h) / sum(ab),
  obp = (sum(h) + sum(bb) + sum(hbp)) / (sum(ab) + sum(bb) + sum(hbp) + sum(sf)),
  slg = ((sum(h) - sum(doubles) - sum(triples) - sum(home_runs)) + (2*sum(doubles)) + (3*sum(triples)) + (4*sum(home_runs))) / sum(ab),
  Distance = mean(Distance..Miles.),
  Season = mean(season)
) %>% arrange(
  Distance
)

series_pitching_summary <- series_pitching %>% group_by(
  Series
) %>% summarize(
  avg_ip = mean(ip),
  hits_allowed = sum(ha),
  ERA = (sum(er)/sum(ip)) * 7,
  walks = sum(bb),
  strikeouts = sum(so),
  batters_faced = sum(bf),
  hr_allowed = sum(hr_a),
  Distance = mean(Distance..Miles.),
  Season = mean(season)
) %>% arrange(
  Distance
)


# visualize #

cor(series_hitting_summary$Distance, series_hitting_summary$ba)
cor(series_hitting_summary$Distance, series_hitting_summary$obp)
cor(series_hitting_summary$Distance, series_hitting_summary$slg)

ggplot(series_hitting_summary, aes(x = Distance, y = ba, color = Season)) +
  geom_point(size = 1.5) +
  facet_wrap(~ Season, ncol = 1) +
  theme_minimal()

ggplot(series_hitting_summary, aes(x = Distance, y = obp, color = Season)) +
  geom_point(size = 1.5) +
  facet_wrap(~ Season, ncol = 1) +
  theme_minimal()


ggplot(series_hitting_summary, aes(x = Distance, y = slg, color = Season)) +
  geom_point(size = 1.5) +
  facet_wrap(~ Season, ncol = 1) +
  theme_minimal()



cor(series_pitching_summary$Distance, series_pitching_summary$ERA)

ggplot(series_pitching_summary, aes(x = Distance, y = ERA, color = Season)) +
  geom_point(size = 1.5) +
  facet_wrap(~ Season, ncol = 1) +
  theme_minimal()

ggplot(series_pitching_summary, aes(x = Distance, y = strikeouts, color = Season)) +
  geom_point(size = 1.5) +
  facet_wrap(~ Season, ncol = 1) +
  theme_minimal()



#######################################################################


### Looking at Home Games Only ###

Home_indGamesHitting <- series_hitting_IndGames %>% filter(
  Distance == 0
)

ggplot(Home_indGamesHitting, aes(x = hits)) + geom_histogram()
ggplot(Home_indGamesHitting, aes(x = strikeout)) + geom_histogram()
ggplot(Home_indGamesHitting, aes(x = walks)) + geom_histogram()
ggplot(Home_indGamesHitting, aes(x = total_bases)) + geom_histogram()


Home_SeriesHitting <- series_hitting_summary %>% filter(
  Distance == 0
)

ggplot(Home_SeriesHitting, aes(x = ba)) + geom_histogram()
ggplot(Home_SeriesHitting, aes(x = obp)) + geom_histogram()
ggplot(Home_SeriesHitting, aes(x = slg)) + geom_histogram()


Hoe_indGamesPitching <- series_pitching_IndGames %>% filter(
  Distance > 0
)

ggplot(Home_indGamesPitching, aes(x = strikeouts)) + geom_histogram()
ggplot(Home_indGamesPitching, aes(x = walks)) + geom_histogram()
ggplot(Home_indGamesPitching, aes(x = hits_allowed)) + geom_histogram()


Away_SeriesPitching <- series_pitching_summary %>% filter(
  Distance > 0
)

ggplot(Home_SeriesPitching, aes(x = ERA)) + geom_histogram()
