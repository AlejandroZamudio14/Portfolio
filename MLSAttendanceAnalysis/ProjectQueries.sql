-- ============================================================
-- Table Creation: Data was imported into tables via CSV files
-- ============================================================

CREATE TABLE TeamInformation_MLS(
	Team VARCHAR(255) PRIMARY KEY,
	City TEXT NOT NULL,
	Conference VARCHAR(255) NOT NULL,
	Founded INT NOT NULL,
	Logo TEXT
)

CREATE TABLE Stadiums_MLS(
	Stadium VARCHAR(255) PRIMARY KEY,
	Capacity INT NOT NULL
)

CREATE TABLE MLS_Schedule(
	ROUND VARCHAR(255),
	Day_Of_Week VARCHAR(255),
	Game_Date DATE,
	Start_Times VARCHAR(255),
	Start_Time_Local TIME,
	Home_Team VARCHAR(255),
	Home_Goals INT,
	Home_Penalties INT,
	Away_Penalties INT,
	Away_Goals INT,
	Away_Team VARCHAR(255),
	Final_Score VARCHAR(255),
	Attendance INT,
	Venue VARCHAR(255),
	Referee VARCHAR(255),
	Notes TEXT,
	PRIMARY KEY(Game_Date, Home_Team, Away_Team),
	CONSTRAINT fk_AwayTeam FOREIGN KEY (Away_Team) REFERENCES TeamInformation_MLS(Team),
	CONSTRAINT fk_HomeTeam FOREIGN KEY (Home_Team) REFERENCES TeamInformation_MLS(Team)
)


CREATE TABLE Standings_MLS(
	POS INT NOT NULL,
	Team VARCHAR(255),
	PLD INT,
	Wins INT,
	Draws INT,
	Losses INT,
	Goals_For INT,
	Goals_Allowed INT,
	Goal_Differential INT,
	Points INT,
	Notes TEXT,
	Conference VARCHAR(255),
	Season INT,
	PRIMARY KEY(Team, Season),
	CONSTRAINT fk_Team FOREIGN KEY (Team) REFERENCES TeamInformation_MLS(Team)
)


CREATE TABLE TeamVenues_MLS(
	Team VARCHAR(255),
	Stadium VARCHAR(255),
	Season INT,
	PRIMARY KEY(Team, Stadium, Season),
	CONSTRAINT fk_Team FOREIGN KEY (Team) REFERENCES TeamInformation_MLS(Team),
	CONSTRAINT fk_Stadium FOREIGN KEY (Stadium) REFERENCES Stadiums_MLS(Stadium)
)


-- ============================================================
-- Tableau Data: Used for Visualizations; exported via csv
-- ============================================================


-- Team Information Map
SELECT 
	Team, SPLIT_PART(City, ', ', 1) AS Team_City, SPLIT_PART(City, ', ', 2) AS Team_State,
	CASE
		WHEN SPLIT_PART(City, ', ', 2) IN ('Ontario', 'British Columbia', 'Quebec') THEN 'Canada'
		ELSE 'United States'
END AS Team_Country, 
Conference, Founded
FROM TeamInformation_MLS;


-- Team Seasons: Used to connect datasets
SELECT 
	DISTINCT EXTRACT(YEAR FROM Game_Date) AS Season, Home_Team
FROM MLS_Schedule
ORDER BY Home_Team, EXTRACT(YEAR FROM Game_Date);


-- Median and Mean Bar Chart
SELECT
    EXTRACT(YEAR FROM Game_Date) AS Season, Home_Team, AVG(Attendance) AS Mean_Attendance,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY Attendance) AS Median_Attendance,
    COUNT(Game_Date) AS Home_Games
FROM MLS_Schedule
GROUP BY EXTRACT(YEAR FROM Game_Date), Home_Team
ORDER BY EXTRACT(YEAR FROM Game_Date);


-- Attendance on Daily Basis
SELECT
    EXTRACT(YEAR FROM Game_Date) AS Season, Game_Date, Home_Team, Attendance
FROM MLS_Schedule
ORDER BY Home_Team, Game_Date;



-- Team Venue and Filled Capacity Info
WITH Stadium_Info AS (
	SELECT 
		EXTRACT(YEAR FROM Game_Date) AS Season, Game_Date, Home_Team, Venue, 
		ROW_NUMBER() OVER(PARTITION BY EXTRACT(YEAR FROM Game_Date), Home_Team, Venue 
		ORDER BY Game_Date) AS Home_GP
	FROM MLS_Schedule
),
PrevGame AS (
	SELECT 
		Season, Home_Team, Venue, MAX(Game_Date) AS Last_Played, MAX(Home_GP) AS Total_GP
	FROM Stadium_Info
	GROUP BY Season, Home_Team, Venue
)
SELECT 
	PG.Season, MS.Home_Team, MS.Venue, AVG(MS.Attendance) AS Avg_Attenance, 
	MAX(PG.Last_Played) AS Last_Played, MAX(PG.Total_GP) AS Total_GP, 
	MAX(SM.Capacity) AS Capacity, 
	(1.0 * AVG(MS.Attendance) / MAX(SM.Capacity)) AS Avg_Fill
FROM MLS_Schedule AS MS 
LEFT JOIN PrevGame AS PG ON PG.Home_Team = MS.Home_Team AND PG.Venue = MS.Venue
LEFT JOIN Stadiums_MLS AS SM ON MS.Venue = SM.Stadium
GROUP BY PG.Season, MS.Home_Team, MS.Venue
ORDER BY MS.Home_Team;



-- ============================================================
-- Analysis of Attendance and Capacity Filled
-- ============================================================


-- 1). How has Attendance changed season by season
WITH Season_Avg_Attendance AS (
	SELECT 
		EXTRACT(YEAR FROM Game_Date) AS Season, AVG(Attendance) AS Average_Attendance
	FROM MLS_Schedule
	GROUP BY EXTRACT(YEAR FROM Game_Date)
	ORDER BY EXTRACT(YEAR FROM Game_Date)
)
SELECT *, 
	ROUND((1.0 * (Average_Attendance - LAG(Average_Attendance, 1) OVER()) / LAG(Average_Attendance, 1) 
	OVER()), 4) * 100 AS Percent_Change
FROM Season_Avg_Attendance;
-- Each season besides 2025 see an increase in average attendance with the ongoing 2026 Season
-- sporting the lowest increase from the previous season.


WITH Season_Avg_Capacity AS(
	SELECT 
		EXTRACT(YEAR FROM MS.Game_Date) AS Season,
		AVG((1.0 * MS.Attendance / SM.Capacity)) AS Average_Capacity_Utilization
	FROM MLS_Schedule AS MS
	INNER JOIN TeamVenues_MLS AS TV ON EXTRACT(YEAR FROM MS.Game_Date) = TV.Season AND MS.Venue = TV.Stadium
	INNER JOIN Stadiums_MLS AS SM ON TV.Stadium = SM.Stadium
	GROUP BY EXTRACT(YEAR FROM MS.Game_Date)
	ORDER BY EXTRACT(YEAR FROM MS.Game_Date)
)
SELECT *, 
	ROUND((1.0 * (Average_Capacity_Utilization - LAG(Average_Capacity_Utilization, 1) OVER()) / LAG(Average_Capacity_Utilization, 1) 
	OVER()), 4) * 100 AS Percent_Change
FROM Season_Avg_Capacity;
-- A look at league wide capacity utilization shows a similar trend, except the ongoing 2026
-- Season has a decrease in average capacity filled.


WITH Season_Totals AS (
	SELECT 
		EXTRACT(YEAR FROM Game_Date) AS Season, SUM(Attendance) AS Total_Attendance, 
		COUNT(Attendance) AS Total_Games
	FROM MLS_Schedule
	GROUP BY EXTRACT(YEAR FROM Game_Date)
	ORDER BY EXTRACT(YEAR FROM Game_Date)
)
SELECT *, 
	ROUND((1.0 * (Total_Attendance - LAG(Total_Attendance, 1) OVER()) / LAG(Total_Attendance, 1) 
	OVER()), 4) * 100 AS Percent_Change
FROM Season_Totals;
-- With a notable increase in the amount of games played in the entire season, there is a increase
-- in total attendance with the exception of the 2025 season. Note: the 2026 season is ongoing,
-- so change in total attendance only reflects half the 2026 Season.

-- =======================================================================================


-- 2). Evaluation at the Team Level
WITH Team_Average_Attendance AS (
	SELECT 
		EXTRACT(YEAR FROM Game_Date) AS Season, Home_Team,
		AVG(Attendance) AS Average_Attendance
	FROM MLS_Schedule
	GROUP BY Home_Team, EXTRACT(YEAR FROM Game_Date)
	ORDER BY Home_Team, EXTRACT(YEAR FROM Game_Date)
)
SELECT *, 
	ROUND((1.0 * (Average_Attendance - LAG(Average_Attendance, 1) 
	OVER(PARTITION BY Home_Team ORDER BY Season)) / LAG(Average_Attendance, 1) 
	OVER(PARTITION BY Home_Team ORDER BY Season)), 4) * 100 AS Percent_Change
FROM Team_Average_Attendance;
-- Data reflects a common trend where many teams have an increase in average attendance in 2023
-- and 2024, however do experience some sort of decrease in the 2025 season. A total of 22 teams
-- saw a decrease in average attendance in 2025


WITH Team_Avg_Capacity AS(
	SELECT 
		EXTRACT(YEAR FROM MS.Game_Date) AS Season, Home_Team,
		AVG((1.0 * MS.Attendance / SM.Capacity)) AS Average_Capacity_Utilization
	FROM MLS_Schedule AS MS
	INNER JOIN TeamVenues_MLS AS TV ON EXTRACT(YEAR FROM MS.Game_Date) = TV.Season AND MS.Venue = TV.Stadium
	INNER JOIN Stadiums_MLS AS SM ON TV.Stadium = SM.Stadium
	GROUP BY Home_Team, EXTRACT(YEAR FROM MS.Game_Date)
	ORDER BY Home_Team, EXTRACT(YEAR FROM MS.Game_Date)
)
SELECT *, 
	ROUND((1.0 * (Average_Capacity_Utilization - LAG(Average_Capacity_Utilization, 1) 
	OVER(PARTITION BY Home_Team ORDER BY Season)) / LAG(Average_Capacity_Utilization, 1) 
	OVER(PARTITION BY Home_Team ORDER BY Season)), 4) * 100 AS Percent_Change
FROM Team_Avg_Capacity;
-- Similar to the average attendance, the data shows that a majority of teams see increases in
-- capacity utilization in 2023 and 2024, however experience some sort of decline in 2025.



-- ============================================================
-- Specific Analysis Pieces
-- ============================================================

-- 3). 2025 Analysis
-- 2025 Saw a notable drop in league average attendance and capticity utilization, both dropping
-- around 5% or more. Because the 2026 Season is still ongoing, the negative percent change is 
-- not as trustworthy. However, its important to evaluate on a team basis
WITH Team_Average_Attendance AS (
	SELECT 
		EXTRACT(YEAR FROM Game_Date) AS Season, Home_Team,
		AVG(Attendance) AS Average_Attendance
	FROM MLS_Schedule
	GROUP BY Home_Team, EXTRACT(YEAR FROM Game_Date)
	ORDER BY Home_Team, EXTRACT(YEAR FROM Game_Date)
),
Seasonal_Change AS (
SELECT *, 
	ROUND((1.0 * (Average_Attendance - LAG(Average_Attendance, 1) 
	OVER(PARTITION BY Home_Team ORDER BY Season)) / LAG(Average_Attendance, 1) 
	OVER(PARTITION BY Home_Team ORDER BY Season)), 4) * 100 AS Percent_Change
FROM Team_Average_Attendance
)
SELECT 
	SC.Season, SC.Home_Team, SC.Percent_Change, 
	1.0 * SM.Wins / (SM.Wins + SM.Losses + SM.Draws) AS Win_Perc
FROM Seasonal_Change AS SC INNER JOIN Standings_MLS AS SM
ON SC.Home_Team = SM.Team AND SC.Season = SM.Season
WHERE SC.Season = 2025
ORDER BY 1.0 * SM.Wins / (SM.Wins + SM.Losses + SM.Draws) DESC;
-- While not completely linear, the data shows that among clubs with lower winning percentages, 
-- there was a tendency for average attendance to decline by a larger percentage. Although 22 
-- clubs experienced a negative change in average attendance, the largest declines were 
-- among clubs with poorer season performance.
WITH Team_Avg_Capacity AS(
	SELECT 
		EXTRACT(YEAR FROM MS.Game_Date) AS Season, Home_Team,
		AVG((1.0 * MS.Attendance / SM.Capacity)) AS Average_Capacity_Utilization
	FROM MLS_Schedule AS MS
	INNER JOIN TeamVenues_MLS AS TV ON EXTRACT(YEAR FROM MS.Game_Date) = TV.Season AND MS.Venue = TV.Stadium
	INNER JOIN Stadiums_MLS AS SM ON TV.Stadium = SM.Stadium
	GROUP BY Home_Team, EXTRACT(YEAR FROM MS.Game_Date)
	ORDER BY Home_Team, EXTRACT(YEAR FROM MS.Game_Date)
),
Capacity_Seasonal_Change AS(
SELECT *, 
	ROUND((1.0 * (Average_Capacity_Utilization - LAG(Average_Capacity_Utilization, 1) 
	OVER(PARTITION BY Home_Team ORDER BY Season)) / LAG(Average_Capacity_Utilization, 1) 
	OVER(PARTITION BY Home_Team ORDER BY Season)), 4) * 100 AS Percent_Change
FROM Team_Avg_Capacity
)
SELECT 
	CSC.Season, CSC.Home_Team, CSC.Percent_Change, 
	1.0 * SM.Wins / (SM.Wins + SM.Losses + SM.Draws) AS Win_Perc
FROM Capacity_Seasonal_Change AS CSC INNER JOIN Standings_MLS AS SM
ON CSC.Home_Team = SM.Team AND CSC.Season = SM.Season
WHERE CSC.Season = 2025
ORDER BY 1.0 * SM.Wins / (SM.Wins + SM.Losses + SM.Draws) DESC;
-- A similar pattern can be seen in the change in capacity utilization. A majority of clubs with 
-- lower winning percentages saw larger negative changes. While most clubs saw a decline in 
-- utilization, the largest changes were among the low performing clubs. It is important to note that 
-- this analysis is limited to attendance and performance data. Additional context such as club 
-- circumstances, stadium or venue changes, scheduling and external issues would be needed 
-- before drawing concrete conclusions. 


-- 4). Highest and Lowest changes per Season
WITH Team_Average_Attendance AS (
	SELECT 
		EXTRACT(YEAR FROM Game_Date) AS Season, Home_Team,
		AVG(Attendance) AS Average_Attendance
	FROM MLS_Schedule
	GROUP BY Home_Team, EXTRACT(YEAR FROM Game_Date)
	ORDER BY Home_Team, EXTRACT(YEAR FROM Game_Date)
),
Seasonal_Change AS (
SELECT *, 
	ROUND((1.0 * (Average_Attendance - LAG(Average_Attendance, 1) 
	OVER(PARTITION BY Home_Team ORDER BY Season)) / LAG(Average_Attendance, 1) 
	OVER(PARTITION BY Home_Team ORDER BY Season)), 4) * 100 AS Percent_Change
FROM Team_Average_Attendance
)
SELECT *, 
	RANK() OVER(PARTITION BY Season ORDER BY Percent_Change DESC) AS PC_Rank
FROM Seasonal_Change
WHERE Season NOT IN (2022, 2026);
-- Attendance Changes:
-- 2023: Best - Inter Miami (40% change), Worst: Houston Dynamo FC (-5% Change)
-- 2024: Best - Vancouver Whitecaps FC (45% change), Worst: San Jose Earthquakes (-5% Change)
-- 2025: Best - San Jose Earthquakes (12% change), Worst: FC Dallas (-42% Change)
WITH Team_Avg_Capacity AS(
	SELECT 
		EXTRACT(YEAR FROM MS.Game_Date) AS Season, Home_Team,
		AVG((1.0 * MS.Attendance / SM.Capacity)) AS Average_Capacity_Utilization
	FROM MLS_Schedule AS MS
	INNER JOIN TeamVenues_MLS AS TV ON EXTRACT(YEAR FROM MS.Game_Date) = TV.Season AND MS.Venue = TV.Stadium
	INNER JOIN Stadiums_MLS AS SM ON TV.Stadium = SM.Stadium
	GROUP BY Home_Team, EXTRACT(YEAR FROM MS.Game_Date)
	ORDER BY Home_Team, EXTRACT(YEAR FROM MS.Game_Date)
),
Capacity_Seasonal_Change AS(
SELECT *, 
	ROUND((1.0 * (Average_Capacity_Utilization - LAG(Average_Capacity_Utilization, 1) 
	OVER(PARTITION BY Home_Team ORDER BY Season)) / LAG(Average_Capacity_Utilization, 1) 
	OVER(PARTITION BY Home_Team ORDER BY Season)), 4) * 100 AS Percent_Change
FROM Team_Avg_Capacity
)
SELECT *, 
	RANK() OVER(PARTITION BY Season ORDER BY Percent_Change DESC) AS PC_Rank
FROM Capacity_Seasonal_Change
WHERE Season NOT IN (2022, 2026);
-- Capacity Utilization Changes:
-- 2023: Best - Inter Miami (40% change), Worst: LA Galaxy (-9% Change)
-- 2024: Best - Vancouver Whitecaps FC (45% change), Worst: San Jose Earthquakes (-5% Change)
-- 2025: Best - San Jose Earthquakes (13% change), Worst: FC Dallas (-42% Change)

-- Explanations: The 2023 season marked Leo Messi's first season in MLS, driving lots of interest
-- and attention to the club. This season also saw another season of fairly poor results by the
-- Houston Dynamo and affects of boycotts by fans of LA Galaxy. 2024 saw the Whitecaps greatly 
-- market season tickets as part of the "Messi Affect", along with the club performing at a higher
-- level. The Earthquakes had struggled heavily this season, posting a poor performance and slight
-- dip in attendance and amount of capacity filled. In 2025, the Earthquakes managed to recover
-- lost attendance by performing much better, thanks to increased investment because of the 
-- upcoming World Cup, their match against Inter Miami, and a match against LAFC at a higher
-- capacity venue. FC Dallas saw attendance and capacity drop significantly due to major 
-- renovations of their current stadium


-- In the 2025 Season, many teams saw notable dips in average attendance and capacity filled in 
-- venues. With much attention on international tournaments such as Gold Cup and Club World Cup,
-- it is possible these tournaments took most attention away from the league. Furthermore, the
-- Messi hype had begun to enter its third, possibly causing a level out of interest to a 
-- consistent basis. While teams continue to acquire international stars, most do not carry the
-- huge hype that came with Messi. 


-- 5). Highest Attended Game Each Season
WITH Venue_Use AS (
	SELECT 
		EXTRACT(YEAR FROM Game_Date) AS Season, Venue, COUNT(Game_Date) AS Games_Played
	FROM MLS_Schedule
	GROUP BY EXTRACT(YEAR FROM Game_Date), Venue
),
Attendance_Ranks AS (
	SELECT 
		EXTRACT(YEAR FROM MS.Game_Date) AS Season, MS.Game_Date, MS.Home_Team, MS.Away_Team, 
		MS.Attendance, MS.Venue,
		RANK() OVER(PARTITION BY EXTRACT(YEAR FROM MS.Game_Date) ORDER BY MS.Attendance DESC) AS Attendance_Rank,
		VU.Games_Played
	FROM MLS_Schedule AS MS INNER JOIN Venue_Use AS VU 
	ON MS.Venue = VU.Venue AND EXTRACT(YEAR FROM MS.Game_Date) = VU.Season
	WHERE Attendance IS NOT NULL
)
SELECT * FROM Attendance_Ranks WHERE Attendance_Rank = 1
-- In 2023, 2024 and 2026, the highest attended matches were at one game use venues with the
-- Rose Bowl hosting the most attended game in this five year stretch. 2022 and 2025 see the home
-- stadiums for Charlotte FC and Atlanta United hosting the highest viewed matches (stadiums 
-- parimarly used for NFL Games). 


-- 6). Expansion Teams (Past 5 Seasons)
WITH Team_Average_Attendance AS (
	SELECT 
		EXTRACT(YEAR FROM Game_Date) AS Season, Home_Team,
		AVG(Attendance) AS Average_Attendance
	FROM MLS_Schedule
	GROUP BY Home_Team, EXTRACT(YEAR FROM Game_Date)
	ORDER BY Home_Team, EXTRACT(YEAR FROM Game_Date)
)
SELECT *, 
	ROUND((1.0 * (Average_Attendance - LAG(Average_Attendance, 1) 
	OVER(PARTITION BY Home_Team ORDER BY Season)) / LAG(Average_Attendance, 1) 
	OVER(PARTITION BY Home_Team ORDER BY Season)), 4) * 100 AS Percent_Change
FROM Team_Average_Attendance
WHERE Home_Team IN ('Austin FC', 'Charlotte FC', 'St. Louis City SC', 'San Diego FC');
-- Charlotte FC has seen the worst percent change with two straight seasons of decline in 
-- average attendance. San Diego FC (first season in 2025) has too little data to make 
-- well backed colunclusions.
WITH Team_Avg_Capacity AS(
	SELECT 
		EXTRACT(YEAR FROM MS.Game_Date) AS Season, Home_Team,
		AVG((1.0 * MS.Attendance / SM.Capacity)) AS Average_Capacity_Utilization
	FROM MLS_Schedule AS MS
	INNER JOIN TeamVenues_MLS AS TV ON EXTRACT(YEAR FROM MS.Game_Date) = TV.Season AND MS.Venue = TV.Stadium
	INNER JOIN Stadiums_MLS AS SM ON TV.Stadium = SM.Stadium
	GROUP BY Home_Team, EXTRACT(YEAR FROM MS.Game_Date)
	ORDER BY Home_Team, EXTRACT(YEAR FROM MS.Game_Date)
)
SELECT *, 
	ROUND((1.0 * (Average_Capacity_Utilization - LAG(Average_Capacity_Utilization, 1) 
	OVER(PARTITION BY Home_Team ORDER BY Season)) / LAG(Average_Capacity_Utilization, 1) 
	OVER(PARTITION BY Home_Team ORDER BY Season)), 4) * 100 AS Percent_Change
FROM Team_Avg_Capacity
WHERE Home_Team IN ('Austin FC', 'Charlotte FC', 'St. Louis City SC', 'San Diego FC');
-- The trends reflected in the average attendance of expansion teams is seen in the capacity
-- utilization with Charlotte FC seeing decline from their peak in 2023. 