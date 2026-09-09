# **MLS Attendance Analysis - Personal Project**
**Date:** August 2026 - September 2026

## **Description**
### **Introduction**
Over the past few years, soccer has garnered a higher following in the United States as attention surrounding both the domestic and international game has increased. From the arrival of Lionel Messi in Major League Soccer (MLS) to the United States co-hosting the 2026 FIFA World Cup, this period of increased attention marks a potential turning point for men's soccer in the country. My interest in this potential growth comes from the domestic side and looking at the consistency in attendance amongst MLS and their clubs. With high-profile players and major international events bringing high attention to the sport, I am interested in determining whether MLS has experienced notable growth in attendance and capacity utilization in recent seasons.

In this project, I analyze MLS attendance and venue capacity data from the 2022 through 2026 seasons, including data from the ongoing 2026 season. Using pgAdmin to create, store, and access data from a locally set-up PostgreSQL server, I query the data for analysis and use the resulting datasets to create a Tableau dashboard. The database contains five tables that were created using SQL CREATE TABLE statements and populated by importing data from CSV files. The project concludes with an analysis of the major findings, as well as recommendations and considerations for further evaluating the growth of the league.

### **The Data**
As mentioned, the local Postgres database contains 5 tables. 

1. The first table, MLS_Schedule, uses schedule data from FBRef to track game dates, home and away teams, venues, scores, attendance, and other game-level information

2. The second table, TeamInformation_MLS, contains information about each MLS club, including its name, location, founding year, conference, and a link to the club's logo on Wikipedia. 

3. The third table, Standings_MLS, contains each team's position in the standings following the conclusion of the regular season, including wins, draws, losses, position, conference, and season. This table does not include 2026 because the season is ongoing.

4. The fourth table, Stadiums_MLS, contains the stadiums appearing in the schedule data and their respective reported capacities.

5. Finally, the fifth table, TeamVenues_MLS, identifies which stadiums served as a particular team's home venue during each season. This accounts for teams that used multiple venues or played individual matches at alternative venues during a season.

### **Materials**
This project repository contains datasets used to populate the database tables and generated the Tableau Visualizations. I also include the SQL file of each query used in the project, containing comments that discuss findings and thoughts. Lastly, I include a markdown file that discuses my findings and recommendation on further analysis. As mentioned, a Tableau dashboard was created to display my findings. This dashboard has been published using Tableau Public, and can be viewed in the following link: 
