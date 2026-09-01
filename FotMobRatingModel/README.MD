## **FotMob Rating Model**

### **Purpose**

The Purpose of this project is to develop a model that replicates the player ratings given to players on the popular Soccer site "FotMob". Using Machine Learning and Statsbomb data from the 2018 and 2022 World Cup, I aim to create a model that intakes the player's stats for the game and predict the rating they might receive. The Final models are split into three separate Linear Regressions that take in data for attackers, midfielders and defenders. Each model is trained on that player's stat line and uses the fotmob rating from the match as the target variable. In the evaluation, I find that the attacker model performs the best, likely due to the chosen variables containing many offensive and passing specific metrics that are great at evaluating Forwards. The Midfielder and Defender models perform fairly similar, with the Defender model producing prediction ratings that tend to be roughly 0.4 rating points off. This is likely due to a different weighting scheme or possibly other variables by the site to generate their ratings. The model was tested on a train/test split of the 2018 world cup data, however the final evaluation looks at the rating accuracy on 4 games in the 2022 World Cup Finals (Both Semifinals, Third Place Game, and Final). Please see the Model development notebook (ipynb file) for the process to develop the models.

### **Areas for Improvement**

In order to use the fotmob rating, each game was manually scraped for each player's ratings. Therefore, only one full tournament was used to train the model and 4 games from the next were used to as a final display. Therefore, the evaluation is very limited and may not fully reflect the accuracy of the models. Further testing and evaluation of more out of sample data (more games from the 2022 and future World Cups) would help check if the model's findings and performance are consistent and accurate.

### **Tableau Dashboard**

A public tableau has been created to display the model's rating for every rated player in the final 4 games of the 2022 World Cup. Please see the link below to view. Please note that there is currently an issue where switching between games that does not have the same team results in a blank page. Selecting another player will fix the issue. This issue is currently being worked on.  

View the Tableau Dashboard: https://public.tableau.com/views/FotMob_RatingModel/Dashboard1?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link
