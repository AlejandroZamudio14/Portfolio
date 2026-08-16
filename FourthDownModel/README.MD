# **NFL Fourth Down Model - Personal Project**
**Date:** June 2026

## **Description** 
Over the past decades, the decision for a football team to "go for it" on 4th down has become a much more common choice. What was once seen as a risky and desperate decision has now become a common way for teams to keep their offense on the field and continue promising drives. With many factors playing into the decision to run a play on 4th down, I am interested in attempting to take those factors and create a model to provide insight in the decision to go for it. 

In this project, I perform analysis of fourth down data for the 2021 to 2025 NFL seasons to evaluate trends and see if I can identify key factors to making the decision to go for it. Through this analysis, I train three different machine learning models (Logistic Regression, Random Forest, XGBoost) with the intent of taking "pre-snap" variables and generating the probability of converting on 4th down. The decision to use three different models is aimed at observing the difference in predictions between the models. The models use the same variables which include: the Yard Line the ball is currently on, the yards to go until a first down is reached, if the possession is in a goal down scenario (4th and Goal), the score differential at the time of the play, the amount of seconds left in the half, the current probability of scoring a touchdown, and a matchup strength variable using EPA developed in my code. 

The folder contains the RMarkdown file and resulting PDF file that serve as a "Report" to document the steps taken to train the model and perform analysis, the files for the ML Models, and the app and manifest file used in the resulting shiny app. 

## **Project Shiny App**
For this project, a Shiny App was created to serve as a visual and interactive use of the models. The app allows you to choose the specific model and insert variable values to generate the probability of converting. For this app, I use a prediction cutoff of 0.55 to display text stating to "Go For It" or "Kick / FG Attempt". This cutoff was chosen as a personal cutoff. The shiny app is accessible below:

**ShinyApp URL:** https://019e8a4b-6786-3a48-10bb-f297f17abb36.share.connect.posit.cloud


