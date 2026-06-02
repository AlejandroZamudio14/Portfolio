library(shiny)
library(ranger)
library(xgboost)

# load models
final_LR <- readRDS("FINAL_LogReg_FDC.rds")
final_RF <- readRDS("FINAL_RanFor_FDC.rds")
final_XGB <- readRDS("FINAL_XGB_FDC.rds")

# must match training order exactly
features <- c(
  "yardline_100",
  "ydstogo",
  "goal_to_go",
  "score_differential",
  "half_seconds_remaining",
  "td_prob",
  "matchup_strength"
)

ui <- fluidPage(
  
  titlePanel("4th Down Conversion Model"),
  
  sidebarLayout(
    sidebarPanel(
      
      selectInput(
        "model_type",
        "Choose Model",
        choices = c("Logistic Regression", "Random Forest", "XGBoost")
      ),
      
      numericInput("yardline_100", "Yardline (100 = own goal, 0 = opponent goal)", 50),
      numericInput("ydstogo", "Yards to Go", 5),
      checkboxInput("goal_to_go", "Goal to Go", FALSE),
      numericInput("score_differential", "Score Differential", 0),
      numericInput("half_seconds_remaining", "Seconds Remaining (Half)", 900),
      numericInput("td_prob", "TD Probability", 0.2),
      numericInput("matchup_strength", "Matchup Strength", 0),
      
      actionButton("predict", "Predict")
    ),
    
    mainPanel(
      uiOutput("prob"),
      uiOutput("recommendation")
    )
  )
)

server <- function(input, output) {
  
  observeEvent(input$predict, {
    
    new_data <- data.frame(
      yardline_100 = input$yardline_100,
      ydstogo = input$ydstogo,
      goal_to_go = as.numeric(input$goal_to_go),
      score_differential = input$score_differential,
      half_seconds_remaining = input$half_seconds_remaining,
      td_prob = input$td_prob,
      matchup_strength = input$matchup_strength
    )
    
    prob <- switch(
      input$model_type,
      
      "Logistic Regression" = predict(final_LR, newdata = new_data, type = "response"),
      
      "Random Forest" = predict(final_RF, data = new_data)$predictions[,2],
      
      "XGBoost" = {
        dnew <- xgb.DMatrix(as.matrix(new_data[, features]))
        predict(final_XGB, dnew)
      }
    )
    
    output$prob <- renderUI({
      div(
        style = "
          border: 2px solid #1f77b4;
          border-radius: 12px;
          padding: 15px;
          font-size: 18px;
          margin-top: 10px;
          background-color: #f8fbff;
        ",
        paste("Conversion Probability:", round(prob, 3))
      )
    })
    
    output$recommendation <- renderUI({
      
      rec <- if (prob > 0.55) {
        "Recommendation: GO FOR IT"
      } else {
        "Recommendation: PUNT / FG"
      }
      
      div(
        style = "
          border: 2px solid #1f77b4;
          border-radius: 12px;
          padding: 15px;
          font-size: 18px;
          margin-top: 10px;
          background-color: #f8fbff;
        ",
        rec
      )
    })
    
  })
}

shinyApp(ui, server)