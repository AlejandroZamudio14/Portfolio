## Alejandro Zamudio: 2023 Rangers Battery Stealing Probability


##################

# Libraries
library(shiny)
library(dplyr)
library(tidyverse)
library(stringr)
library(xgboost)
library(caret)

##################

# Import Model and Dummy Variables
model <- readRDS("stolen_base_xgb_model.rds")
dummy <- readRDS("stolen_base_dummy_vars.rds")

# Import Data for Pitcher, Catcher and Runner Profiles
Pitchers <- read.csv("Profiles/Rangers2023_PitcherProfiles.csv", fileEncoding = "UTF-8")
Catchers <- read.csv("Profiles/Rangers2023_CatcherProfiles.csv", fileEncoding = "UTF-8")
Runners <- read.csv("Profiles/MLB2023_RunnerProfiles.csv", fileEncoding = "Windows-1252")

##################

# Define UI for application that draws a histogram
ui <- fluidPage(
  
  tags$head(
    tags$style(HTML("
    
    body {
      background-color: #f5f6f8;
      font-family: Arial, sans-serif;
    }

    .container-fluid {
      max-width: 1200px;
      margin: 0 auto;
      padding: 20px 30px;
    }

  "))
  ),
  
  
  # Application description
  # Application header
  div(
    style = "
    background-color: #003278;
    color: white;
    padding: 35px 40px;
    margin-bottom: 30px;
    text-align: center;
    border-radius: 8px;
  ",
    
    h1(
      "Base Stealing Matchup Predictor",
      style = "
      font-weight: bold;
      margin-top: 0;
      margin-bottom: 15px;
    "
    ),
    
    p(
      "This application uses a matchup-based model to estimate the probability ",
      "of a successful stolen base against specific pitcher-catcher combinations. ",
      "Pitcher and catcher profiles are based on the 2023 Texas Rangers, while ",
      "runner profiles include 2023 MLB runners with at least 7 stolen-base ",
      "attempts to second base.",
      style = "
      max-width: 900px;
      margin: 0 auto;
      font-size: 16px;
      line-height: 1.6;
    "
    )
  ),
  
  br(),
  
  # Matchup selection box
  div(
    style = "
    background-color: white;
    border: 2px solid #003278;
    border-radius: 8px;
    padding: 20px 25px 10px 25px;
    margin-bottom: 30px;
  ",
    
    h3(
      "Matchup Selection",
      style = "
      color: #003278;
      font-weight: bold;
      text-align: center;
      margin-top: 0;
      margin-bottom: 20px;
    "
    ),
    
    fluidRow(
      
      column(
        width = 3,
        selectInput(
          "pitcher",
          "Pitcher",
          choices = unique(Pitchers$player_name)
        )
      ),
      
      column(
        width = 3,
        selectInput(
          "catcher",
          "Catcher",
          choices = unique(Catchers$Catcher_Full_Name)
        )
      ),
      
      column(
        width = 3,
        selectInput(
          "runner",
          "Runner",
          choices = unique(Runners$Runner_Full_Name)
        )
      ),
      
      column(
        width = 3,
        selectInput(
          "target_base",
          "Target Base",
          choices = c("2B", "3B"),
          selected = "2B"
        )
      )
      
    )
  ),


  # Prediction section
  div(
    style = "
    background-color: white;
    border: 2px solid #003278;
    border-radius: 10px;
    padding: 30px 25px;
    margin-bottom: 35px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  ",
    
    h3(
      "Matchup Prediction",
      style = "
      color: #003278;
      font-weight: bold;
      margin-top: 0;
      margin-bottom: 20px;
    "
    ),
    
    # Final probability
    div(
      style = "
      font-size: 64px;
      font-weight: bold;
      color: #003278;
      line-height: 1;
    ",
      textOutput("final_probability", inline = TRUE)
    ),
    
    div(
      style = "
      font-size: 16px;
      color: #666666;
      margin-top: 10px;
      margin-bottom: 25px;
    ",
      "Probability of Steal"
    ),
    
    # Pitch-specific predictions
    uiOutput("pitch_prediction_display")
    
  ),
  
  # Matchup details
  h3(
    "Matchup Details",
    style = "
    text-align: center;
    color: #003278;
    font-weight: bold;
    margin-top: 45px;
    margin-bottom: 25px;
  "
  ),
  
  fluidRow(
    
    column(
      width = 4,
      div(
        style = "
        background-color: white;
        border: 2px solid #003278;
        border-radius: 10px;
        padding: 20px;
        min-height: 300px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        text-align: center;
      ",
        
        uiOutput("pitcher_details")
      )
    ),
    
    column(
      width = 4,
      div(
        style = "
        background-color: white;
        border: 2px solid #003278;
        border-radius: 10px;
        padding: 20px;
        min-height: 300px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        text-align: center;
      ",
        
        uiOutput("catcher_details")
      )
    ),
    
    column(
      width = 4,
      div(
        style = "
        background-color: white;
        border: 2px solid #003278;
        border-radius: 10px;
        padding: 20px;
        min-height: 300px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        text-align: center;
      ",
        
        uiOutput("runner_details")
      )
    )
    
  ),
  # About the model
  div(
    style = "
    background-color: white;
    border: 2px solid #003278;
    border-radius: 10px;
    padding: 25px 30px;
    margin-top: 40px;
    margin-bottom: 20px;
  ",
    
    h3(
      "About the Model",
      style = "
      color: #003278;
      font-weight: bold;
      margin-top: 0;
      margin-bottom: 15px;
    "
    ),
    
    p(
      "The Matchup model estimates the probability of a successful stolen base ",
      "using pitcher, catcher, runner, and pitch characteristics. The model was ",
      "trained using 2023 MLB stolen-base opportunities and uses player-level ",
      "profiles to construct matchup-specific predictions.",
      style = "
      font-size: 15px;
      line-height: 1.6;
      color: #444444;
    "
    ),
    
    div(
      style = "
      display: flex;
      justify-content: center;
      gap: 50px;
      flex-wrap: wrap;
      margin-top: 20px;
      padding-top: 15px;
      border-top: 1px solid #dddddd;
    ",
      
      div(
        strong("Model"),
        br(),
        "XGBoost"
      ),
      
      div(
        strong("Training Season"),
        br(),
        "2023"
      ),
      
      div(
        strong("Pitcher/Catcher Profiles"),
        br(),
        "2023 Texas Rangers"
      ),
      
      div(
        strong("Runner Profiles"),
        br(),
        "2023 MLB runners with ≥7 steals to 2B"
      )
      
    ),
    
    p(
      "*Predictions are estimates based on historical data and should not be interpreted as guarantees of individual outcomes.*",
      style = "
      font-size: 12px;
      color: #777777;
      text-align: center;
      margin-top: 20px;
      margin-bottom: 0;
    "
    )
  )
)

# Define server logic required to draw a histogram
server <- function(input, output) {

  # Selected pitcher
  selected_pitcher <- reactive({
    
    Pitchers %>%
      filter(
        player_name == input$pitcher
      )
    
  })
  
  
  # Selected catcher
  selected_catcher <- reactive({
    
    catcher_data <- Catchers %>%
      filter(
        Catcher_Full_Name == input$catcher
      )
    
    # Use selected base if available
    if (input$target_base %in% catcher_data$Base) {
      
      catcher_data %>%
        filter(Base == input$target_base)
      
    } else {
      
      # Otherwise use 2B as fallback
      catcher_data %>%
        filter(Base == "2B")
      
    }
    
  })
  
  
  # Selected runner
  selected_runner <- reactive({
    
    runner_data <- Runners %>%
      filter(
        Runner_Full_Name == input$runner
      )
    
    # Use selected base if available
    if (input$target_base %in% runner_data$Target.Base) {
      
      runner_data %>%
        filter(Target.Base == input$target_base)
      
    } else {
      
      # Otherwise use 2B as fallback
      runner_data %>%
        filter(Target.Base == "2B")
      
    }
    
  })
  
  
  output$pitcher_profile <- renderTable({
    selected_pitcher()
  })
  
  
  output$catcher_profile <- renderTable({
    selected_catcher()
  })
  
  
  output$runner_profile <- renderTable({
    selected_runner()
  })
  
  output$pitcher_details <- renderUI({
    
    pitcher <- selected_pitcher() %>%
      filter(pitch_classification != "Other") %>%
      mutate(
        pitch_usage = Count / sum(Count)
      )
    
    tagList(
      
      div(
        style = "
        color: #003278;
        font-size: 14px;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
      ",
        "Pitcher"
      ),
      
      h3(
        pitcher$player_name[1],
        style = "
        margin-top: 0;
        margin-bottom: 20px;
        color: #222222;
      "
      ),
      
      lapply(seq_len(nrow(pitcher)), function(i) {
        
        div(
          style = "
          background-color: #f5f6f8;
          border-radius: 6px;
          padding: 10px;
          margin-bottom: 8px;
        ",
          
          div(
            style = "
            font-weight: bold;
            color: #003278;
            font-size: 15px;
          ",
            pitcher$pitch_classification[i]
          ),
          
          div(
            style = "
            color: #555555;
            font-size: 13px;
            margin-top: 3px;
          ",
            paste0(
              round(pitcher$pitch_usage[i] * 100, 1),
              "% usage | ",
              round(pitcher$avg_speed[i], 1)
            )
          )
          
          
          
          
        )
        
      }),
      div(
        style = "
        margin-top: 15px;
        font-size: 11px;
        color: #777777;
      ",
        "2023 MLB Pitcher Profiles"
      )
      
    )
    
  })
  
  output$catcher_details <- renderUI({
    
    catcher <- selected_catcher()
    
    tagList(
      
      div(
        style = "
        color: #003278;
        font-size: 14px;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
      ",
        "Catcher"
      ),
      
      h3(
        catcher$Catcher_Full_Name[1],
        style = "
        margin-top: 0;
        margin-bottom: 20px;
        color: #222222;
      "
      ),
      
      div(
        style = "
        background-color: #f5f6f8;
        border-radius: 6px;
        padding: 10px;
        margin-bottom: 8px;
      ",
        
        div(
          style = "
          font-weight: bold;
          color: #003278;
          font-size: 15px;
        ",
          "Throw Speed"
        ),
        
        div(
          style = "
          color: #555555;
          font-size: 13px;
          margin-top: 3px;
        ",
          round(catcher$avg_TS[1], 1)
        )
        
      ),
      
      div(
        style = "
        background-color: #f5f6f8;
        border-radius: 6px;
        padding: 10px;
        margin-bottom: 8px;
      ",
        
        div(
          style = "
          font-weight: bold;
          color: #003278;
          font-size: 15px;
        ",
          "Exchange"
        ),
        
        div(
          style = "
          color: #555555;
          font-size: 13px;
          margin-top: 3px;
        ",
          round(catcher$avg_Exchange[1], 2)
        )
        
      ),
      
      div(
        style = "
        background-color: #f5f6f8;
        border-radius: 6px;
        padding: 10px;
        margin-bottom: 8px;
      ",
        
        div(
          style = "
          font-weight: bold;
          color: #003278;
          font-size: 15px;
        ",
          "Accuracy"
        ),
        
        div(
          style = "
          color: #555555;
          font-size: 13px;
          margin-top: 3px;
        ",
          round(catcher$avg_Accuracy[1], 1)
        )
        
      ),

        
      
      div(
        style = "
        margin-top: 15px;
        font-size: 11px;
        color: #777777;
      ",
        "Statcast Normalized Metrics"
      )
      
    )
    
  })
  
  output$runner_details <- renderUI({
    
    runner <- selected_runner()
    
    tagList(
      
      div(
        style = "
        color: #003278;
        font-size: 14px;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
      ",
        "Runner"
      ),
      
      h3(
        runner$Runner_Full_Name[1],
        style = "
        margin-top: 0;
        margin-bottom: 20px;
        color: #222222;
      "
      ),
      
      div(
        style = "
        background-color: #f5f6f8;
        border-radius: 6px;
        padding: 10px;
        margin-bottom: 8px;
      ",
        
        div(
          style = "
          font-weight: bold;
          color: #003278;
          font-size: 15px;
        ",
          "Lead Distance Gained"
        ),
        
        div(
          style = "
          color: #555555;
          font-size: 13px;
          margin-top: 3px;
        ",
          round(runner$avg_LeadGained[1], 1),
          " ft"
        )
        
      ),
      
      div(
        style = "
        background-color: #f5f6f8;
        border-radius: 6px;
        padding: 10px;
        margin-bottom: 8px;
      ",
        
        div(
          style = "
          font-weight: bold;
          color: #003278;
          font-size: 15px;
        ",
          "Pitcher First Move"
        ),
        
        div(
          style = "
          color: #555555;
          font-size: 13px;
          margin-top: 3px;
        ",
          round(runner$avg_FirstMove[1], 1),
          " ft"
        )
        
      ),
      
      div(
        style = "
        background-color: #f5f6f8;
        border-radius: 6px;
        padding: 10px;
      ",
        
        div(
          style = "
          font-weight: bold;
          color: #003278;
          font-size: 15px;
        ",
          "At Pitch Release"
        ),
        
        div(
          style = "
          color: #555555;
          font-size: 13px;
          margin-top: 3px;
        ",
          round(runner$avg_atRelease[1], 1),
          " ft"
        )
        
      ),
      
      div(
        style = "
        margin-top: 15px;
        font-size: 11px;
        color: #777777;
      ",
        "2023 MLB Runner Profile"
      )
      
    )
    
  })
  
  
  prediction_data <- reactive({
    
    pitcher <- selected_pitcher()
    catcher <- selected_catcher()
    runner <- selected_runner()
    
    pitcher <- pitcher %>%
      filter(pitch_classification != "Other") %>%
      mutate(
        pitch_usage = Count / sum(Count),
        
        p_throws = pitcher$p_throws[1],
        
        release_speed = avg_speed,
        release_extension = avg_extension,
        flight_time = avg_flight_time,
        
        Throw_Speed = catcher$avg_TS[1],
        Exchange = catcher$avg_Exchange[1],
        
        Lead_Distance_Gained = runner$avg_LeadGained[1],
        Pitcher_First_Move = runner$avg_FirstMove[1],
        At_Pitch_Release = runner$avg_atRelease[1],
        
        target_base = input$target_base
      )
    
    pitcher
    
  })
  
  model_data <- reactive({
    
    prediction_data() %>%
      select(
        p_throws,
        release_speed,
        release_extension,
        flight_time,
        pitch_classification,
        Throw_Speed,
        Exchange,
        Lead_Distance_Gained,
        Pitcher_First_Move,
        At_Pitch_Release,
        target_base
      )
    
  })
  
  model_matrix <- reactive({
    
    data <- model_data()
    
    matrix_data <- data.frame(
      p_throws.L = ifelse(data$p_throws == "L", 1, 0),
      p_throws.R = ifelse(data$p_throws == "R", 1, 0),
      
      release_speed = data$release_speed,
      release_extension = data$release_extension,
      flight_time = data$flight_time,
      
      pitch_classification.Break =
        ifelse(data$pitch_classification == "Break", 1, 0),
      
      pitch_classification.Fast =
        ifelse(data$pitch_classification == "Fast", 1, 0),
      
      pitch_classification.Other =
        ifelse(data$pitch_classification == "Other", 1, 0),
      
      pitch_classification.Off =
        ifelse(data$pitch_classification == "Off", 1, 0),
      
      Throw_Speed = data$Throw_Speed,
      Exchange = data$Exchange,
      Lead_Distance_Gained = data$Lead_Distance_Gained,
      Pitcher_First_Move = data$Pitcher_First_Move,
      At_Pitch_Release = data$At_Pitch_Release,
      
      target_base.2B =
        ifelse(data$target_base == "2B", 1, 0),
      
      target_base.3B =
        ifelse(data$target_base == "3B", 1, 0)
    )
    
    as.matrix(matrix_data)
    
  })
  
  pitch_predictions <- reactive({
    
    predict(
      model,
      newdata = model_matrix()
    )
    
  })
  
  final_probability <- reactive({
    
    sum(
      pitch_predictions() * prediction_data()$pitch_usage
    )
    
  })
  
  output$pitch_prediction_display <- renderUI({
    
    predictions <- pitch_predictions()
    pitch_types <- prediction_data()$pitch_classification
    
    div(
      style = "
      display: flex;
      justify-content: center;
      gap: 55px;
      margin-top: 10px;
    ",
      
      lapply(seq_along(predictions), function(i) {
        
        div(
          style = "
          text-align: center;
          min-width: 80px;
        ",
          
          div(
            style = "
            font-size: 26px;
            font-weight: bold;
            color: #003278;
          ",
            paste0(round(predictions[i] * 100, 1), "%")
          ),
          
          div(
            style = "
            font-size: 14px;
            color: #555555;
            margin-top: 3px;
          ",
            pitch_types[i]
          )
          
        )
        
      })
    )
    
  })
  
  output$final_probability <- renderText({
    
    paste0(
      round(final_probability() * 100, 1),
      "%"
    )
    
  })
  
}

# Run the application 
shinyApp(ui = ui, server = server)
