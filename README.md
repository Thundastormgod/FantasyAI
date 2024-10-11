            ** Fantasy Premier League Team Generator**
This repository contains a Flask web application that generates a random Fantasy Premier League (FPL) team based on player performance predictions using machine learning. The app collects data from the official Fantasy Premier League API, trains models to predict player points, and generates random teams from an optimized pool of players.

**Table of Contents**
Overview
Features
Installation
Usage
Model Training
API Endpoints
Technologies Used
License

**Overview**
This project aims to generate random Fantasy Premier League teams based on predicted player points. It collects data from the FPL API, preprocesses the data, and uses machine learning models—Random Forest and XGBoost—to predict player points based on selected features (such as goals, assists, and minutes played). An optimal pool of players is created from the top 30 predicted point-scorers, and the app generates a random team every time you request one.

**Features**
Data Collection: Fetches live data from the Fantasy Premier League API.
Model Training: Trains a Random Forest and XGBoost model to predict player points based on historical and real-time features.
Team Generation: Generates a random team of FPL players based on a pool of top-performing players.
Optimal Pool: Selects the top 30 players with the highest predicted points for optimal team selection.
Flask Web App: Simple interface to view and generate teams.

**Installation**
Clone the repository:

bash
Copy code
git clone https://github.com/Thundastormgod/FantasyAI


**Create a virtual environment:**

bash
Copy code
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Set up environment variables: Create a .env file to store any necessary API keys and environment configurations (e.g., FPL_API_URL).

**Run the application:**

python run.py or app.py
The app will be available at http://localhost:5000.

**Usage**
Once the app is running, navigate to http://localhost:5000 in your browser. You will see an interface with a button to generate a random Fantasy Premier League team.

The app collects real-time data from the FPL API.
It runs predictions on player points using trained machine learning models.
From the pool of top 30 predicted players, it randomly selects a team adhering to FPL squad restrictions (e.g., number of players per position and maximum number from each club).


**Model Training**
The project uses two machine learning models to predict player points:

Random Forest
XGBoost
Features Used for Training
Goals
Assists
Minutes played
Clean sheets
Cards
Bonus points
Form
Selected percentage
Transfers in/out

**
**Steps****

Data Preprocessing: Clean and preprocess FPL data for training.

Model Training: Train Random Forest and XGBoost models using player historical data.

Prediction: Use the trained models to predict points for each player.

Top Player Selection: Select the top 30 players with the highest predicted points.
API Endpoints.

The Model is evaluated using a  RMSE - Real Mean Squared Error, Training loss and Evaluation loss.


The application exposes several API endpoints for interaction:

/generate-team: Generates a random FPL team from the optimal player pool.
/train-model: Re-trains the machine learning models on new data (admin-only endpoint).
/get-players: Returns player data, including predicted points.

Technologies Used
Python: Core language for backend and model training.
Flask: Web framework for creating the API and frontend interface.
Random Forest & XGBoost: Machine learning models for player point prediction.
Fantasy Premier League API: Source for real-time player data.
Pandas & Scikit-learn: For data preprocessing and model development.
NumPy: Efficient numerical computations.
SQLAlchemy: ORM for database interaction (if needed).
License
This project is licensed under the MIT License - see the LICENSE file for details.
