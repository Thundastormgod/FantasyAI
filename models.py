import pandas as pd
import joblib
import numpy as np
import logging
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from data import get_fpl_players

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_fpl_points(row: pd.Series) -> float:
    """
    Calculate FPL points based on the FPL point allocation system.

    Args:
        row (pd.Series): A row from the DataFrame containing player stats.

    Returns:
        float: Calculated FPL points for the player divided by 3 for realism.
    """
    points = 0
    if row['minutes'] >= 60:
        points += 1  # Minimum point for playing 60 minutes
    
    points += row['goals_scored'] * 4  # Goals scored
    points += row['assists'] * 3       # Assists
    points += row['clean_sheets'] * 4  # Clean sheets for defenders and goalkeepers
    points += row['bonus']             # Bonus points
    points += row['saves'] * 0.5       # Saves by goalkeepers
    points -= row['own_goals'] * 2     # Own goals
    
    # Points for yellow and red cards
    points -= row['yellow_cards'] * 1
    points -= row['red_cards'] * 3

    return points / 3  # Divide total points by 3 for realism

def prepare_data(players_data: list[dict]) -> pd.DataFrame:
    """Prepare and clean player data for modeling.

    Args:
        players_data (list of dict): Raw player data fetched from API.

    Returns:
        pd.DataFrame: Cleaned DataFrame ready for modeling.
    """
    df = pd.DataFrame(players_data)
    features = ['minutes', 'goals_scored', 'assists', 'clean_sheets', 'goals_conceded', 
                'own_goals', 'penalties_saved', 'penalties_missed', 'yellow_cards', 
                'red_cards', 'saves', 'bonus', 'bps', 'influence', 'creativity', 'threat', 
                'ict_index', 'form']

    # Convert string values to float and handle missing values
    for feature in features:
        df[feature] = pd.to_numeric(df[feature], errors='coerce')

    # Calculate total points using FPL system
    df['total_points'] = df.apply(calculate_fpl_points, axis=1)
    
    # Drop rows with missing target values
    df.dropna(subset=['total_points'], inplace=True)

    # Fill or drop missing feature values as needed
    df.fillna(0, inplace=True)

    return df

def train_base_models(players_data: list[dict]) -> None:
    """Train Random Forest and XGBoost models and save them."""
    logging.info("Preparing data for model training...")
    data = prepare_data(players_data)

    # Define features and target
    features = ['minutes', 'goals_scored', 'assists', 'clean_sheets', 'goals_conceded', 
                'own_goals', 'penalties_saved', 'penalties_missed', 'yellow_cards', 
                'red_cards', 'saves', 'bonus', 'bps', 'influence', 'creativity', 'threat', 
                'ict_index', 'form']
    X = data[features]
    y = data['total_points']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    # Train XGBoost model
    xgb_model = XGBRegressor(n_estimators=100, random_state=42)
    xgb_model.fit(X_train_scaled, y_train)

    # Save models and scaler
    joblib.dump(rf_model, 'random_forest_model.pkl')
    joblib.dump(xgb_model, 'xgboost_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    logging.info("Base models and scaler saved successfully.")

    # Evaluate models
    rf_train_score = rf_model.score(X_train_scaled, y_train)
    rf_test_score = rf_model.score(X_test_scaled, y_test)
    xgb_train_score = xgb_model.score(X_train_scaled, y_train)
    xgb_test_score = xgb_model.score(X_test_scaled, y_test)
    
    logging.info(f"Random Forest Train R2 Score: {rf_train_score:.4f}")
    logging.info(f"Random Forest Test R2 Score: {rf_test_score:.4f}")
    logging.info(f"XGBoost Train R2 Score: {xgb_train_score:.4f}")
    logging.info(f"XGBoost Test R2 Score: {xgb_test_score:.4f}")

def train_meta_model(players_data: list[dict]) -> None:
    """Train a meta-model using predictions from base models."""
    logging.info("Preparing data for meta-model training...")
    data = prepare_data(players_data)

    # Define features and target
    features = ['minutes', 'goals_scored', 'assists', 'clean_sheets', 'goals_conceded', 
                'own_goals', 'penalties_saved', 'penalties_missed', 'yellow_cards', 
                'red_cards', 'saves', 'bonus', 'bps', 'influence', 'creativity', 'threat', 
                'ict_index', 'form']
    X = data[features]
    y = data['total_points']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Load base models
    rf_model = joblib.load('random_forest_model.pkl')
    xgb_model = joblib.load('xgboost_model.pkl')

    # Get predictions from base models
    rf_train_pred = rf_model.predict(X_train_scaled)
    xgb_train_pred = xgb_model.predict(X_train_scaled)
    rf_test_pred = rf_model.predict(X_test_scaled)
    xgb_test_pred = xgb_model.predict(X_test_scaled)

    # Prepare meta-model training data
    meta_X_train = pd.DataFrame({'rf': rf_train_pred, 'xgb': xgb_train_pred})
    meta_X_test = pd.DataFrame({'rf': rf_test_pred, 'xgb': xgb_test_pred})

    # Train the meta-model
    meta_model = LinearRegression()
    meta_model.fit(meta_X_train, y_train)

    # Save the meta-model
    joblib.dump(meta_model, 'meta_model.pkl')
    logging.info("Meta-model saved successfully.")

    # Evaluate meta-model
    meta_test_pred = meta_model.predict(meta_X_test)
    meta_test_score = mean_squared_error(y_test, meta_test_pred, squared=False)
    logging.info(f"Meta-model Test RMSE: {meta_test_score:.4f}")

def predict_points(players: dict) -> float:
    model_paths = {
        'rf_model': 'random_forest_model.pkl',
        'xgb_model': 'xgboost_model.pkl',
        'meta_model': 'meta_model.pkl',
        'scaler': 'scaler.pkl'
    }
    
    # Load models and scaler
    rf_model = joblib.load(model_paths['rf_model'])
    xgb_model = joblib.load(model_paths['xgb_model'])
    meta_model = joblib.load(model_paths['meta_model'])
    scaler = joblib.load(model_paths['scaler'])

    # Define features
    features = ['minutes', 'goals_scored', 'assists', 'clean_sheets', 
                'goals_conceded', 'own_goals', 'penalties_saved', 
                'penalties_missed', 'yellow_cards', 'red_cards', 
                'saves', 'bonus', 'bps', 'influence', 'creativity', 
                'threat', 'ict_index', 'form']

    # Prepare the features for prediction
    player_features = np.array([players.get(feature, 0) for feature in features]).reshape(1, -1)
    
    # Scale the features
    player_features_scaled = scaler.transform(player_features)

    # Get predictions from both models
    rf_pred = rf_model.predict(player_features_scaled)
    xgb_pred = xgb_model.predict(player_features_scaled)

    # Prepare for meta-model
    combined_features = pd.DataFrame({'rf': rf_pred, 'xgb': xgb_pred})
    combined_pred = meta_model.predict(combined_features)

    return combined_pred[0]


