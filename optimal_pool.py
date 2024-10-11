import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any
from data import fetch_fpl_data  # Ensure this is the correct function name
from load_model import load_models

def get_team_mapping() -> Dict[int, str]:
    """Fetches team data and creates a mapping from team ID to team name."""
    data = fetch_fpl_data()  # Fetch data using the correct function name
    teams = data.get('teams', [])
    team_map = {team['id']: team['name'] for team in teams}
    return team_map

# Position ID to position name mapping
POSITION_MAP = {
    1: "GK",
    2: "DEF",
    3: "MID",
    4: "FWD"
}

def calculate_fpl_points(row: pd.Series, is_captain: bool = False) -> float:
    """
    Calculate FPL points based on the FPL point allocation system.

    Args:
        row (pd.Series): A row from the DataFrame containing player stats.
        is_captain (bool): Flag indicating if the player is the captain.

    Returns:
        float: Calculated FPL points for the player, adjusted for captaincy.
    """
    points = 0
    if row['minutes'] >= 60:
        points += 1  # Minimum point for playing 60 minutes
    
    points += row['goals_scored'] * 4  # Goals scored
    points += row['assists'] * 3       # Assists
    points += row['clean_sheets'] * 4  # Clean sheets for defenders and goalkeepers
    points += row['bonus']              # Bonus points
    points += row['saves'] * 0.5        # Saves by goalkeepers
    points -= row['own_goals'] * 2      # Own goals
    
    # Points for yellow and red cards
    points -= row['yellow_cards'] * 1
    points -= row['red_cards'] * 3

    # Adjust points for captaincy
    if is_captain:
        points *= 2  # Double points for captain

    return points / 3  # Divide by 3 for more realistic scoring

def get_optimal_pool() -> Dict[str, List[Dict[str, Any]]]:
    """Generates the optimal pool of players for each position, limited to the top 10 players by predicted points using a hybrid model."""
    
    # Paths to your model files
    model_paths = {
        'rf_model': 'random_forest_model.pkl',
        'xgb_model': 'xgboost_model.pkl',
        'meta_model': 'meta_model.pkl',
        'scaler': 'scaler.pkl'
    }

    # Load the models and scaler
    try:
        rf_model, xgb_model, meta_model, scaler = load_models(model_paths)
    except Exception as e:
        logging.error(f"Failed to load models or scaler: {e}")
        return {}

    # Fetch player and team data
    players_data = fetch_fpl_data().get('elements', [])
    team_map = get_team_mapping()
    
    if not players_data:
        logging.error("Failed to fetch player data.")
        return {}

    # Define feature names
    features = ['minutes', 'goals_scored', 'assists', 'clean_sheets', 'goals_conceded', 
                'own_goals', 'penalties_saved', 'penalties_missed', 'yellow_cards', 
                'red_cards', 'saves', 'bonus', 'bps', 'influence', 'creativity', 'threat', 
                'ict_index', 'form']

    # Initialize the optimal pool with empty lists for each position
    optimal_pool = {position: [] for position in POSITION_MAP.values()}

    for player in players_data:
        try:
            # Prepare the features for prediction as a DataFrame
            player_features = {feature: player.get(feature, 0) for feature in features}
            player_df = pd.DataFrame([player_features], columns=features)

            # Scale the features
            player_features_scaled = scaler.transform(player_df)

            # Get predictions from both models
            rf_pred = rf_model.predict(player_features_scaled)
            xgb_pred = xgb_model.predict(player_features_scaled)

            # Combine predictions using the meta-model
            combined_pred = meta_model.predict(np.hstack((rf_pred.reshape(-1, 1), xgb_pred.reshape(-1, 1))))

            # Divide combined_pred by 3 for more realistic points
            realistic_points = combined_pred[0] / 3

            # Get the player's position using POSITION_MAP
            position_id = player.get('element_type')
            position = POSITION_MAP.get(position_id, 'Unknown')

            # Get the player's team name using the dynamic team_map
            team_id = player.get('team')
            team_name = team_map.get(team_id, "Unknown Team")

            # Debugging information
            logging.debug(f"Processing player {player.get('first_name', 'Unknown')} {player.get('second_name', 'Unknown')}")
            logging.debug(f"Team ID: {team_id}, Team Name: {team_name}")

            # Create a dictionary to store player info
            player_info = {
                'name': f"{player.get('first_name', 'Unknown')} {player.get('second_name', 'Unknown')}",
                'team': team_name,
                'position': position,
                'predicted_points': realistic_points  # Store realistic points
            }

            # Add player to the appropriate position list in the optimal pool
            if position in optimal_pool:
                optimal_pool[position].append(player_info)

        except Exception as e:
            logging.warning(f"Error processing player {player.get('first_name', 'Unknown')} {player.get('second_name', 'Unknown')}: {e}")
            continue

    # Keep only the top 10 players for each position based on predicted points
    for position, players in optimal_pool.items():
        # Sort players by predicted points in descending order and keep only the top 10
        optimal_pool[position] = sorted(players, key=lambda x: x['predicted_points'], reverse=True)[:10]

    return optimal_pool

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    optimal_pool = get_optimal_pool()
    logging.info(f"Optimal Pool: {optimal_pool}")
