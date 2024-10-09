import joblib
import numpy as np
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_models():
    """Load the pre-trained models and scaler."""
    model_paths = {
        'rf_model': 'random_forest_model.pkl',
        'xgb_model': 'xgboost_model.pkl',
        'meta_model': 'meta_model.pkl',
        'scaler': 'scaler.pkl'
    }

    models = {name: joblib.load(path) for name, path in model_paths.items()}
    return models

def predicted_points(player: dict) -> float:
    """Predict the FPL points for a given player.

    Args:
        player (dict): A dictionary containing player statistics.

    Returns:
        float: Predicted FPL points.
    """
    models = load_models()

    # Define features
    features = ['minutes', 'goals_scored', 'assists', 'clean_sheets', 
                'goals_conceded', 'own_goals', 'penalties_saved', 
                'penalties_missed', 'yellow_cards', 'red_cards', 
                'saves', 'bonus', 'bps', 'influence', 'creativity', 
                'threat', 'ict_index', 'form']

    # Prepare the features for prediction
    player_features = np.array([player.get(feature, 0) for feature in features]).reshape(1, -1)
    
    # Scale the features
    player_features_scaled = models['scaler'].transform(player_features)

    # Get predictions from both models
    rf_pred = models['rf_model'].predict(player_features_scaled)
    xgb_pred = models['xgb_model'].predict(player_features_scaled)

    # Prepare for meta-model
    combined_features = pd.DataFrame({'rf': rf_pred, 'xgb': xgb_pred})
    combined_pred = models['meta_model'].predict(combined_features)

    predicted_points_value = combined_pred[0]
    
    # Log the player's name and predicted points
    player_name = f"{player.get('first_name', 'Unknown')} {player.get('second_name', 'Unknown')}"
    logging.info(f"Predicted points for {player_name}: {predicted_points_value:.2f}")

    return predicted_points_value
