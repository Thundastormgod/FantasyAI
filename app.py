from flask import Flask, render_template, jsonify
from flask_cors import CORS
from predictor import predicted_points  # Rename your function for clarity
from optimal_pool import get_optimal_pool  
from data import fetch_fpl_data
from team_generator import generate_team
import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load player data
def load_player_data():
    """Fetch and validate player data."""
    players_data = fetch_fpl_data()
    if not players_data:
        logging.warning("No player data available. Some endpoints may not function correctly.")
    return players_data

players_data = load_player_data()

@app.route('/')
def home():
    """Render the index.html as the default page."""
    return render_template('index.html')

@app.route('/api/predicted-points', methods=['GET'])
def predicted_points():
    """API to predict points for all players."""
    if players_data is None:
        return jsonify({"error": "Player data is not available."}), 503

    predictions = []
    logging.info("Predicting points for players.")

    required_features = ['minutes', 'goals_scored', 'assists', 'clean_sheets', 
                         'goals_conceded', 'own_goals', 'penalties_saved', 
                         'penalties_missed', 'yellow_cards', 'red_cards', 
                         'saves', 'bonus', 'bps', 'influence', 'creativity', 
                         'threat', 'ict_index', 'form']

    for player in players_data:
        if isinstance(player, dict) and all(feature in player for feature in required_features):
            try:
                predicted_points_value = predicted_points(player)  # Updated function name
                predictions.append({
                    'name': f"{player['first_name']} {player['second_name']}",
                    'predicted_points': predicted_points_value
                })
                logging.info(f"Predicted points for {player['first_name']} {player['second_name']}: {predicted_points_value}")
            except Exception as e:
                logging.error(f"Error predicting points for {player.get('first_name', '')} {player.get('second_name', '')}: {str(e)}")
        else:
            logging.warning(f"Skipping invalid player data: {player}")

    # Sort predictions by predicted points in descending order
    predictions.sort(key=lambda x: x['predicted_points'], reverse=True)

    if not predictions:
        return jsonify({"message": "No predictions available at this time."}), 200

    return jsonify(predictions)

@app.route('/api/random-team')
def random_team():
    """Generate a random team from the optimal pool."""
    try:
        team = generate_team()
        if not team:
            logging.error("Random team generation failed.")
            return jsonify({'error': 'Failed to generate random team'}), 500
        
        logging.info(f"Generated team: {team}")  
        return jsonify({'team': team})
    
    except Exception as e:
        logging.error(f"Error in random_team endpoint: {str(e)}")
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(debug=debug_mode)
