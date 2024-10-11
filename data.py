import requests
from typing import List, Dict, Any
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_fpl_data() -> Dict[str, Any]:
    """
    Fetches data from the FPL API including players and teams.
    
    Returns:
        dict: A dictionary containing players and teams data.
    """
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    
    try:
        logging.info(f"Fetching data from {url}")
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
        logging.info("Successfully fetched FPL data")
        return data
    except requests.RequestException as e:
        logging.error(f"Error fetching FPL data: {e}")
        return {}

def get_fpl_players() -> List[Dict[str, Any]]:
    """
    Retrieves player data from the FPL API.
    
    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each representing a player.
    """
    data = fetch_fpl_data()
    players = data.get('elements', [])
    logging.info(f"Fetched {len(players)} players")
    return players

def get_fpl_teams() -> Dict[int, str]:
    """
    Retrieves team data from the FPL API and creates a mapping from team ID to team name.
    
    Returns:
        Dict[int, str]: A dictionary mapping team IDs to team names.
    """
    data = fetch_fpl_data()
    teams = data.get('teams', [])
    team_map = {team['id']: team['name'] for team in teams}
    logging.info(f"Fetched team data for {len(teams)} teams")
    return team_map

# Example usage:
if __name__ == "__main__":
    # Fetch and print player and team data for debugging
    players = get_fpl_players()
    teams = get_fpl_teams()
    
    logging.info(f"Players: {players[:5]}")  # Print first 5 players for inspection
    logging.info(f"Teams: {teams}")  # Print the team mapping
