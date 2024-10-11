import random
from typing import List, Dict, Any
from optimal_pool import get_optimal_pool

def generate_random_team(optimal_pool: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Generate a random team from the optimal pool in the desired format."""
    team = []
    
    try:
        # Ensure there are enough players in each category
        required_counts = {
            'GK': 2,
            'DEF': 5,
            'MID': 5,
            'FWD': 3
        }
        
        for position, count in required_counts.items():
            players = optimal_pool.get(position, [])
            if len(players) < count:
                raise ValueError(f'Not enough players for position {position}: {len(players)} available, {count} required.')
            team.extend(random.sample(players, count))
        
        # Format team output
        formatted_team = [
            {
                'name': player['name'],
                'team': player['team'],
                'position': player['position'],
                'predicted_points': player['predicted_points']
            }
            for player in team
        ]
    
    except ValueError as e:
        # Log the error and return an empty list
        print(f"Error generating team: {e}")
        return []
    
    return formatted_team

def generate_team() -> List[Dict[str, Any]]:
    """Fetch optimal pool and generate a random team."""
    optimal_pool = get_optimal_pool()
    if not optimal_pool:
        print("Could not generate team due to data fetching error.")
        return []
    
    # Print the optimal pool for debugging
    print("Optimal Pool:", optimal_pool)

    return generate_random_team(optimal_pool)

if __name__ == "__main__":
    # Print the generated team for debugging
    team = generate_team()
    print("Generated Team:", team)
