import os
import joblib
import logging
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def load_models(model_paths: dict) -> tuple:
    """
    Load the Random Forest, XGBoost, and Meta-model along with the associated scaler.

    Parameters:
    - model_paths (dict): Dictionary containing paths to the saved models and scaler.
        Example: {
            'rf_model': 'random_forest_model.pkl',
            'xgb_model': 'xgboost_model.pkl',
            'meta_model': 'meta_model.pkl',
            'scaler': 'scaler.pkl'
        }

    Returns:
    - tuple: Loaded Random Forest model, XGBoost model, Meta-model, and scaler.

    Raises:
    - FileNotFoundError: If any of the model or scaler files are missing.
    - ValueError: If any of the loaded models are of unexpected types.
    """
    
    # Extract paths from the dictionary
    rf_model_path = model_paths.get("rf_model")
    xgb_model_path = model_paths.get("xgb_model")
    meta_model_path = model_paths.get("meta_model")
    scaler_path = model_paths.get("scaler")
    
    # Check if all files exist
    missing_files = [path for path in [rf_model_path, xgb_model_path, meta_model_path, scaler_path] if path is None or not os.path.exists(path)]
    if missing_files:
        missing_files_str = ', '.join(missing_files)
        logging.error(f"Missing files or None values: {missing_files_str}")
        raise FileNotFoundError(f"One or more model or scaler files are missing or None: {missing_files_str}")

    # Load the models and scaler
    try:
        rf_model = joblib.load(rf_model_path)
        xgb_model = joblib.load(xgb_model_path)
        meta_model = joblib.load(meta_model_path)
        scaler = joblib.load(scaler_path)
    except Exception as e:
        logging.error(f"Error loading models or scaler: {e}")
        raise RuntimeError(f"Error loading models or scaler: {e}")

    # Validate the types of loaded models
    if not isinstance(rf_model, RandomForestRegressor):
        raise ValueError("Loaded Random Forest model is of unexpected type.")
    if not isinstance(xgb_model, XGBRegressor):
        raise ValueError("Loaded XGBoost model is of unexpected type.")
    if not isinstance(meta_model, LinearRegression):
        raise ValueError("Loaded Meta-model is of unexpected type.")
    if not isinstance(scaler, StandardScaler):
        raise ValueError("Loaded scaler is of unexpected type.")
    
    return rf_model, xgb_model, meta_model, scaler

# Example usage:
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model_paths = {
        'rf_model': 'random_forest_model.pkl',
        'xgb_model': 'xgboost_model.pkl',
        'meta_model': 'meta_model.pkl',
        'scaler': 'scaler.pkl'
    }
    try:
        rf_model, xgb_model, meta_model, scaler = load_models(model_paths)
        logging.info("Models and scaler loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load models and scaler: {e}")
