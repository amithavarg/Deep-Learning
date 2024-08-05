import pandas as pd
import logging

def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        logging.info("Data loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None
