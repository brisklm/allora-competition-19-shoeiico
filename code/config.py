import os
from datetime import datetime
import numpy as np
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
except ImportError:
    SentimentIntensityAnalyzer = None
try:
    import optuna
except ImportError:
    optuna = None
data_base_path = os.path.join(os.getcwd(), 'data')
model_file_path = os.path.join(data_base_path, 'model.pkl')
scaler_file_path = os.path.join(data_base_path, 'scaler.pkl')
training_price_data_path = os.path.join(data_base_path, 'price_data.csv')
selected_features_path = os.path.join(data_base_path, 'selected_features.json')
best_model_info_path = os.path.join(data_base_path, 'best_model.json')
sol_source_path = os.path.join(data_base_path, os.getenv('SOL_SOURCE', 'raw_sol.csv'))
eth_source_path = os.path.join(data_base_path, os.getenv('ETH_SOURCE', 'raw_eth.csv'))
features_sol_path = os.path.join(data_base_path, os.getenv('FEATURES_PATH', 'features_sol.csv'))
features_eth_path = os.path.join(data_base_path, os.getenv('FEATURES_PATH_ETH', 'features_eth.csv'))
TOKEN = os.getenv('TOKEN', 'BTC')
TIMEFRAME = os.getenv('TIMEFRAME', '8h')
TRAINING_DAYS = int(os.getenv('TRAINING_DAYS', 365))
MINIMUM_DAYS = 180
REGION = os.getenv('REGION', 'com')
DATA_PROVIDER = os.getenv('DATA_PROVIDER', 'binance')
MODEL = os.getenv('MODEL', 'LSTM_Hybrid')
CG_API_KEY = os.getenv('CG_API_KEY', 'CG-xA5NyokGEVbc4bwrvJPcpZvT')
HELIUS_API_KEY = os.getenv('HELIUS_API_KEY', '70ed65ce-4750-4fd5-83bd-5aee9aa79ead')
HELIUS_RPC_URL = os.getenv('HELIUS_RPC_URL', 'https://mainnet.helius-rpc.com')
BITQUERY_API_KEY = os.getenv('BITQUERY_API_KEY', 'ory_at_LmFLzUutMY8EVb-P_PQVP9ntfwUVTV05LMal7xUqb2I.vxFLfMEoLGcu4XoVi47j-E2bspraTSrmYzCt1A4y2k')
FEATURES = ['log_return', 'volume', 'vader_sentiment', 'technical_indicators']
if SentimentIntensityAnalyzer is not None:

    def analyze_sentiment(text):
        sia = SentimentIntensityAnalyzer()
        return sia.polarity_scores(text)['compound']
else:

    def analyze_sentiment(text):
        return 0.0
if optuna is not None:

    def objective(trial):
        max_depth = trial.suggest_int('max_depth', 3, 10)
        num_leaves = trial.suggest_int('num_leaves', 20, 100)
        learning_rate = trial.suggest_loguniform('learning_rate', 0.001, 0.1)
        score = np.random.rand()
        return score

    def optimize_model(n_trials=100):
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        return (study.best_params, study.best_value)
else:

    def optimize_model(n_trials=100):
        return ({}, 0.0)

def handle_nan(data):
    return np.nan_to_num(data, nan=0.0, posinf=1000000000.0, neginf=-1000000000.0)

def check_low_variance(data, threshold=0.01):
    variances = np.var(data, axis=0)
    low_variance_features = np.where(variances < threshold)[0]
    return low_variance_features