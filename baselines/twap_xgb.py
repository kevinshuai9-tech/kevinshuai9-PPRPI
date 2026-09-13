# train_signal_fusion_xgboost.py
"""
Train an XGBoost model to fuse microstructure signals for intelligent execution.

Input:
    - twap_outputs/policy_compare_results.csv (from enhanced TWAP backtest)
    - twap_outputs/signal_history.pkl (optional, if recorded separately)

Output:
    - models/signal_fusion_xgb.json
    - models/signal_fusion_report.pdf
    - models/feature_importance.csv

Integration:
    The trained model can be loaded in enhanced_multi_scale_twap_policy
    to replace manual signal weighting.
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Global signal names (must match _extract_micro_signals)
SIGNAL_NAMES = ['ofi', 'momentum', 'liquidity', 'consumption']

# Market state features (optional but recommended)
STATE_NAMES = ['high_volatility', 'low_liquidity', 'trending', 'opening', 'closing']

# Feature columns
FEATURE_COLS = SIGNAL_NAMES + STATE_NAMES + ['steps_left', 'inventory_ratio']


def load_signal_data_from_backtest(results_csv: str = "twap_outputs/policy_compare_results.csv",
                                   signal_pkl: str = "twap_outputs/signal_history.pkl") -> pd.DataFrame:
    """
    Load signal-performance pairs from backtest results.
    
    Two modes:
    1. If signal_history.pkl exists → use detailed per-step signals
    2. Else → approximate from episode-level data (less ideal)
    """
    if os.path.exists(signal_pkl):
        logger.info("Loading detailed signal history from pickle...")
        with open(signal_pkl, 'rb') as f:
            signal_history = pickle.load(f)
        
        records = []
        for key, episodes in signal_history.items():
            code, date = key
            for ep in episodes:
                for step_data in ep:  # each step in episode
                    signals = step_data.get('signals', {})
                    market_state = step_data.get('market_state', {})
                    steps_left = step_data.get('steps_left', 1)
                    inventory_ratio = step_data.get('inventory_ratio', 1.0)
                    reward = step_data.get('reward', 0.0)  # bp improvement
                    
                    record = {
                        'code': code,
                        'date': date,
                        'reward': reward,
                        'steps_left': steps_left,
                        'inventory_ratio': inventory_ratio
                    }
                    record.update({k: signals.get(k, 0.0) for k in SIGNAL_NAMES})
                    record.update({k: float(market_state.get(k, False)) for k in STATE_NAMES})
                    records.append(record)
        
        if records:
            return pd.DataFrame(records)
        else:
            logger.warning("Signal history is empty.")
    
    # Fallback: use episode-level approximation (not recommended for production)
    logger.info("Falling back to episode-level approximation...")
    if not os.path.exists(results_csv):
        raise FileNotFoundError(f"Results file not found: {results_csv}")
    
    df = pd.read_csv(results_csv)
    # Filter for enhanced strategy episodes
    enhanced_df = df[df['policy'].str.contains('enhanced', case=False, na=False)].copy()
    
    if enhanced_df.empty:
        raise ValueError("No enhanced strategy episodes found in results.")
    
    # Simulate signals (placeholder - in real use, you MUST record per-step signals)
    np.random.seed(42)
    records = []
    for _, row in enhanced_df.iterrows():
        n_steps = 20  # assume 20 steps per episode
        for i in range(n_steps):
            record = {
                'code': row['code'],
                'date': row['date'],
                'reward': row['price_performance_bp'] / n_steps,  # distribute reward
                'steps_left': n_steps - i,
                'inventory_ratio': (n_steps - i) / n_steps
            }
            # Random signals (replace with real data!)
            for sig in SIGNAL_NAMES:
                record[sig] = np.random.uniform(-1, 1)
            for state in STATE_NAMES:
                record[state] = np.random.choice([0, 1])
            records.append(record)
    
    return pd.DataFrame(records)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare and engineer features for training."""
    df = df.copy()
    
    # Add interaction features
    df['ofi_x_liquidity'] = df['ofi'] * df['liquidity']
    df['momentum_x_trending'] = df['momentum'] * df['trending']
    df['steps_left_norm'] = df['steps_left'] / df['steps_left'].max()
    
    # Clip extreme values
    for col in SIGNAL_NAMES:
        df[col] = np.clip(df[col], -2.0, 2.0)
    
    return df


def train_xgboost_model(df: pd.DataFrame, output_dir: str = "models"):
    """Train XGBoost model to predict reward (bp improvement) from signals."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Feature matrix and target
    feature_cols = FEATURE_COLS + ['ofi_x_liquidity', 'momentum_x_trending', 'steps_left_norm']
    X = df[feature_cols].copy()
    y = df['reward'].copy()
    
    logger.info(f"Training on {len(X)} samples with {len(feature_cols)} features")
    
    # Handle missing values
    X = X.fillna(0.0)
    
    # Time-series aware split (if date available)
    if 'date' in df.columns:
        df_sorted = df.sort_values('date')
        split_idx = int(0.8 * len(df_sorted))
        train_idx = df_sorted.index[:split_idx]
        test_idx = df_sorted.index[split_idx:]
        X_train, X_test = X.loc[train_idx], X.loc[test_idx]
        y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
    
    # XGBoost regressor
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42,
        early_stopping_rounds=20,
        eval_metric='rmse'
    )
    
    logger.info("Training XGBoost model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )
    
    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    logger.info(f"Test MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    
    # Save model
    model_path = os.path.join(output_dir, "signal_fusion_xgb.json")
    model.save_model(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Feature importance
    importance = model.feature_importances_
    feat_imp_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    imp_path = os.path.join(output_dir, "feature_importance.csv")
    feat_imp_df.to_csv(imp_path, index=False)
    logger.info(f"Feature importance saved to {imp_path}")
    
    # Generate report
    generate_training_report(X_test, y_test, y_pred, feat_imp_df, output_dir)
    
    return model, feat_imp_df


def generate_training_report(X_test, y_test, y_pred, feat_imp_df, output_dir: str):
    """Generate comprehensive training report with plots."""
    plt.figure(figsize=(15, 10))
    
    # 1. Prediction vs Actual
    plt.subplot(2, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Reward (bp)')
    plt.ylabel('Predicted Reward (bp)')
    plt.title('Prediction vs Actual')
    
    # 2. Residuals
    plt.subplot(2, 2, 2)
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Reward')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    
    # 3. Feature Importance
    plt.subplot(2, 2, 3)
    top_features = feat_imp_df.head(10)
    plt.barh(top_features['feature'], top_features['importance'])
    plt.xlabel('Importance')
    plt.title('Top 10 Feature Importance')
    plt.gca().invert_yaxis()
    
    # 4. Reward Distribution
    plt.subplot(2, 2, 4)
    plt.hist(y_test, bins=30, alpha=0.7, label='Actual')
    plt.hist(y_pred, bins=30, alpha=0.7, label='Predicted')
    plt.xlabel('Reward (bp)')
    plt.ylabel('Frequency')
    plt.title('Reward Distribution')
    plt.legend()
    
    plt.tight_layout()
    report_path = os.path.join(output_dir, "signal_fusion_report.pdf")
    plt.savefig(report_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Training report saved to {report_path}")


def load_trained_model(model_path: str = "models/signal_fusion_xgb.json"):
    """Load trained XGBoost model for inference."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    return model


def predict_signal_reward(model, signals: dict, market_state: dict, steps_left: int, inventory_ratio: float) -> float:
    """
    Predict the expected bp improvement from current signals.
    
    Args:
        model: Trained XGBoost model
        signals: dict with keys in SIGNAL_NAMES
        market_state: dict with keys in STATE_NAMES
        steps_left: int
        inventory_ratio: float (0~1)
    
    Returns:
        predicted_reward: float (bp)
    """
    # Build feature vector
    features = {}
    for sig in SIGNAL_NAMES:
        features[sig] = signals.get(sig, 0.0)
    for state in STATE_NAMES:
        features[state] = float(market_state.get(state, False))
    features['steps_left'] = steps_left
    features['inventory_ratio'] = inventory_ratio
    
    # Interaction features
    features['ofi_x_liquidity'] = features['ofi'] * features['liquidity']
    features['momentum_x_trending'] = features['momentum'] * features['trending']
    features['steps_left_norm'] = steps_left / 100.0  # normalize assuming max 100 steps
    
    # Create DataFrame
    feature_cols = FEATURE_COLS + ['ofi_x_liquidity', 'momentum_x_trending', 'steps_left_norm']
    X = pd.DataFrame([features])[feature_cols].fillna(0.0)
    
    # Predict
    pred = model.predict(X)[0]
    return float(pred)


# ======================
# Integration Helper for Strategy
# ======================

class XGBSignalFusionEngine:
    """Wrapper for easy integration into trading strategy."""
    
    def __init__(self, model_path: str = "models/signal_fusion_xgb.json"):
        self.model = load_trained_model(model_path)
        self.feature_cols = FEATURE_COLS + ['ofi_x_liquidity', 'momentum_x_trending', 'steps_left_norm']
    
    def get_composite_signal(self, signals: dict, market_state: dict, steps_left: int, inventory_ratio: float) -> float:
        """
        Convert predicted reward into a normalized signal (-1 to 1).
        Higher reward → more aggressive execution.
        """
        reward = predict_signal_reward(self.model, signals, market_state, steps_left, inventory_ratio)
        # Normalize reward to [-1, 1] range (adjust scale based on your data)
        composite_signal = np.tanh(reward / 5.0)  # 5bp = saturation point
        return composite_signal


# ======================
# Main Training Function
# ======================

def main():
    """Train the XGBoost signal fusion model."""
    logger.info("Starting XGBoost signal fusion training...")
    
    try:
        # Load data
        df = load_signal_data_from_backtest()
        logger.info(f"Loaded {len(df)} signal-reward pairs")
        
        # Prepare features
        df = prepare_features(df)
        
        # Train model
        model, feat_imp = train_xgboost_model(df, output_dir="models")
        
        # Test inference
        test_signals = {name: 0.5 for name in SIGNAL_NAMES}
        test_state = {name: False for name in STATE_NAMES}
        test_state['trending'] = True
        
        reward_pred = predict_signal_reward(model, test_signals, test_state, steps_left=10, inventory_ratio=0.5)
        logger.info(f"Test prediction: {reward_pred:.3f} bp")
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
