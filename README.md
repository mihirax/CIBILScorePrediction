# Crypto Wallet CIBIL Score Prediction

A machine learning system that generates CIBIL-like credit scores for cryptocurrency wallets based on their transaction history and behavioral patterns.

## Objective

Create a creditworthiness scoring system that mimics traditional CIBIL scores to assess wallet reliability and credit risk.

## Architecture Overview

```
Input: JSON Transaction Data
            ↓
    Data Preprocessing
            ↓
    Feature Engineering
            ↓
    CIBIL Score Creation
            ↓
    XGBoost Model Training
            ↓
    Predictions & Analysis
            ↓
    Output: CSV + Visualizations
```

## Method Chosen: XGBoost Regression

**Why XGBoost?**
- **Handles Mixed Data Types**: Excellent with numerical and categorical features
- **Feature Importance**: Built-in interpretability for understanding score drivers
- **Robust Performance**: Handles outliers and missing values well
- **Proven Track Record**: Industry standard for financial scoring models
- **Gradient Boosting**: Iteratively improves predictions by learning from errors

## Processing Flow

### 1. Data Loading & Preprocessing
```python
# Load JSON transaction data
# Clean column names
# Convert data types (timestamps, numerics)
# Remove invalid/empty records
```

### 2. Feature Engineering
```python
# Transaction-level features:
- txn_value_usd (amount × price)
- txn_hour, txn_weekday (temporal patterns)
- account_age_days
- txn_duration_days

# User-level aggregations:
- total_txn_value, avg_txn_value, txn_count
- txn_std, median_txn_value, min/max values
- unique_assets, unique_action_types
- transaction frequency, portfolio diversity
- wealth_indicator (log-transformed total value)
```

### 3. CIBIL Score Creation (Percentile-Based)
```python
Score Components (Total: 1000 points):
- Wealth Indicator: 250 points (25%)
- Activity Frequency: 200 points (20%)
- Value Consistency: 150 points (15%)
- Portfolio Diversity: 150 points (15%)
- Account Experience: 150 points (15%)
- Engagement Spread: 100 points (10%)

Final Score = 300 + weighted_percentile_sum (capped at 950)
```

### 4. Model Training Pipeline
```python
# Feature scaling using RobustScaler
# Train-test split (80-20)
# XGBoost hyperparameters:
  - n_estimators: 200
  - max_depth: 4
  - learning_rate: 0.05
  - subsample: 0.8
  - colsample_bytree: 0.8
```

### 5. Output Generation
- **Predictions CSV**: All wallets with actual vs predicted scores
- **Feature Importance**: Top driving factors for creditworthiness
- **Analysis Plots**: Score distributions and behavioral insights

## Repository Structure

```
CIBILScorePrediction/
├── main.py                     # Complete model pipeline
├── README.md                   # This file
├── analysis.md                 # Detailed wallet behavior analysis
├── requirements.txt            # Python dependencies
├── user-wallet-transactions.json        # Data directory
├── visualizations.png                   # Generated files

```

## Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Model
```bash
# Place your JSON file in the data/ directory
python main.py
```

### Expected Outputs
1. **Console Output**: Model performance metrics, feature importance
2. **cibil_score_predictions.csv**: Wallet scores and metadata
3. **feature_importance.csv**: Feature ranking by importance
4. **cibil_analysis_plots.png**: Visualization dashboard

## Model Performance Metrics

The model is evaluated using:
- **R² Score**: Explained variance (target: > 0.3)
- **MAE (Mean Absolute Error)**: Average prediction error in score points
- **RMSE**: Root mean squared error for outlier sensitivity

## Key Features Influencing CIBIL Score

1. **Wealth Indicator** (25%): Log-transformed total transaction value
2. **Transaction Frequency** (20%): Daily transaction rate
3. **Value Consistency** (15%): Ratio of average to standard deviation
4. **Portfolio Diversity** (15%): Unique assets per transaction ratio
5. **Account Age** (15%): Days since account creation
6. **Activity Spread** (10%): Temporal engagement pattern

## Score Interpretation

| Score Range | Credit Rating | Wallet Behavior |
|-------------|---------------|-----------------|
| 300-500     | Poor          | Low activity, minimal diversity |
| 500-650     | Fair          | Moderate activity, some consistency |
| 650-750     | Good          | Regular transactions, diverse portfolio |
| 750-850     | Very Good     | High activity, consistent patterns |
| 850-950     | Excellent     | Premium behavior, maximum diversity |

## Configuration Options

### Model Hyperparameters
```python
# Modify in main.py CIBILScorePredictor.__init__()
XGBRegressor(
    n_estimators=200,      # Number of boosting rounds
    max_depth=4,           # Maximum tree depth
    learning_rate=0.05,    # Step size shrinkage
    subsample=0.8,         # Sample ratio per tree
    colsample_bytree=0.8,  # Feature ratio per tree
    random_state=42
)
```

### Score Weighting
```python
# Modify in create_cibil_score() method
score_components = {
    'wealth': 250,        # Wealth indicator weight
    'activity': 200,      # Activity frequency weight
    'consistency': 150,   # Value consistency weight
    'diversity': 150,     # Portfolio diversity weight
    'experience': 150,    # Account age weight
    'engagement': 100     # Activity spread weight
}
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or suggestions, please open an issue in the GitHub repository.

---

**Note**: This scoring model is for educational and research purposes. Always validate with domain experts before using in production financial applications.
