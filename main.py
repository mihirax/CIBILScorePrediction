import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class CIBILScorePredictor:
    def __init__(self):
        self.scaler = RobustScaler()
        self.model = XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        self.feature_cols = []
        
    def load_and_preprocess_data(self, json_path):
        """Load JSON data and perform preprocessing"""
        print("Loading and preprocessing data...")
        
        # Load JSON data
        with open(json_path, "r") as f:
            data = json.load(f)
        
        df = pd.json_normalize(data)
        
        # Column name cleanup
        df.columns = [col.strip('_').replace('.$', '_').replace('.', '_') for col in df.columns]
        
        # Focus on relevant columns
        keep_cols = [
            'userWallet', 'txHash', 'timestamp', 'blockNumber',
            'actionData_type', 'actionData_amount', 'actionData_assetSymbol',
            'actionData_assetPriceUSD',
            'createdAt_date', 'updatedAt_date'
        ]
        df = df[keep_cols].copy()
        
        # Remove rows with empty asset symbols
        df = df[df['actionData_assetSymbol'].astype(str).str.strip() != '']
        
        # Convert numeric columns
        df['actionData_amount'] = pd.to_numeric(df['actionData_amount'], errors='coerce')
        df['actionData_assetPriceUSD'] = pd.to_numeric(df['actionData_assetPriceUSD'], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df['createdAt_date'] = pd.to_datetime(df['createdAt_date'], errors='coerce')
        df['updatedAt_date'] = pd.to_datetime(df['updatedAt_date'], errors='coerce')
        
        # Drop rows with NaN after conversion
        df.dropna(subset=['actionData_amount', 'actionData_assetPriceUSD', 'createdAt_date', 'updatedAt_date'], inplace=True)
        
        print(f"Data loaded: {len(df)} transactions for {df['userWallet'].nunique()} unique wallets")
        return df
    
    def engineer_features(self, df):
        """Create features from transaction data"""
        print("Engineering features...")
        
        # Basic feature engineering
        df['txn_value_usd'] = df['actionData_amount'] * df['actionData_assetPriceUSD']
        df['txn_hour'] = df['timestamp'].dt.hour
        df['txn_weekday'] = df['timestamp'].dt.weekday
        df['createdAt_date'] = pd.to_datetime(df['createdAt_date']).dt.tz_localize(None)
        df['updatedAt_date'] = pd.to_datetime(df['updatedAt_date']).dt.tz_localize(None)
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
        
        df['account_age_days'] = (df['updatedAt_date'] - df['createdAt_date']).dt.total_seconds() / (3600 * 24)
        df['txn_duration_days'] = (df['updatedAt_date'] - df['timestamp']).dt.total_seconds() / (3600 * 24)
        
        # Aggregation to user level
        agg_df = df.groupby('userWallet').agg({
            'txn_value_usd': ['sum', 'mean', 'count', 'std', 'median', 'min', 'max'],
            'actionData_amount': ['mean', 'std', 'median'],
            'account_age_days': 'max',
            'txn_duration_days': ['mean', 'std'],
            'txn_hour': 'nunique',
            'txn_weekday': 'nunique',
            'actionData_type': pd.Series.nunique,
            'actionData_assetSymbol': pd.Series.nunique,
        }).reset_index()
        
        # Flatten column names
        agg_df.columns = ['userWallet', 'total_txn_value', 'avg_txn_value', 'txn_count', 'txn_std',
                          'median_txn_value', 'min_txn_value', 'max_txn_value',
                          'avg_amount', 'std_amount', 'median_amount',
                          'account_age_days', 'avg_txn_duration', 'std_txn_duration',
                          'unique_hours', 'unique_weekdays', 'unique_action_types', 'unique_assets']
        
        # Handle missing values
        agg_df.fillna(0, inplace=True)
        
        # Create additional meaningful features
        agg_df['txn_frequency'] = agg_df['txn_count'] / (agg_df['account_age_days'] + 1)
        agg_df['value_consistency'] = agg_df['avg_txn_value'] / (agg_df['txn_std'] + 1)
        agg_df['portfolio_diversity'] = agg_df['unique_assets'] / (agg_df['txn_count'] + 1)
        agg_df['activity_spread'] = agg_df['unique_hours'] * agg_df['unique_weekdays']
        agg_df['wealth_indicator'] = np.log1p(agg_df['total_txn_value'])
        
        print(f"Features engineered for {len(agg_df)} users")
        return agg_df
    
    def create_cibil_score(self, agg_df):
        """Create CIBIL-like score using percentile-based approach"""
        print("Creating CIBIL scores...")
        
        def create_percentile_score(series, weight):
            """Convert series to percentile ranks and apply weight"""
            return (series.rank(pct=True) * weight).fillna(0)
        
        # Create balanced score components
        score_components = {
            'wealth': create_percentile_score(agg_df['wealth_indicator'], 250),
            'activity': create_percentile_score(agg_df['txn_frequency'], 200),
            'consistency': create_percentile_score(agg_df['value_consistency'], 150),
            'diversity': create_percentile_score(agg_df['portfolio_diversity'], 150),
            'experience': create_percentile_score(agg_df['account_age_days'], 150),
            'engagement': create_percentile_score(agg_df['activity_spread'], 100)
        }
        
        # Combine components
        total_score = sum(score_components.values())
        agg_df['cibil_score'] = np.clip(300 + total_score, 300, 950)
        
        # Add realistic noise
        np.random.seed(42)
        noise = np.random.normal(0, 20, len(agg_df))
        agg_df['cibil_score'] = np.clip(agg_df['cibil_score'] + noise, 300, 950)
        
        print(f"CIBIL scores created - Mean: {agg_df['cibil_score'].mean():.2f}, Std: {agg_df['cibil_score'].std():.2f}")
        return agg_df
    
    def train_model(self, agg_df):
        """Train XGBoost model"""
        print("Training XGBoost model...")
        
        # Prepare features and target
        self.feature_cols = [col for col in agg_df.columns if col not in ['userWallet', 'cibil_score']]
        X = agg_df[self.feature_cols]
        y = agg_df['cibil_score']
        
        # Feature scaling
        X_scaled = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nXGBoost Performance Metrics:")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R² Score: {r2:.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        return X_scaled, feature_importance
    
    def generate_predictions(self, agg_df, X_scaled):
        """Generate predictions and save to CSV"""
        print("Generating predictions...")
        
        # Predict on all data
        all_predictions = self.model.predict(X_scaled)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'userWallet': agg_df['userWallet'],
            'actual_cibil_score': agg_df['cibil_score'],
            'predicted_cibil_score': all_predictions,
            'total_txn_value': agg_df['total_txn_value'],
            'txn_count': agg_df['txn_count'],
            'account_age_days': agg_df['account_age_days'],
            'unique_assets': agg_df['unique_assets'],
            'txn_frequency': agg_df['txn_frequency'],
            'portfolio_diversity': agg_df['portfolio_diversity']
        })
        
        # Round scores to integers
        results_df['actual_cibil_score'] = results_df['actual_cibil_score'].round(0).astype(int)
        results_df['predicted_cibil_score'] = results_df['predicted_cibil_score'].round(0).astype(int)
        
        # Sort by predicted score (highest first)
        results_df = results_df.sort_values('predicted_cibil_score', ascending=False).reset_index(drop=True)
        
        # Save to CSV
        results_df.to_csv('cibil_score_predictions.csv', index=False)
        print(f"Saved predictions to 'cibil_score_predictions.csv'")
        print(f"Total users: {len(results_df)}")
        
        return results_df
    
    def generate_analysis_plots(self, results_df):
        """Generate plots for analysis"""
        print("Generating analysis plots...")
        
        # Score distribution by ranges
        score_ranges = ['300-400', '400-500', '500-600', '600-700', '700-800', '800-900', '900-950']
        range_counts = []
        
        for range_label in score_ranges:
            start, end = map(int, range_label.split('-'))
            count = len(results_df[(results_df['predicted_cibil_score'] >= start) & 
                                  (results_df['predicted_cibil_score'] < end)])
            range_counts.append(count)
        
        # Score distribution plot
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.bar(score_ranges, range_counts, color='skyblue', edgecolor='navy')
        plt.title('CIBIL Score Distribution by Ranges', fontsize=14, fontweight='bold')
        plt.xlabel('Score Range')
        plt.ylabel('Number of Wallets')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        # Score histogram
        plt.subplot(2, 2, 2)
        plt.hist(results_df['predicted_cibil_score'], bins=50, color='lightgreen', alpha=0.7, edgecolor='darkgreen')
        plt.title('CIBIL Score Distribution (Detailed)', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted CIBIL Score')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.3)
        
        # Transaction value vs CIBIL score
        plt.subplot(2, 2, 3)
        plt.scatter(results_df['total_txn_value'], results_df['predicted_cibil_score'], alpha=0.6, s=30)
        plt.title('Transaction Value vs CIBIL Score', fontsize=14, fontweight='bold')
        plt.xlabel('Total Transaction Value (USD)')
        plt.ylabel('Predicted CIBIL Score')
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
        
        # Transaction count vs CIBIL score
        plt.subplot(2, 2, 4)
        plt.scatter(results_df['txn_count'], results_df['predicted_cibil_score'], alpha=0.6, s=30, color='orange')
        plt.title('Transaction Count vs CIBIL Score', fontsize=14, fontweight='bold')
        plt.xlabel('Transaction Count')
        plt.ylabel('Predicted CIBIL Score')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('cibil_analysis_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return score_ranges, range_counts

def main():
    """Main execution function"""
    # Initialize predictor
    predictor = CIBILScorePredictor()
    
    # Load and preprocess data
    df = predictor.load_and_preprocess_data("/content/user-wallet-transactions.json")
    
    # Engineer features
    agg_df = predictor.engineer_features(df)
    
    # Create CIBIL scores
    agg_df = predictor.create_cibil_score(agg_df)
    
    # Train model
    X_scaled, feature_importance = predictor.train_model(agg_df)
    
    # Generate predictions
    results_df = predictor.generate_predictions(agg_df, X_scaled)
    
    # Generate analysis plots
    score_ranges, range_counts = predictor.generate_analysis_plots(results_df)
    
    # Save feature importance
    feature_importance.to_csv('feature_importance.csv', index=False)
    
    print("\nAnalysis complete! Check generated files:")
    print("- cibil_score_predictions.csv")
    print("- feature_importance.csv") 
    print("- cibil_analysis_plots.png")

if __name__ == "__main__":
    main()
