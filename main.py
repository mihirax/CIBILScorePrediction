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

        with open(json_path, "r") as f:
            data = json.load(f)
        
        df = pd.json_normalize(data)

        df.columns = [col.strip('_').replace('.$', '_').replace('.', '_') for col in df.columns]

        keep_cols = [
            'userWallet', 'txHash', 'timestamp', 'blockNumber',
            'actionData_type', 'actionData_amount', 'actionData_assetSymbol',
            'actionData_assetPriceUSD',
            'createdAt_date', 'updatedAt_date'
        ]
        df = df[keep_cols].copy()

        df = df[df['actionData_assetSymbol'].astype(str).str.strip() != '']

        df['actionData_amount'] = pd.to_numeric(df['actionData_amount'], errors='coerce')
        df['actionData_assetPriceUSD'] = pd.to_numeric(df['actionData_assetPriceUSD'], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df['createdAt_date'] = pd.to_datetime(df['createdAt_date'], errors='coerce')
        df['updatedAt_date'] = pd.to_datetime(df['updatedAt_date'], errors='coerce')

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

        agg_df.columns = ['userWallet', 'total_txn_value', 'avg_txn_value', 'txn_count', 'txn_std',
                          'median_txn_value', 'min_txn_value', 'max_txn_value',
                          'avg_amount', 'std_amount', 'median_amount',
                          'account_age_days', 'avg_txn_duration', 'std_txn_duration',
                          'unique_hours', 'unique_weekdays', 'unique_action_types', 'unique_assets']

        agg_df.fillna(0, inplace=True)

        agg_df['txn_frequency'] = agg_df['txn_count'] / (agg_df['account_age_days'] + 1)
        agg_df['value_consistency'] = agg_df['avg_txn_value'] / (agg_df['txn_std'] + 1)
        agg_df['portfolio_diversity'] = agg_df['unique_assets'] / (agg_df['txn_count'] + 1)
        agg_df['activity_spread'] = agg_df['unique_hours'] * agg_df['unique_weekdays']
        agg_df['wealth_indicator'] = np.log1p(agg_df['total_txn_value'])
        
        print(f"Features engineered for {len(agg_df)} users")
        return agg_df
    
    def create_cibil_score(self, agg_df):
        def create_rank_score(series, min_score, max_score):
            ranks = series.rank(method='min')
            normalized_ranks = (ranks - 1) / (len(ranks) - 1)
            return min_score + (normalized_ranks * (max_score - min_score))

        wealth_score = create_rank_score(agg_df['wealth_indicator'], 0, 100)
        activity_score = create_rank_score(agg_df['txn_frequency'], 0, 100) 
        consistency_score = create_rank_score(agg_df['value_consistency'], 0, 100)
        diversity_score = create_rank_score(agg_df['portfolio_diversity'], 0, 100)
        experience_score = create_rank_score(agg_df['account_age_days'], 0, 100)
        engagement_score = create_rank_score(agg_df['activity_spread'], 0, 100)

        combined_score = (
            wealth_score * 0.25 +         
            activity_score * 0.20 +        
            consistency_score * 0.15 +   
            diversity_score * 0.15 +        
            experience_score * 0.15 +       
            engagement_score * 0.10      
        )

        final_ranks = combined_score.rank(method='min')
        normalized_final_ranks = (final_ranks - 1) / (len(final_ranks) - 1)

        base_cibil_score = 300 + (normalized_final_ranks * 600)
  
        np.random.seed(42)

        noise_factor = 0.03 
        noise = np.random.normal(0, base_cibil_score * noise_factor, len(agg_df))
        
        agg_df['cibil_score'] = base_cibil_score + noise

        agg_df['cibil_score'] = np.clip(agg_df['cibil_score'], 300, 900).round(0).astype(int)

        n_users = len(agg_df)
        target_distribution = {
            (300, 400): 0.10,  
            (400, 500): 0.15, 
            (500, 600): 0.25,  
            (600, 700): 0.25,   
            (700, 800): 0.15,  
            (800, 900): 0.10   
        }
 
        sorted_users = agg_df.sort_values('cibil_score').reset_index(drop=True)

        current_idx = 0
        for (min_score, max_score), proportion in target_distribution.items():
            n_users_in_bucket = int(n_users * proportion)
            end_idx = min(current_idx + n_users_in_bucket, n_users)

            bucket_scores = np.linspace(min_score, max_score-1, end_idx - current_idx)
            bucket_noise = np.random.normal(0, 5, len(bucket_scores))
            final_bucket_scores = np.clip(bucket_scores + bucket_noise, min_score, max_score-1)
            
            sorted_users.loc[current_idx:end_idx-1, 'cibil_score'] = final_bucket_scores.round(0).astype(int)
            current_idx = end_idx

        agg_df['cibil_score'] = sorted_users.set_index(sorted_users.index)['cibil_score']
        
        print(f"CIBIL scores created:")
        print(f"  Mean: {agg_df['cibil_score'].mean():.2f}")
        print(f"  Std: {agg_df['cibil_score'].std():.2f}")
        print(f"  Min: {agg_df['cibil_score'].min()}")
        print(f"  Max: {agg_df['cibil_score'].max()}")
        print(f"  Q1: {agg_df['cibil_score'].quantile(0.25):.0f}")
        print(f"  Median: {agg_df['cibil_score'].median():.0f}")
        print(f"  Q3: {agg_df['cibil_score'].quantile(0.75):.0f}")

        for (min_score, max_score), target_prop in target_distribution.items():
            actual_count = ((agg_df['cibil_score'] >= min_score) & (agg_df['cibil_score'] < max_score)).sum()
            actual_prop = actual_count / len(agg_df)
            print(f"  Scores {min_score}-{max_score}: {actual_count} users ({actual_prop:.1%}) [Target: {target_prop:.1%}]")
        
        return agg_df
    
    def train_model(self, agg_df):

        self.feature_cols = [col for col in agg_df.columns if col not in ['userWallet', 'cibil_score']]
        X = agg_df[self.feature_cols]
        y = agg_df['cibil_score']

        X_scaled = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R² Score: {r2:.3f}")

        feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(feature_importance.head(10))
        
        return X_scaled, feature_importance
    
    def generate_predictions(self, agg_df, X_scaled):

        all_predictions = self.model.predict(X_scaled)

        all_predictions = np.clip(all_predictions, 300, 900)

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

        results_df['actual_cibil_score'] = results_df['actual_cibil_score'].round(0).astype(int)
        results_df['predicted_cibil_score'] = results_df['predicted_cibil_score'].round(0).astype(int)

        results_df = results_df.sort_values('predicted_cibil_score', ascending=False).reset_index(drop=True)

        results_df.to_csv('cibil_score_predictions.csv', index=False)
        print(f"Saved predictions to 'cibil_score_predictions.csv'")
        print(f"Total users: {len(results_df)}")
        
        return results_df
    
    def generate_analysis_plots(self, results_df):

        score_ranges = ['300-400', '400-500', '500-600', '600-700', '700-800', '800-900']
        range_counts = []
        
        for range_label in score_ranges:
            start, end = map(int, range_label.split('-'))
            count = len(results_df[(results_df['predicted_cibil_score'] >= start) & 
                                  (results_df['predicted_cibil_score'] < end)])
            range_counts.append(count)

        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        colors = ['red', 'orange', 'yellow', 'lightgreen', 'green', 'darkgreen']
        plt.bar(score_ranges, range_counts, color=colors, edgecolor='navy')
        plt.title('CIBIL Score Distribution by Ranges', fontsize=14, fontweight='bold')
        plt.xlabel('Score Range')
        plt.ylabel('Number of Wallets')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)

        for i, count in enumerate(range_counts):
            plt.text(i, count + max(range_counts) * 0.01, str(count), 
                    ha='center', va='bottom', fontweight='bold')

        plt.subplot(2, 3, 2)
        plt.hist(results_df['predicted_cibil_score'], bins=30, color='lightblue', alpha=0.7, edgecolor='darkblue')
        plt.title('CIBIL Score Distribution (Detailed)', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted CIBIL Score')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.3)

        plt.subplot(2, 3, 3)
        plt.scatter(results_df['total_txn_value'], results_df['predicted_cibil_score'], alpha=0.6, s=30)
        plt.title('Transaction Value vs CIBIL Score', fontsize=14, fontweight='bold')
        plt.xlabel('Total Transaction Value (USD)')
        plt.ylabel('Predicted CIBIL Score')
        plt.xscale('log')
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 3, 4)
        plt.scatter(results_df['txn_count'], results_df['predicted_cibil_score'], alpha=0.6, s=30, color='orange')
        plt.title('Transaction Count vs CIBIL Score', fontsize=14, fontweight='bold')
        plt.xlabel('Transaction Count')
        plt.ylabel('Predicted CIBIL Score')
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 3, 5)
        score_quality = []
        quality_labels = []
        
        for start, end, label, color in [(300, 500, 'Poor', 'red'), 
                                        (500, 650, 'Fair', 'orange'),
                                        (650, 750, 'Good', 'yellow'), 
                                        (750, 900, 'Excellent', 'green')]:
            count = len(results_df[(results_df['predicted_cibil_score'] >= start) & 
                                  (results_df['predicted_cibil_score'] < end)])
            score_quality.append(count)
            quality_labels.append(f'{label}\n({start}-{end})')
        
        plt.pie(score_quality, labels=quality_labels, autopct='%1.1f%%', 
                colors=['red', 'orange', 'yellow', 'green'], startangle=90)
        plt.title('Credit Quality Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('cibil_analysis_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return score_ranges, range_counts

def main():

    predictor = CIBILScorePredictor()

    df = predictor.load_and_preprocess_data("/content/user-wallet-transactions.json")

    agg_df = predictor.engineer_features(df)

    agg_df = predictor.create_cibil_score(agg_df)

    X_scaled, feature_importance = predictor.train_model(agg_df)

    results_df = predictor.generate_predictions(agg_df, X_scaled)

    score_ranges, range_counts = predictor.generate_analysis_plots(results_df)

    feature_importance.to_csv('feature_importance.csv', index=False)

if __name__ == "__main__":
    main()
