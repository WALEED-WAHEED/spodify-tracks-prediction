"""
Linear Regression Modeling, Model Validation, and Predictions
for Spotify Tracks Prediction

Features used:
- Audio Features: Danceability, Energy, Valence, Tempo, Loudness, Duration_min
- Metadata features: explicit

Advanced improvements:
- Feature engineering (polynomial features, interactions)
- Outlier handling
- Multiple model types with hyperparameter tuning
- Feature selection
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 10

class SpotifyLinearRegression:
    """Advanced Linear Regression Model for Spotify Tracks Prediction"""
    
    def __init__(self, csv_path='dataset.csv', use_polynomial=True, handle_outliers=True):
        """
        Initialize the model
        
        Parameters:
        -----------
        csv_path : str
            Path to the dataset CSV file
        use_polynomial : bool
            Whether to use polynomial features (default: True)
        handle_outliers : bool
            Whether to handle outliers (default: True)
        """
        self.csv_path = csv_path
        self.use_polynomial = use_polynomial
        self.handle_outliers = handle_outliers
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.poly_features = None
        self.feature_selector = None
        self.feature_names = None
        self.original_feature_names = None
        
    def load_data(self):
        """Load and inspect the dataset"""
        print("=" * 60)
        print("LOADING DATA")
        print("=" * 60)
        
        self.df = pd.read_csv(self.csv_path)
        print(f"\nDataset shape: {self.df.shape}")
        print(f"\nColumns: {list(self.df.columns)}")
        print(f"\nFirst few rows:")
        print(self.df.head())
        print(f"\nData types:")
        print(self.df.dtypes)
        print(f"\nMissing values:")
        print(self.df.isnull().sum())
        print(f"\nDataset info:")
        print(self.df.info())
        
        return self.df
    
    def preprocess_data(self, target_column='popularity'):
        """
        Preprocess the data and prepare features with advanced techniques
        
        Parameters:
        -----------
        target_column : str
            Name of the target variable (default: 'popularity')
        """
        print("\n" + "=" * 60)
        print("PREPROCESSING DATA")
        print("=" * 60)
        
        # Create a copy to avoid modifying original
        df_processed = self.df.copy()
        
        # Convert duration_ms to Duration_min
        df_processed['Duration_min'] = df_processed['duration_ms'] / 60000.0
        
        # Select features as specified by client
        audio_features = ['danceability', 'energy', 'valence', 'tempo', 'loudness', 'Duration_min']
        metadata_features = ['explicit']
        
        # Combine all features
        feature_columns = audio_features + metadata_features
        
        # Check if all features exist
        missing_features = [f for f in feature_columns if f not in df_processed.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Extract features and target
        self.X = df_processed[feature_columns].copy()
        self.y = df_processed[target_column].copy()
        
        # Handle explicit column (convert boolean to int if needed)
        if self.X['explicit'].dtype == 'bool':
            self.X['explicit'] = self.X['explicit'].astype(int)
        elif self.X['explicit'].dtype == 'object':
            self.X['explicit'] = (self.X['explicit'] == 'True').astype(int)
        
        # Store original feature names
        self.original_feature_names = list(self.X.columns)
        
        # Remove any rows with missing values
        mask = ~(self.X.isnull().any(axis=1) | self.y.isnull())
        self.X = self.X[mask]
        self.y = self.y[mask]
        
        print(f"\nBefore outlier handling: {len(self.X)} samples")
        
        # Handle outliers if requested
        if self.handle_outliers:
            initial_count = len(self.X)
            # Remove outliers from target variable using IQR method
            Q1 = self.y.quantile(0.25)
            Q3 = self.y.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers in features instead of removing
            for col in self.X.columns:
                Q1_feat = self.X[col].quantile(0.25)
                Q3_feat = self.X[col].quantile(0.75)
                IQR_feat = Q3_feat - Q1_feat
                lower_feat = Q1_feat - 1.5 * IQR_feat
                upper_feat = Q3_feat + 1.5 * IQR_feat
                self.X[col] = self.X[col].clip(lower=lower_feat, upper=upper_feat)
            
            # Remove extreme outliers from target
            mask = (self.y >= lower_bound) & (self.y <= upper_bound)
            self.X = self.X[mask]
            self.y = self.y[mask]
            
            removed = initial_count - len(self.X)
            if removed > 0:
                print(f"Outlier handling: Removed {removed} samples ({removed/initial_count*100:.2f}%)")
        
        print(f"\nSelected features: {self.original_feature_names}")
        print(f"\nFeature statistics:")
        print(self.X.describe())
        print(f"\nTarget variable statistics:")
        print(self.y.describe())
        print(f"\nFinal dataset shape: X={self.X.shape}, y={self.y.shape}")
        
        return self.X, self.y
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets
        
        Parameters:
        -----------
        test_size : float
            Proportion of dataset to include in the test split
        random_state : int
            Random state for reproducibility
        """
        print("\n" + "=" * 60)
        print("SPLITTING DATA")
        print("=" * 60)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        
        print(f"\nTraining set: X_train={self.X_train.shape}, y_train={self.y_train.shape}")
        print(f"Testing set: X_test={self.X_test.shape}, y_test={self.y_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def scale_features(self):
        """Scale features using StandardScaler and optionally add polynomial features"""
        print("\n" + "=" * 60)
        print("SCALING FEATURES")
        print("=" * 60)
        
        # Fit scaler on training data and transform both train and test
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        
        # Add polynomial features if requested
        if self.use_polynomial:
            print("\nAdding polynomial features (degree=2 with interactions only)...")
            self.poly_features = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
            X_train_scaled = self.poly_features.fit_transform(X_train_scaled)
            X_test_scaled = self.poly_features.transform(X_test_scaled)
            
            # Get polynomial feature names
            poly_feature_names = self.poly_features.get_feature_names_out(self.original_feature_names)
            print(f"Polynomial features created: {X_train_scaled.shape[1]} features (from {len(self.original_feature_names)} original)")
        else:
            poly_feature_names = self.original_feature_names
        
        # Feature selection - select best features
        print("\nApplying feature selection...")
        k_best = min(30, X_train_scaled.shape[1])  # Select top 30 features or all if less
        self.feature_selector = SelectKBest(score_func=f_regression, k=k_best)
        X_train_scaled = self.feature_selector.fit_transform(X_train_scaled, self.y_train)
        X_test_scaled = self.feature_selector.transform(X_test_scaled)
        
        # Get selected feature names
        if self.use_polynomial:
            selected_mask = self.feature_selector.get_support()
            self.feature_names = [poly_feature_names[i] for i in range(len(poly_feature_names)) if selected_mask[i]]
        else:
            selected_mask = self.feature_selector.get_support()
            self.feature_names = [poly_feature_names[i] for i in range(len(poly_feature_names)) if selected_mask[i]]
        
        print(f"Selected {X_train_scaled.shape[1]} best features out of {len(poly_feature_names)}")
        
        # Convert back to DataFrame
        self.X_train_scaled = pd.DataFrame(
            X_train_scaled, 
            columns=self.feature_names, 
            index=self.X_train.index
        )
        self.X_test_scaled = pd.DataFrame(
            X_test_scaled, 
            columns=self.feature_names, 
            index=self.X_test.index
        )
        
        print("\nFeatures scaled and engineered successfully!")
        print(f"\nScaled training features statistics:")
        print(self.X_train_scaled.describe())
        
        return self.X_train_scaled, self.X_test_scaled
    
    def train_best_model(self, use_scaled=True):
        """
        Train multiple models and select the best one
        
        Parameters:
        -----------
        use_scaled : bool
            Whether to use scaled features (default: True)
        """
        print("\n" + "=" * 60)
        print("TRAINING AND SELECTING BEST MODEL")
        print("=" * 60)
        
        # Select features
        if use_scaled:
            X_train = self.X_train_scaled
            X_test = self.X_test_scaled
        else:
            X_train = self.X_train
            X_test = self.X_test
        
        models = {}
        results = []
        
        # 1. Linear Regression
        print("\n1. Testing Linear Regression...")
        model_lr = LinearRegression()
        model_lr.fit(X_train, self.y_train)
        y_pred = model_lr.predict(X_test)
        r2 = r2_score(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        models['Linear Regression'] = model_lr
        results.append({'model': 'Linear Regression', 'r2': r2, 'rmse': rmse, 'model_obj': model_lr})
        print(f"   R²: {r2:.4f}, RMSE: {rmse:.4f}")
        
        # 2. Ridge Regression with quick tuning
        print("\n2. Testing Ridge Regression...")
        ridge_params = {'alpha': [1.0, 10.0, 100.0]}
        ridge = Ridge()
        grid_ridge = GridSearchCV(ridge, ridge_params, cv=3, scoring='r2', n_jobs=1, verbose=0)
        grid_ridge.fit(X_train, self.y_train)
        y_pred = grid_ridge.predict(X_test)
        r2 = r2_score(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        models['Ridge Regression'] = grid_ridge.best_estimator_
        results.append({'model': 'Ridge Regression', 'r2': r2, 'rmse': rmse, 'model_obj': grid_ridge.best_estimator_})
        print(f"   R²: {r2:.4f}, RMSE: {rmse:.4f}, Best alpha: {grid_ridge.best_params_['alpha']}")
        
        # 3. Lasso Regression with quick tuning
        print("\n3. Testing Lasso Regression...")
        lasso_params = {'alpha': [1.0, 10.0, 100.0]}
        lasso = Lasso(max_iter=2000)
        grid_lasso = GridSearchCV(lasso, lasso_params, cv=3, scoring='r2', n_jobs=1, verbose=0)
        grid_lasso.fit(X_train, self.y_train)
        y_pred = grid_lasso.predict(X_test)
        r2 = r2_score(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        models['Lasso Regression'] = grid_lasso.best_estimator_
        results.append({'model': 'Lasso Regression', 'r2': r2, 'rmse': rmse, 'model_obj': grid_lasso.best_estimator_})
        print(f"   R²: {r2:.4f}, RMSE: {rmse:.4f}, Best alpha: {grid_lasso.best_params_['alpha']}")
        
        # 4. Random Forest with optimized parameters (faster)
        print("\n4. Testing Random Forest...")
        rf = RandomForestRegressor(
            n_estimators=50,  # Reduced for faster training
            max_depth=12,      # Reduced depth
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42, 
            n_jobs=1
        )
        rf.fit(X_train, self.y_train)
        y_pred = rf.predict(X_test)
        r2 = r2_score(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        models['Random Forest'] = rf
        results.append({'model': 'Random Forest', 'r2': r2, 'rmse': rmse, 'model_obj': rf})
        print(f"   R²: {r2:.4f}, RMSE: {rmse:.4f}")
        
        # 5. Gradient Boosting with optimized parameters (faster)
        print("\n5. Testing Gradient Boosting...")
        gb = GradientBoostingRegressor(
            n_estimators=50,  # Reduced for faster training
            learning_rate=0.1,
            max_depth=4,      # Reduced depth
            random_state=42
        )
        gb.fit(X_train, self.y_train)
        y_pred = gb.predict(X_test)
        r2 = r2_score(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        models['Gradient Boosting'] = gb
        results.append({'model': 'Gradient Boosting', 'r2': r2, 'rmse': rmse, 'model_obj': gb})
        print(f"   R²: {r2:.4f}, RMSE: {rmse:.4f}")
        
        # Find best model
        results_df = pd.DataFrame(results)
        best_result = results_df.loc[results_df['r2'].idxmax()]
        self.best_model_name = best_result['model']
        self.model = best_result['model_obj']
        
        print(f"\n{'='*60}")
        print(f"BEST MODEL: {self.best_model_name}")
        print(f"{'='*60}")
        print(f"R² Score: {best_result['r2']:.4f}")
        print(f"RMSE: {best_result['rmse']:.4f}")
        
        # Print all results
        print(f"\nAll Models Comparison:")
        print(results_df[['model', 'r2', 'rmse']].to_string(index=False))
        
        return self.model
    
    def validate_model(self, use_scaled=True, cv_folds=5):
        """
        Validate the model using cross-validation and test set
        
        Parameters:
        -----------
        use_scaled : bool
            Whether to use scaled features
        cv_folds : int
            Number of folds for cross-validation
        """
        print("\n" + "=" * 60)
        print("MODEL VALIDATION")
        print("=" * 60)
        
        # Select features
        if use_scaled:
            X_train = self.X_train_scaled
            X_test = self.X_test_scaled
        else:
            X_train = self.X_train
            X_test = self.X_test
        
        # Cross-validation (reduced folds for faster execution)
        print(f"\nPerforming {cv_folds}-fold cross-validation...")
        # Use fewer folds for tree-based models to speed up
        if 'Forest' in self.best_model_name or 'Boosting' in self.best_model_name:
            cv_folds_actual = 3  # Reduce to 3 folds for ensemble models
            print(f"Using {cv_folds_actual} folds for faster execution...")
        else:
            cv_folds_actual = cv_folds
        
        kfold = KFold(n_splits=cv_folds_actual, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            self.model, X_train, self.y_train, 
            cv=kfold, scoring='neg_mean_squared_error', n_jobs=1
        )
        cv_rmse = np.sqrt(-cv_scores)
        cv_r2 = cross_val_score(
            self.model, X_train, self.y_train, 
            cv=kfold, scoring='r2', n_jobs=1
        )
        
        print(f"\nCross-Validation Results:")
        print(f"  RMSE: {cv_rmse.mean():.4f} (+/- {cv_rmse.std() * 2:.4f})")
        print(f"  R² Score: {cv_r2.mean():.4f} (+/- {cv_r2.std() * 2:.4f})")
        
        # Training set predictions
        y_train_pred = self.model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        train_r2 = r2_score(self.y_train, y_train_pred)
        
        print(f"\nTraining Set Performance:")
        print(f"  RMSE: {train_rmse:.4f}")
        print(f"  MAE: {train_mae:.4f}")
        print(f"  R² Score: {train_r2:.4f}")
        
        # Test set predictions
        y_test_pred = self.model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        
        print(f"\nTest Set Performance:")
        print(f"  RMSE: {test_rmse:.4f}")
        print(f"  MAE: {test_mae:.4f}")
        print(f"  R² Score: {test_r2:.4f}")
        
        # Store predictions
        self.y_train_pred = y_train_pred
        self.y_test_pred = y_test_pred
        
        return {
            'cv_rmse': cv_rmse.mean(),
            'cv_r2': cv_r2.mean(),
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'train_r2': train_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2
        }
    
    def make_predictions(self, X_new=None, use_scaled=True):
        """
        Make predictions on new data
        
        Parameters:
        -----------
        X_new : pd.DataFrame or None
            New data to make predictions on. If None, uses test set
        use_scaled : bool
            Whether to use scaled features
        
        Returns:
        --------
        predictions : np.array
            Predicted values
        """
        print("\n" + "=" * 60)
        print("MAKING PREDICTIONS")
        print("=" * 60)
        
        if X_new is None:
            # Use test set
            if use_scaled:
                X_pred = self.X_test_scaled
            else:
                X_pred = self.X_test
            predictions = self.y_test_pred
            print("\nUsing test set for predictions")
        else:
            # Preprocess new data
            X_new_processed = X_new.copy()
            
            # Scale
            if use_scaled:
                X_new_scaled = self.scaler.transform(X_new_processed)
                
                # Apply polynomial features if used
                if self.use_polynomial:
                    X_new_scaled = self.poly_features.transform(X_new_scaled)
                
                # Apply feature selection
                X_new_scaled = self.feature_selector.transform(X_new_scaled)
                X_pred = pd.DataFrame(X_new_scaled, columns=self.feature_names)
            else:
                X_pred = X_new_processed
            
            predictions = self.model.predict(X_pred)
            print(f"\nMade predictions on {len(predictions)} new samples")
        
        print(f"\nPrediction statistics:")
        print(f"  Mean: {predictions.mean():.4f}")
        print(f"  Std: {predictions.std():.4f}")
        print(f"  Min: {predictions.min():.4f}")
        print(f"  Max: {predictions.max():.4f}")
        print(f"\nFirst 10 predictions:")
        for i, pred in enumerate(predictions[:10]):
            print(f"  Sample {i+1}: {pred:.4f}")
        
        return predictions
    
    def visualize_results(self, save_plots=True):
        """
        Create enhanced visualizations for model results
        
        Parameters:
        -----------
        save_plots : bool
            Whether to save plots to files
        """
        print("\n" + "=" * 60)
        print("CREATING VISUALIZATIONS")
        print("=" * 60)
        
        # Create a comprehensive figure with multiple subplots
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Get feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            importances = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False).head(15)
        elif hasattr(self.model, 'coef_'):
            importances = pd.DataFrame({
                'Feature': self.feature_names,
                'Coefficient': self.model.coef_
            }).sort_values('Coefficient', key=abs, ascending=False).head(15)
            importances['Importance'] = np.abs(importances['Coefficient'])
        else:
            importances = None
        
        # 1. Feature importance/coefficients (Top Left)
        ax1 = fig.add_subplot(gs[0, 0])
        if importances is not None:
            colors = ['#2E86AB' if x >= 0 else '#A23B72' for x in importances.get('Coefficient', importances['Importance'])]
            bars = ax1.barh(importances['Feature'], importances['Importance'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
            ax1.set_xlabel('Importance / |Coefficient|', fontweight='bold')
            ax1.set_title('Top Feature Importances', fontweight='bold', fontsize=12)
            for i, (bar, val) in enumerate(zip(bars, importances['Importance'])):
                ax1.text(val + 0.01, i, f'{val:.3f}', va='center', ha='left', fontweight='bold', fontsize=8)
        ax1.grid(axis='x', alpha=0.3, linestyle='--')
        ax1.set_facecolor('#F8F9FA')
        
        # 2. Actual vs Predicted - Training (Top Middle)
        ax2 = fig.add_subplot(gs[0, 1])
        scatter = ax2.scatter(self.y_train, self.y_train_pred, alpha=0.6, s=30, 
                             c=self.y_train, cmap='viridis', edgecolors='black', linewidth=0.5)
        ax2.plot([self.y_train.min(), self.y_train.max()], 
                [self.y_train.min(), self.y_train.max()], 
                'r--', lw=2.5, label='Perfect Prediction', zorder=3)
        ax2.set_xlabel('Actual Popularity', fontweight='bold')
        ax2.set_ylabel('Predicted Popularity', fontweight='bold')
        ax2.set_title('Training Set: Actual vs Predicted', fontweight='bold', fontsize=12)
        ax2.legend(loc='upper left', framealpha=0.9)
        ax2.grid(alpha=0.3, linestyle='--')
        ax2.set_facecolor('#F8F9FA')
        plt.colorbar(scatter, ax=ax2, label='Actual Popularity')
        train_r2 = r2_score(self.y_train, self.y_train_pred)
        ax2.text(0.05, 0.95, f'R² = {train_r2:.4f}', transform=ax2.transAxes,
                fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontweight='bold')
        
        # 3. Actual vs Predicted - Test (Top Right)
        ax3 = fig.add_subplot(gs[0, 2])
        scatter = ax3.scatter(self.y_test, self.y_test_pred, alpha=0.6, s=30, 
                             c=self.y_test, cmap='plasma', edgecolors='black', linewidth=0.5)
        ax3.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], 
                'r--', lw=2.5, label='Perfect Prediction', zorder=3)
        ax3.set_xlabel('Actual Popularity', fontweight='bold')
        ax3.set_ylabel('Predicted Popularity', fontweight='bold')
        ax3.set_title('Test Set: Actual vs Predicted', fontweight='bold', fontsize=12)
        ax3.legend(loc='upper left', framealpha=0.9)
        ax3.grid(alpha=0.3, linestyle='--')
        ax3.set_facecolor('#F8F9FA')
        plt.colorbar(scatter, ax=ax3, label='Actual Popularity')
        test_r2 = r2_score(self.y_test, self.y_test_pred)
        ax3.text(0.05, 0.95, f'R² = {test_r2:.4f}', transform=ax3.transAxes,
                fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                fontweight='bold')
        
        # 4. Residuals plot - Training (Middle Left)
        ax4 = fig.add_subplot(gs[1, 0])
        residuals_train = self.y_train - self.y_train_pred
        scatter4 = ax4.scatter(self.y_train_pred, residuals_train, alpha=0.6, s=30, 
                   c=residuals_train, cmap='coolwarm', edgecolors='black', linewidth=0.5)
        ax4.axhline(y=0, color='r', linestyle='--', linewidth=2.5, label='Zero Residual')
        ax4.set_xlabel('Predicted Popularity', fontweight='bold')
        ax4.set_ylabel('Residuals', fontweight='bold')
        ax4.set_title('Training Set: Residuals Plot', fontweight='bold', fontsize=12)
        ax4.legend(loc='best', framealpha=0.9)
        ax4.grid(alpha=0.3, linestyle='--')
        ax4.set_facecolor('#F8F9FA')
        plt.colorbar(scatter4, ax=ax4, label='Residual Value')
        
        # 5. Residuals plot - Test (Middle Middle)
        ax5 = fig.add_subplot(gs[1, 1])
        residuals_test = self.y_test - self.y_test_pred
        scatter5 = ax5.scatter(self.y_test_pred, residuals_test, alpha=0.6, s=30, 
                   c=residuals_test, cmap='coolwarm', edgecolors='black', linewidth=0.5)
        ax5.axhline(y=0, color='r', linestyle='--', linewidth=2.5, label='Zero Residual')
        ax5.set_xlabel('Predicted Popularity', fontweight='bold')
        ax5.set_ylabel('Residuals', fontweight='bold')
        ax5.set_title('Test Set: Residuals Plot', fontweight='bold', fontsize=12)
        ax5.legend(loc='best', framealpha=0.9)
        ax5.grid(alpha=0.3, linestyle='--')
        ax5.set_facecolor('#F8F9FA')
        plt.colorbar(scatter5, ax=ax5, label='Residual Value')
        
        # 6. Distribution of residuals - Test (Middle Right)
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.hist(residuals_test, bins=50, color='#2E86AB', alpha=0.7, edgecolor='black', linewidth=1.2)
        ax6.axvline(x=0, color='r', linestyle='--', linewidth=2.5, label='Zero Residual')
        ax6.set_xlabel('Residual Value', fontweight='bold')
        ax6.set_ylabel('Frequency', fontweight='bold')
        ax6.set_title('Test Set: Residual Distribution', fontweight='bold', fontsize=12)
        ax6.legend(loc='best', framealpha=0.9)
        ax6.grid(alpha=0.3, linestyle='--', axis='y')
        ax6.set_facecolor('#F8F9FA')
        mean_res = residuals_test.mean()
        std_res = residuals_test.std()
        ax6.text(0.05, 0.95, f'Mean: {mean_res:.2f}\nStd: {std_res:.2f}', 
                transform=ax6.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontweight='bold')
        
        # 7. Feature importance (Bottom Left)
        ax7 = fig.add_subplot(gs[2, 0])
        if importances is not None:
            importances_sorted = importances.sort_values('Importance', ascending=True)
            colors_abs = ['#2E86AB' if x >= 0 else '#A23B72' for x in importances_sorted.get('Coefficient', importances_sorted['Importance'])]
            bars = ax7.barh(importances_sorted['Feature'], importances_sorted['Importance'], 
                           color=colors_abs, alpha=0.8, edgecolor='black', linewidth=1.2)
            ax7.set_xlabel('Importance', fontweight='bold')
            ax7.set_title('Feature Importance Ranking', fontweight='bold', fontsize=12)
            for i, (bar, val) in enumerate(zip(bars, importances_sorted['Importance'])):
                ax7.text(val + 0.01, i, f'{val:.3f}', va='center', ha='left', 
                        fontweight='bold', fontsize=8)
        ax7.grid(axis='x', alpha=0.3, linestyle='--')
        ax7.set_facecolor('#F8F9FA')
        
        # 8. Prediction error distribution (Bottom Middle)
        ax8 = fig.add_subplot(gs[2, 1])
        errors = np.abs(residuals_test)
        ax8.hist(errors, bins=50, color='#F18F01', alpha=0.7, edgecolor='black', linewidth=1.2)
        ax8.axvline(x=errors.mean(), color='r', linestyle='--', linewidth=2.5, 
                   label=f'Mean Error: {errors.mean():.2f}')
        ax8.set_xlabel('Absolute Error', fontweight='bold')
        ax8.set_ylabel('Frequency', fontweight='bold')
        ax8.set_title('Test Set: Absolute Error Distribution', fontweight='bold', fontsize=12)
        ax8.legend(loc='best', framealpha=0.9)
        ax8.grid(alpha=0.3, linestyle='--', axis='y')
        ax8.set_facecolor('#F8F9FA')
        
        # 9. Model performance metrics (Bottom Right)
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        train_rmse = np.sqrt(mean_squared_error(self.y_train, self.y_train_pred))
        train_mae = mean_absolute_error(self.y_train, self.y_train_pred)
        train_r2 = r2_score(self.y_train, self.y_train_pred)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, self.y_test_pred))
        test_mae = mean_absolute_error(self.y_test, self.y_test_pred)
        test_r2 = r2_score(self.y_test, self.y_test_pred)
        
        metrics_text = f"""
        MODEL PERFORMANCE METRICS
        
        Model: {self.best_model_name}
        
        Training Set:
        ──────────────────────
        R² Score:  {train_r2:.4f}
        RMSE:      {train_rmse:.4f}
        MAE:       {train_mae:.4f}
        
        Test Set:
        ──────────────────────
        R² Score:  {test_r2:.4f}
        RMSE:      {test_rmse:.4f}
        MAE:       {test_mae:.4f}
        
        Features:   {len(self.original_feature_names)} original
                    {len(self.feature_names)} engineered
        """
        
        ax9.text(0.1, 0.5, metrics_text, transform=ax9.transAxes,
                fontsize=10, verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8, pad=15),
                fontweight='bold')
        
        # Add main title
        fig.suptitle('Spotify Tracks Popularity Prediction - Model Analysis', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('linear_regression_results.png', dpi=300, bbox_inches='tight', facecolor='white')
            print("\nPlot saved as 'linear_regression_results.png'")
        
        plt.show()
        
        # 2. Separate Feature importance plot
        if importances is not None:
            plt.figure(figsize=(12, 7))
            importances_display = importances.head(20).sort_values('Importance', ascending=True)
            colors_abs = ['#2E86AB' if x >= 0 else '#A23B72' for x in importances_display.get('Coefficient', importances_display['Importance'])]
            bars = plt.barh(importances_display['Feature'], importances_display['Importance'], 
                           color=colors_abs, alpha=0.8, edgecolor='black', linewidth=1.5)
            plt.xlabel('Importance / |Coefficient|', fontweight='bold', fontsize=12)
            plt.ylabel('Feature', fontweight='bold', fontsize=12)
            plt.title('Feature Importance Analysis', fontweight='bold', fontsize=14, pad=20)
            plt.grid(axis='x', alpha=0.3, linestyle='--')
            plt.gca().set_facecolor('#F8F9FA')
            
            for i, (bar, val) in enumerate(zip(bars, importances_display['Importance'])):
                plt.text(val + 0.01, i, f'{val:.3f}', va='center', ha='left', 
                        fontweight='bold', fontsize=9)
            
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#2E86AB', alpha=0.8, label='Positive Impact'),
                Patch(facecolor='#A23B72', alpha=0.8, label='Negative Impact')
            ]
            plt.legend(handles=legend_elements, loc='lower right', framealpha=0.9, fontsize=10)
            
            plt.tight_layout()
            
            if save_plots:
                plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight', facecolor='white')
                print("Plot saved as 'feature_importance.png'")
            
            plt.show()
        
        print("\nVisualizations created successfully!")
    
    def run_full_pipeline(self, target_column='popularity', test_size=0.2, 
                         random_state=42, use_scaled=True, cv_folds=3):
        """
        Run the complete modeling pipeline
        
        Parameters:
        -----------
        target_column : str
            Name of the target variable
        test_size : float
            Proportion of dataset for testing
        random_state : int
            Random state for reproducibility
        use_scaled : bool
            Whether to use scaled features
        cv_folds : int
            Number of cross-validation folds
        """
        print("\n" + "=" * 60)
        print("SPOTIFY TRACKS PREDICTION - ADVANCED MODEL PIPELINE")
        print("=" * 60)
        print(f"Configuration:")
        print(f"  - Polynomial features: {self.use_polynomial}")
        print(f"  - Outlier handling: {self.handle_outliers}")
        
        # Load data
        self.load_data()
        
        # Preprocess
        self.preprocess_data(target_column=target_column)
        
        # Split data
        self.split_data(test_size=test_size, random_state=random_state)
        
        # Scale features
        if use_scaled:
            self.scale_features()
        
        # Train best model
        self.train_best_model(use_scaled=use_scaled)
        
        # Validate model
        metrics = self.validate_model(use_scaled=use_scaled, cv_folds=cv_folds)
        
        # Make predictions
        self.make_predictions(use_scaled=use_scaled)
        
        # Visualize results
        self.visualize_results()
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        return metrics


def main():
    """Main function to run the advanced linear regression modeling"""
    
    # Initialize the model with advanced features
    model = SpotifyLinearRegression(
        csv_path='dataset.csv',
        use_polynomial=True,      # Use polynomial features for better performance
        handle_outliers=True      # Handle outliers
    )
    
    # Run the complete pipeline
    metrics = model.run_full_pipeline(
        target_column='popularity',
        test_size=0.2,
        random_state=42,
        use_scaled=True,
        cv_folds=3  # Reduced for faster execution
    )
    
    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL MODEL SUMMARY")
    print("=" * 60)
    print(f"\nBest Model: {model.best_model_name}")
    print(f"\nTest Set R² Score: {metrics['test_r2']:.4f}")
    print(f"Test Set RMSE: {metrics['test_rmse']:.4f}")
    print(f"Test Set MAE: {metrics['test_mae']:.4f}")
    print(f"\nCross-Validation R² Score: {metrics['cv_r2']:.4f}")
    print(f"Cross-Validation RMSE: {metrics['cv_rmse']:.4f}")


if __name__ == "__main__":
    main()
