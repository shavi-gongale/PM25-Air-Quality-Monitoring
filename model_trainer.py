import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_models(X_train, y_train):
    """
    Train multiple machine learning models for PM2.5 prediction.
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training target values
        
    Returns:
        dict: Trained models
    """
    models = {}
    
    # Model Tree (Decision Tree Regressor)
    model_tree = DecisionTreeRegressor(random_state=42)
    model_tree.fit(X_train, y_train)
    models['Model Tree'] = model_tree
    
    # NLR (Non-Linear Regression - using Random Forest as a proxy)
    nlr = RandomForestRegressor(n_estimators=50, random_state=42)
    nlr.fit(X_train, y_train)
    models['NLR'] = nlr
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    
    # Logistic Regression (adapted for regression by binning)
    # Since Logistic Regression is for classification, we'll bin the PM2.5 values
    # into categories and then predict probabilities for each category
    # For regression tasks, we'll use the expected value based on these probabilities
    
    # First, create bins for PM2.5 values
    bins = np.linspace(0, np.max(y_train) * 1.1, 20)  # 20 bins from 0 to max value
    y_train_binned = np.digitize(y_train, bins) - 1  # Bin indices (0-based)
    
    # Train logistic regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train_binned)
    
    # Create a wrapper function to convert classification output to regression
    class LogisticRegressionWrapper:
        def __init__(self, model, bins):
            self.model = model
            self.bins = bins
            
        def predict(self, X):
            # Get probabilities for each bin
            proba = self.model.predict_proba(X)
            
            # Calculate expected value using bin centers and probabilities
            bin_centers = (self.bins[:-1] + self.bins[1:]) / 2
            expected_values = np.zeros(X.shape[0])
            
            for i in range(X.shape[0]):
                expected_values[i] = np.sum(proba[i] * bin_centers)
                
            return expected_values
    
    # Add the wrapped logistic regression to the models dictionary
    models['Logistic Regression'] = LogisticRegressionWrapper(lr, bins)
    
    return models

def evaluate_models(models, X_test, y_test):
    """
    Evaluate the performance of the trained models.
    
    Args:
        models (dict): Trained models
        X_test (np.ndarray): Testing features
        y_test (np.ndarray): Testing target values
        
    Returns:
        dict: Evaluation metrics for each model
    """
    results = {}
    
    for name, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store results
        results[name] = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    return results
