import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from model import load_model, get_preprocessor
import matplotlib.pyplot as plt
import seaborn as sns

# Load test datasets
X_test_airbnb = pd.read_csv('data/airbnb_test.csv')
y_test_airbnb = pd.read_csv('data/airbnb_test_labels.csv')

X_test_zillow = pd.read_csv('data/zillow_test.csv')
y_test_zillow = pd.read_csv('data/zillow_test_labels.csv')

# Evaluate Airbnb models
print("Airbnb Models:")
for model_name in ["linear_regression", "random_forest", "xgboost"]:
    base_model = load_model(f'models/{model_name}_base_model_airbnb.joblib')
    tuned_model = load_model(f'models/{model_name}_tuned_model_airbnb.joblib')

    print(f"\n{model_name} Base Model:")
    test_mae = mean_absolute_error(y_test_airbnb, base_model.predict(X_test_airbnb))
    test_r2 = r2_score(y_test_airbnb, base_model.predict(X_test_airbnb))
    print(f"Test MAE: {test_mae:.2f}, Test R-squared: {test_r2:.2f}")

    print(f"\n{model_name} Tuned Model:")
    test_mae = mean_absolute_error(y_test_airbnb, tuned_model.predict(X_test_airbnb))
    test_r2 = r2_score(y_test_airbnb, tuned_model.predict(X_test_airbnb))
    print(f"Test MAE: {test_mae:.2f}, Test R-squared: {test_r2:.2f}")

#preprocess zillow test data for lstm
X_test_zillow_lstm = X_test_zillow.values.reshape(X_test_zillow.shape[0], 1, X_test_zillow.shape[1])
X_test_zillow_lstm = X_test_zillow_lstm.astype(np.float32)
X_test_zillow_lstm = np.nan_to_num(X_test_zillow_lstm)
y_test_zillow = np.nan_to_num(y_test_zillow)

# Evaluate Zillow models
print("\nZillow Models:")
for model_name, is_keras in [("n_beats", True), ("lstm", True)]:
    if model_name=="lstm":
        base_model = load_model(f'models/{model_name}_base_model_zillow.h5', is_keras=True)
        tuned_model = load_model(f'models/{model_name}_tuned_model_zillow.h5', is_keras=True)

        print(f"\n{model_name} Base Model:")
        test_mae = mean_absolute_error(y_test_zillow, base_model.predict(X_test_zillow_lstm))
        test_r2 = r2_score(y_test_zillow, base_model.predict(X_test_zillow_lstm))
        print(f"Test MAE: {test_mae:.2f}, Test R-squared: {test_r2:.2f}")

        print(f"\n{model_name} Tuned Model:")
        test_mae = mean_absolute_error(y_test_zillow, tuned_model.predict(X_test_zillow_lstm))
        test_r2 = r2_score(y_test_zillow, tuned_model.predict(X_test_zillow_lstm))
        print(f"Test MAE: {test_mae:.2f}, Test R-squared: {test_r2:.2f}")

    else:

        base_model = load_model(f'models/{model_name}_base_model_zillow.h5', is_keras=True)
        tuned_model = load_model(f'models/{model_name}_tuned_model_zillow.h5', is_keras=True)

        print(f"\n{model_name} Base Model:")
        test_mae = mean_absolute_error(y_test_zillow, base_model.predict(X_test_zillow))
        test_r2 = r2_score(y_test_zillow, base_model.predict(X_test_zillow))
        print(f"Test MAE: {test_mae:.2f}, Test R-squared: {test_r2:.2f}")

        print(f"\n{model_name} Tuned Model:")
        test_mae = mean_absolute_error(y_test_zillow, tuned_model.predict(X_test_zillow))
        test_r2 = r2_score(y_test_zillow, tuned_model.predict(X_test_zillow))
        print(f"Test MAE: {test_mae:.2f}, Test R-squared: {test_r2:.2f}")

        # Visualizations for N-Beats model
        if model_name == "n_beats":
            y_pred = tuned_model.predict(X_test_zillow)

            # Actual vs. Predicted Values Plot
            plt.figure(figsize=(10, 6))
            plt.scatter(range(len(y_test_zillow)), y_test_zillow, color='blue', label='Actual', alpha=0.5)
            plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted', alpha=0.5)
            plt.title('Actual vs Predicted Values (N-Beats Model)')
            plt.xlabel('Index')
            plt.ylabel('Values')
            plt.legend()
            plt.show()

        