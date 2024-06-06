import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from model import (get_preprocessor, linear_regression_model, random_forest_model, xgboost_model,
                   create_base_n_beats_model, create_base_lstm_model,n_beats_model_builder,lstm_model_builder, save_model, tune_hyperparameters, setup_neural_network_tuning)
from keras_tuner import HyperParameters
from sklearn.metrics import mean_absolute_error, r2_score


# Ensure the 'models' directory exists
os.makedirs('models', exist_ok=True)

# Load datasets
airbnb_data = pd.read_csv('data/all_airbnb_processed.csv')
zillow_data = pd.read_csv('data/all_zillow_processed.csv')

# Define hyperparameters
param_grid_lr = {'regressor__fit_intercept': [True, False]}
param_grid_rf = {'regressor__n_estimators': [1, 2, 3], 'regressor__max_depth': [None, 1, 2]}
param_grid_xgb = {'regressor__n_estimators': [1, 2, 3], 'regressor__learning_rate': [0.01, 0.05, 0.1], 'regressor__max_depth': [3, 5, 7]}

# Airbnb feature specification
airbnb_numerical_features = ['accommodates', 'beds', 'minimum_nights', 'availability_365', 'number_of_reviews', 
                             'review_scores_rating', 'review_scores_accuracy', 'review_scores_value', 'amenities_count']
airbnb_categorical_features = ['neighbourhood_cleansed', 'room_type', 'city']

# Zillow feature specification
zillow_numerical_features = [col for col in zillow_data.columns if col not in ['RegionName','RegionType','StateName', 'State', 'City', 'Metro', 'CountyName', '2023-12-31']]
zillow_categorical_features = ['RegionName','City']

# Get preprocessors
preprocessor_airbnb = get_preprocessor(airbnb_numerical_features, airbnb_categorical_features)
preprocessor_zillow = get_preprocessor(zillow_numerical_features, zillow_categorical_features)

# Initialize base models for Airbnb
models_airbnb = {
    "Linear Regression": linear_regression_model(preprocessor_airbnb),
    "Random Forest": random_forest_model(preprocessor_airbnb),
    "XGBoost": xgboost_model(preprocessor_airbnb)
}


# Load and split data
X_airbnb = airbnb_data.drop('price', axis=1)
y_airbnb = airbnb_data['price']
y_airbnb = airbnb_data['price'].fillna(airbnb_data['price'].mean())


scaler = StandardScaler()

# Apply the scaler to the numerical features
if zillow_numerical_features:
    zillow_data[zillow_numerical_features] = scaler.fit_transform(zillow_data[zillow_numerical_features])

# For the Zillow data, ensure all categorical columns are transformed
if 'RegionName' in zillow_data.columns:
    zillow_data.drop(columns=['RegionType','StateName','State','Metro','CountyName'],inplace=True)
    zillow_data = pd.get_dummies(zillow_data, columns=['RegionName','City'])

# Load and split data
X_zillow = zillow_data.drop('2023-12-31', axis=1)
y_zillow = zillow_data['2023-12-31']

# Split the data for modeling
X_train_airbnb, X_temp_airbnb, y_train_airbnb, y_temp_airbnb = train_test_split(X_airbnb, y_airbnb, test_size=0.3, random_state=42)
X_val_airbnb, X_test_airbnb, y_val_airbnb, y_test_airbnb = train_test_split(X_temp_airbnb, y_temp_airbnb, test_size=0.5, random_state=42)

X_train_zillow, X_temp_zillow, y_train_zillow, y_temp_zillow = train_test_split(X_zillow, y_zillow, test_size=0.3, random_state=42)
X_val_zillow, X_test_zillow, y_val_zillow, y_test_zillow = train_test_split(X_temp_zillow, y_temp_zillow, test_size=0.5, random_state=42)

# Save the preprocessed test data as CSV files for Airbnb
X_test_airbnb.to_csv('data/airbnb_test.csv', index=False)
y_test_airbnb.to_csv('data/airbnb_test_labels.csv', index=False, header=['price'])

# Save the preprocessed test data as CSV files for Zillow
X_test_zillow.to_csv('data/zillow_test.csv', index=False)
y_test_zillow.to_csv('data/zillow_test_labels.csv', index=False, header=['2023-12-31'])

# Initialize base models for Airbnb
linear_regression_base = linear_regression_model(preprocessor_airbnb)
random_forest_base = random_forest_model(preprocessor_airbnb)
xgboost_base = xgboost_model(preprocessor_airbnb)

# Train base models for Airbnb
linear_regression_base.fit(X_train_airbnb, y_train_airbnb)
val_mae = mean_absolute_error(y_val_airbnb, linear_regression_base.predict(X_val_airbnb))
val_r2 = r2_score(y_val_airbnb, linear_regression_base.predict(X_val_airbnb))
print(f"linear_regression_base\nValidation MAE: {val_mae:.2f}, Validation R-squared: {val_r2:.2f}")

random_forest_base.fit(X_train_airbnb, y_train_airbnb)
val_mae = mean_absolute_error(y_val_airbnb, random_forest_base.predict(X_val_airbnb))
val_r2 = r2_score(y_val_airbnb, random_forest_base.predict(X_val_airbnb))
print(f"random_forest_base\nValidation MAE: {val_mae:.2f}, Validation R-squared: {val_r2:.2f}")

xgboost_base.fit(X_train_airbnb, y_train_airbnb)
val_mae = mean_absolute_error(y_val_airbnb, xgboost_base.predict(X_val_airbnb))
val_r2 = r2_score(y_val_airbnb, xgboost_base.predict(X_val_airbnb))
print(f"xgboost_base\nValidation MAE: {val_mae:.2f}, Validation R-squared: {val_r2:.2f}")


# Save base models for Airbnb
save_model(linear_regression_base, 'models/linear_regression_base_model_airbnb.joblib')
save_model(random_forest_base, 'models/random_forest_base_model_airbnb.joblib')
save_model(xgboost_base, 'models/xgboost_base_model_airbnb.joblib')

# Tune and retrain Airbnb models
best_lr = tune_hyperparameters(linear_regression_base, param_grid_lr, X_train_airbnb, y_train_airbnb)
test_mae = mean_absolute_error(y_test_airbnb, best_lr.predict(X_test_airbnb))
test_r2 = r2_score(y_test_airbnb, best_lr.predict(X_test_airbnb))
print(f"best_lr\nValidation MAE: {val_mae:.2f}, Validation R-squared: {val_r2:.2f}")

best_rf = tune_hyperparameters(random_forest_base, param_grid_rf, X_train_airbnb, y_train_airbnb)
test_mae = mean_absolute_error(y_test_airbnb, best_rf.predict(X_test_airbnb))
test_r2 = r2_score(y_test_airbnb, best_rf.predict(X_test_airbnb))
print(f"best_rf\nValidation MAE: {val_mae:.2f}, Validation R-squared: {val_r2:.2f}")

best_xgb = tune_hyperparameters(xgboost_base, param_grid_xgb, X_train_airbnb, y_train_airbnb)
test_mae = mean_absolute_error(y_test_airbnb, best_xgb.predict(X_test_airbnb))
test_r2 = r2_score(y_test_airbnb, best_xgb.predict(X_test_airbnb))
print(f"best_xgb\nValidation MAE: {val_mae:.2f}, Validation R-squared: {val_r2:.2f}")

# Save tuned Airbnb models
save_model(best_lr, 'models/linear_regression_tuned_model_airbnb.joblib')
save_model(best_rf, 'models/random_forest_tuned_model_airbnb.joblib')
save_model(best_xgb, 'models/xgboost_tuned_model_airbnb.joblib')

# Convert data for LSTM (reshape)
X_train_zillow_lstm = X_train_zillow.values.reshape(X_train_zillow.shape[0], 1, X_train_zillow.shape[1])
X_val_zillow_lstm = X_val_zillow.values.reshape(X_val_zillow.shape[0], 1, X_val_zillow.shape[1])

# Check for NaN values and drop or fill them
X_train_zillow_lstm = X_train_zillow_lstm.astype(np.float32)  # Ensuring data type is float32
X_val_zillow_lstm = X_val_zillow_lstm.astype(np.float32)

# Filling NaN values if there are any
X_train_zillow_lstm = np.nan_to_num(X_train_zillow_lstm)
X_val_zillow_lstm = np.nan_to_num(X_val_zillow_lstm)

# Ensure the target variables are also correctly formatted
y_train_zillow = y_train_zillow.astype(np.float32)
y_val_zillow = y_val_zillow.astype(np.float32)
y_train_zillow = np.nan_to_num(y_train_zillow)
y_val_zillow = np.nan_to_num(y_val_zillow)

# Initialize and fit base models for Zillow
input_shape = X_train_zillow.shape[1]  # Feature dimension for N-Beats
n_beats_base = create_base_n_beats_model(input_shape)
lstm_base = create_base_lstm_model(input_shape)

# Fit base models
n_beats_base.fit(X_train_zillow, y_train_zillow, epochs=100, validation_data=(X_val_zillow, y_val_zillow))
val_mae = mean_absolute_error(y_val_zillow, n_beats_base.predict(X_val_zillow))
val_r2 = r2_score(y_val_zillow, n_beats_base.predict(X_val_zillow))
print(f"n_beats_base\nValidation MAE: {val_mae:.2f}, Validation R-squared: {val_r2:.2f}")

lstm_base.fit(X_train_zillow_lstm, y_train_zillow, epochs=100, validation_data=(X_val_zillow_lstm, y_val_zillow))
val_mae = mean_absolute_error(y_val_zillow, lstm_base.predict(X_val_zillow_lstm))
val_r2 = r2_score(y_val_zillow, lstm_base.predict(X_val_zillow_lstm))
print(f"lstm_base\nValidation MAE: {val_mae:.2f}, Validation R-squared: {val_r2:.2f}")

# Save base models for Zillow
n_beats_base.save('models/n_beats_base_model_zillow.h5')
lstm_base.save('models/lstm_base_model_zillow.h5')

# Adjust input shapes and data shapes as needed
best_n_beats = setup_neural_network_tuning(n_beats_model_builder, X_train_zillow, y_train_zillow, X_val_zillow, y_val_zillow)
val_mae = mean_absolute_error(y_val_zillow, best_n_beats.predict(X_val_zillow))
val_r2 = r2_score(y_val_zillow, best_n_beats.predict(X_val_zillow))
print(f"best_n_beats\nValidation MAE: {val_mae:.2f}, Validation R-squared: {val_r2:.2f}")

best_lstm = setup_neural_network_tuning(lstm_model_builder, X_train_zillow_lstm, y_train_zillow, X_val_zillow_lstm, y_val_zillow)
val_mae = mean_absolute_error(y_val_zillow, best_lstm.predict(X_val_zillow_lstm))
val_r2 = r2_score(y_val_zillow, best_lstm.predict(X_val_zillow_lstm))
print(f"best_lstm\nValidation MAE: {val_mae:.2f}, Validation R-squared: {val_r2:.2f}")

# Save the best models from tuning
best_n_beats.save('models/n_beats_tuned_model_zillow.h5')
best_lstm.save('models/lstm_tuned_model_zillow.h5')

