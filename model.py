import joblib
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.layers import Input
import tensorflow as tf
from tensorflow.keras.models import Model
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from keras_tuner import RandomSearch, Hyperband, HyperParameters


# Create the 'neural_net_tuning' directory if it doesn't exist
os.makedirs('neural_net_tuning', exist_ok=True)

#for standardizing and encoding
def get_preprocessor(numerical_features, categorical_features):
    scaler = StandardScaler()
    encoder = OneHotEncoder(handle_unknown='ignore')
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', scaler, numerical_features),
            ('cat', encoder, categorical_features)
        ])
    return preprocessor

#base model -linear regression
def linear_regression_model(preprocessor):
    return Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

#base model -random forest regression
def random_forest_model(preprocessor):
    return Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])

#base model -xgboost
def xgboost_model(preprocessor):
    return Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(objective='reg:squarederror', random_state=42))
    ])

#BASE NEURAL NETWORKS
def create_base_n_beats_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')
    return model

def create_base_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, input_shape=(1, input_shape), return_sequences=True),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')
    return model

# FOR HYPERTUNING
def n_beats_model_builder(hp):
    inputs = Input(shape=(15425,))
    x = Dense(units=hp.Int('units', min_value=32, max_value=32, step=32), activation='relu')(inputs)
    x = Dense(units=hp.Int('units', min_value=32, max_value=32, step=32), activation='relu')(x)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])), loss='mean_squared_error')
    return model

def lstm_model_builder(hp):
    inputs = Input(shape=(1, 15425))
    x = LSTM(units=hp.Int('units', min_value=20, max_value=20, step=20), return_sequences=True)(inputs)
    x = LSTM(units=hp.Int('units', min_value=20, max_value=20, step=20))(x)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])), loss='mean_squared_error')
    return model

#saving within the environment
def save_model(model, filename):
    if 'Sequential' in str(type(model)):
        model.save(filename)
    else:
        joblib.dump(model, filename)

#loading the models
def load_model(filename, is_keras=False):
    if is_keras:
        return tf.keras.models.load_model(filename)
    else:
        return joblib.load(filename)

#hypertuning airbnb models
def tune_hyperparameters(model, param_grid, X, y):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)
    grid_search.fit(X, y)
    return grid_search.best_estimator_

#hypertuning zillow models
def setup_neural_network_tuning(model_builder, X_train, y_train, X_val, y_val):
    hp = HyperParameters()
    project_dir = os.path.join('neural_net_tuning', 'checkpoints', 'n_beats_lstm_tuning')
    os.makedirs(project_dir, exist_ok=True)  # Create the project directory if it doesn't exist

    tuner = RandomSearch(
        model_builder,
        objective='val_loss',
        max_trials=25,  # Adjust as necessary
        executions_per_trial=10,
        directory=project_dir,  # Use the project directory for checkpoints
        project_name='n_beats_lstm_tuning',
        overwrite=True,
        hyperparameters=hp
    )
    tuner.search(X_train, y_train, epochs=100, validation_data=(X_val, y_val))  # Adjust epochs as necessary
    best_model = tuner.get_best_models(num_models=1)[0]
    return best_model