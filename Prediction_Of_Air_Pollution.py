# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from statsmodels.tsa.seasonal import seasonal_decompose
from xgboost import XGBRegressor

# Step 1: Load and Clean Data
# Load the original AirPollution.csv
dataset = pd.read_csv('AirPollution.csv')

# Combine year, month, day, and hour into a single datetime field
dataset['datetime'] = pd.to_datetime(dataset[['year', 'month', 'day', 'hour']])
dataset.drop(['year', 'month', 'day', 'hour', 'No'], axis=1, inplace=True)

# Rename columns for consistency
dataset.columns = ['pollution', 'dew', 'temp', 'pressure', 'w_dir', 'w_speed', 'snow', 'rain', 'datetime']

# Handle missing values in the `pollution` column
dataset['pollution'] = dataset['pollution'].fillna(0)

# Encode categorical feature `w_dir`
encoder = LabelEncoder()
dataset['w_dir'] = encoder.fit_transform(dataset['w_dir'])

# Set `datetime` as the index
dataset.set_index('datetime', inplace=True)

# Filter out rows with zero pollution values for meaningful analysis
dataset_filtered = dataset[dataset['pollution'] > 0].copy()

# Save the cleaned dataset (optional)
dataset_filtered.to_csv('AirPollution_Cleaned.csv')

# Step 2: Normalize Data for Model Training
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_dataset = scaler.fit_transform(dataset_filtered.values)

# Step 3: Create Supervised Learning Problem
def to_supervised(data, lookback=4):
    X, Y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, :])
        Y.append(data[i, 0])
    return np.array(X), np.array(Y)

X, Y = to_supervised(scaled_dataset)

# Split data into train and test sets
n_train = int(0.8 * len(X))
X_train, X_test = X[:n_train], X[n_train:]
Y_train, Y_test = Y[:n_train], Y[n_train:]

# Step 4: Feature Selection with RFE
X_features = dataset_filtered.drop(['pollution'], axis=1)
y_target = dataset_filtered['pollution']
rfe = RFE(estimator=RandomForestRegressor(), n_features_to_select=5)
X_selected = rfe.fit_transform(X_features, y_target)
selected_features = X_features.columns[rfe.support_]
print("Selected Features:", selected_features)

# Plot feature importance for selected features
selected_importances = rfe.estimator_.feature_importances_
plt.figure(figsize=(10, 6))
plt.bar(selected_features, selected_importances, color='skyblue')
plt.title("Feature Importance (Random Forest - Selected Features)")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Step 5: LSTM Model Training
lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_train, Y_train, epochs=20, batch_size=32)

# LSTM Predictions
Y_pred_lstm = lstm_model.predict(X_test)

# Evaluate LSTM
rmse_lstm = np.sqrt(mean_squared_error(Y_test, Y_pred_lstm))
print(f"LSTM RMSE: {rmse_lstm:.4f}")

# Step 6: Random Forest Model for Comparison
rf_model = RandomForestRegressor()
rf_model.fit(X_train.reshape(X_train.shape[0], -1), Y_train)
Y_pred_rf = rf_model.predict(X_test.reshape(X_test.shape[0], -1))

# Evaluate Random Forest
rmse_rf = np.sqrt(mean_squared_error(Y_test, Y_pred_rf))
print(f"Random Forest RMSE: {rmse_rf:.4f}")

# Step 7: XGBoost Model
xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
xgb_model.fit(X_train.reshape(X_train.shape[0], -1), Y_train)
Y_pred_xgb = xgb_model.predict(X_test.reshape(X_test.shape[0], -1))

# Evaluate XGBoost
rmse_xgb = np.sqrt(mean_squared_error(Y_test, Y_pred_xgb))
print(f"XGBoost RMSE: {rmse_xgb:.4f}")

# Rescale Predictions for Plotting
Y_test_actual = scaler.inverse_transform(np.concatenate([Y_test.reshape(-1, 1), X_test[:, -1, 1:]], axis=1))[:, 0]
Y_pred_lstm_actual = scaler.inverse_transform(np.concatenate([Y_pred_lstm, X_test[:, -1, 1:]], axis=1))[:, 0]
Y_pred_rf_actual = scaler.inverse_transform(np.concatenate([Y_pred_rf.reshape(-1, 1), X_test[:, -1, 1:]], axis=1))[:, 0]
Y_pred_xgb_actual = scaler.inverse_transform(np.concatenate([Y_pred_xgb.reshape(-1, 1), X_test[:, -1, 1:]], axis=1))[:, 0]

# Step 8: Visualization

# Plot 1: Actual vs Predicted (LSTM, RF, XGBoost)
plt.figure(figsize=(14, 6))
plt.plot(Y_test_actual[:100], color='red', label='Actual')
plt.plot(Y_pred_lstm_actual[:100], color='green', label='LSTM Predicted')
plt.plot(Y_pred_rf_actual[:100], color='blue', label='Random Forest Predicted')
plt.plot(Y_pred_xgb_actual[:100], color='purple', label='XGBoost Predicted')
plt.legend()
plt.title("Actual vs Predicted: LSTM, Random Forest, XGBoost")
plt.show()

# Plot 2: Residual Analysis
residuals = Y_test_actual[:100] - Y_pred_lstm_actual[:100]
plt.figure(figsize=(12, 6))
plt.scatter(range(len(residuals)), residuals, alpha=0.6, label='Residuals')
plt.axhline(0, color='red', linestyle='--')
plt.title("Residual Analysis")
plt.xlabel("Sample Index")
plt.ylabel("Residuals")
plt.legend()
plt.show()

# Plot 3: Seasonal Plots (Hour and Month)
dataset_filtered['hour'] = dataset_filtered.index.hour
hourly_avg = dataset_filtered.groupby('hour')['pollution'].mean()
plt.figure(figsize=(10, 6))
sns.barplot(x=hourly_avg.index, y=hourly_avg.values, palette='Blues_d')
plt.title("Average PM2.5 by Hour of the Day")
plt.xlabel("Hour")
plt.ylabel("PM2.5 Concentration")
plt.show()

dataset_filtered['month'] = dataset_filtered.index.month
monthly_avg = dataset_filtered.groupby('month')['pollution'].mean()
plt.figure(figsize=(10, 6))
sns.barplot(x=monthly_avg.index, y=monthly_avg.values, palette='Oranges_d')
plt.title("Average PM2.5 by Month")
plt.xlabel("Month")
plt.ylabel("PM2.5 Concentration")
plt.show()

# Plot 4: Correlation Heatmap
plt.figure(figsize=(10, 8))
corr_matrix = dataset_filtered.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True)
plt.title("Correlation Heatmap of Features")
plt.show()

# Plot 5: Time-Series Decomposition
if len(dataset_filtered) >= 48:  # Minimum size: twice the period
    decomposed = seasonal_decompose(dataset_filtered['pollution'], model='additive', period=24)
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    decomposed.observed.plot(ax=axes[0], legend=False)
    axes[0].set_ylabel("Observed")
    axes[0].set_title("Time-Series Decomposition of PM2.5")
    decomposed.trend.plot(ax=axes[1], legend=False)
    axes[1].set_ylabel("Trend")
    decomposed.seasonal.plot(ax=axes[2], legend=False)
    axes[2].set_ylabel("Seasonal")
    decomposed.resid.plot(ax=axes[3], legend=False)
    axes[3].set_ylabel("Residuals")
    axes[3].set_xlabel("Date")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("Not enough data points for decomposition.")

# Plot: Actual Interrupted and LSTM Continues
plt.figure(figsize=(14, 6))
transition_point = len(Y_test_actual) // 2
plt.plot(range(transition_point), Y_test_actual[:transition_point], color='red', label='Actual (Observed)')
plt.plot(range(transition_point, len(Y_test_actual)), Y_pred_lstm_actual[transition_point:], color='green', label='LSTM Predicted')
plt.axvline(x=transition_point, color='blue', linestyle='--', label='Transition Point')
plt.title("Actual Interrupted and LSTM Continues")
plt.xlabel("Index")
plt.ylabel("PM2.5 Concentration")
plt.legend()
plt.tight_layout()
plt.show()

# Plot: Actual Interrupted and Random Forest Continues
plt.figure(figsize=(14, 6))
plt.plot(range(transition_point), Y_test_actual[:transition_point], color='red', label='Actual (Observed)')
plt.plot(range(transition_point, len(Y_test_actual)), Y_pred_rf_actual[transition_point:], color='blue', label='Random Forest Predicted')
plt.axvline(x=transition_point, color='purple', linestyle='--', label='Transition Point')
plt.title("Actual Interrupted and Random Forest Continues")
plt.xlabel("Index")
plt.ylabel("PM2.5 Concentration")
plt.legend()
plt.tight_layout()
plt.show()
