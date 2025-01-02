# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from xgboost import XGBRegressor

# Step 1: Load and Clean Data
# Load datasets
pollution_data = pd.read_csv('AirPollution.csv')
traffic_data = pd.read_csv('TrafficData.csv')
health_data = pd.read_csv('PublicHealthData.csv')

# Combine year, month, day, and hour into a single datetime field for pollution data
pollution_data['datetime'] = pd.to_datetime(pollution_data[['year', 'month', 'day', 'hour']])
pollution_data.drop(['year', 'month', 'day', 'hour', 'No'], axis=1, inplace=True)
pollution_data.columns = ['pollution', 'dew', 'temp', 'pressure', 'w_dir', 'w_speed', 'snow', 'rain', 'datetime']
pollution_data.set_index('datetime', inplace=True)

# Clean and prepare traffic data
traffic_data['datetime'] = pd.to_datetime(traffic_data['datetime'])
traffic_data.set_index('datetime', inplace=True)

# Clean and prepare public health data
health_data['datetime'] = pd.to_datetime(health_data['datetime'])
health_data.set_index('datetime', inplace=True)

# Merge datasets with an outer join to avoid losing data
merged_data = pollution_data.merge(traffic_data, how='outer', left_index=True, right_index=True)
merged_data = merged_data.merge(health_data, how='outer', left_index=True, right_index=True)

# Handle missing values
numeric_cols = merged_data.select_dtypes(include=[np.number]).columns
merged_data[numeric_cols] = merged_data[numeric_cols].fillna(merged_data[numeric_cols].median())

categorical_cols = merged_data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    merged_data[col] = merged_data[col].fillna(merged_data[col].mode()[0])

# Encode categorical features
encoder = LabelEncoder()
merged_data['w_dir'] = encoder.fit_transform(merged_data['w_dir'])
merged_data['congestion_level'] = encoder.fit_transform(merged_data['congestion_level'])

# Step 2: Normalize Data
# Exclude non-numeric columns
non_numeric_cols = merged_data.select_dtypes(exclude=[np.number]).columns
numeric_data = merged_data.drop(columns=non_numeric_cols)

# Normalize numeric data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(numeric_data)

# Reintegrate non-numeric columns
encoded_data = pd.DataFrame(scaled_data, columns=numeric_data.columns, index=merged_data.index)
for col in non_numeric_cols:
    encoded_data[col] = merged_data[col]

# Debug: Check processed data
print("Processed Data Description:\n", encoded_data.describe())

# Step 3: Create Supervised Learning Problem
def to_supervised(data, lookback=4):
    X, Y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, :])
        Y.append(data[i, 0])  # Target is PM2.5 (pollution)
    return np.array(X), np.array(Y)

X, Y = to_supervised(scaled_data)

# Split data into train and test sets
n_train = int(0.8 * len(X))
X_train, X_test = X[:n_train], X[n_train:]
Y_train, Y_test = Y[:n_train], Y[n_train:]

# Step 4: Feature Selection
X_features = numeric_data.drop(['pollution'], axis=1)
y_target = numeric_data['pollution']
rfe = RFE(estimator=RandomForestRegressor(), n_features_to_select=5)
rfe.fit(X_features, y_target)

selected_features = X_features.columns[rfe.support_]
print("Selected Features:", selected_features)

# Step 5: Train LSTM Model
lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_train, Y_train, epochs=20, batch_size=32)

# Predict with LSTM
Y_pred_lstm = lstm_model.predict(X_test)

# Step 6: Train XGBoost Model
xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
xgb_model.fit(X_train.reshape(X_train.shape[0], -1), Y_train)
Y_pred_xgb = xgb_model.predict(X_test.reshape(X_test.shape[0], -1))

# Evaluate Models
rmse_lstm = np.sqrt(mean_squared_error(Y_test, Y_pred_lstm))
rmse_xgb = np.sqrt(mean_squared_error(Y_test, Y_pred_xgb))
print(f"LSTM RMSE: {rmse_lstm:.4f}")
print(f"XGBoost RMSE: {rmse_xgb:.4f}")

# Step 7: Visualizations
# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap of Features")
plt.show()

# Seasonal pollution trends
merged_data['hour'] = merged_data.index.hour
hourly_avg = merged_data.groupby('hour')['pollution'].mean()
plt.figure(figsize=(14, 6))
sns.barplot(x=hourly_avg.index, y=hourly_avg.values, palette='Blues_d')
plt.title("Average PM2.5 by Hour of the Day")
plt.xlabel("Hour")
plt.ylabel("PM2.5 Concentration")
plt.show()

merged_data['month'] = merged_data.index.month
monthly_avg = merged_data.groupby('month')['pollution'].mean()
plt.figure(figsize=(14, 6))
sns.barplot(x=monthly_avg.index, y=monthly_avg.values, palette='Oranges_d')
plt.title("Average PM2.5 by Month")
plt.xlabel("Month")
plt.ylabel("PM2.5 Concentration")
plt.show()

# Traffic impact on PM2.5
plt.figure(figsize=(14, 6))
sns.scatterplot(data=merged_data, x='vehicle_count', y='pollution', hue='congestion_level', palette='coolwarm')
plt.title("Traffic Impact on PM2.5 Levels")
plt.xlabel("Vehicle Count")
plt.ylabel("PM2.5 Concentration")
plt.show()

# Public health impact visualization
plt.figure(figsize=(14, 6))
sns.scatterplot(data=merged_data, x='respiratory_illness_cases', y='pollution', hue='region', palette='Set2')
plt.title("Public Health Impact (Respiratory Cases vs PM2.5)")
plt.xlabel("Respiratory Illness Cases")
plt.ylabel("PM2.5 Concentration")
plt.show()

# Model Predictions: Actual vs Predicted
plt.figure(figsize=(14, 6))
plt.plot(Y_test[:100], label='Actual')
plt.plot(Y_pred_lstm[:100], label='LSTM Predicted')
plt.plot(Y_pred_xgb[:100], label='XGBoost Predicted')
plt.legend()
plt.title("Actual vs Predicted: LSTM and XGBoost")
plt.show()

# Save cleaned data
merged_data.to_csv('MergedData_Cleaned.csv')
