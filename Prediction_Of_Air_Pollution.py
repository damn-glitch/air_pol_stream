import streamlit as st
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
from statsmodels.tsa.seasonal import seasonal_decompose

#----------------------------------------------
# Streamlit Page Configuration
#----------------------------------------------
st.set_page_config(
    page_title="Air Pollution Analysis",
    page_icon=":cloud:",
    layout="wide",
    initial_sidebar_state="expanded"
)

#----------------------------------------------
# Helper Functions
#----------------------------------------------
def to_supervised(data, lookback=4):
    """Transform a time series dataset into a supervised learning problem."""
    X, Y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, :])
        Y.append(data[i, 0])
    return np.array(X), np.array(Y)

@st.cache_data
def load_data(csv_file):
    """Load and pre-process Air Pollution data."""
    dataset = pd.read_csv(csv_file)
    dataset['datetime'] = pd.to_datetime(dataset[['year', 'month', 'day', 'hour']])
    dataset.drop(['year', 'month', 'day', 'hour', 'No'], axis=1, inplace=True)
    
    # Rename columns
    dataset.columns = [
        'pollution', 'dew', 'temp', 'pressure', 'w_dir', 
        'w_speed', 'snow', 'rain', 'datetime'
    ]
    
    # Handle missing values in the `pollution` column
    dataset['pollution'] = dataset['pollution'].fillna(0)
    
    # Encode categorical feature `w_dir`
    encoder = LabelEncoder()
    dataset['w_dir'] = encoder.fit_transform(dataset['w_dir'])
    
    # Set `datetime` as the index
    dataset.set_index('datetime', inplace=True)

    # Filter out rows with zero pollution values for meaningful analysis
    dataset_filtered = dataset[dataset['pollution'] > 0].copy()
    
    return dataset_filtered

@st.cache_data
def scale_data(dataset_filtered):
    """Scale data using MinMaxScaler."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_dataset = scaler.fit_transform(dataset_filtered.values)
    return scaler, scaled_dataset

def plot_feature_importance(features, importances):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(features, importances, color='skyblue')
    ax.set_title("Feature Importance (Random Forest - Selected Features)")
    ax.set_xlabel("Features")
    ax.set_ylabel("Importance")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

#----------------------------------------------
# Streamlit Sidebar
#----------------------------------------------
st.sidebar.title("Air Pollution Analysis")
st.sidebar.markdown(
    """
    This demo analyzes air pollution data using:
    - LSTM
    - Random Forest
    - XGBoost
    """
)

#----------------------------------------------
# Main Application
#----------------------------------------------
st.title("Air Pollution Modeling and Visualization")

#----------------------------------------------
# Step 1: Load Data
#----------------------------------------------
st.header("1. Load and Clean Data")

uploaded_file = st.file_uploader("Upload AirPollution.csv", type=["csv"])

if uploaded_file is not None:
    with st.spinner("Loading data..."):
        dataset_filtered = load_data(uploaded_file)
    
    st.write("Data Preview:")
    st.dataframe(dataset_filtered.head())
    
    st.write("Data Summary:")
    st.write(dataset_filtered.describe())
    
    #------------------------------------------
    # Step 2: Normalize Data
    #------------------------------------------
    st.header("2. Data Normalization")
    scaler, scaled_dataset = scale_data(dataset_filtered)
    
    st.success("Data has been successfully normalized!")
    
    #------------------------------------------
    # Step 3: Create Supervised Learning Problem
    #------------------------------------------
    st.header("3. Create Supervised Learning Problem")
    lookback = st.slider("Select Lookback Window:", min_value=1, max_value=10, value=4, step=1)
    
    with st.spinner("Transforming data into supervised format..."):
        X, Y = to_supervised(scaled_dataset, lookback=lookback)
        n_train = int(0.8 * len(X))
        X_train, X_test = X[:n_train], X[n_train:]
        Y_train, Y_test = Y[:n_train], Y[n_train:]
    
    st.write(f"Training set size: {X_train.shape[0]}")
    st.write(f"Test set size: {X_test.shape[0]}")

    #------------------------------------------
    # Step 4: Feature Selection (RFE)
    #------------------------------------------
    st.header("4. Feature Selection with Random Forest (RFE)")
    X_features = dataset_filtered.drop(['pollution'], axis=1)
    y_target = dataset_filtered['pollution']

    rfe = RFE(estimator=RandomForestRegressor(), n_features_to_select=5)
    with st.spinner("Performing RFE..."):
        X_selected = rfe.fit_transform(X_features, y_target)
    
    selected_features = X_features.columns[rfe.support_]
    st.write("Selected Features:", list(selected_features))
    
    fig_features = plot_feature_importance(selected_features, rfe.estimator_.feature_importances_)
    st.pyplot(fig_features)

    #------------------------------------------
    # Step 5: LSTM Model Training
    #------------------------------------------
    st.header("5. LSTM Model Training")
    epochs = st.slider("Number of Epochs for LSTM:", min_value=1, max_value=50, value=10, step=1)
    with st.spinner("Training LSTM model..."):
        lstm_model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        lstm_model.compile(optimizer='adam', loss='mean_squared_error')
        lstm_model.fit(X_train, Y_train, epochs=epochs, batch_size=32, verbose=0)

    Y_pred_lstm = lstm_model.predict(X_test)
    rmse_lstm = np.sqrt(mean_squared_error(Y_test, Y_pred_lstm))
    st.metric(label="LSTM RMSE", value=f"{rmse_lstm:.4f}")

    #------------------------------------------
    # Step 6: Random Forest Model
    #------------------------------------------
    st.header("6. Random Forest Model")
    with st.spinner("Training Random Forest model..."):
        rf_model = RandomForestRegressor()
        rf_model.fit(X_train.reshape(X_train.shape[0], -1), Y_train)
        Y_pred_rf = rf_model.predict(X_test.reshape(X_test.shape[0], -1))
    
    rmse_rf = np.sqrt(mean_squared_error(Y_test, Y_pred_rf))
    st.metric(label="Random Forest RMSE", value=f"{rmse_rf:.4f}")

    #------------------------------------------
    # Step 7: XGBoost Model
    #------------------------------------------
    st.header("7. XGBoost Model")
    with st.spinner("Training XGBoost model..."):
        xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
        xgb_model.fit(X_train.reshape(X_train.shape[0], -1), Y_train)
        Y_pred_xgb = xgb_model.predict(X_test.reshape(X_test.shape[0], -1))
    
    rmse_xgb = np.sqrt(mean_squared_error(Y_test, Y_pred_xgb))
    st.metric(label="XGBoost RMSE", value=f"{rmse_xgb:.4f}")

    #------------------------------------------
    # Rescale Predictions for Plotting
    #------------------------------------------
    st.header("8. Visualization of Results")
    with st.spinner("Preparing plots..."):
        # We need to reconstruct the shape for the scaler.
        # We'll combine each prediction with the last known features of X_test,
        # then invert transform. This approach is approximate for demonstration.
        
        # Actual
        Y_test_actual = scaler.inverse_transform(
            np.concatenate([Y_test.reshape(-1, 1), X_test[:, -1, 1:]], axis=1)
        )[:, 0]
        # LSTM
        Y_pred_lstm_actual = scaler.inverse_transform(
            np.concatenate([Y_pred_lstm, X_test[:, -1, 1:]], axis=1)
        )[:, 0]
        # Random Forest
        Y_pred_rf_actual = scaler.inverse_transform(
            np.concatenate([Y_pred_rf.reshape(-1, 1), X_test[:, -1, 1:]], axis=1)
        )[:, 0]
        # XGBoost
        Y_pred_xgb_actual = scaler.inverse_transform(
            np.concatenate([Y_pred_xgb.reshape(-1, 1), X_test[:, -1, 1:]], axis=1)
        )[:, 0]

        # Plot 1: Actual vs Predicted
        st.subheader("Plot 1: Actual vs Predicted (LSTM, RF, XGBoost)")
        fig1, ax1 = plt.subplots(figsize=(14, 6))
        ax1.plot(Y_test_actual[:100], color='red', label='Actual')
        ax1.plot(Y_pred_lstm_actual[:100], color='green', label='LSTM')
        ax1.plot(Y_pred_rf_actual[:100], color='blue', label='Random Forest')
        ax1.plot(Y_pred_xgb_actual[:100], color='purple', label='XGBoost')
        ax1.legend()
        ax1.set_title("Actual vs Predicted: First 100 Samples")
        st.pyplot(fig1)

        # Plot 2: Residual Analysis (LSTM as example)
        st.subheader("Plot 2: Residual Analysis (LSTM)")
        residuals = Y_test_actual[:100] - Y_pred_lstm_actual[:100]
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.scatter(range(len(residuals)), residuals, alpha=0.6, label='Residuals')
        ax2.axhline(0, color='red', linestyle='--')
        ax2.set_title("Residual Analysis (LSTM)")
        ax2.set_xlabel("Sample Index")
        ax2.set_ylabel("Residuals")
        ax2.legend()
        st.pyplot(fig2)

        # Seasonal Plots
        st.subheader("Plot 3: Seasonal Plots (Hour and Month)")
        dataset_filtered['hour'] = dataset_filtered.index.hour
        hourly_avg = dataset_filtered.groupby('hour')['pollution'].mean()
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.barplot(x=hourly_avg.index, y=hourly_avg.values, palette='Blues_d', ax=ax3)
        ax3.set_title("Average PM2.5 by Hour of the Day")
        ax3.set_xlabel("Hour")
        ax3.set_ylabel("PM2.5")
        st.pyplot(fig3)

        dataset_filtered['month'] = dataset_filtered.index.month
        monthly_avg = dataset_filtered.groupby('month')['pollution'].mean()
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        sns.barplot(x=monthly_avg.index, y=monthly_avg.values, palette='Oranges_d', ax=ax4)
        ax4.set_title("Average PM2.5 by Month")
        ax4.set_xlabel("Month")
        ax4.set_ylabel("PM2.5")
        st.pyplot(fig4)

        # Correlation Heatmap
        st.subheader("Plot 4: Correlation Heatmap")
        fig5, ax5 = plt.subplots(figsize=(10, 8))
        corr_matrix = dataset_filtered.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True, ax=ax5)
        ax5.set_title("Correlation Heatmap of Features")
        st.pyplot(fig5)

        # Plot 5: Time-Series Decomposition
        st.subheader("Plot 5: Time-Series Decomposition (if data is sufficient)")
        if len(dataset_filtered) >= 48:
            decomposed = seasonal_decompose(dataset_filtered['pollution'], model='additive', period=24)
            fig6, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
            
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
            st.pyplot(fig6)
        else:
            st.warning("Not enough data points for decomposition.")

        # Plot: Actual Interrupted and LSTM Continues
        st.subheader("Additional: Actual Interrupted and LSTM Continues")
        fig7, ax7 = plt.subplots(figsize=(14, 6))
        transition_point = len(Y_test_actual) // 2
        ax7.plot(range(transition_point), Y_test_actual[:transition_point], color='red', label='Actual (Observed)')
        ax7.plot(
            range(transition_point, len(Y_test_actual)),
            Y_pred_lstm_actual[transition_point:], color='green', label='LSTM Predicted'
        )
        ax7.axvline(x=transition_point, color='blue', linestyle='--', label='Transition Point')
        ax7.set_title("Actual Interrupted and LSTM Continues")
        ax7.set_xlabel("Index")
        ax7.set_ylabel("PM2.5 Concentration")
        ax7.legend()
        plt.tight_layout()
        st.pyplot(fig7)

        # Plot: Actual Interrupted and Random Forest Continues
        st.subheader("Additional: Actual Interrupted and Random Forest Continues")
        fig8, ax8 = plt.subplots(figsize=(14, 6))
        ax8.plot(range(transition_point), Y_test_actual[:transition_point], color='red', label='Actual (Observed)')
        ax8.plot(
            range(transition_point, len(Y_test_actual)),
            Y_pred_rf_actual[transition_point:], color='blue', label='Random Forest Predicted'
        )
        ax8.axvline(x=transition_point, color='purple', linestyle='--', label='Transition Point')
        ax8.set_title("Actual Interrupted and Random Forest Continues")
        ax8.set_xlabel("Index")
        ax8.set_ylabel("PM2.5 Concentration")
        ax8.legend()
        plt.tight_layout()
        st.pyplot(fig8)

    st.success("All plots generated successfully!")

else:
    st.warning("Please upload a valid CSV file to proceed.")
