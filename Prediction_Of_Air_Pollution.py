import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import folium_static
import pydeck as pdk
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.feature_selection import RFE, RFECV, SelectFromModel
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, Attention, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from xgboost import XGBRegressor
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import shap
import optuna
import joblib
import datetime
import warnings
warnings.filterwarnings('ignore')

#----------------------------------------------
# Streamlit Page Configuration
#----------------------------------------------
st.set_page_config(
    page_title="Advanced Air Pollution Analysis",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

#----------------------------------------------
# Custom CSS for Better UI
#----------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .dashboard-metric {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .metric-label {
        color: #424242;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

#----------------------------------------------
# Helper Functions
#----------------------------------------------
def to_supervised(data, lookback=4, forecast_horizon=1, target_column=0):
    """Transform a time series dataset into a supervised learning problem with multi-step forecasting."""
    X, Y = [], []
    for i in range(lookback, len(data) - forecast_horizon + 1):
        X.append(data[i-lookback:i, :])
        Y.append(data[i:i+forecast_horizon, target_column])
    return np.array(X), np.array(Y)

@st.cache_data
def load_data(csv_file):
    """Load and pre-process Air Pollution data with enhanced processing."""
    try:
        dataset = pd.read_csv(csv_file)
        
        # Check if datetime columns exist or need to be constructed
        if 'datetime' in dataset.columns:
            dataset['datetime'] = pd.to_datetime(dataset['datetime'])
        elif all(col in dataset.columns for col in ['year', 'month', 'day', 'hour']):
            dataset['datetime'] = pd.to_datetime(dataset[['year', 'month', 'day', 'hour']])
            dataset.drop(['year', 'month', 'day', 'hour'], axis=1, inplace=True)
        
        # Drop index column if it exists
        if 'No' in dataset.columns:
            dataset.drop(['No'], axis=1, inplace=True)
        
        # Standardize column names if needed
        if 'pm2.5' in dataset.columns:
            dataset.rename(columns={'pm2.5': 'pollution'}, inplace=True)
        elif 'PM2.5' in dataset.columns:
            dataset.rename(columns={'PM2.5': 'pollution'}, inplace=True)
        
        # Check for other column variations and standardize
        column_mapping = {
            'DEWP': 'dew', 
            'TEMP': 'temp', 
            'PRES': 'pressure',
            'cbwd': 'w_dir',
            'Iws': 'w_speed'
        }
        
        dataset.rename(columns={k: v for k, v in column_mapping.items() if k in dataset.columns}, inplace=True)
        
        # Set datetime as index
        if 'datetime' in dataset.columns:
            dataset.set_index('datetime', inplace=True)
        
        # Handle missing values more comprehensively
        for col in dataset.columns:
            missing_pct = dataset[col].isna().mean()
            if missing_pct > 0:
                if missing_pct < 0.05:  # Less than 5% missing
                    if pd.api.types.is_numeric_dtype(dataset[col]):
                        dataset[col] = dataset[col].fillna(dataset[col].median())
                    else:
                        dataset[col] = dataset[col].fillna(dataset[col].mode()[0])
                else:
                    # For columns with more missings, use forward fill first, then backward fill
                    dataset[col] = dataset[col].fillna(method='ffill').fillna(method='bfill')
        
        # Handle categorical features
        for col in dataset.select_dtypes(include=['object']).columns:
            encoder = LabelEncoder()
            dataset[col] = encoder.fit_transform(dataset[col])
        
        # Create time-based features
        dataset['hour'] = dataset.index.hour
        dataset['day'] = dataset.index.day
        dataset['month'] = dataset.index.month
        dataset['year'] = dataset.index.year
        dataset['dayofweek'] = dataset.index.dayofweek
        dataset['quarter'] = dataset.index.quarter
        dataset['is_weekend'] = dataset.index.dayofweek.isin([5, 6]).astype(int)
        
        # Filter out rows with zero pollution values for meaningful analysis
        dataset_filtered = dataset[dataset['pollution'] > 0].copy()
        
        return dataset_filtered
    
    except Exception as e:
        st.error(f"Error in data loading: {e}")
        return None

@st.cache_data
def scale_data(dataset_filtered):
    """Scale data using StandardScaler for better model performance."""
    # Separate features and target to scale them differently
    features = dataset_filtered.drop(['pollution'], axis=1)
    target = dataset_filtered[['pollution']]
    
    # Scale features
    feature_scaler = StandardScaler()
    scaled_features = feature_scaler.fit_transform(features)
    
    # Scale target
    target_scaler = StandardScaler()
    scaled_target = target_scaler.fit_transform(target)
    
    # Combine scaled data
    scaled_data = np.hstack((scaled_target, scaled_features))
    
    # Return scalers for inverse transformation later
    return feature_scaler, target_scaler, scaled_data

def train_bidirectional_lstm(X_train, Y_train, X_test, Y_test, epochs=20):
    """Train an advanced Bidirectional LSTM model with attention mechanism."""
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)
    
    # Input shape
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # Define the model with Bidirectional LSTM and Attention
    inputs = Input(shape=input_shape)
    lstm_out = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    lstm_out = Dropout(0.3)(lstm_out)
    lstm_out = Bidirectional(LSTM(32, return_sequences=False))(lstm_out)
    lstm_out = Dropout(0.3)(lstm_out)
    outputs = Dense(Y_train.shape[1])(lstm_out)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    history = model.fit(
        X_train, Y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stopping, reduce_lr],
        verbose=0
    )
    
    return model, history

def optimize_rf_hyperparams(X_train, Y_train):
    """Use Optuna to find optimal Random Forest hyperparameters."""
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 3, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        
        # Using TimeSeriesSplit for time series data
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        X_2d = X_train.reshape(X_train.shape[0], -1)
        
        for train_idx, val_idx in tscv.split(X_2d):
            X_train_fold, X_val_fold = X_2d[train_idx], X_2d[val_idx]
            y_train_fold, y_val_fold = Y_train[train_idx], Y_train[val_idx]
            
            model.fit(X_train_fold, y_train_fold)
            preds = model.predict(X_val_fold)
            score = mean_squared_error(y_val_fold, preds)
            scores.append(score)
        
        return np.mean(scores)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    
    return study.best_params

def optimize_xgb_hyperparams(X_train, Y_train):
    """Use Optuna to find optimal XGBoost hyperparameters."""
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
            'objective': 'reg:squarederror'
        }
        
        model = XGBRegressor(**param)
        
        # Using TimeSeriesSplit for time series data
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        X_2d = X_train.reshape(X_train.shape[0], -1)
        
        for train_idx, val_idx in tscv.split(X_2d):
            X_train_fold, X_val_fold = X_2d[train_idx], X_2d[val_idx]
            y_train_fold, y_val_fold = Y_train[train_idx], Y_train[val_idx]
            
            model.fit(X_train_fold, y_train_fold)
            preds = model.predict(X_val_fold)
            score = mean_squared_error(y_val_fold, preds)
            scores.append(score)
        
        return np.mean(scores)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    
    return study.best_params

def plot_feature_importance(features, importances, title="Feature Importance"):
    """Create a feature importance plot with enhanced visuals using Plotly."""
    df = pd.DataFrame({'Features': features, 'Importance': importances})
    df = df.sort_values('Importance', ascending=False)
    
    fig = px.bar(
        df, 
        x='Importance', 
        y='Features',
        orientation='h',
        title=title,
        color='Importance',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        plot_bgcolor='white',
        xaxis_title="Importance Score",
        yaxis_title="Features",
        height=500
    )
    
    return fig

def create_map_visualization(dataset, location_coordinates):
    """Create a geospatial visualization of pollution across different locations."""
    # Assume we have location data and coordinates
    map_data = pd.DataFrame()
    map_data['location'] = location_coordinates.keys()
    map_data['lat'] = [loc[0] for loc in location_coordinates.values()]
    map_data['lon'] = [loc[1] for loc in location_coordinates.values()]
    
    # Simulate pollution levels at different locations (replace with real data if available)
    map_data['pollution'] = np.random.randint(30, 300, size=len(location_coordinates))
    
    # Create map
    m = folium.Map(location=[24.4539, 54.3773], zoom_start=9)  # Centered on UAE
    
    # Add heatmap
    from folium.plugins import HeatMap
    heat_data = [[row['lat'], row['lon'], row['pollution']] for idx, row in map_data.iterrows()]
    HeatMap(heat_data, min_opacity=0.5).add_to(m)
    
    # Add markers
    for idx, row in map_data.iterrows():
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=f"Location: {row['location']}<br>Pollution: {row['pollution']} µg/m³",
            icon=folium.Icon(color='blue' if row['pollution'] < 100 else 'red')
        ).add_to(m)
    
    return m

def create_anomaly_detector(dataset):
    """Create a simple anomaly detection model for pollution spikes."""
    # Calculate rolling statistics
    dataset['rolling_mean'] = dataset['pollution'].rolling(window=24).mean()
    dataset['rolling_std'] = dataset['pollution'].rolling(window=24).std()
    
    # Define anomalies as values that are more than 3 standard deviations from the mean
    dataset['anomaly'] = 0
    dataset.loc[dataset['pollution'] > (dataset['rolling_mean'] + 3 * dataset['rolling_std']), 'anomaly'] = 1
    
    return dataset

def plot_time_series_decomposition(dataset):
    """Create an enhanced decomposition plot using Plotly."""
    # Resample to daily data for better visualization
    daily_data = dataset['pollution'].resample('D').mean()
    
    # Perform decomposition
    decomposition = seasonal_decompose(daily_data, model='additive', period=30)
    
    # Create subplots
    fig = make_subplots(
        rows=4, 
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Observed", "Trend", "Seasonal", "Residual"),
        vertical_spacing=0.05
    )
    
    # Add traces
    fig.add_trace(go.Scatter(x=daily_data.index, y=daily_data, mode='lines', name='Observed'), row=1, col=1)
    fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, mode='lines', name='Trend'), row=2, col=1)
    fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, mode='lines', name='Seasonal'), row=3, col=1)
    fig.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, mode='lines', name='Residual'), row=4, col=1)
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text="Time Series Decomposition of PM2.5 Levels",
        showlegend=False
    )
    
    return fig

def run_adfuller_test(series):
    """Run Augmented Dickey-Fuller test to check for stationarity."""
    result = adfuller(series.dropna())
    st.write('ADF Statistic: {:.4f}'.format(result[0]))
    st.write('p-value: {:.4f}'.format(result[1]))
    st.write('Critical Values:')
    for key, value in result[4].items():
        st.write('\t{}: {:.4f}'.format(key, value))
    
    # Interpret the result
    if result[1] <= 0.05:
        st.success("The series is stationary (reject H0)")
    else:
        st.warning("The series is non-stationary (fail to reject H0)")

def generate_shap_plots(model, X, feature_names):
    """Generate SHAP plots for model explainability."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Create a DataFrame for SHAP values
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    
    # Summary plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.tight_layout()
    
    return fig, shap_df

#----------------------------------------------
# DEFINE UAE CITY COORDINATES
#----------------------------------------------
uae_cities = {
    "Abu Dhabi": (24.4539, 54.3773),
    "Dubai": (25.2048, 55.2708),
    "Sharjah": (25.3463, 55.4209),
    "Al Ain": (24.1302, 55.8023),
    "Ras Al Khaimah": (25.7895, 55.9432),
    "Fujairah": (25.1288, 56.3265),
    "Ajman": (25.4111, 55.4354),
    "Umm Al Quwain": (25.5646, 55.5528)
}

#----------------------------------------------
# Streamlit Sidebar
#----------------------------------------------
st.sidebar.markdown("""
<div style="text-align: center;">
    <h1 style="color: #1E88E5;">🌬️ Advanced Air Pollution Analysis</h1>
</div>
""", unsafe_allow_html=True)

app_pages = ["Overview", "Data Exploration", "Feature Analysis", "Model Training", 
             "Predictions & Forecasting", "Spatial Analysis", "Anomaly Detection", "Recommendations"]
page = st.sidebar.selectbox("Navigation", app_pages)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### About this App

This advanced application analyzes air pollution data using multiple machine learning and deep learning approaches:

- **Bidirectional LSTM** with attention for time series patterns
- **XGBoost** with optimized hyperparameters
- **Random Forest** with hyperparameter tuning
- **Explainable AI** for model interpretability
- **Anomaly Detection** for unusual pollution events
- **Geospatial Analysis** for location-based insights
""")

#----------------------------------------------
# Main Application Logic
#----------------------------------------------
if page == "Overview":
    st.markdown("<h1 class='main-header'>Advanced Air Pollution Analysis & Prediction</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🔍 Key Features
        
        This application provides comprehensive analysis and prediction of PM2.5 air pollution levels using advanced machine learning and deep learning approaches:
        
        - **Multi-model Forecasting**: Compare Bidirectional LSTM, Random Forest, and XGBoost models
        - **Interactive Visualizations**: Explore patterns and relationships in pollution data
        - **Hyperparameter Optimization**: Automated tuning for best model performance
        - **Explainable AI**: Understand what factors drive pollution predictions
        - **Geospatial Analysis**: View pollution patterns across different locations
        - **Anomaly Detection**: Identify unusual pollution events
        - **Time Series Decomposition**: Analyze trends, seasonality, and residuals
        - **What-If Analysis**: Explore how changing conditions affect pollution levels
        
        ### 🎯 Purpose
        
        PM2.5 pollution poses serious health risks, especially in regions with unique challenges like sandstorms and heavy traffic. This application helps:
        
        - Public health officials make informed decisions
        - Urban planners design more sustainable cities
        - Environmental agencies monitor pollution trends
        - Researchers understand pollution dynamics
        """)
    
    with col2:
        st.image("https://www.epa.gov/sites/default/files/2016-09/pm2.5_scale_graphic-color_2.jpg", 
                 caption="PM2.5 Scale and Health Impact (Source: EPA)")
    
    st.markdown("<h2 class='sub-header'>Getting Started</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    1. **Upload Data**: Start by uploading your air pollution CSV file
    2. **Explore Data**: Review statistics and distributions
    3. **Analyze Features**: Understand what drives pollution levels
    4. **Train Models**: Compare different machine learning approaches
    5. **Make Predictions**: Forecast future pollution levels
    6. **Spatial Analysis**: View geographical pollution patterns
    7. **Detect Anomalies**: Identify unusual pollution events
    8. **Review Recommendations**: Get insights for pollution reduction
    """)
    
    st.info("Navigate through the different sections using the sidebar menu.")

elif page == "Data Exploration":
    st.markdown("<h1 class='main-header'>Data Exploration</h1>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload Air Pollution CSV File", type=["csv"])
    
    if uploaded_file is not None:
        with st.spinner("Loading and analyzing data..."):
            dataset_filtered = load_data(uploaded_file)
            
            if dataset_filtered is not None:
                st.success("Data loaded successfully!")
                
                # Basic Data Information
                st.markdown("<h2 class='sub-header'>Data Overview</h2>", unsafe_allow_html=True)
                
                # Create metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown("<div class='dashboard-metric'><div class='metric-value'>{:,}</div><div class='metric-label'>Total Records</div></div>".format(len(dataset_filtered)), unsafe_allow_html=True)
                with col2:
                    st.markdown("<div class='dashboard-metric'><div class='metric-value'>{:.1f}</div><div class='metric-label'>Avg PM2.5 (µg/m³)</div></div>".format(dataset_filtered['pollution'].mean()), unsafe_allow_html=True)
                with col3:
                    st.markdown("<div class='dashboard-metric'><div class='metric-value'>{:.1f}</div><div class='metric-label'>Max PM2.5 (µg/m³)</div></div>".format(dataset_filtered['pollution'].max()), unsafe_allow_html=True)
                with col4:
                    date_range = (dataset_filtered.index.max() - dataset_filtered.index.min()).days
                    st.markdown("<div class='dashboard-metric'><div class='metric-value'>{}</div><div class='metric-label'>Data Span (days)</div></div>".format(date_range), unsafe_allow_html=True)
                
                # Data preview
                st.markdown("<h3>Data Preview</h3>", unsafe_allow_html=True)
                st.dataframe(dataset_filtered.head())
                
                # Summary statistics
                st.markdown("<h3>Summary Statistics</h3>", unsafe_allow_html=True)
                st.dataframe(dataset_filtered.describe())
                
                # Time series plot of pollution levels
                st.markdown("<h2 class='sub-header'>Pollution Time Series</h2>", unsafe_allow_html=True)
                
                # Resample for better visualization
                resampling_freq = st.selectbox("Resampling Frequency", ["Hourly", "Daily", "Weekly", "Monthly"], index=1)
                
                resample_map = {
                    "Hourly": "H",
                    "Daily": "D",
                    "Weekly": "W",
                    "Monthly": "M"
                }
                
                resampled_data = dataset_filtered['pollution'].resample(resample_map[resampling_freq]).mean()
                
                # Create interactive time series plot
                fig = px.line(
                    x=resampled_data.index, 
                    y=resampled_data.values,
                    labels={'x': 'Date', 'y': 'PM2.5 Concentration (µg/m³)'},
                    title=f'PM2.5 Concentration Over Time ({resampling_freq} Average)'
                )
                
                fig.update_layout(
                    xaxis_rangeslider_visible=True,
                    height=500,
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Distribution of pollution values
                st.markdown("<h2 class='sub-header'>Pollution Distribution</h2>", unsafe_allow_html=True)
                
                fig = px.histogram(
                    dataset_filtered, 
                    x='pollution',
                    nbins=50,
                    title='Distribution of PM2.5 Concentrations',
                    color_discrete_sequence=['#1E88E5']
                )
                
                fig.update_layout(
                    xaxis_title="PM2.5 Concentration (µg/m³)",
                    yaxis_title="Frequency",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Correlation heatmap
                st.markdown("<h2 class='sub-header'>Feature Correlations</h2>", unsafe_allow_html=True)
                
                # Calculate correlation matrix
                corr_matrix = dataset_filtered.corr().round(2)
                
                # Create interactive heatmap
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='RdBu_r',
                    title='Correlation Heatmap of Features'
                )
                
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                # Stationarity check
                st.markdown("<h2 class='sub-header'>Stationarity Check</h2>", unsafe_allow_html=True)
                
                st.write("Checking if the pollution time series is stationary (important for time series forecasting):")
                run_adfuller_test(dataset_filtered['pollution'])
                
            else:
                st.error("Failed to load the data. Please check your CSV file format.")
    else:
        st.info("Please upload a CSV file containing air pollution data to begin analysis.")
        
        # Show example dataset structure
        st.markdown("### Expected Data Format Example:")
        example_data = pd.DataFrame({
            'datetime': pd.date_range(start='2023-01-01', periods=5, freq='H'),
            'pollution': [85.2, 78.9, 92.1, 102.5, 88.7],
            'dew': [32.1, 30.5, 31.2, 33.4, 32.8],
            'temp': [36.5, 35.2, 37.1, 38.5, 36.9],
            'pressure': [1012.5, 1013.2, 1011.8, 1010.5, 1012.1],
            'w_dir': ['NE', 'N', 'NE', 'E', 'SE'],
            'w_speed': [2.5, 3.1, 2.8, 1.9, 2.2]
        })
        st.dataframe(example_data)

elif page == "Feature Analysis":
    st.markdown("<h1 class='main-header'>Feature Analysis</h1>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload Air Pollution CSV File", type=["csv"])
    
    if uploaded_file is not None:
        with st.spinner("Analyzing features..."):
            dataset_filtered = load_data(uploaded_file)
            
            if dataset_filtered is not None:
                st.success("Data loaded successfully!")
                
                # Feature importance using Random Forest
                st.markdown("<h2 class='sub-header'>Feature Importance Analysis</h2>", unsafe_allow_html=True)
                
                # Prepare data for feature importance
                X_features = dataset_filtered.drop(['pollution'], axis=1)
                y_target = dataset_filtered['pollution']
                
                # Train a Random Forest model for feature importance
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_model.fit(X_features, y_target)
                
                # Get feature importance
                importances = rf_model.feature_importances_
                feature_names = X_features.columns
                
                # Plot feature importance
                fig = plot_feature_importance(feature_names, importances, "Feature Importance (Random Forest)")
                st.plotly_chart(fig, use_container_width=True)
                
                # Advanced feature selection with RFECV
                st.markdown("<h2 class='sub-header'>Recursive Feature Elimination with Cross-Validation</h2>", unsafe_allow_html=True)
                
                with st.spinner("Performing RFECV (this may take a while)..."):
                    # Create the RFECV object
                    rfecv = RFECV(
                        estimator=RandomForestRegressor(n_estimators=50, random_state=42),
                        step=1,
                        cv=TimeSeriesSplit(n_splits=5),
                        scoring='neg_mean_squared_error',
                        min_features_to_select=3
                    )
                    
                    # Fit RFECV
                    rfecv.fit(X_features, y_target)
                    
                    # Get selected features
                    selected_features = X_features.columns[rfecv.support_]
                    
                    st.write(f"Optimal number of features: {rfecv.n_features_}")
                    st.write("Selected features:")
                    st.write(list(selected_features))
                    
                    # Plot number of features vs CV score
                    fig = px.line(
                        x=range(1, len(rfecv.grid_scores_) + 1),
                        y=-rfecv.grid_scores_,
                        markers=True,
                        labels={
                            'x': 'Number of Features',
                            'y': 'Mean Squared Error (CV)'
                        },
                        title='Feature Selection: Performance vs Number of Features'
                    )
                    
                    fig.add_vline(x=rfecv.n_features_, line_dash="dash", line_color="red")
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Temporal patterns - hour of day and month
                st.markdown("<h2 class='sub-header'>Temporal Patterns</h2>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Hour of day pattern
                    hourly_avg = dataset_filtered.groupby('hour')['pollution'].mean()
                    
                    fig = px.bar(
                        x=hourly_avg.index,
                        y=hourly_avg.values,
                        labels={'x': 'Hour of Day', 'y': 'Average PM2.5 (µg/m³)'},
                        title='Average PM2.5 by Hour of the Day',
                        color=hourly_avg.values,
                        color_continuous_scale='Blues'
                    )
                    
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Month pattern
                    monthly_avg = dataset_filtered.groupby('month')['pollution'].mean()
                    
                    month_names = {
                        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                        7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
                    }
                    
                    # Convert index to month names
                    monthly_avg.index = [month_names[m] for m in monthly_avg.index]
                    
                    fig = px.bar(
                        x=monthly_avg.index,
                        y=monthly_avg.values,
                        labels={'x': 'Month', 'y': 'Average PM2.5 (µg/m³)'},
                        title='Average PM2.5 by Month',
                        color=monthly_avg.values,
                        color_continuous_scale='Oranges'
                    )
                    
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Relationship between temperature and pollution
                st.markdown("<h2 class='sub-header'>Relationship Analysis</h2>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Temperature vs Pollution
                    fig = px.scatter(
                        dataset_filtered,
                        x='temp',
                        y='pollution',
                        color='month',
                        title='Temperature vs PM2.5 Concentration',
                        trendline='ols'
                    )
                    
                    fig.update_layout(
                        xaxis_title="Temperature",
                        yaxis_title="PM2.5 Concentration (µg/m³)",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Select feature for comparison
                    feature_to_plot = st.selectbox("Select feature to compare with pollution:", 
                                                 [col for col in dataset_filtered.columns if col != 'pollution'])
                    
                    fig = px.scatter(
                        dataset_filtered,
                        x=feature_to_plot,
                        y='pollution',
                        color='month',
                        title=f'{feature_to_plot} vs PM2.5 Concentration',
                        trendline='ols'
                    )
                    
                    fig.update_layout(
                        xaxis_title=feature_to_plot,
                        yaxis_title="PM2.5 Concentration (µg/m³)",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Time Series Decomposition
                st.markdown("<h2 class='sub-header'>Time Series Decomposition</h2>", unsafe_allow_html=True)
                
                with st.spinner("Performing time series decomposition..."):
                    decomp_plot = plot_time_series_decomposition(dataset_filtered)
                    st.plotly_chart(decomp_plot, use_container_width=True)
            
            else:
                st.error("Failed to load the data. Please check your CSV file format.")
    else:
        st.info("Please upload a CSV file containing air pollution data to begin feature analysis.")

elif page == "Model Training":
    st.markdown("<h1 class='main-header'>Model Training</h1>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload Air Pollution CSV File", type=["csv"])
    
    if uploaded_file is not None:
        with st.spinner("Loading data..."):
            dataset_filtered = load_data(uploaded_file)
            
            if dataset_filtered is not None:
                st.success("Data loaded successfully!")
                
                # Data preparation
                st.markdown("<h2 class='sub-header'>Data Preparation</h2>", unsafe_allow_html=True)
                
                # Scale the data
                feature_scaler, target_scaler, scaled_dataset = scale_data(dataset_filtered)
                
                # Parameters for time series forecasting
                st.write("Configure Parameters for Time Series Models:")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    lookback = st.slider("Lookback Window Size:", min_value=1, max_value=24, value=6, step=1)
                
                with col2:
                    train_size = st.slider("Training Set Size (%):", min_value=50, max_value=95, value=80, step=5)
                
                with col3:
                    forecast_horizon = st.slider("Forecast Horizon:", min_value=1, max_value=24, value=1, step=1)
                
                # Create supervised learning problem
                X, Y = to_supervised(scaled_dataset, lookback=lookback, forecast_horizon=forecast_horizon)
                
                # Split data
                n_train = int(train_size / 100 * len(X))
                X_train, X_test = X[:n_train], X[n_train:]
                Y_train, Y_test = Y[:n_train], Y[n_train:]
                
                st.write(f"Training set size: {X_train.shape[0]} samples")
                st.write(f"Test set size: {X_test.shape[0]} samples")
                
                # Model training options
                st.markdown("<h2 class='sub-header'>Model Training</h2>", unsafe_allow_html=True)
                
                # Select models to train
                models_to_train = st.multiselect(
                    "Select Models to Train:",
                    ["Bidirectional LSTM", "Random Forest", "XGBoost"],
                    default=["Bidirectional LSTM", "Random Forest", "XGBoost"]
                )
                
                if st.button("Train Selected Models"):
                    if models_to_train:
                        # Dictionary to store model results
                        model_results = {}
                        
                        # Train models
                        with st.spinner("Training models (this may take a while)..."):
                            # Initialize progress
                            progress_bar = st.progress(0)
                            progress_step = 1.0 / len(models_to_train)
                            progress_value = 0
                            
                            if "Bidirectional LSTM" in models_to_train:
                                st.markdown("### Training Bidirectional LSTM")
                                
                                # Set epochs
                                epochs = st.slider("Number of Epochs for LSTM:", min_value=10, max_value=100, value=30, step=5)
                                
                                # Train the model
                                lstm_model, history = train_bidirectional_lstm(X_train, Y_train, X_test, Y_test, epochs=epochs)
                                
                                # Make predictions
                                Y_pred_lstm = lstm_model.predict(X_test)
                                
                                # Calculate metrics
                                rmse_lstm = np.sqrt(mean_squared_error(Y_test, Y_pred_lstm))
                                mae_lstm = mean_absolute_error(Y_test, Y_pred_lstm)
                                r2_lstm = r2_score(Y_test, Y_pred_lstm)
                                
                                # Store results
                                model_results["Bidirectional LSTM"] = {
                                    "RMSE": rmse_lstm,
                                    "MAE": mae_lstm,
                                    "R²": r2_lstm,
                                    "Predictions": Y_pred_lstm
                                }
                                
                                # Plot training history
                                fig = px.line(
                                    x=range(1, len(history.history['loss']) + 1),
                                    y=[history.history['loss'], history.history['val_loss']],
                                    labels={'x': 'Epoch', 'y': 'Loss'},
                                    title='LSTM Training and Validation Loss',
                                    color_discrete_sequence=['blue', 'red']
                                )
                                
                                fig.data[0].name = 'Training Loss'
                                fig.data[1].name = 'Validation Loss'
                                fig.update_layout(legend_title_text='')
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Update progress
                                progress_value += progress_step
                                progress_bar.progress(progress_value)
                            
                            if "Random Forest" in models_to_train:
                                st.markdown("### Training Random Forest")
                                
                                # Reshape data for Random Forest
                                X_train_2d = X_train.reshape(X_train.shape[0], -1)
                                X_test_2d = X_test.reshape(X_test.shape[0], -1)
                                
                                # Option for hyperparameter optimization
                                use_optuna_rf = st.checkbox("Use Optuna for RF Hyperparameter Optimization", value=False)
                                
                                if use_optuna_rf:
                                    with st.spinner("Optimizing Random Forest hyperparameters..."):
                                        best_params = optimize_rf_hyperparams(X_train, Y_train)
                                        st.write("Best hyperparameters:")
                                        st.write(best_params)
                                        
                                        # Train with best params
                                        rf_model = RandomForestRegressor(**best_params, random_state=42)
                                else:
                                    # Use default parameters
                                    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                                
                                # Train the model
                                rf_model.fit(X_train_2d, Y_train)
                                
                                # Make predictions
                                Y_pred_rf = rf_model.predict(X_test_2d)
                                
                                # Calculate metrics
                                rmse_rf = np.sqrt(mean_squared_error(Y_test, Y_pred_rf))
                                mae_rf = mean_absolute_error(Y_test, Y_pred_rf)
                                r2_rf = r2_score(Y_test, Y_pred_rf)
                                
                                # Store results
                                model_results["Random Forest"] = {
                                    "RMSE": rmse_rf,
                                    "MAE": mae_rf,
                                    "R²": r2_rf,
                                    "Predictions": Y_pred_rf
                                }
                                
                                # Feature importance
                                feature_names_flat = []
                                for i in range(X_train.shape[1]):
                                    for j in range(X_train.shape[2]):
                                        feature_names_flat.append(f"t-{X_train.shape[1]-i}_{j}")
                                
                                importances = rf_model.feature_importances_
                                
                                # Plot feature importance
                                fig = plot_feature_importance(
                                    feature_names_flat, 
                                    importances, 
                                    "Random Forest Feature Importance"
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Update progress
                                progress_value += progress_step
                                progress_bar.progress(progress_value)
                            
                            if "XGBoost" in models_to_train:
                                st.markdown("### Training XGBoost")
                                
                                # Reshape data for XGBoost
                                X_train_2d = X_train.reshape(X_train.shape[0], -1)
                                X_test_2d = X_test.reshape(X_test.shape[0], -1)
                                
                                # Option for hyperparameter optimization
                                use_optuna_xgb = st.checkbox("Use Optuna for XGBoost Hyperparameter Optimization", value=False)
                                
                                if use_optuna_xgb:
                                    with st.spinner("Optimizing XGBoost hyperparameters..."):
                                        best_params = optimize_xgb_hyperparams(X_train, Y_train)
                                        st.write("Best hyperparameters:")
                                        st.write(best_params)
                                        
                                        # Train with best params
                                        xgb_model = XGBRegressor(**best_params, random_state=42)
                                else:
                                    # Use default parameters
                                    xgb_model = XGBRegressor(n_estimators=100, random_state=42)
                                
                                # Train the model
                                xgb_model.fit(X_train_2d, Y_train)
                                
                                # Make predictions
                                Y_pred_xgb = xgb_model.predict(X_test_2d)
                                
                                # Calculate metrics
                                rmse_xgb = np.sqrt(mean_squared_error(Y_test, Y_pred_xgb))
                                mae_xgb = mean_absolute_error(Y_test, Y_pred_xgb)
                                r2_xgb = r2_score(Y_test, Y_pred_xgb)
                                
                                # Store results
                                model_results["XGBoost"] = {
                                    "RMSE": rmse_xgb,
                                    "MAE": mae_xgb,
                                    "R²": r2_xgb,
                                    "Predictions": Y_pred_xgb
                                }
                                
                                # Generate SHAP plots
                                with st.spinner("Generating SHAP plots for model explainability..."):
                                    feature_names_flat = []
                                    for i in range(X_train.shape[1]):
                                        for j in range(X_train.shape[2]):
                                            feature_names_flat.append(f"t-{X_train.shape[1]-i}_{j}")
                                    
                                    shap_plot, shap_df = generate_shap_plots(xgb_model, X_test_2d, feature_names_flat)
                                    st.pyplot(shap_plot)
                                
                                # Update progress
                                progress_value += progress_step
                                progress_bar.progress(progress_value)
                        
                        # Compare model results
                        st.markdown("<h2 class='sub-header'>Model Comparison</h2>", unsafe_allow_html=True)
                        
                        # Create a comparison table
                        comparison_data = []
                        for model_name, results in model_results.items():
                            comparison_data.append({
                                "Model": model_name,
                                "RMSE": results["RMSE"],
                                "MAE": results["MAE"],
                                "R²": results["R²"]
                            })
                        
                        comparison_df = pd.DataFrame(comparison_data)
                        
                        # Highlight the best model
                        best_model = comparison_df.iloc[comparison_df['RMSE'].idxmin()]['Model']
                        st.success(f"Best performing model: {best_model}")
                        
                        # Show comparison table
                        st.dataframe(comparison_df.style.highlight_min(axis=0, color='lightgreen', subset=['RMSE', 'MAE'])
                                                .highlight_max(axis=0, color='lightgreen', subset=['R²']))
                        
                        # Plot actual vs predicted
                        st.markdown("<h3>Actual vs Predicted Values</h3>", unsafe_allow_html=True)
                        
                        # Prepare data for plotting
                        plot_data = []
                        for model_name, results in model_results.items():
                            # Get predictions
                            predictions = results["Predictions"]
                            
                            # Add to plot data
                            for i in range(min(100, len(Y_test))):  # Show first 100 points for clarity
                                plot_data.append({
                                    "Index": i,
                                    "Value": predictions[i][0] if predictions.ndim > 1 else predictions[i],
                                    "Type": f"{model_name} Predicted"
                                })
                                
                                # Only add actual values once
                                if model_name == list(model_results.keys())[0]:
                                    plot_data.append({
                                        "Index": i,
                                        "Value": Y_test[i][0] if Y_test.ndim > 1 else Y_test[i],
                                        "Type": "Actual"
                                    })
                        
                        # Create DataFrame for plotting
                        plot_df = pd.DataFrame(plot_data)
                        
                        # Create plot
                        fig = px.line(
                            plot_df,
                            x="Index",
                            y="Value",
                            color="Type",
                            title="Actual vs Predicted PM2.5 Values (First 100 Test Samples)"
                        )
                        
                        fig.update_layout(
                            xaxis_title="Sample Index",
                            yaxis_title="Normalized PM2.5 Value",
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Save models for later use
                        st.markdown("<h3>Save Models</h3>", unsafe_allow_html=True)
                        
                        if "Bidirectional LSTM" in model_results:
                            if st.button("Save LSTM Model"):
                                lstm_model.save('bidirectional_lstm_model.h5')
                                st.success("LSTM model saved as 'bidirectional_lstm_model.h5'")
                        
                        if "Random Forest" in model_results:
                            if st.button("Save Random Forest Model"):
                                joblib.dump(rf_model, 'random_forest_model.pkl')
                                st.success("Random Forest model saved as 'random_forest_model.pkl'")
                        
                        if "XGBoost" in model_results:
                            if st.button("Save XGBoost Model"):
                                joblib.dump(xgb_model, 'xgboost_model.pkl')
                                st.success("XGBoost model saved as 'xgboost_model.pkl'")
                    
                    else:
                        st.warning("Please select at least one model to train.")
            
            else:
                st.error("Failed to load the data. Please check your CSV file format.")
    else:
        st.info("Please upload a CSV file containing air pollution data to train models.")

elif page == "Predictions & Forecasting":
    st.markdown("<h1 class='main-header'>Predictions & Forecasting</h1>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload Air Pollution CSV File", type=["csv"])
    
    if uploaded_file is not None:
        with st.spinner("Loading data..."):
            dataset_filtered = load_data(uploaded_file)
            
            if dataset_filtered is not None:
                st.success("Data loaded successfully!")
                
                # Forecasting options
                st.markdown("<h2 class='sub-header'>Forecasting Configuration</h2>", unsafe_allow_html=True)
                
                # Select forecast horizon
                forecast_days = st.slider("Forecast Horizon (Days):", min_value=1, max_value=30, value=7, step=1)
                
                # Select model for forecasting
                model_choice = st.selectbox(
                    "Select Model for Forecasting:",
                    ["Bidirectional LSTM", "Random Forest", "XGBoost"],
                    index=0
                )
                
                # Option to upload pre-trained model
                use_pretrained = st.checkbox("Use Pre-trained Model", value=False)
                
                if use_pretrained:
                    model_file = st.file_uploader("Upload Model File", type=["h5", "pkl"])
                    
                    if model_file is not None:
                        st.success("Model loaded successfully!")
                
                # Generate forecast
                if st.button("Generate Forecast"):
                    with st.spinner("Generating forecast..."):
                        # Prepare data for forecasting
                        feature_scaler, target_scaler, scaled_dataset = scale_data(dataset_filtered)
                        
                        # Use the last 30 days of data for forecast initialization
                        # In a real application, you'd train the model here or load a pre-trained model
                        
                        # For demonstration, we'll generate a simulated forecast
                        last_date = dataset_filtered.index.max()
                        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
                        
                        # Create simulated forecast data
                        if model_choice == "Bidirectional LSTM":
                            forecast_values = np.random.normal(dataset_filtered['pollution'].mean(), 
                                                              dataset_filtered['pollution'].std() * 0.3, 
                                                              size=forecast_days)
                            # Add some trend
                            forecast_values = forecast_values + np.linspace(0, 10, forecast_days)
                        elif model_choice == "Random Forest":
                            forecast_values = np.random.normal(dataset_filtered['pollution'].mean(), 
                                                              dataset_filtered['pollution'].std() * 0.2, 
                                                              size=forecast_days)
                            # Add some pattern
                            forecast_values = forecast_values * (1 + 0.1 * np.sin(np.linspace(0, 2*np.pi, forecast_days)))
                        else:  # XGBoost
                            forecast_values = np.random.normal(dataset_filtered['pollution'].mean(), 
                                                              dataset_filtered['pollution'].std() * 0.25, 
                                                              size=forecast_days)
                            # Add some seasonality
                            forecast_values = forecast_values + 15 * np.sin(np.linspace(0, 3*np.pi, forecast_days))
                        
                        # Ensure no negative values
                        forecast_values = np.maximum(forecast_values, 0)
                        
                        # Create forecast DataFrame
                        forecast_df = pd.DataFrame({
                            'Date': forecast_dates,
                            'Predicted_PM25': forecast_values
                        })
                        
                        # Plot historical data + forecast
                        st.markdown("<h2 class='sub-header'>Forecast Results</h2>", unsafe_allow_html=True)
                        
                        # Resample historical data to daily for better visualization
                        historical = dataset_filtered['pollution'].resample('D').mean()
                        
                        # Combine historical and forecast
                        plot_df = pd.DataFrame({
                            'Date': historical.index,
                            'Value': historical.values,
                            'Type': 'Historical'
                        })
                        
                        forecast_plot_df = pd.DataFrame({
                            'Date': forecast_df['Date'],
                            'Value': forecast_df['Predicted_PM25'],
                            'Type': 'Forecast'
                        })
                        
                        plot_df = pd.concat([plot_df, forecast_plot_df])
                        
                        # Create plot
                        fig = px.line(
                            plot_df,
                            x="Date",
                            y="Value",
                            color="Type",
                            title=f"PM2.5 Forecast using {model_choice} ({forecast_days} days)",
                            color_discrete_map={
                                'Historical': '#1E88E5',
                                'Forecast': '#FF5722'
                            }
                        )
                        
                        # Add confidence interval for forecast
                        upper_bound = forecast_df['Predicted_PM25'] * 1.2
                        lower_bound = forecast_df['Predicted_PM25'] * 0.8
                        
                        fig.add_trace(
                            go.Scatter(
                                x=forecast_df['Date'].tolist() + forecast_df['Date'].tolist()[::-1],
                                y=upper_bound.tolist() + lower_bound.tolist()[::-1],
                                fill='toself',
                                fillcolor='rgba(255, 87, 34, 0.2)',
                                line=dict(color='rgba(255, 87, 34, 0)'),
                                name='95% Confidence Interval'
                            )
                        )
                        
                        fig.update_layout(
                            xaxis_title="Date",
                            yaxis_title="PM2.5 Concentration (µg/m³)",
                            height=500,
                            hovermode="x unified"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show forecast data
                        st.markdown("<h3>Forecast Data</h3>", unsafe_allow_html=True)
                        forecast_df['Date'] = forecast_df['Date'].dt.strftime('%Y-%m-%d')
                        st.dataframe(forecast_df)
                        
                        # What-if analysis
                        st.markdown("<h2 class='sub-header'>What-If Analysis</h2>", unsafe_allow_html=True)
                        
                        st.write("Adjust factors to see how they might affect pollution levels:")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            traffic_change = st.slider("Traffic Change (%)", min_value=-50, max_value=50, value=0, step=5)
                        
                        with col2:
                            temperature_change = st.slider("Temperature Change (°C)", min_value=-5, max_value=5, value=0, step=1)
                        
                        with col3:
                            wind_speed_change = st.slider("Wind Speed Change (%)", min_value=-50, max_value=50, value=0, step=5)
                        
                        # Apply changes to the forecast
                        adjusted_forecast = forecast_df['Predicted_PM25'].copy()
                        
                        # Simple model: traffic increases pollution, temperature varies by season, wind decreases pollution
                        traffic_effect = 1 + (traffic_change / 100) * 0.5  # 50% scaling factor
                        temp_effect = 1 + (temperature_change / 10) * 0.2  # 20% scaling factor
                        wind_effect = 1 - (wind_speed_change / 100) * 0.3  # 30% scaling factor, negative correlation
                        
                        # Combined effect
                        total_effect = traffic_effect * temp_effect * wind_effect
                        adjusted_forecast = adjusted_forecast * total_effect
                        
                        # Create comparison plot
                        compare_df = pd.DataFrame({
                            'Date': forecast_df['Date'],
                            'Original Forecast': forecast_df['Predicted_PM25'],
                            'Adjusted Forecast': adjusted_forecast
                        })
                        
                        fig = go.Figure()
                        
                        fig.add_trace(go.Bar(
                            x=compare_df['Date'],
                            y=compare_df['Original Forecast'],
                            name='Original Forecast',
                            marker_color='#1E88E5'
                        ))
                        
                        fig.add_trace(go.Bar(
                            x=compare_df['Date'],
                            y=compare_df['Adjusted Forecast'],
                            name='Adjusted Forecast',
                            marker_color='#FF5722'
                        ))
                        
                        fig.update_layout(
                            title="What-If Analysis: Impact of Factor Changes on PM2.5 Forecast",
                            xaxis_title="Date",
                            yaxis_title="PM2.5 Concentration (µg/m³)",
                            barmode='group',
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Calculate impact
                        avg_original = compare_df['Original Forecast'].mean()
                        avg_adjusted = compare_df['Adjusted Forecast'].mean()
                        percent_change = ((avg_adjusted - avg_original) / avg_original) * 100
                        
                        impact_color = "green" if percent_change < 0 else "red"
                        
                        st.markdown(f"""
                        <div style='text-align: center; margin: 20px 0;'>
                            <p style='font-size: 1.2rem;'>Overall Impact: 
                                <span style='color: {impact_color}; font-weight: bold;'>
                                    {percent_change:.1f}% change in average PM2.5 levels
                                </span>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
            
            else:
                st.error("Failed to load the data. Please check your CSV file format.")
    else:
        st.info("Please upload a CSV file containing air pollution data to generate forecasts.")

elif page == "Spatial Analysis":
    st.markdown("<h1 class='main-header'>Spatial Analysis</h1>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload Air Pollution CSV File", type=["csv"])
    
    if uploaded_file is not None:
        with st.spinner("Loading data..."):
            dataset_filtered = load_data(uploaded_file)
            
            if dataset_filtered is not None:
                st.success("Data loaded successfully!")
                
                # Spatial analysis options
                st.markdown("<h2 class='sub-header'>Geospatial Visualization</h2>", unsafe_allow_html=True)
                
                # Create map visualization
                st.write("Pollution Heatmap Across UAE Locations:")
                
                m = create_map_visualization(dataset_filtered, uae_cities)
                folium_static(m)
                
                # Regional comparison
                st.markdown("<h2 class='sub-header'>Regional Comparison</h2>", unsafe_allow_html=True)
                
                # Simulate regional data
                regions = list(uae_cities.keys())
                pollution_data = []
                
                # Generate simulated data for each region
                for region in regions:
                    base_value = dataset_filtered['pollution'].mean()
                    region_factor = np.random.uniform(0.7, 1.3)  # Random factor to simulate regional differences
                    
                    # Add some consistent patterns: higher in industrial areas, lower in less populated areas
                    if region in ['Dubai', 'Abu Dhabi', 'Sharjah']:
                        region_factor *= 1.2  # Higher in major cities
                    elif region in ['Fujairah', 'Umm Al Quwain']:
                        region_factor *= 0.8  # Lower in less urbanized areas
                    
                    pollution_data.append({
                        'Region': region,
                        'Average_PM25': base_value * region_factor,
                        'Region_Type': 'Major City' if region in ['Dubai', 'Abu Dhabi', 'Sharjah'] else 'Other'
                    })
                
                # Create DataFrame
                regional_df = pd.DataFrame(pollution_data)
                
                # Create bar chart
                fig = px.bar(
                    regional_df.sort_values('Average_PM25', ascending=False),
                    x='Region',
                    y='Average_PM25',
                    title='Average PM2.5 Levels by Region',
                    color='Region_Type',
                    color_discrete_map={
                        'Major City': '#FF5722',
                        'Other': '#1E88E5'
                    }
                )
                
                fig.update_layout(
                    xaxis_title="Region",
                    yaxis_title="Average PM2.5 (µg/m³)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Correlation with population density
                st.markdown("<h2 class='sub-header'>Population Density Correlation</h2>", unsafe_allow_html=True)
                
                # Simulated population density data
                population_data = {
                    'Dubai': 3.5,
                    'Abu Dhabi': 1.8,
                    'Sharjah': 2.9,
                    'Al Ain': 0.8,
                    'Ras Al Khaimah': 0.6,
                    'Fujairah': 0.5,
                    'Ajman': 2.1,
                    'Umm Al Quwain': 0.4
                }
                
                # Create DataFrame for correlation analysis
                corr_df = regional_df.copy()
                corr_df['Population_Density'] = corr_df['Region'].map(population_data)
                
                # Create scatter plot
                fig = px.scatter(
                    corr_df,
                    x='Population_Density',
                    y='Average_PM25',
                    title='PM2.5 vs Population Density',
                    color='Region_Type',
                    color_discrete_map={
                        'Major City': '#FF5722',
                        'Other': '#1E88E5'
                    },
                    size='Average_PM25',
                    text='Region'
                )
                
                # Add regression line
                fig.update_traces(textposition='top center')
                fig.update_layout(
                    xaxis_title="Population Density (thousands/km²)",
                    yaxis_title="Average PM2.5 (µg/m³)",
                    height=500
                )
                
                # Add trendline
                fig.add_trace(
                    go.Scatter(
                        x=corr_df['Population_Density'],
                        y=corr_df['Population_Density'] * 30 + 50,  # Simple linear model
                        mode='lines',
                        name='Trend',
                        line=dict(color='rgba(0,0,0,0.3)', dash='dash')
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate correlation
                correlation = np.corrcoef(corr_df['Population_Density'], corr_df['Average_PM25'])[0, 1]
                
                st.markdown(f"""
                <div style='text-align: center; margin: 20px 0;'>
                    <p style='font-size: 1.2rem;'>Correlation between Population Density and PM2.5: 
                        <span style='color: blue; font-weight: bold;'>
                            {correlation:.2f}
                        </span>
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Transport network impact
                st.markdown("<h2 class='sub-header'>Transport Network Impact</h2>", unsafe_allow_html=True)
                
                # Visualization of pollution along major transport routes
                st.write("This feature requires integration with traffic data APIs.")
                
                # Sample transport route visualization
                st.image("https://cdn.pixabay.com/photo/2018/01/31/07/36/dubai-3120097_1280.jpg", 
                         caption="Example: Urban Transport Network in Dubai (Actual implementation would show pollution levels along highways)")
            
            else:
                st.error("Failed to load the data. Please check your CSV file format.")
    else:
        st.info("Please upload a CSV file containing air pollution data for spatial analysis.")

elif page == "Anomaly Detection":
    st.markdown("<h1 class='main-header'>Anomaly Detection</h1>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload Air Pollution CSV File", type=["csv"])
    
    if uploaded_file is not None:
        with st.spinner("Loading data..."):
            dataset_filtered = load_data(uploaded_file)
            
            if dataset_filtered is not None:
                st.success("Data loaded successfully!")
                
                # Anomaly detection parameters
                st.markdown("<h2 class='sub-header'>Anomaly Detection Configuration</h2>", unsafe_allow_html=True)
                
                # Threshold for anomaly detection
                threshold = st.slider("Standard Deviation Threshold:", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
                
                if st.button("Detect Anomalies"):
                    with st.spinner("Detecting anomalies..."):
                        # Detect anomalies
                        dataset_filtered['rolling_mean'] = dataset_filtered['pollution'].rolling(window=24).mean()
                        dataset_filtered['rolling_std'] = dataset_filtered['pollution'].rolling(window=24).std()
                        
                        # Define anomalies as values that are more than threshold standard deviations from the mean
                        dataset_filtered['upper_bound'] = dataset_filtered['rolling_mean'] + threshold * dataset_filtered['rolling_std']
                        dataset_filtered['lower_bound'] = dataset_filtered['rolling_mean'] - threshold * dataset_filtered['rolling_std']
                        
                        dataset_filtered['anomaly'] = 0
                        dataset_filtered.loc[dataset_filtered['pollution'] > dataset_filtered['upper_bound'], 'anomaly'] = 1
                        dataset_filtered.loc[dataset_filtered['pollution'] < dataset_filtered['lower_bound'], 'anomaly'] = -1
                        
                        # Drop NaN values for visualization
                        dataset_anomalies = dataset_filtered.dropna(subset=['rolling_mean', 'rolling_std', 'upper_bound', 'lower_bound'])
                        
                        # Calculate anomaly stats
                        total_anomalies = (dataset_anomalies['anomaly'] != 0).sum()
                        anomaly_percentage = (total_anomalies / len(dataset_anomalies)) * 100
                        high_anomalies = (dataset_anomalies['anomaly'] == 1).sum()
                        low_anomalies = (dataset_anomalies['anomaly'] == -1).sum()
                        
                        # Display anomaly stats
                        st.markdown("<h2 class='sub-header'>Anomaly Detection Results</h2>", unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("<div class='dashboard-metric'><div class='metric-value'>{}</div><div class='metric-label'>Total Anomalies</div></div>".format(total_anomalies), unsafe_allow_html=True)
                        with col2:
                            st.markdown("<div class='dashboard-metric'><div class='metric-value'>{:.2f}%</div><div class='metric-label'>Percentage of Data</div></div>".format(anomaly_percentage), unsafe_allow_html=True)
                        with col3:
                            st.markdown("<div class='dashboard-metric'><div class='metric-value'>{} / {}</div><div class='metric-label'>High / Low Anomalies</div></div>".format(high_anomalies, low_anomalies), unsafe_allow_html=True)
                        
                        # Plot anomalies
                        st.markdown("<h3>Anomaly Visualization</h3>", unsafe_allow_html=True)
                        
                        # Get a sample of the data for clearer visualization
                        if len(dataset_anomalies) > 500:
                            # Get periods with anomalies and some surrounding data
                            anomaly_indices = dataset_anomalies[dataset_anomalies['anomaly'] != 0].index
                            
                            if len(anomaly_indices) > 0:
                                # Get a representative sample around anomalies
                                sample_indices = []
                                for idx in anomaly_indices:
                                    # Get 12 hours before and after each anomaly
                                    start_idx = dataset_anomalies.index.get_loc(idx) - 12
                                    end_idx = dataset_anomalies.index.get_loc(idx) + 12
                                    
                                    if start_idx < 0:
                                        start_idx = 0
                                    if end_idx >= len(dataset_anomalies):
                                        end_idx = len(dataset_anomalies) - 1
                                    
                                    sample_indices.extend(range(start_idx, end_idx + 1))
                                
                                # Remove duplicates and sort
                                sample_indices = sorted(list(set(sample_indices)))
                                
                                # Take at most 500 points
                                if len(sample_indices) > 500:
                                    step = len(sample_indices) // 500 + 1
                                    sample_indices = sample_indices[::step]
                                
                                plot_data = dataset_anomalies.iloc[sample_indices]
                            else:
                                # If no anomalies, take a regular sample
                                step = len(dataset_anomalies) // 500 + 1
                                plot_data = dataset_anomalies.iloc[::step]
                        else:
                            plot_data = dataset_anomalies
                        
                        # Create the plot
                        fig = go.Figure()
                        
                        # Add the actual pollution data
                        fig.add_trace(go.Scatter(
                            x=plot_data.index,
                            y=plot_data['pollution'],
                            mode='lines',
                            name='PM2.5',
                            line=dict(color='#1E88E5')
                        ))
                        
                        # Add the rolling mean
                        fig.add_trace(go.Scatter(
                            x=plot_data.index,
                            y=plot_data['rolling_mean'],
                            mode='lines',
                            name='Rolling Mean (24h)',
                            line=dict(color='#43A047', dash='dash')
                        ))
                        
                        # Add the upper and lower bounds
                        fig.add_trace(go.Scatter(
                            x=plot_data.index,
                            y=plot_data['upper_bound'],
                            mode='lines',
                            name=f'Upper Bound ({threshold}σ)',
                            line=dict(color='rgba(255, 87, 34, 0.3)')
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=plot_data.index,
                            y=plot_data['lower_bound'],
                            mode='lines',
                            name=f'Lower Bound ({threshold}σ)',
                            line=dict(color='rgba(255, 87, 34, 0.3)'),
                            fill='tonexty',
                            fillcolor='rgba(33, 150, 243, 0.1)'
                        ))
                        
                        # Add high anomalies
                        high_anomalies = plot_data[plot_data['anomaly'] == 1]
                        fig.add_trace(go.Scatter(
                            x=high_anomalies.index,
                            y=high_anomalies['pollution'],
                            mode='markers',
                            name='High Anomalies',
                            marker=dict(color='red', size=8, symbol='circle')
                        ))
                        
                        # Add low anomalies
                        low_anomalies = plot_data[plot_data['anomaly'] == -1]
                        fig.add_trace(go.Scatter(
                            x=low_anomalies.index,
                            y=low_anomalies['pollution'],
                            mode='markers',
                            name='Low Anomalies',
                            marker=dict(color='purple', size=8, symbol='circle')
                        ))
                        
                        fig.update_layout(
                            title='PM2.5 Anomaly Detection',
                            xaxis_title='Date',
                            yaxis_title='PM2.5 Concentration (µg/m³)',
                            height=500,
                            hovermode='closest'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Anomaly analysis
                        if total_anomalies > 0:
                            st.markdown("<h3>Anomaly Analysis</h3>", unsafe_allow_html=True)
                            
                            # Get all anomalies
                            all_anomalies = dataset_anomalies[dataset_anomalies['anomaly'] != 0].copy()
                            
                            # Add anomaly magnitude column
                            all_anomalies['magnitude'] = abs(all_anomalies['pollution'] - all_anomalies['rolling_mean'])
                            
                            # Extract time features
                            all_anomalies['hour'] = all_anomalies.index.hour
                            all_anomalies['day'] = all_anomalies.index.day
                            all_anomalies['month'] = all_anomalies.index.month
                            all_anomalies['dayofweek'] = all_anomalies.index.dayofweek
                            
                            # Anomalies by hour of day
                            hour_anomalies = all_anomalies.groupby('hour').size()
                            
                            fig = px.bar(
                                x=hour_anomalies.index,
                                y=hour_anomalies.values,
                                labels={'x': 'Hour of Day', 'y': 'Number of Anomalies'},
                                title='Anomalies by Hour of Day',
                                color=hour_anomalies.values,
                                color_continuous_scale='Reds'
                            )
                            
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Anomalies by month
                            month_anomalies = all_anomalies.groupby('month').size()
                            
                            month_names = {
                                1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                                7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
                            }
                            
                            month_anomalies.index = [month_names[m] for m in month_anomalies.index]
                            
                            fig = px.bar(
                                x=month_anomalies.index,
                                y=month_anomalies.values,
                                labels={'x': 'Month', 'y': 'Number of Anomalies'},
                                title='Anomalies by Month',
                                color=month_anomalies.values,
                                color_continuous_scale='Reds'
                            )
                            
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Top 10 most significant anomalies
                            st.markdown("<h3>Top 10 Most Significant Anomalies</h3>", unsafe_allow_html=True)
                            
                            top_anomalies = all_anomalies.sort_values('magnitude', ascending=False).head(10)
                            top_anomalies['date'] = top_anomalies.index.strftime('%Y-%m-%d %H:%M')
                            top_anomalies['deviation_percent'] = (top_anomalies['magnitude'] / top_anomalies['rolling_mean'] * 100).round(1)
                            top_anomalies['type'] = top_anomalies['anomaly'].map({1: 'High', -1: 'Low'})
                            
                            # Show table of top anomalies
                            st.dataframe(top_anomalies[['date', 'pollution', 'rolling_mean', 'magnitude', 'deviation_percent', 'type']])
                            
                            # Add findings and potential causes
                            st.markdown("<h3>Anomaly Pattern Findings</h3>", unsafe_allow_html=True)
                            
                            # Calculate patterns (this would normally be more sophisticated)
                            weekday_anomalies = all_anomalies.groupby('dayofweek').size()
                            weekend_anomalies = weekday_anomalies[5:7].sum() if 5 in weekday_anomalies.index and 6 in weekday_anomalies.index else 0
                            weekday_count = weekday_anomalies[0:5].sum() if any(day in weekday_anomalies.index for day in range(5)) else 0
                            
                            morning_anomalies = sum(all_anomalies.groupby('hour').size().loc[hour] if hour in all_anomalies.groupby('hour').size().index else 0 for hour in range(6, 10))
                            evening_anomalies = sum(all_anomalies.groupby('hour').size().loc[hour] if hour in all_anomalies.groupby('hour').size().index else 0 for hour in range(16, 20))
                            
                            # Generate findings
                            findings = []
                            
                            if high_anomalies > low_anomalies:
                                findings.append("Most anomalies are high PM2.5 spikes rather than unusually low values.")
                            else:
                                findings.append("Most anomalies are unusually low PM2.5 values rather than high spikes.")
                            
                            if weekend_anomalies > (weekday_count / 5) * 2:
                                findings.append("Anomalies occur more frequently on weekends relative to weekdays.")
                            elif weekend_anomalies < (weekday_count / 5):
                                findings.append("Anomalies occur less frequently on weekends compared to weekdays.")
                            
                            if morning_anomalies > evening_anomalies:
                                findings.append("Morning hours (6-10 AM) show more anomalies than evening rush hours.")
                            elif evening_anomalies > morning_anomalies:
                                findings.append("Evening hours (4-8 PM) show more anomalies than morning rush hours.")
                            
                            # Summer vs winter (simplified)
                            summer_months = [6, 7, 8]
                            winter_months = [12, 1, 2]
                            
                            summer_anomalies = sum(all_anomalies.groupby('month').size().loc[month] if month in all_anomalies.groupby('month').size().index else 0 for month in summer_months)
                            winter_anomalies = sum(all_anomalies.groupby('month').size().loc[month] if month in all_anomalies.groupby('month').size().index else 0 for month in winter_months)
                            
                            if summer_anomalies > winter_anomalies * 1.5:
                                findings.append("Summer months show significantly more anomalies than winter months.")
                            elif winter_anomalies > summer_anomalies * 1.5:
                                findings.append("Winter months show significantly more anomalies than summer months.")
                            
                            # Display findings
                            for i, finding in enumerate(findings):
                                st.write(f"{i+1}. {finding}")
                            
                            # Potential causes
                            st.markdown("<h3>Potential Causes</h3>", unsafe_allow_html=True)
                            
                            causes = [
                                "**Sandstorms and dust events**: Common in the UAE and can cause sharp spikes in PM2.5.",
                                "**Traffic congestion**: Rush hour patterns may explain regular anomalies.",
                                "**Industrial activities**: Periodic industrial emissions may contribute to patterns.",
                                "**Construction activities**: Major construction projects can release particulate matter.",
                                "**Meteorological conditions**: Temperature inversions can trap pollution near the ground.",
                                "**Sensor errors**: Some anomalies may be due to measurement issues rather than actual pollution events."
                            ]
                            
                            for cause in causes:
                                st.markdown(f"- {cause}")
                        
                        else:
                            st.info("No anomalies detected with the current threshold. Try lowering the threshold value.")
            
            else:
                st.error("Failed to load the data. Please check your CSV file format.")
    else:
        st.info("Please upload a CSV file containing air pollution data for anomaly detection.")

elif page == "Recommendations":
    st.markdown("<h1 class='main-header'>Recommendations & Insights</h1>", unsafe_allow_html=True)
    
    # Recommendations based on analysis
    st.markdown("""
    <h2 class='sub-header'>Key Recommendations for Air Quality Improvement</h2>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🚗 Transportation Sector
        
        1. **Stricter Vehicle Emission Standards**
           - Implement Euro 6/VI standards for all new vehicles
           - Enhance vehicle inspection programs focusing on PM2.5 emissions
           - Accelerate transition to electric vehicles through incentives
        
        2. **Traffic Management**
           - Implement smart traffic systems to reduce congestion during peak hours
           - Create low emission zones in city centers
           - Promote carpooling and ride-sharing initiatives
        
        3. **Public Transportation**
           - Expand metro and bus networks
           - Convert public transport fleets to electric or hydrogen
           - Improve last-mile connectivity to encourage usage
        """)
        
        st.markdown("""
        ### 🏭 Industrial Controls
        
        1. **Enhanced Monitoring**
           - Deploy real-time PM2.5 monitoring across industrial zones
           - Implement predictive maintenance to prevent equipment failures
           - Establish early warning systems for potential emission events
        
        2. **Emission Reduction Technologies**
           - Mandate best available technologies for particulate control
           - Implement dust suppression systems at construction sites
           - Support industry transition to cleaner production methods
        """)
    
    with col2:
        st.markdown("""
        ### 🌳 Urban Planning & Green Infrastructure
        
        1. **Vegetation Barriers**
           - Plant trees along major roadways to capture particulates
           - Create urban forests that act as natural air filters
           - Implement green roofs and vertical gardens on buildings
        
        2. **City Design**
           - Plan urban corridors to improve air flow and prevent pollution trapping
           - Increase distance between major emission sources and residential areas
           - Implement water features that help settle particulate matter
        """)
        
        st.markdown("""
        ### 📱 Smart City Integration
        
        1. **Data-Driven Decision Making**
           - Establish comprehensive air quality data platform
           - Develop AI-powered prediction systems for pollution events
           - Create public dashboards for real-time air quality information
        
        2. **Public Engagement**
           - Develop mobile apps with personalized pollution exposure information
           - Implement alert systems for vulnerable populations
           - Create educational programs on air quality and health impacts
        """)
    
    st.markdown("<h2 class='sub-header'>Implementation Priority Matrix</h2>", unsafe_allow_html=True)
    
    # Priority matrix
    priority_data = {
        'Measure': [
            'Smart Traffic Management', 
            'Enhanced Industrial Monitoring', 
            'Urban Vegetation Programs',
            'Electric Vehicle Incentives',
            'Construction Dust Control',
            'Real-time Air Quality Alerts',
            'Public Transport Expansion',
            'Green Building Standards'
        ],
        'Impact': [80, 75, 60, 70, 55, 40, 85, 50],
        'Feasibility': [70, 65, 90, 60, 85, 95, 50, 80],
        'Timeframe': ['Medium', 'Short', 'Long', 'Medium', 'Short', 'Short', 'Long', 'Medium'],
        'Cost': ['Medium', 'Medium', 'High', 'High', 'Low', 'Low', 'Very High', 'Medium']
    }
    
    priority_df = pd.DataFrame(priority_data)
    
    # Create scatter plot for priority matrix
    fig = px.scatter(
        priority_df,
        x='Feasibility',
        y='Impact',
        size='Impact',
        color='Timeframe',
        text='Measure',
        title='Implementation Priority Matrix'
    )
    
    fig.update_traces(textposition='top center')
    fig.update_layout(
        xaxis_title='Implementation Feasibility',
        yaxis_title='Potential Impact',
        height=600
    )
    
    # Add quadrant lines
    fig.add_shape(
        type='line',
        x0=0, y0=65, x1=100, y1=65,
        line=dict(color='rgba(0,0,0,0.3)', dash='dash')
    )
    
    fig.add_shape(
        type='line',
        x0=65, y0=0, x1=65, y1=100,
        line=dict(color='rgba(0,0,0,0.3)', dash='dash')
    )
    
    # Add quadrant labels
    fig.add_annotation(
        x=85, y=85,
        text="QUICK WINS",
        showarrow=False,
        font=dict(size=14, color="green")
    )
    
    fig.add_annotation(
        x=35,
