import numpy as np
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler

class AdvancedForecaster:
    def __init__(self, data):
        self.data = data
        self.scaler = StandardScaler()
    
    
    def sarima_forecast(self, steps):
        """Generate SARIMA forecast"""
        try:
            model = SARIMAX(self.data['close'],
                           order=(1, 1, 1),
                           seasonal_order=(1, 1, 1, 12))
            results = model.fit(disp=False)
            forecast = results.forecast(steps=steps)
            return forecast
        except:
            # Return simple moving average forecast if SARIMA fails
            last_value = self.data['close'].rolling(window=20).mean().iloc[-1]
            return np.array([last_value] * steps)

    def prophet_forecast(self, steps):
        """Generate Prophet forecast"""
        # Prepare data for Prophet
        df_prophet = pd.DataFrame({
            'ds': self.data.index,
            'y': self.data['close']
        })

        # Initialize and fit Prophet model
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05
        )
        
        # Add additional regressors
        df_features = self.prepare_features()
        for column in ['MA5', 'MA20', 'volatility', 'volume_ma5']:
            if column in df_features.columns:
                df_prophet[column] = df_features[column]
                model.add_regressor(column)

        # Fit the model
        model.fit(df_prophet)

        # Create future dataframe
        future_dates = pd.date_range(
            start=self.data.index[-1] + pd.Timedelta(days=1),
            periods=steps
        )
        future = pd.DataFrame({'ds': future_dates})

        # Add regressors to future dataframe
        for column in ['MA5', 'MA20', 'volatility', 'volume_ma5']:
            if column in df_features.columns:
                future[column] = df_features[column].iloc[-1]

        # Make prediction
        forecast = model.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    def ensemble_forecast(self, steps):
        """Combine multiple forecasting methods"""
        # Get individual forecasts
        sarima_pred = self.sarima_forecast(steps)
        prophet_forecast = self.prophet_forecast(steps)
        
        # Extract Prophet predictions
        prophet_pred = prophet_forecast['yhat'].values
        prophet_lower = prophet_forecast['yhat_lower'].values
        prophet_upper = prophet_forecast['yhat_upper'].values
        
        # Combine forecasts (weighted average)
        weights = [0.3, 0.7]  # Giving more weight to Prophet
        ensemble_forecast = weights[0] * sarima_pred + weights[1] * prophet_pred
        
        # Create forecast dates
        forecast_dates = prophet_forecast['ds']
        
        # Create confidence intervals
        forecast_df = pd.DataFrame({
            'forecast': ensemble_forecast,
            'upper_bound': prophet_upper,
            'lower_bound': prophet_lower
        }, index=forecast_dates)
        
        return forecast_df
