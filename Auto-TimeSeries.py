                                       '1-PYCARET'
# LOADING DATA
import pandas as pd
from pycaret.datasets import get_data
data = get_data('pycaret_downloads')
data['Date'] = pd.to_datetime(data['Date'])
data = data.groupby('Date').sum()
data = data.asfreq('D')
data.head()

# plot the data
data.plot()

# with functional API
from pycaret.time_series import *
setup(data, fh = 7, fold = 3, session_id = 123)# with new object-oriented API
from pycaret.internal.pycaret_experiment import TimeSeriesExperiment
exp = TimeSeriesExperiment()
exp.setup(data, fh = 7, fold = 3, session_id = 123)

check_stats()

# functional API(EXPLORATORY DATA ANALYSIS)
plot_model(plot = 'ts')# object-oriented API
exp.plot_model(plot = 'ts')

# cross-validation plot
plot_model(plot = 'cv')

# ACF plot
plot_model(plot = 'acf')

# Diagnostics plot
plot_model(plot = 'diagnostics')

# Decomposition plot
plot_model(plot = 'decomp_stl')

#Model Training and Selectin(functional API)
best = compare_models()# object-oriented API
best = exp.compare_models()

# create the best model
prophet = create_model('prophet')
print(prophet)

#tune model
tuned_prophet = tune_model(prophet)
print(tuned_prophet)

#Plot model
plot_model(best, plot = 'forecast')

# forecast in unknown future
plot_model(best, plot = 'forecast', data_kwargs = {'fh' : 30})

# in-sample plot
plot_model(best, plot = 'insample')

# residuals plot
plot_model(best, plot = 'residuals')

# diagnostics plot
plot_model(best, plot = 'diagnostics')

# finalize model
final_best = finalize_model(best)# generate predictions
predict_model(final_best, fh = 90)


                                '2-MERLION'
from merlion.utils
import TimeSeries
from ts_datasets.anomaly
import NAB
time_series, metadata = NAB(subset = "realKnownCause")[3]
train_data = TimeSeries.from_pd(time_series[metadata.trainval])
test_data = TimeSeries.from_pd(time_series[~metadata.trainval])
test_labels = TimeSeries.from_pd(metadata.anomaly[~metadata.trainval])

from merlion.plot
import plot_anoms
import matplotlib.pyplot as plt
fig, ax = model.plot_anomaly(time_series = test_data)
plot_anoms(ax = ax, anomaly_labels = test_labels)
plt.show()

from merlion.evaluate.anomaly
import TSADMetric
p = TSADMetric.Precision.value(ground_truth = test_labels, predict = test_pred)
r = TSADMetric.Recall.value(ground_truth = test_labels, predict = test_pred)
f1 = TSADMetric.F1.value(ground_truth = test_labels, predict = test_pred)
mttd = TSADMetric.MeanTimeToDetect.value(ground_truth = test_labels, predict = test_pred)
print(f "Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}\n"
  f "Mean Time To Detect: {mttd}")

from merlion.utils
import TimeSeries
from ts_datasets.forecast
import M4

# Data loader returns pandas DataFrames, which we convert to Merlion TimeSeries
time_series, metadata = M4(subset = "Hourly")[0]
train_data = TimeSeries.from_pd(time_series[metadata.trainval])
test_data = TimeSeries.from_pd(time_series[~metadata.trainval])
from merlion.models.defaults
import DefaultForecasterConfig, DefaultForecaster
model = DefaultForecaster(DefaultForecasterConfig())
model.train(train_data = train_data)
test_pred, test_err = model.forecast(time_stamps = test_data.time_stamps)

import matplotlib.pyplot as plt
fig, ax = model.plot_forecast(time_series = test_data, plot_forecast_uncertainty = True)
plt.show()


                               '3-AUTO TIMESERIES'
from auto_ts import auto_timeseries
model = auto_timeseries( score_type='rmse', time_interval='Month', non_seasonal_pdq=None, seasonality=False, seasonal_period=12, model_type=['Prophet'], verbose=2)

df = pd.read_csv("Amazon_Stock_Price.csv", usecols=['Date', 'Close'])
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

train_df = df.iloc[:2800]
test_df = df.iloc[2800:]

train_df.Close.plot(figsize=(15,8), title= 'AMZN Stock Price', fontsize=14, label='Train')
test_df.Close.plot(figsize=(15,8), title= 'AMZN Stock Price', fontsize=14, label='Test')
plt.legend()
plt.grid()
plt.show()

model = auto_timeseries(forecast_period=219, score_type='rmse', time_interval='D', model_type='best')
model.fit(traindata= train_df, ts_column="Date", target="Close")

model.get_leaderboard()
model.plot_cv_scores()

future_predictions = model.predict(testdata=219)

                               '4-TIME FRESH'
# Importing libraries
import pandas as pd
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute, make_forecasting_frame
from tsfresh.feature_extraction import ComprehensiveFCParameters, settings

# Reading the data
data = pd.read_csv('../input/air-passengers/AirPassengers.csv')

# Some preprocessing for time component:
data.columns = ['month','Passengers']
data['month'] = pd.to_datetime(data['month'],infer_datetime_format=True,format='%y%m')
data.index = data.month
df_air = data.drop(['month'], axis = 1)

# Use Forecasting frame from tsfresh for rolling forecast training
df_shift, y_air = make_forecasting_frame(df_air["Passengers"], kind="Passengers", max_timeshift=12, rolling_direction=1)
print(df_shift)


# Getting Comprehensive Features
extraction_settings = ComprehensiveFCParameters()
X = extract_features(df_shift, column_id="id", column_sort="time", column_value="value", impute_function=impute,
                     show_warnings=False,
                     default_fc_parameters=extraction_settings
                     )


                            '5-DARTS'
#Loading the package
from darts import TimeSeries
from darts.models import ExponentialSmoothing
import matplotlib.pyplot as plt

# Reading the data
data = pd.read_csv('../input/air-passengers/AirPassengers.csv')
series = TimeSeries.from_dataframe(data, 'Month', '#Passengers')
print(series)


# Splitting the series in train and validation set
train, val = series.split_before(pd.Timestamp('19580101'))

# Applying a simple Exponential Smoothing model
model = ExponentialSmoothing()
model.fit(train)

# Getting and plotting the predictions
prediction = model.predict(len(val))series.plot(label='actual')
prediction.plot(label='forecast', lw=3)
plt.legend()