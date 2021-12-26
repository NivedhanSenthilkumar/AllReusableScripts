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