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