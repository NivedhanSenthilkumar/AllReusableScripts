
"JP PIMPORT LIBRARIES"
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_validate

#machine learning classifiers
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import classification_report, confusion_matrix

##PURE AUTOML
from lazypredict.Supervised import LazyRegressor, LazyClassifier
from autosklearn.classification import AutoSklearnClassifier
from __future__ import print_function
import sys,tempfile, urllib, os
from autoviml.Auto_ViML import Auto_ViML
from tpot import TPOTClassifier
import h2o
from h2o.automl import H2OAutoML


                                  '1-INBUILT CUSTOM FUNCTION'
#1.1 - REGRESSION
scoring = {'R2-Square': make_scorer(r2_score),
           'MSE': make_scorer(mean_squared_error),
           'MAE': make_scorer(mean_absolute_error)}

# Instantiate the machine learning classifiers
lin_model = LinearRegression()
svr_model = SVR()
rf_model = RandomForestRegressor()
gb_model = GradientBoostingRegressor()
dt_model = DecisionTreeRegressor()
lgbm_model = LGBMRegressor()
ridge_model = Ridge()
lasso_model = Lasso()
knn_model = KNeighborsRegressor()

# Define the models evaluation function
def Regression(X, y, folds):
    lin = cross_validate(lin_model, X, y, cv=folds, scoring=scoring)
    svr = cross_validate(svr_model, X, y, cv=folds, scoring=scoring)
    rf = cross_validate(rf_model, X, y, cv=folds, scoring=scoring)
    gb = cross_validate(gb_model, X, y, cv=folds, scoring=scoring)
    dt = cross_validate(dt_model, X, y, cv=folds, scoring=scoring)
    lgbm = cross_validate(lgbm_model, X, y, cv=folds, scoring=scoring)
    ridge = cross_validate(ridge_model, X, y, cv=folds, scoring=scoring)
    lasso = cross_validate(lasso_model, X, y, cv=folds, scoring=scoring)
    knn = cross_validate(knn_model, X, y, cv=folds, scoring=scoring)

# Create a data frame with the models perfoamnce metrics scores
    models_scores_table = pd.DataFrame({'SVR Regression': [
                                          svr['test_R2-Square'].mean(),
                                          svr['test_MSE'].mean(),
                                          svr['test_MAE'].mean()],

                                          'LGBM Regression': [
                                              lgbm['test_R2-Square'].mean(),
                                              lgbm['test_MSE'].mean(),
                                              lgbm['test_MAE'].mean()],

                                          'Linear Regression': [
                                              lin['test_R2-Square'].mean(),
                                              lin['test_MSE'].mean(),
                                              lin['test_MAE'].mean()],

                                          'Ridge Regression': [
                                              ridge['test_R2-Square'].mean(),
                                              ridge['test_MSE'].mean(),
                                              ridge['test_MAE'].mean()],

                                          'Lasso Regression': [
                                              lasso['test_R2-Square'].mean(),
                                              lasso['test_MSE'].mean(),
                                              lasso['test_MAE'].mean()],

                                          'KNN Regression': [
                                              knn['test_R2-Square'].mean(),
                                              knn['test_MSE'].mean(),
                                              knn['test_MAE'].mean()],

                                          'XGB Regression': [
                                              gb['test_R2-Square'].mean(),
                                              gb['test_MSE'].mean(),
                                              gb['test_MAE'].mean()],

                                          'DecisionTree Regression': [
                                              dt['test_R2-Square'].mean(),
                                              dt['test_MSE'].mean(),
                                              dt['test_MAE'].mean()],

                                          'Randomforest Regression': [
                                              rf['test_R2-Square'].mean(),
                                              rf['test_MSE'].mean(),
                                              rf['test_MAE'].mean()]},

                                          index=['R2', 'MSE', 'MAE'])

                                      # Add 'Best Score' column
                                      models_scores_table['Best Score'] = models_scores_table.idxmin(axis=1)
    # Return models performance metrics scores data frame
    return (models_scores_table)

#1.2 - CLASSIFICATION
scoring = {'accuracy': make_scorer(accuracy_score),
           'precision': make_scorer(precision_score),
           'recall': make_scorer(recall_score),
           'f1_score': make_scorer(f1_score)}

# Instantiate the machine learning classifiers
log_model = LogisticRegression(max_iter=10000)
svc_model = LinearSVC(dual=False)
dtr_model = DecisionTreeClassifier()
rfc_model = RandomForestClassifier()
gnb_model = GaussianNB()
gb_model = GradientBoostingClassifier()


# Define the models evaluation function
def classification(X, y, folds):
    # Perform cross-validation to each machine learning classifier
    log = cross_validate(log_model, X, y, cv=folds, scoring=scoring)
    svc = cross_validate(svc_model, X, y, cv=folds, scoring=scoring)
    dtr = cross_validate(dtr_model, X, y, cv=folds, scoring=scoring)
    rfc = cross_validate(rfc_model, X, y, cv=folds, scoring=scoring)
    gnb = cross_validate(gnb_model, X, y, cv=folds, scoring=scoring)
    gb = cross_validate(gb_model, X, y, cv=folds, scoring=scoring)

    # Create a data frame with the models perfoamnce metrics scores
    models_scores_table = pd.DataFrame({'Logistic Regression': [log['test_accuracy'].mean(),
                                                                log['test_precision'].mean(),
                                                                log['test_recall'].mean(),
                                                                log['test_f1_score'].mean()],

                                        'Support Vector Classifier': [svc['test_accuracy'].mean(),
                                                                      svc['test_precision'].mean(),
                                                                      svc['test_recall'].mean(),
                                                                      svc['test_f1_score'].mean()],

                                        'Decision Tree-Classifier': [dtr['test_accuracy'].mean(),
                                                                     dtr['test_precision'].mean(),
                                                                     dtr['test_recall'].mean(),
                                                                     dtr['test_f1_score'].mean()],

                                        'Random Forest-Classifier': [rfc['test_accuracy'].mean(),
                                                                     rfc['test_precision'].mean(),
                                                                     rfc['test_recall'].mean(),
                                                                     rfc['test_f1_score'].mean()],

                                        'GradientBoosting-Classifier': [gb['test_accuracy'].mean(),
                                                                        gb['test_precision'].mean(),
                                                                        gb['test_recall'].mean(),
                                                                        gb['test_f1_score'].mean()],

                                        'GaussianNaiveBayes-Classifier': [gnb['test_accuracy'].mean(),
                                                                          gnb['test_precision'].mean(),
                                                                          gnb['test_recall'].mean(),
                                                                          gnb['test_f1_score'].mean()]},

                                       index=['Accuracy', 'Precision', 'Recall', 'F1 Score'])
    # Add 'Best Score' column
    models_scores_table['Best Score'] = models_scores_table.idxmax(axis=1)
    # Return models performance metrics scores data frame
    return (models_scores_table)


                                     'PYCARET'

from pycaret.classification import *
clf = setup(data, target='Loan Status', use_gpu=True)

##1-DATA PREPARATION
#1.1- REGRESSION
from pycaret.regression import *
s = setup(data, target='Price', transform_target=True, log_experiment=True,
                                             experiment_name='diamond')

#1.2-CLASSIFICATION
from pycaret.classification import *
clf = setup(data, target='CLASS', use_gpu=True)
exp_clf101 = setup(data=dfr, target='Timely response?', session_id=123)

from pycaret.classification import *
clf = setup(data, target='Class', imputation_type="iterative")


' 2-MODEL BUILDING'
# Enable models
models(internal=True)[['Name', 'GPU Enabled']]
 # compare baseline models
best = compare_models()
print(best)

' 3 - Hyperparameter Tuning'
# train dt using default hyperparameters
dt = create_model('dt')
#tune hyperparameters with scikit-learn (default)
tuned_dt_sklearn = tune_model(dt)
# tune hyperparameters with scikit-optimize
tuned_dt_skopt = tune_model(dt, search_library='scikit-optimize')
# tune hyperparameters with optuna
tuned_dt_optuna = tune_model(dt, search_library='optuna')
# tune hyperparameters with tune-sklearn
tuned_dt_tuneskl = tune_model(dt, search_library='tune-sklearn')
# check the residuals of trained model
plot_model(best, plot='residuals_interactive')
# check feature importance
plot_model(best, plot='feature')
# finalize the model
finalbest = finalize_model(best)
print(finalbest)
# save model to disk
save_model(finalbest, 'diamond-pipeline')
# within notebook (notice ! sign infront)
!mlflow
ui
# on command line in the same folder
mlflow
ui

Now open your browser and type “https: // localhost: 5000”.
# load model
from pycaret.regression import load_model
pipeline = load_model('C:/Users/moezs/mlruns/1/b8c10d259b294b28a3e233a9d2c209c0/artifacts/model/model')
# print pipeline
print(pipeline)
# generate predictions
from pycaret.regression import predict_model
predictions = predict_model(pipeline, data=data)
predictions.head()
'Get - Metrics'
# check all metrics used for model evaluation
get_metrics()
# add Log Loss metric in pycaret
from sklearn.metrics import log_loss

add_metric('logloss', 'LogLoss', log_loss, greater_is_better=False)

# Visualization
# 1 - Scatter plot
import plotly.express as px

fig = px.scatter(x=data['Carat Weight'], y=data['Price'],
                 facet_col=data['Cut'], opacity=0.25, template='plotly_dark',
                 trendline='ols',
                 trendline_color_override='red',
                 title='SARAH GETS A DIAMOND - A CASE STUDY')
fig.show()

# 2 - Histogram
fig = px.histogram(data, x=["Price"], template='plotly_dark',
                   title='Histogram of Price')
fig.show()


                                  '2-AUTOSKLEARN'
#1-CLASSIFICATION
model = AutoSklearnClassifier(time_left_for_this_task=120, per_run_time_limit=30, n_jobs=-1)
model.fit(X_train, y_train)
# summarize
print(model.sprint_statistics())
# evaluate best model
y_hat = model.predict(X_test)
# to display the entire pipeline that performed the best
model.show_models()

#2-REGRESSION
X, y = sklearn.datasets.load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = \sklearn.model_selection.train_test_split(X, y, random_state=1)

automl = autosklearn.regression.AutoSklearnRegressor(
    time_left_for_this_task=120,
    per_run_time_limit=30,
    tmp_folder='/tmp/autosklearn_regression_example_tmp',
)
automl.fit(X_train, y_train, dataset_name='diabetes')
print(automl.leaderboard())

train_predictions = automl.predict(X_train)
print("Train R2 score:", sklearn.metrics.r2_score(y_train, train_predictions))
test_predictions = automl.predict(X_test)
print("Test R2 score:", sklearn.metrics.r2_score(y_test, test_predictions))

                                  '3-AUTO-VIML'
BASE_DIR = '/tmp'
OUTPUT_FILE = os.path.join(BASE_DIR, 'churn_data.csv')
churn_data=urllib.request.urlretrieve('https://raw.githubusercontent.com/srivatsan88/YouTubeLI/master/dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv', OUTPUT_FILE)
churn_df = pd.read_csv(OUTPUT_FILE)
churn_df.head()
size = int(0.7*churn_df.shape[0])
train_df = churn_df[:size]
test_df = churn_df[size:]


target='Churn'
model, features, trainm, testm = Auto_ViML(train_df, target, test_df, sample_submission='',
                                    scoring_parameter='',
                                    hyper_param='RS',feature_reduction=True,
                                     Boosting_Flag=True,Binning_Flag=False,
                                    Add_Poly=0, Stacking_Flag=False,
                                    Imbalanced_Flag=True,
                                    verbose=1)

"""hyper_param: Tuning options are GridSearch ('GS') and RandomizedSearch ('RS'). Default is 'GS'.
feature_reduction: Default = 'True' but it can be set to False if you don't want automatic    
Boosting Flag: you have 4 possible choices (default is False):                               
  None = This will build a Linear Model                                                  
  False = This will build a Random Forest or Extra Trees model (also known as Bagging)        
  True = This will build an XGBoost model                                                     
  CatBoost = THis will build a CatBoost model (provided you have CatBoost installed)          
"""

print(confusion_matrix(test_df[target].values,testm['Churn_XGBoost_predictions'].values))
print(confusion_matrix(test_df[target].values,testm['Churn_Logistic Regression_predictions'].values))
print(confusion_matrix(test_df[target].values,testm['Churn_Ensembled_predictions'].values))
print(classification_report(test_df[target].values,testm['Churn_XGBoost_predictions'].values))
print(classification_report(test_df[target].values,testm['Churn_Ensembled_predictions'].values))


                      '4-TPOT'
# define and perform the search for best model
tpot_model = TPOTClassifier(generations=5, population_size=20, cv=5, random_state=42, verbosity=2)
tpot_model.fit(X_train, y_train)
#evaluate the best model
print(tpot_model.score(X_test, y_test))
# export the corresponding Python code for the optimized pipeline to a text file
tpot_model.export('tpot_exported_pipeline.py')


                             '5-Lazy predict'
#1-LazyClassifier
Classifier = LazyClassifier(ignore_warnings=False, custom_metric=None)
models, predictions = Classifier.fit(X_train, X_test, y_train, y_test)

#2-Lazyregressor
reg = LazyRegressor(ignore_warnings=False, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)


                                    '6-h20'
h2o.init()
churn_df = h2o.import_file('https://raw.githubusercontent.com/srivatsan88/YouTubeLI/master/dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv')
churn_train,churn_test,churn_valid = churn_df.split_frame(ratios=[.7, .15])

y = "Churn"
x = churn_df.columns
x.remove(y)
x.remove("customerID")

aml = H2OAutoML(max_models = 10, seed = 10, exclude_algos = ["StackedEnsemble", "DeepLearning"], verbosity="info", nfolds=0)
aml.train(x = x, y = y, training_frame = churn_train, validation_frame=churn_valid)

lb = aml.leaderboard
lb.head()
churn_pred=aml.leader.predict(churn_test)
churn_pred.head()
aml.leader.model_performance(churn_test)
model_ids = list(aml.leaderboard['model_id'].as_data_frame().iloc[:,0])
#se = h2o.get_model([mid for mid in model_ids if "StackedEnsemble_AllModels" in mid][0])
#metalearner = h2o.get_model(se.metalearner()['name'])
model_ids
h2o.get_model([mid for mid in model_ids if "XGBoost" in mid][0])
out = h2o.get_model([mid for mid in model_ids if "XGBoost" in mid][0])
print(out.params)
out.convert_H2OXGBoostParams_2_XGBoostParams()
out
out_gbm = h2o.get_model([mid for mid in model_ids if "GBM" in mid][0])
out.confusion_matrix()
out.varimp_plot()
aml.leader.download_mojo(path = "./")


                                        '7-autogluon'
BASE_DIR = '/tmp'
OUTPUT_FILE = os.path.join(BASE_DIR, 'churn_data.csv')
churn_data=urllib.request.urlretrieve('https://raw.githubusercontent.com/srivatsan88/YouTubeLI/master/dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv', OUTPUT_FILE)
size = int(0.8*churn_master_df.shape[0])
train_df = churn_master_df[:size]
test_df = churn_master_df[size:]

train_data = task.Dataset(df=train_df)
test_data = task.Dataset(df=test_df)

print(train_data.head())
print(test_data.describe())

label_column = 'Churn'
train_data[label_column].describe()

predictor = task.fit(train_data=train_data, label=label_column, eval_metric='accuracy')
y_test = test_data[label_column]

test_data_nolab = test_data.drop(labels=[label_column],axis=1)
print(test_data_nolab.head())

y_pred = predictor.predict(test_data_nolab)
print("Predictions:  ", y_pred)
perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)

print(predictor.problem_type)
print(predictor.feature_types)

predictor.predict_proba(test_data_nolab)
predictor.leaderboard()


hp_tune = True
rf_options = { 'n_estimators': 100}
gbm_options = {'num_boost_round': 100,'num_leaves': ag.space.Int(lower=6, upper=20, default=8)}
hyperparameters = {'RF':rf_options, 'GBM': gbm_options}

time_limits = 2*60
num_trials = 1
search_strategy = 'skopt'

predictor = task.fit(train_data=train_data, tuning_data=test_data, label=label_column,
                     time_limits=time_limits, num_trials=num_trials,
                     hyperparameter_tune=hp_tune, hyperparameters=hyperparameters,
                     search_strategy=search_strategy, nthreads_per_trial=1, ngpus_per_trial=1)
predictor.fit_summary()
predictor.leaderboard()

                                'TIME SERIES'
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
