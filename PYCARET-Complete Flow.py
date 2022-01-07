
from pycaret.classification import *
clf = setup(data, target='Loan Status', use_gpu=True)

'1-PYCARET'
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

                                   Now
                                   open
                                   your
                                   browser and type “https: // localhost: 5000”.

                                   # load model
                                   from pycaret.regression import load_model

                                   pipeline = load_model(
                                       'C:/Users/moezs/mlruns/1/b8c10d259b294b28a3e233a9d2c209c0/artifacts/model/model')
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

