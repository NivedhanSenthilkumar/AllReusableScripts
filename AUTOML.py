
"IMPORT LIBRARIES"
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
                                  # Define dictionary with performance metrics
                                  scoring = {
                                      'R2-Square': make_scorer(r2_score),
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
                                      '''
                                      X : data set features
                                      y : data set target
                                      folds : number of cross-validation folds

                                      '''

                                      # Perform cross-validation to each machine learning classifier
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


                                  '2-AUTOSKLEARN'
#1-CLASSIFICATION
# define and perform the search for best model
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


