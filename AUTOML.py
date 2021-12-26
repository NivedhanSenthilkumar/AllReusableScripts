

                                  '2-AUTOSKLEARN'
#import the package
from autosklearn.classification import AutoSklearnClassifier
# define and perform the search for best model
model = AutoSklearnClassifier(time_left_for_this_task=120, per_run_time_limit=30, n_jobs=-1)
model.fit(X_train, y_train)
# summarize
print(model.sprint_statistics())
# evaluate best model
y_hat = model.predict(X_test)
# to display the entire pipeline that performed the best
model.show_models()

                                  '3-AUTO-VIML'
from __future__ import print_function
import sys,tempfile, urllib, os
import pandas as pd

BASE_DIR = '/tmp'
OUTPUT_FILE = os.path.join(BASE_DIR, 'churn_data.csv')
churn_data=urllib.request.urlretrieve('https://raw.githubusercontent.com/srivatsan88/YouTubeLI/master/dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv', OUTPUT_FILE)
churn_df = pd.read_csv(OUTPUT_FILE)
churn_df.head()
size = int(0.7*churn_df.shape[0])
train_df = churn_df[:size]
test_df = churn_df[size:]

from autoviml.Auto_ViML import Auto_ViML
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

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(test_df[target].values,testm['Churn_XGBoost_predictions'].values))
print(confusion_matrix(test_df[target].values,testm['Churn_Logistic Regression_predictions'].values))
print(confusion_matrix(test_df[target].values,testm['Churn_Ensembled_predictions'].values))
print(classification_report(test_df[target].values,testm['Churn_XGBoost_predictions'].values))
print(classification_report(test_df[target].values,testm['Churn_Ensembled_predictions'].values))


                      '4-TPOT'
pip install tpot
from tpot import TPOTClassifier
# define and perform the search for best model
tpot_model = TPOTClassifier(generations=5, population_size=20, cv=5, random_state=42, verbosity=2)
tpot_model.fit(X_train, y_train)
#evaluate the best model
print(tpot_model.score(X_test, y_test))
# export the corresponding Python code for the optimized pipeline to a text file
tpot_model.export('tpot_exported_pipeline.py')

                    '5-lazy predict'
 'INSTALLATION'
pip install lazypredict
from lazypredict.Supervised import LazyRegressor, LazyClassifier


                            'MODEL - Building'
1 - Building all classifier models
# LazyClassifier Instance and fiting data
Classifier = LazyClassifier(ignore_warnings=False, custom_metric=None)
models, predictions = Classifier.fit(X_train, X_test, y_train, y_test)



2 - Building all regressor models
#Lazyregressor Instance and fiting data
reg = LazyRegressor(ignore_warnings=False, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)


                      '6-h20'
import h2o
h2o.init()
from h2o.automl import H2OAutoML
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
out.params
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
predictor.fit_summary()
predictor.leaderboard()

