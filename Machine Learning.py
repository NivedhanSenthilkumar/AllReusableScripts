

                           "1-FEATURE ENGINEERING"
#1-FEATURE TOOLS
import featuretools as ft
es = ft.demo.load_retail()
print(es)
feature_matrix_sessions, features_defs = ft.dfs(dataframes=dataframes,
                                                relationships=relationships,
                                                target_dataframe_name="sessions")
feature_matrix_sessions.head(5)


#2-AUTOFEAT
import pandas as pd
from AutoFeat import AutoFeatRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
data = load_boston()
X = data.data
y= data.target
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.3,random_state =0)
model = AutoFeatRegressor()
df = model.fit_transform(X, y)
pred = model.predict(X_test)
print("Final R^2: %.4f" % model.score(df, y))


                              '2-FEATURE SELECTION'
#FORWARD SELECTION
#VERSION1 - Hardcode the maximum number of features to select
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
sfs = SequentialFeatureSelector(estimator=KNeighborsClassifier(n_neighbors=3),
                          n_features_to_select=3)
features = sfs.fit(X, y)
sfs.get_support()
sfs.get_params([deep])


##Version2 - Select all the best features
linreg = LinearRegression()
linreg_forward = sfs(estimator=linreg, k_features = 'best', forward=True,
                     verbose=2, scoring='r2')
sfs_forward = linreg_forward.fit(X_train, y_train)

# print the selected feature names when k_features = (5, 15)
print('Features selelected using forward selection are: ')
print(sfs_forward.k_feature_names_)
# print the R-squared value
print('\nR-Squared: ', sfs_forward.k_score_)


#Version3 -  Select the best features within a particular minimum and maximum
linreg = LinearRegression()
linreg_forward = sfs(estimator = linreg, k_features = (5,15), forward = True,
                     verbose = 2, scoring = 'r2')
sfs_forward = linreg_forward.fit(X_train, y_train)
# print the selected feature names when k_features = (5, 15)
print('Features selelected using forward selection are: ')
print(linreg_forward.k_feature_names_)
# print the R-squared value
print('\nR-Squared: ', sfs_forward.k_score_)


#BACKWARD ELIMINATION
linreg = LinearRegression()
linreg_backward = sfs(estimator = linreg, k_features = 'best', forward = False,
                     verbose = 2, scoring = 'r2')
sfs_backward = linreg_backward.fit(X_train, y_train)
print('Features selelected using backward elimination are: ')
print(sfs_backward.k_feature_names_)
print('\nR-Squared: ', sfs_backward.k_score_)


#RECURSIVE FEATURE ELIMINATION
linreg_rfe = LinearRegression()
rfe_model = RFE(estimator=linreg_rfe, n_features_to_select = 12)
rfe_model = rfe_model.fit(X_train, y_train)
feat_index = pd.Series(data = rfe_model.ranking_, index = X_train.columns)

# select the features with rank = 1
# 'index' returns the indices of a series (i.e. features with rank=1)
signi_feat_rfe = feat_index[feat_index==1].index
# print the significant features obtained from RFE
print(signi_feat_rfe)


                       "3-Feature importance"""
#1-Predictive Power Score
pps.predictors(train_df, "Overall_Experience")[['x', 'y', 'ppscore']]

#2-RANKING FEATURES
import featurewiz as FW
outputs = FW.featurewiz(dataname, target, corr_limit=0.70, verbose=2, sep=',',
		header=0, test_data='',feature_engg='', category_encoders='',
		dask_xgboost_flag=False, nrows=None)

#3-FEATURE IMPORTANC PLOT (REGRESSION,CLASSIFICATION)
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=1)
# define the model
model = RandomForestRegressor()
# fit the model
model.fit(X, y)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
important_features = pd.DataFrame({'Features': X_train_xfs.columns,'Importance': xgb_model.feature_importances_})
fe_imp=important_features.sort_values(by='Importance',ascending=False)


X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
# define the model
model = RandomForestClassifier()
# fit the model
model.fit(X, y)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()



#4-Visualize Importance
def visualize_importance(model, df_X, df_Y):
    result = permutation_importance(model,
                                    df_X, df_Y,
                                    n_repeats=10, n_jobs=2,
                                    random_state = random_seed)

    sorted_idx = result.importances_mean.argsort()
    fig, ax = plt.subplots(figsize =(20,10))
    ax.barh(df_X.columns[sorted_idx],
            result.importances[sorted_idx].mean(axis =1).T),
    ax.set_title("Permutation Importances")
    fig.tight_layout()
    plt.show()
    return sorted_idx


#5-Partial Independence Plots
def partial_dependence(model, df_X, sorted_idx, threshold=10):
    fig, ax = plt.subplots(figsize=(30, 5))
    ax.set_title("%d Most important features" % threshold)
    plot_partial_dependence(model, df_X,
                            df_X.columns[sorted_idx][::-1][:threshold],
                            n_cols=threshold,
                            n_jobs=-1,
                            grid_resolution=100, ax=ax)



                    "4-HYPERPARAMETER TUNING"
#1-Optuna - Randomforest
RANDOM_SEED = 42
kfolds = KFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
def randomforest_objective(trial):
    _n_estimators = trial.suggest_int("n_estimators", 50, 200)
    _max_depth = trial.suggest_int("max_depth", 5, 20)
    _min_samp_split = trial.suggest_int("min_samples_split", 2, 10)
    _min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 10)
    _max_features = trial.suggest_int("max_features", 10, 50)
    rf = RandomForestRegressor(
        max_depth=_max_depth,
        min_samples_split=_min_samp_split,
        min_samples_leaf=_min_samples_leaf,
        max_features=_max_features,
        n_estimators=_n_estimators,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    scores = cross_val_score(rf, X, y, cv=kfolds, scoring="r2")
    return scores.mean()

def tune(objective):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    params = study.best_params
    best_score = study.best_value
    print(f"Best score: {best_score} \nOptimized parameters: {params}")
    return params

##Apply Functions
randomforest_params = tune(randomforest_objective)

#2-GRIDSEARCH CV





                    """ENSEMBLE MODELLING"""
#1-VOTING REGRESSOR
r1 = xg.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.05, max_delta_step=0, max_depth=6,
             min_child_weight=2,  monotone_constraints='()',
             n_estimators=150, n_jobs=-1, num_parallel_tree=1,
             objective='reg:squarederror', random_state=7685, reg_alpha=3,
             reg_lambda=5, scale_pos_weight=42.80000000000001, subsample=0.2,
             tree_method='auto', validate_parameters=1, verbosity=0)

r2 = GradientBoostingRegressor(alpha=0.9, ccp_alpha=0.0, criterion='friedman_mse',
                          init=None, learning_rate=0.1, loss='ls', max_depth=4,
                          max_features=1.0, max_leaf_nodes=None,
                          min_impurity_decrease=0.4, min_impurity_split=None,
                          min_samples_leaf=2, min_samples_split=2,
                          min_weight_fraction_leaf=0.0, n_estimators=110,
                          n_iter_no_change=None, presort='deprecated',
                          random_state=7685, subsample=0.25, tol=0.0001,
                          validation_fraction=0.1, verbose=0, warm_start=False)

r3 = AdaBoostRegressor(base_estimator=None, learning_rate=0.0005, loss='linear',
                  n_estimators=140, random_state=7685)

model1 = VotingRegressor([('ada', r3), ('xgboost', r1),('gbr',r2)])
model1.fit(X,Y)
ypred=model1.predict(testconcat)



                         'PERFORMANCE METRICS'
                          '1 - CLASSIFICATION'
#1.1 - LOG metrics"""
def full_log_likelihood(w, X, y):
    score = np.dot(X, w).reshape(1, X.shape[0])
    return np.sum(-np.log(1 + np.exp(score))) + np.sum(y * score)

def null_log_likelihood(w, X, y):
    z = np.array([w if i == 0 else 0.0 for i, w in enumerate(w.reshape(1, X.shape[1])[0])]).reshape(X.shape[1], 1)
    score = np.dot(X, z).reshape(1, X.shape[0])
    return np.sum(-np.log(1 + np.exp(score))) + np.sum(y * score)

def mcfadden_rsquare(w, X, y):
    return 1.0 - (full_log_likelihood(w, X, y) / null_log_likelihood(w, X, y))

def mcfadden_adjusted_rsquare(w, X, y):
    k = float(X.shape[1])
    return 1.0 - ((full_log_likelihood(w, X, y) - k) / null_log_likelihood(w, X, y))

#Scores for classification Algorihthms
# create an empty dataframe to store the scores for various algorithms
score_card = pd.DataFrame(columns=['Probability Cutoff', 'AUC Score', 'Precision Score', 'Recall Score',
                                   'Accuracy Score', 'Kappa Score', 'f1-score'])
# append the result table for all performance scores
# performance measures considered for model comparision are 'AUC Score', 'Precision Score', 'Recall Score','Accuracy Score',
# 'Kappa Score', and 'f1-score'
# compile the required information in a user defined function
def update_score_card(model, cutoff):
    # let 'y_pred_prob' be the predicted values of y
    y_pred_prob = logreg.predict(X_test)
    # convert probabilities to 0 and 1 using 'if_else'
    y_pred = [0 if x < cutoff else 1 for x in y_pred_prob]
    # assign 'score_card' as global variable
    global score_card
    # append the results to the dataframe 'score_card'
    # 'ignore_index = True' do not consider the index labels
    score_card = score_card.append({'Probability Cutoff': cutoff,
                                    'AUC Score': metrics.roc_auc_score(y_test, y_pred),
                                    'Precision Score': metrics.precision_score(y_test, y_pred),
                                    'Recall Score': metrics.recall_score(y_test, y_pred),
                                    'Accuracy Score': metrics.accuracy_score(y_test, y_pred),
                                    'Kappa Score': metrics.cohen_kappa_score(y_test, y_pred),
                                    'f1-score': metrics.f1_score(y_test, y_pred)},
                                   ignore_index=True)
score_card = pd.DataFrame(index=['Backward Elimination', 'RFECV'])
score_card['No of features'] = [len(back_feat), len(rfe_feat)]
score_card['Best score'] = [back_mod.k_score_, np.mean(score)]

#1.3 - Cost based method
# define a function to calculate the total_cost for a cut-off value
# pass the actual values of y, predicted probabilities of y, cost for FN and FP
def calculate_total_cost(actual_value, predicted_value, cost_FN, cost_FP):
    # pass the actual and predicted values to calculate the confusion matrix
    cm = confusion_matrix(actual_value, predicted_value)
    # create an array of the confusion matrix
    cm_array = np.array(cm)
    # return the total_cost
    return cm_array[1, 0] * cost_FN + cm_array[0, 1] * cost_FP


# create an empty dataframe to store the cost for different probability cut-offs
df_total_cost = pd.DataFrame(columns=['cut-off', 'total_cost'])
# initialize i to '0' corresponding to the 1st row in the dataframe
i = 0
# use for loop to calculate 'total_cost' for each cut-off probability value
# call the function 'calculate_total_cost' to calculate the cost
# pass the actual y-values
# calculate the predicted y-values from 'y_pred_prob' for the cut-off probability value
# assign the costs 3.5 and 2 to False Negatives and False Positives respectively
# add the obtained 'cut_off' and 'total_cost' at the ith index of the dataframe
for cut_off in range(10, 100):
    total_cost = calculate_total_cost(y_test, y_pred_prob.map(lambda x: 1 if x > (cut_off / 100) else 0), 3.5, 2)
    df_total_cost.loc[i] = [(cut_off / 100), total_cost]
    # increment the value of 'i' for each row index in the dataframe 'df_total_cost'
    i += 1

#Classfication Report
def get_test_report(model):
    test_pred = model.predict(X_test)
    return (classification_report(y_test, test_pred))


#Calculate TP,TN,FP,FN
def positivenegatives(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 1:
            TP += 1
        if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
            FP += 1
        if y_actual[i] == y_hat[i] == 0:
            TN += 1
        if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
            FN += 1
    return (TP, FP, TN, FN)


#CONFUSION MATRIX
def Confusionmatrix(actualvalue, predictedvalue):
    cm = confusion_matrix(actualvalue, predictedvalue)
    hm = sns.heatmap(cm)
    return cm, hm

#Custom Function
def Classificationerrormetric(model):
                               ypred = model.predict(xtest)
                               scorecard = pd.DataFrame({
            'Accuracy': sklearn.metrics.accuracy_score(ytest, ypred),
            'Area Under Curve': sklearn.metrics.roc_curve(ytest, ypred),
            'F1': sklearn.metrics.f1_score(ytest, ypred),
            'Recall': sklearn.metrics.recall_score(ytest, ypred),
            'Precision': sklearn.metrics.precision_score(ytest, ypred),
            'Log Loss': sklearn.metrics.log_loss(ytest, ypred)},
            index=['ERROR', 'ACCURACY', 'ROC-AUC', 'F1', 'Recall', 'Precision', 'LogLoss'])
        return scorecard.head(1)


                            '2-REGRESSION'
def Regressionerrormetric(model):
                               ypred = model.predict(xtest)
                               scorecard = pd.DataFrame({
            'Mean Absolute Error': metrics.mean_absolute_error(ytest, ypred),
            'Mean Squared Error': metrics.mean_squared_error(ytest, ypred),
            'Root Mean Squared Error': np.sqrt(((ypred - ytest) ** 2).mean()),
            'Mean Absolute Percentage error': np.mean(np.abs((ytest - ypred) / ytest)) * 100,
            'Mean Squared Log Error': metrics.mean_squared_log_error(ytest, ypred),
            'Root Mean Square Log error': np.sqrt(metrics.mean_squared_log_error(ytest, ypred)),
            'Overall Error': np.abs((ytest - ypred)).sum(),
            'Overall Error Percentage': (np.abs((ytest - ypred) / (ytest)).sum()) * 100},
            index=['ERROR', 'MAE', 'MSE', 'RMSE', 'MAPE', 'MSLE', 'RMSLE', 'OE', 'OEP'])
        return scorecard.head(1)


                         'General Formulas'
mape = mean_absolute_error(Y_actual, Y_Predicted) * 100
rmse = np.sqrt(((ypred - ytest) ** 2).mean())
OverallError = np.abs((ytest - ypred)),
OverallErrorPercentage = np.abs((ytest - ypred) / (ytest)) * 100
mape = (np.mean(np.abs((actual - predicted) / actual)) * 100)



"misc"
# logistic regression for feature importance
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
# define the model
model = LogisticRegression()
# fit the model
model.fit(X, y)
# get importance
importance = model.coef_[0]
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()


# decision tree for feature importance on a regression problem
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=1)
# define the model
model = DecisionTreeRegressor()
# fit the model
model.fit(X, y)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()

# xgboost for feature importance on a regression problem
# define dataset
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=1)
# define the model
model = XGBRegressor()
# fit the model
model.fit(X, y)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()


# xgboost for feature importance on a classification problem
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
# define the model
model = XGBClassifier()
# fit the model
model.fit(X, y)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()

# decision tree for feature importance on a classification problem
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
# define the model
model = DecisionTreeClassifier()
# fit the model
model.fit(X, y)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()

