






## DATA SEGREGATION
numdata = df.select_dtypes(include = np.number)
catdata = df.select_dtypes(exclude = np.number)



                               "EXPLORATORY DATA ANALYSIS"
# Numerical Data Describe
def numericalattributes(X):
    Output = pd.DataFrame()
    Output['Variables'] = X.columns
    Output['Skewness'] = X.skew().values
    Output ['Kurtosis'] = X.kurt().values
    Output ['Standarddeviation'] = X.std().values
    Output ['Variance'] = X.var().values
    Output ['Mean'] = X.mean().values
    Output ['Median'] = X.median().values
    Output ['Minimum'] = X.min().values
    Output ['Maximum'] = X.max().values
    Output ['Sum'] = X.sum().values
    Output ['Count'] = X.count().values
    return Output


#VALUE COUNT
def valuecounts(catdata):
    for i in catdata.columns:
        print(catdata[i].value_counts())


#Null before imputing
Total = concat.isnull().sum().sort_values(ascending=False)
Percent = (concat.isnull().sum()*100/len(concat)).sort_values(ascending=False)
Missingdata = pd.concat([Total, Percent], axis = 1, keys = ['Total', 'Percentage of Missing Values'])
Missingdata

#Dropping Columns based on Null
df = df.drop(['Consumer disputed?'],axis=1)


                                "STATISTICAL TESTS"
#1-Numerical Data
numdata.corr()

#2-Categorical data with categorical data
#2.1-CHISQUARE TEST
def FunctionChisq(inpData, TargetVariable, CategoricalVariablesList):
    from scipy.stats import chi2_contingency

    # Creating an empty list of final selected predictors
    SelectedPredictors = []

    for predictor in CategoricalVariablesList:
        CrossTabResult = pd.crosstab(index=inpData[TargetVariable], columns=inpData[predictor])
        ChiSqResult = chi2_contingency(CrossTabResult)

        # If the ChiSq P-Value is <0.05, that means we reject H0
        if (ChiSqResult[1] < 0.05):
            print(predictor, 'is correlated with', TargetVariable, '| P-Value:', ChiSqResult[1])
            SelectedPredictors.append(predictor)
        else:
            print(predictor, 'is NOT correlated with', TargetVariable, '| P-Value:', ChiSqResult[1])
    return (SelectedPredictors)

CategoricalVariables= ['Product', 'Sub-product', 'Issue', 'Sub-issue', 'Company',
       'Consumer consent provided?', 'Submitted via', 'Timely response?',
      'Duration',
       'Regions']
# Calling the function
FunctionChisq(inpData=df,
              TargetVariable='Company response to consumer',
              CategoricalVariablesList= CategoricalVariables)

#2.2-ANOVA TEST











                                "ENSEMBLE COMBINATIONS"
#1-Randomforest with Adaboost and Gradient Boosting
from sklearn.ensemble import VotingRegressor
r1 = AdaBoostRegressor()
r2 = RandomForestRegressor()
r3 =  GradientBoostingRegressor()
model1 = VotingRegressor([('ada', r1), ('rf', r2),('gbr',r3)])
model1.fit(xtrain,ytrain)
prediction1 = model1.predict(xtest)

#2-Extra trees with lightgbm
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor
r1 = ExtraTreesRegressor()
r2 = LGBMRegressor()
model2 = VotingRegressor([('lightgbm', r2), ('ExtraTrees', r1)])
model2.fit(xtrain,ytrain)
prediction2 = model2.predict(xtest)

#3-ALL TREES MODEL
r1 = ExtraTreesRegressor()
r2 = RandomForestRegressor()
model4 = VotingRegressor([('et', r1), ('rf', r2)])
model4.fit(xtrain,ytrain)
prediction4 = model4.predict(xtest)

#4- All boosting models
from xgboost import XGBRegressor
r1 = AdaBoostRegressor()
r2 = XGBRegressor()
r3 =  GradientBoostingRegressor()
model3 = VotingRegressor([('ada', r1), ('XGBRegressor', r2),('xgboost', r3)])
model3.fit(xtrain,ytrain)
prediction3 = model3.predict(xtest)















