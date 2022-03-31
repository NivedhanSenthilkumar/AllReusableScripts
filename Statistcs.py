

coef = ols.stats()


                              """1-ANOVA TEST"""
# Defining a function to find the statistical relationship with all the categorical variables
def FunctionAnova(inpData, TargetVariable, ContinuousPredictorList):
    from scipy.stats import f_oneway

    # Creating an empty list of final selected predictors
    SelectedPredictors = []

    print('##### ANOVA Results ##### \n')
    for predictor in ContinuousPredictorList:
        CategoryGroupLists = inpData.groupby(TargetVariable)[predictor].apply(list)
        AnovaResults = f_oneway(*CategoryGroupLists)

        # If the ANOVA P-Value is <0.05, that means we reject H0
        if (AnovaResults[1] < 0.05):
            print(predictor, 'is correlated with', TargetVariable, '| P-Value:', AnovaResults[1])
            SelectedPredictors.append(predictor)
        else:
            print(predictor, 'is NOT correlated with', TargetVariable, '| P-Value:', AnovaResults[1])

    return (SelectedPredictors)

# Calling the function to check which categorical variables are correlated with target
ContinuousVariables=['Loan Amount','Funded Amount', 'Funded Amount Investor','Interest Rate','Home Ownership','Debit to Income','Open Account','Revolving Balance','Revolving Utilities','Total Accounts','Total Received Interest','Total Received Late Fee','Recoveries', 'Collection Recovery Fee',
                   'Last week Pay','Total Collection Amount','Total Current Balance','Total Revolving Credit Limit']
FunctionAnova(inpData=LoanData, TargetVariable='Loan Status', ContinuousPredictorList=ContinuousVariables)


                                       """2-chisquare test"""
# Writing a function to find the correlation of all categorical variables with the Target variable
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

CategoricalVariables=['Term', 'Grade', 'Employment Duration', 'Verification Status',
       'Delinquency - two years','Inquires - six months', 'Public Record', 'Initial List Status','Collection 12 months Medical','Application Type','Sub Grade']
# Calling the function
FunctionChisq(inpData=LoanData,
              TargetVariable='Loan Status',
              CategoricalVariablesList= CategoricalVariables)