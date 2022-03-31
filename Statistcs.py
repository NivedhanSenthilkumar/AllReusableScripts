

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