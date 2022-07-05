


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

# Separating Numerical and Categorical features to conduct respective test
df_hypo_num = df_Hypothesis.select_dtypes(include = np.number).drop('Loan/No Loan',axis=1)
df_hypo_cat = df_Hypothesis.select_dtypes(include ='O')

# categorical vs target
# chi-square test for independence of attributes:
#   H0: variables are independent
#   H1: variables are dependent

c_pvals = []
for i in df_hypo_cat.columns:
    obs = pd.crosstab(df_hypo_cat[i], df_train_cleaned['Loan/No Loan'])
    stat, pval, dof, exp = st.chi2_contingency(obs)
    c_pvals.append(pval)

pd.DataFrame({'Features': df_hypo_cat.columns, 'p_value': c_pvals})
# All p_values < 0.05(alpha), Rejecting Null Hypothesis, so all the categorical features have some relationship with Target Variable




                                    "T-TEST"
                                  'Paired T-TEST'
#STEP1 : Take shapiro test to check normality
stats.shapiro(bef_scr)
stats.shapiro(aft_scr)

# Test of Normality - Shapiro  test
# Ho: skew=0 (normal)
# Ha : skew!=0 (not normal)

# if pval> sig lvl . Ho is selected else Ha is selected
#For proceeding to ttest the data(before and after) both should always be normal

# Hypothesis:
# Ho : mu of before  >=  mu of after
# Ha : mu of before  <  mu of after
ttstat,twosid_pval = stats.ttest_rel(bef_scr,aft_scr)
print('T stat:',ttstat)
print('Two sided Pval:',twosid_pval)
print('One sided pval:',twosid_pval/2)
# if pval> sig lvl . Ho is selected else Ha is selected


                              # numerical vs target
                              # Two tailed Two-Sample T-Test:
                              #   H0: mean1 = mean2
                              #   H1: mean1 !=mean2

n_pvals = []
for i in df_hypo_num.columns:
    a = df_train_cleaned[df_train_cleaned['Loan/No Loan'] == 0][i]
    b = df_train_cleaned[df_train_cleaned['Loan/No Loan'] == 1][i]
    tstat, pval = st.ttest_ind(a, b)
    n_pvals.append(pval)

pd.DataFrame({'Features': df_hypo_num.columns, 'p_value': n_pvals})
# All p_values < 0.05(alpha), Rejecting Null Hypothesis, so all the Numerical features have some relationship with Target Variable




                                """3-CORRELATION"""
#1-PEARSON
df.corr()

#2-SPEARMAN
df.corr(method='spearman')