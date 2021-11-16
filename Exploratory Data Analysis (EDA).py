                           "1 - INITIALIZATION"
                        """ 1.1 - PIP INSTALL """
!pip install shapash
!pip install pycaret
!pip install pandas_profiling
!pip install python-firebase
!pip install scrapy
!pip install sweetviz
!pip install dtale
!pip install datapane
!pip install altair
!pip install pycaret[full]
!pip install lazypredict
!pip install autosklearn
!pip install tpot
!pip install autoviml
!pip install autoviz
!pip install Dora


                      """ 1.2 - PIP UPGRADE"""
import os
import tkinter as tk

root= tk.Tk()

canvas1 = tk.Canvas(root, width = 300, height = 350, bg = 'lightsteelblue2', relief = 'raised')
canvas1.pack()

label1 = tk.Label(root, text='Upgrade PIP', bg = 'lightsteelblue2')
label1.config(font=('helvetica', 20))
canvas1.create_window(150, 80, window=label1)

def upgradePIP ():
    os.system('start cmd /k python.exe -m pip install --upgrade pip')
button1 = tk.Button(text='      Upgrade PIP     ', command=upgradePIP, bg='green', fg='white', font=('helvetica', 12, 'bold'))
canvas1.create_window(150, 180, window=button1)
root.mainloop()


                    """ 1.3 - Import LIBRARIES"""
##1-GENERAL
import pandas as pd
from wordcloud import WordCloud
import numpy as np
import random

##2-Visualization Library
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import datapane as dp
from matplotlib.colors import ListedColormap

##3-EDA Reports
import pandas_profiling as pp
import sweetviz as sv
import dtale as de
import autoviz as au
from Dora import Dora

##4-scaling
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import RobustScaler

## ENCODING
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder

## TRANSFORMATION
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import PowerTransformer

##5-Imputation
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


##5-STATISTICS
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels
import statsmodels.api as sm
import scipy.stats as stats
import statistics
from scipy import stats
from statsmodels.stats import weightstats as stests
from scipy.stats import shapiro
from statsmodels.stats import power
import statsmodels.formula.api as smf
from scipy.stats.mstats import winsorize



##6-train-test split
from sklearn.model_selection import train_test_split



## MACHINE LEARNING
#Feature selection
from sklearn.feature_selection import RFE
from mlxtend.feature_selection import SequentialFeatureSelector as sfs

##AutoML
from lazypredict.Supervised import LazyRegressor, LazyClassifier
from autosklearn.classification import AutoSklearnClassifier
from tpot import TPOTClassifier
from autoviml.Auto_ViML import Auto_ViML
import pycaret
import merlion
import auto_ts


#Sampling Techniques - Classification
from imblearn.under_sampling import RandomUnderSampler

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm  import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import OrthogonalMatchingPursuit
from IPython.display import Image  
from sklearn.ensemble import RandomForestClassifier


# performance metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score 
from sklearn.metrics import fbeta_score

#Hyperparamter Tuning
from sklearn.model_selection import GridSearchCV
from sklearn import tree

#STACKING  MODELS
from sklearn.ensemble import VotingRegressor

## WARNNINGS
from warnings import filterwarnings
filterwarnings('ignore')
pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.float_format = '{:.6f}'.format
pd.options.display.max_columns = None


                             """'2 - DATA PREPROCESSING'"""
'Source data import'
data = pd.read_excel('D:/ORIGINALS -COURSE ONES/SLC/Miniproject/SLC Mini Projects/SLC Mini Projects 2.0/GermanCredit.xlsx')
print(data)

'Sanity Check'
print(data.dtypes)
print(data.shape)
print(data.size)

'Dropping unwanted variables'
data = data.drop(['Columnname'],axis=1)


'Data Segregation'
#Method 1
numdata = data.select_dtypes(include = np.number)
catdata = data.select_dtypes(exclude = np.number)

#Method 2
catdata = ipl.loc[:,['Team','Tournament','Player']]
numdata = ipl[ipl.columns.difference(['Team','Tournament','Player'])]

'Datatype Conversion'
for i in numdata:
    numdata[i] = numdata[i].astype('float')

'Feature understanding'
# 5 POINT summary
ipl.describe()
numdata.describe(include = 'all')
catdata.describe(include = 'all')

#Info
ipl.info()
catdata.info()
numdata.info()

## Numercialessentials
def numessentials(x):
    return pd.DataFrame([x.skew(), x.sum(), x.mean(), x.median(),  x.std(), x.var(),x.min(),x.max()],
                  index=['skew', 'SUM', 'MEAN','MEDIAN', 'STD', 'VAR', 'MIN','MAX'])


## Numerical Attributes - My Function
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


## Variable Summary
def variablesummary(x):
    uc = x.mean()+(2*x.std())
    lc = x.mean()-(2*x.std())
    for i in x:
        if i<lc or i>uc:
            count = 1
        else:
            count = 0
    outlier_flag = count
    return pd.Series([x.corr(),x.count(), x.isnull().sum(), x.sum(), x.mean(), x.median(),  x.std(), x.var(), x.min(), x.quantile(0.01), x.quantile(0.05),x.quantile(0.10),x.quantile(0.25),x.quantile(0.50),x.quantile(0.75), x.quantile(0.90),x.quantile(0.95), x.quantile(0.99),x.max() , lc , uc,outlier_flag],
                  index=['N', 'NMISS', 'SUM', 'MEAN','MEDIAN', 'STD', 'VAR', 'MIN', 'P1' , 'P5' ,'P10' ,'P25' ,'P50' ,'P75' ,'P90' ,'P95' ,'P99' ,'MAX','LC','UC','outlier_flag'])
# UC = MEAN + 2 STD


## VALUE - COUNTS
for i in catdata.columns:
    print(catdata[i].value_counts())

for i in numdata.columns:
    print(numdata[i].value_counts())


                    '2.7 - Null value Finding'
Total = data.isnull().sum().sort_values(ascending=False)
Percent = (data.isnull().sum()*100/len(data)).sort_values(ascending=False)
missingdata = pd.concat([Total, Percent], axis = 1, keys = ['Total', 'Percentage of Missing Values'])    
print(missingdata)


                         '2.8 - Null value Dropping'
# If the missing percentage of null values is greater than 40% drop those values      


                         '2.9 - Imputing'
#1-KNN Imputer
imputer = KNNImputer(n_neighbors=5)
concat = pd.DataFrame(imputer.fit_transform(concat),columns = concat.columns)

#2-Iterative Imputer
imp_mean = IterativeImputer(random_state=0)
imp_mean.fit([[7, 2, 3], [4, np.nan, 6], [10, 5, 9]])
IterativeImputer(random_state=0)
X = [[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]]
imp_mean.transform(X)

# 3 - Hot deck Imputer
df.fillna(method='ffill', inplace=True)


                                      """ OUTLIERS"""
#1-Fix Outlier Range
q1 = df.quantile(0.25)
q3 = df.quantile(0.75)
IQR = q3 - q1
upper_range = q3 + (IQR*1.5)
lower_range = q1 - (IQR*1.5)
extreme_upper_range = q3 + (IQR*3)
extreme_lower_range = q1 - (IQR*3)

#2-Find Count of Outliers
pd.DataFrame(((df_num < extreme_lower_range) | (df_num > extreme_upper_range)).sum(),
             columns = ['No. of Outliers']).sort_values(by = 'No. of Outliers', ascending = False)

#3-Find Percentage of Outliers
pd.DataFrame(((df_num < extreme_lower_range) | (df_num > extreme_upper_range)).sum(),
             columns = ['No. of Outliers']).sort_values(by = 'No. of Outliers', ascending = False) / len(df)


'Outlier Treatment'
#1-Capping (Winzorization)
for i in numdata.columns:
    q1 = numdata[i].quantile(0.25)
    q3 = numdata[i].quantile(0.75)
    IQR = q3-q1
    UB = q3 + 1.5*IQR
    LB = q1 - 1.5*IQR
    UC = numdata[i].quantile(0.99)
    LC = numdata[i].quantile(0.01)
    for ind1 in numdata[i].index:
        if numdata.loc[ind1,i] > UB:
            numdata.loc[ind1,i] = UC
        if numdata.loc[ind1,i] < LB:
            numdata.loc[ind1,i] = LC

#2-Inbuilt Function method
a = np.array([10, 4, 9, 8, 5, 3, 7, 2, 1, 6])
b = np.array([5, 4, 9, 1, 5, 3, 7, 2, 1, 6])
winsorize(a,b, limits=[0.1, 0.2])
masked_array(data=[8, 4, 8, 8, 5, 3, 7, 2, 2, 6],
             mask=False,
       fill_value=999999)


                            'Scaling'
#1-MINMAX SCALER
mm = MinMaxScaler()
numdatamm = mm.fit_transform(numdata)
numdatamm = pd.DataFrame(numdatamm,columns = numdata.columns)

#2-ROBUST SCALER
rs = RobustScaler()
numdatars = rs.fit_transform(numdata)
numdatars = pd.DataFrame(numdatars,columns = numdata.columns)

#3-STANDARD SCALER
sc = StandardScaler()
numdatasc = sc.fit_transform(numdata)
numdatasc = pd.DataFrame(numdatasc,columns = numdata.columns)

#4-MaxAbs Scaler
scaler = MaxAbsScaler()
df_scaled = scaler.fit_transform(numdata.values)
df_scaled = pd.DataFrame(df_scaled,columns = numdata.columns)


#5-Quantile Transformer Scaler
scaler = QuantileTransformer()
df_scaled = scaler.fit_transform(numdata.values)
df_scaled = pd.DataFrame(df_scaled,columns = numdata.columns)


#6-Unit Vector Scaler/Normalizer
from sklearn.preprocessing import Normalizer
scaler = Normalizer(norm = 'l2')
"norm = 'l2' is default"
df_scaled[col_names] = scaler.fit_transform(features.values)



                     """Variable Inflation Factor"""
def variableinflation(X):
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return(vif)
#NOTE : VIF should always be done after scaling (column normalization)


                       """ CORRELATION"""
correlation = X.corr()
sns.heatmap(correlation, annot = True)

#Absolute correlation with Target variable
abs(df.corr()['Variable Name']).sort_values(ascending = False)

def correlation(X):
    correlation = X.corr()
    heatmap = sns.heatmap(correlation,annot = True)
    return correlation,heatmap

def visual(X):
        distplot = sns.distplot(X.columns)
        scatterplot = sns.scatterplot(X.columns)


                         'TRANSFORMATION'
#Log Transformation
np.log2(numdata1['Item_Outlet_Sales']+0.000000001).skew()

#Power Transformation
np.power(numdata1['Item_Outlet_Sales'],0.39).skew()

#Squareroot Transformation
np.sqrt(df.iloc[:,0])

#Boxcox Transformation
sales_box,lam= stats.boxcox(numdata1['Item_Outlet_Sales'])
pd.DataFrame(sales_box).skew()

#Custom Transformer
transformer = FunctionTransformer(np.log2, validate = True)
df_scaled[col_names] = transformer.transform(features.values)

#PowerTransformer
scaler = PowerTransformer(method = 'box-cox')
df_scaled[col_names] = scaler.fit_transform(numdata.values)
#NOTE-method = 'box-cox' or 'yeo-johnson''



                     '2.11 - Encoding Categorical variable'
#1 - One hot Encoding
dummy_var = pd.get_dummies(data = catdata, drop_first = True)

#2 - Factorize Encoding
for i in catdata:
    catdata[i] = catdata[i].factorize()[0]

#3 - Ordinal Encoding
ce_ord = ce.OrdinalEncoder(cols = ['color'])
ce_ord.fit_transform(X, y['outcome'])

# Label Encoding
le = LabelEncoder()
encoded = le.fit_transform(np.ravel(X))

# Count Encoder
ce = CountEncoder(cols = ['Manufacturer', 'Category', 'Leather_interior', 'Fuel_type', 'Gear_box_type',
                          'Drive_wheels', 'Wheel', 'Color', 'Model_1', 'Model_2'],
                  handle_unknown = 0)
X = ce.fit_transform(X)
test_df = ce.transform(test_df)

#Binary Encoder
ce_bin = ce.BinaryEncoder(cols = ['color'])
ce_bin.fit_transform(X, y)

# Base Encoder
ce_basen = ce.BaseNEncoder(cols = ['color'])
ce_basen.fit_transform(X, y)

# Hashing Encoder
ce_hash = ce.HashingEncoder(cols = ['color'])
ce_hash.fit_transform(X, y)

#Helmert Encoder
ce_helmert = ce.HelmertEncoder(cols = ['color'])
ce_helmert.fit_transform(X2, y2)

#Leave one out Encoder
ce_leave = ce.LeaveOneOutEncoder(cols = ['color'])
ce_leave.fit(X3, y3['outcome'])
ce_leave.transform(X3, y3['outcome'])

# Sum Encoder
ce_sum = ce.SumEncoder(cols = ['color'])
ce_sum.fit_transform(X2, y2)

#6- Polynomial Encoder(onehot)
ce_poly = ce.PolynomialEncoder(cols=['color'])
ce_poly.fit_transform(X2, y2)

#7 - Backward difference encoder(onehot)
ce_backward = ce.BackwardDifferenceEncoder(cols = ['color'])
ce_backward.fit_transform(X2, y2)



                            'Concatenating Data'
# 'axis=1' concats the dataframes along columns 
X = pd.concat([df_num, dummy_var], axis = 1)


                         """VISUALIZATION"""
                     """ Univariate Analysis"""
#1 - Distplot Analysis
row = 3
col = 1
count = 1
for i in numdata:
    plt.subplot(row,col,count)
    plt.title(i)
    sns.distplot(data[i])
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(20,20))



# 2 -Countplot
row = 3
col = 1
count = 1
for i in numdata:
    plt.subplot(row,col,count)
    plt.title(i)
    sns.distplot(data[i])
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(10,10))


# 3 - Boxplot Analysis
plt.figure(figsize = (15,8))
numdata.boxplot()
plt.title('Distribution of all Numeric Variables', fontsize = 15)
plt.xticks(rotation = 'vertical', fontsize = 15)
plt.show()

----------------- ALITER------------------------------------------------------------------------------------
row = 3
col = 1
count = 1
for i in numdata:
    plt.subplot(row,col,count)
    plt.title(i)
    sns.boxplot(numdata[i])
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(10,10))


#4- Wordcloud
wc = WordCloud(background_color='white').generate(' '.join(total_tag_list))
plt.figure(figsize=(12.5, 12.5))
plt.title('WordCloud showing most popular Genre')
plt.xticks([])
plt.yticks([])
plt.imshow(wc);
plt.show()
    
    

                         """DATAPANE"""
df = pd.read_csv('https://covid.ourworldindata.org/data/vaccinations/vaccinations-by-manufacturer.csv', parse_dates=['date'])
df = df.groupby(['vaccine', 'date'])['total_vaccinations'].sum().reset_index()

plot = alt.Chart(df).mark_area(opacity=0.4, stroke='black').encode(
    x='date:T',
    y=alt.Y('total_vaccinations:Q'),
    color=alt.Color('vaccine:N', scale=alt.Scale(scheme='set1')),
    tooltip='vaccine:N'
).interactive().properties(width='container')

total_df = df[df["date"] == df["date"].max()].sort_values("total_vaccinations", ascending=False).reset_index(drop=True)
total_styled = total_df.style.bar(subset=["total_vaccinations"], color='#5fba7d', vmax=total_df["total_vaccinations"].sum())

dp.Report("## Vaccination Report",
    dp.Plot(plot, caption="Vaccinations by manufacturer over time"),
    dp.Table(total_styled, caption="Current vaccination totals by manufacturer")
).save(path='report.html', open=True)


------------------------------------ALITER------------------------------------------------------------------------------

# Scripts to create df and chart
# Once you have the df and the chart, simply use
r = dp.Report(
  dp.Markdown('My simple report'), #add description to the report
  dp.Table(df), #create a table
  dp.Plot(chart) #create a chart
)
# Publish your report. Make sure to have visibility='PUBLIC' if you want to share your report
r.publish(name='stock_report', visibility='PUBLIC')


                               """3 - EDA Reports"""
 # 3.1-Pandas Profiling
report = pp.ProfileReport(data)
report.to_file('profile_report.html')

#3.2- Sweetviz
advert_report = sv.analyze(data)
advert_report.show_html('Advertising.html')

# 3.3-dtale
d = dtale.show(data)
d.open_browser() 

#3.4 - Autoviz
#importing Autoviz class
from autoviz.AutoViz_Class import AutoViz_Class#Instantiate the AutoViz class
AV = AutoViz_Class()
df = AV.AutoViz('car_design.csv')

#3.5 -Dora
# Import required module
from Dora import Dora
# Create object
dora = Dora()
# Add dataset path as argument 
dora.configure(output = 'A', data = 'data.csv')
# Display dataset
dora.data

#3.6 - Dataprep
#1- Prepare report
from dataprep.datasets import load_dataset
from dataprep.eda import create_report
df = load_dataset("titanic")
create_report(df).show_browser()

# 2 - EDA
## Plot correlation
from dataprep.eda import plot_correlation
plot_correlation(df)
## Missing values
from dataprep.eda import plot_missing
plot_missing(df)
## Distribution
from dataprep.eda import plot
plot(df)
## Difference between datframes
from dataprep.eda import plot_diff
plot_diff([df1, df2])


                        'TARGET CLASS BALANCING - CLASSFICATION'
                            'Random Under Sampling'
rus = RandomUnderSampler(return_indices =  True)
xresampled,yresampled,idxresampled  =  rus.fit_sample(X,Y)

                             'Random Over Sampling'
rus = RandomUnderSampler(return_indices =  True)
xresampled,yresampled,idxresampled  =  rus.fit_sample(X,Y)


                                 'SMOTE'
s=SMOTE()
imbal=smote.fit_transform(x,y)




                           """  Train Test Split"""
xtrain, xtest, ytrain, ytest = train_test_split( X, Y, test_size=0.33, random_state=42)



                           "Predictive Power Score"
pps.predictors(train_df, "Overall_Experience")[['x', 'y', 'ppscore']]

                             'TIME FUNCTIONS'
## Splitting date into day,month,year
concat["day"] = concat['Orderdate'].map(lambda x: x.day)
concat["month"] = concat['Orderdate'].map(lambda x: x.month)
concat["year"] = concat['Orderdate'].map(lambda x: x.year)
concat['dayofweek'] = pd.to_datetime(concat['Orderdate']).dt.dayofweek

##  Split into Weeks
def Week(x):
    if x >= 1 and x<= 8:
        return 1
    elif x >= 9 and x<= 16:
        return 2
    elif x >= 17 and x<= 24:
        return 3
    else:
        return 4

## Split into Weekday/Weekend
def Weekdayend(x):
    if x >= 0 and x<= 4:
        return 1
    else:
        return 2


## Segregate a month into 4 parts
def Monthsplitup(x):
    if x >= 1 and x<= 10:
        return 1
    elif x >= 11 and x<= 20:
        return 2
    elif x >= 21 and x<= 25:
        return 3
    else:
        return 4


## Segregate into Quarters
def Quarter(x):
    if x >= 1 and x<= 3:
        return 1
    elif x >= 4 and x<= 6:
        return 2
    elif x >= 7 and x<= 9:
        return 3
    else:
        return 4

## Appying the Functions
concat['Week'] = concat['day'].apply(Week)
concat['Weekday/Weekend'] = concat['dayofweek'].apply(Weekdayend)
concat['Monthsplitup'] = concat['day'].apply(Monthsplitup)
concat['Quarter'] = concat['month'].apply(Quarter)


#Python program to convert time from 12 hour to 24 hour format
def convert24(str1):# Function to convert the date format
    # Checking if last two elements of time
    # is AM and first two elements are 12
    if str1[-2:] == "AM" and str1[:2] == "12":
        return "00" + str1[2:-2]

    # remove the AM
    elif str1[-2:] == "AM":
        return str1[:-2]

    # Checking if last two elements of time
    # is PM and first two elements are 12
    elif str1[-2:] == "PM" and str1[:2] == "12":
        return str1[:-2]

    else:
        return str(int(str1[:2]) + 12) + str1[2:8]# add 12 to hours and remove PM

# Applying Function
print(convert24("08:05:45 PM"))

-----------------------------------ALITER------------------------------------------------------------------
#import datetime
from datetime import datetime
#sample input time to be converted
inputTime = "08:05:45 PM"
#Create datetime object from string
in_time = datetime.strptime(inputTime, "%I:%M:%S %p")
#convert to 24 hour format
out_time = datetime.strftime(in_time, "%H:%M:%S")
#print result
print(out_time)


## Time to seconds
def time2seconds(time):
  if type(time) != str:
    return time
  parts = [float(p) for p in time.split(':')]
  parts = [p * (60 ** i) for i, p in enumerate(reversed(parts))]
  return sum(parts)

'1-CUSTOM FUNCTIONS'
#1-PRICE SPLIT
def pricesplit(x):
    if x>=0 and x<= 500:
        return 1
    if x>=501 and x<= 1000:
        return 2
    if x>=1001 and x<= 1500:
        return 3
    if x>=1501 and x<= 2000:
        return 4
    if x>=2001 and x<= 2500:
        return 5
    if x>=2501 and x<= 3000:
        return 6
    if x>=3001 and x<= 3500:
        return 7
    if x>=3501 and x<= 4000:
        return 8
    if x>=4001 and x<= 4500:
        return 9
    if x>=4501 and x<= 5000:
        return 10
    if x>=5001 and x<= 5500:
        return 11
    if x>=5501 and x<= 6000:
        return 12
    else:
        return 13


## Iterating Dictionary





## Writing to excel
import pandas as pd
df1 = pd.DataFrame({'Data': ['a', 'b', 'c', 'd']})
df2 = pd.DataFrame({'Data': [1, 2, 3, 4]})
df3 = pd.DataFrame({'Data': [1.1, 1.2, 1.3, 1.4]})

writer = pd.ExcelWriter('multiple.xlsx', engine='xlsxwriter')

df1.to_excel(writer, sheet_name='Sheeta')
df2.to_excel(writer, sheet_name='Sheetb')
df3.to_excel(writer, sheet_name='Sheetc')

writer.save()

##--ALITER
import csv
# my data rows as dictionary objects
mydict = [{'branch': 'COE', 'cgpa': '9.0', 'name': 'Nikhil', 'year': '2'},
          {'branch': 'COE', 'cgpa': '9.1', 'name': 'Sanchit', 'year': '2'},
          {'branch': 'IT', 'cgpa': '9.3', 'name': 'Aditya', 'year': '2'},
          {'branch': 'SE', 'cgpa': '9.5', 'name': 'Sagar', 'year': '1'},
          {'branch': 'MCE', 'cgpa': '7.8', 'name': 'Prateek', 'year': '3'},
          {'branch': 'EP', 'cgpa': '9.1', 'name': 'Sahil', 'year': '2'}]

# field names
fields = ['name', 'branch', 'year', 'cgpa']
# name of csv file


                           '1 - String Split'
                           df[['code', 'name_of_code']] = df["code"].str.split(" ", 1, expand=True)

                           '2 - Separating Number from String'


                           def find_number(text):
                               num = re.findall(r'[0-9]+', text)
                               return " ".join(num)


                           data['Reviewss'] = data['Reviewss'].apply(lambda x: find_number(x))

                           '3 - Strip '
                           df["name_of_code"] = df["name_of_code"].

                           '4 - Removing whitespace'


                           def removewhitespace(x):
                               """
                               Helper function to remove any blank space from a string
                               x: a string
                               """
                               try:
                                   # Remove spaces inside of the string
                                   x = "".join(x.split())

                               except:
                                   pass
                               return x

#Add values in a row separated by comma
def sum_of_number(listx):
                        a = []
                        n = 0
                        if len(listx) == 1:
                            a = listx
                            n = a
                               else:
                                   a = str(listx).split(',')
                                   for i in range(0, len(a)):
                                       n = n + int(a[i])
                return n


                           --ALITER
                           df['Sum'] = df['Scores'].apply(lambda x: sum(map(float, x.split(','))))

                           '5 - SEGMENTING FUNCTION'
                           1)categorical


                           def regions(x):
                               if x in ['WA', 'MT', 'OR', 'ID', 'WY', 'CA', 'NV', 'UT', 'CO', 'AK']:
                                   return ('West')
                               if x in ['AZ', 'NM', 'TX', 'OK', 'HI']:
                                   return ('Southwest')
                               if x in ['ND', 'SD', 'NE', 'KS', 'MN', 'IA', 'MO', 'WI', 'IL', 'IN', 'MI', 'OH']:
                                   return ('Midwest')
                               if x in ['LA', 'AR', 'MS', 'AL', 'GA', 'FL', 'KY', 'TN', 'SC', 'NC', 'VA', 'DC', 'WV',
                                        'DE', 'MD', 'PR', 'VI']:
                                   return ('Southeast')
                               if x in ['PA', 'NJ', 'NY', 'CT', 'RI', 'MA', 'NH', 'VT', 'ME', 'GU']:
                                   return ('Northeast')
                               if x in ['AE', 'AP', 'AA']:
                                   return ('Armedforces')

#ENCRYPTION and DECRYPTION
from cryptography.fernet import Fernet
    # we will be encryting the below string.
message = "hello geeks"

                           # generate a key for encryptio and decryption
                           # You can use fernet to generate
                           # the key or use random key generator
                           # here I'm using fernet to generate key

key = Fernet.generate_key()

                           # Instance the Fernet class with the key

fernet = Fernet(key)

                           # then use the Fernet class instance
                           # to encrypt the string string must must
                           # be encoded to byte string before encryption
encMessage = fernet.encrypt(message.encode())

print("original string: ", message)
print("encrypted string: ", encMessage)

                           # decrypt the encrypted string with the
                           # Fernet instance of the key,
                           # that was used for encrypting the string
                           # encoded byte string is returned by decrypt method,
                           # so decode it to string with decode methos
decMessage = fernet.decrypt(encMessage).decode()
print("decrypted string: ", decMessage)



 """Replace"""
for i in numdata:
                numdata[i] = numdata[i].replace('-', None)
                               # Replacing Empty values with None


                           # Count of "None" Values
                           (numdata['Not Out'] == 'None').count()