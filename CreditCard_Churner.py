# EDA Libraries and Settings

import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline
sns.set(rc = {'figure.figsize': (5, 5)})
sns.set_style('whitegrid')


# Examine the dataset

pd.read_csv('Bank-Churners.csv').head()


# Drop the ID column and examine again

bc0 = pd.read_csv('Bank-Churners.csv').drop(columns = ['CLIENTNUM'])
bc0.info()
bc0.describe(include = 'all')


# Examine the distribution of customers' status

freqe = len(bc0[bc0['Attrition_Flag'] == 'Existing Customer'])
freqa = len(bc0[bc0['Attrition_Flag'] == 'Attrited Customer'])
size = len(bc0)/100

sns.histplot(bc0, x = 'Attrition_Flag', hue = 'Attrition_Flag', multiple='stack')

pd.DataFrame({'Attrition Flag': ['Existing Customer', 'Attrited Customer'],
              'Count': [freqe, freqa], 'Proportion (%)': [round(freqe/size, 2), round(freqa/size, 2)]})


# Divide features into categorical and numerical

cat = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']

num = ['Customer_Age', 'Dependent_count', 'Months_on_book', 'Total_Revolving_Bal (RM)', 
       'Total_Trans_Amt (RM)', 'Total_Trans_Ct']


# Check unique values in cat and count percentage of 'Unknown' values

for c in cat:
    print(c, ' : ', np.unique(bc0[c]), '\n')
    if 'Unknown' in np.unique(bc0[c]):
        print('    Unknown : ', len(bc0[bc0[c] == 'Unknown']), '/', len(bc0[c]),
              ' = ', round(len(bc0[bc0[c] == 'Unknown'])*100/len(bc0[c]), 2), '%', '\n', sep = '')
        
        
# Drop rows with 'Unknown' values & check the distribution again

bc = bc0
for c in cat:
    bc = bc[bc[c] != 'Unknown']

bc.info()

freqe = len(bc[bc['Attrition_Flag'] == 'Existing Customer'])
freqa = len(bc[bc['Attrition_Flag'] == 'Attrited Customer'])
size = len(bc)/100

pd.DataFrame({'Attrition Flag': ['Existing Customer', 'Attrited Customer'],
              'Count': [freqe, freqa], 'Proportion (%)': [round(freqe/size, 2), round(freqa/size, 2)]})

chart = sns.countplot(data = bc, x = 'Attrition_Flag')
chart.set_xticklabels(chart.get_xticklabels(), rotation = 45, ha = 'right')
for rect in chart.patches:
    chart.text (rect.get_x() + rect.get_width()  / 2, rect.get_height() + 0.75, rect.get_height(),
                horizontalalignment = 'center'); chart


# Initial idea on the distributions of categorical features

fig, ax = plt.subplots(1, len(cat), figsize = (20, 5), sharey = True)
for i in range(len(cat)):
    chart = sns.countplot(data = bc, x = cat[i], ax = ax[i])
    chart.set_xticklabels(chart.get_xticklabels(), rotation = 45, ha = 'right')
    for rect in chart.patches:
        chart.text(rect.get_x() + rect.get_width()  / 2, rect.get_height() + 0.75, rect.get_height(),
                   horizontalalignment = 'center'); chart
        
        
# Comparison between attrited and existing customers with respect to categorical features
# Left-hand side: count plots
# Right-hand side: percentage plots

fig, ax = plt.subplots(len(cat), 2, figsize = (20, 10*len(cat)), sharey = False)

for i in range(len(cat)):
    cnt = bc.groupby(cat[i])['Attrition_Flag'].value_counts(normalize = False).reset_index(name = 'count')
    sns.histplot(x = cat[i] , hue = 'Attrition_Flag', weights = 'count', multiple = 'stack', data = cnt, 
                 shrink = 0.8, ax = ax[i,0])
    perc = bc.groupby(cat[i])['Attrition_Flag'].value_counts(normalize = True).mul(100).reset_index(name = 'percentage')
    sns.histplot(x = cat[i] , hue = 'Attrition_Flag', weights = 'percentage', multiple = 'stack', data = perc, 
                 shrink = 0.8, ax = ax[i,1]).set(ylabel = 'Percentage')
    
    
# Comparison between attrited and existing customers with respect to numerical features
# Left-hand side: boxplots
# Right-hand side: histogram

fig, ax = plt.subplots(len(num), 2, figsize = (20, 10*len(num)), sharey = False)

for i in range(len(num)):
    sns.boxplot(x = bc['Attrition_Flag'], y = bc[num[i]], ax = ax[i,0]);
    sns.histplot(bc, x = num[i], hue = 'Attrition_Flag', ax = ax[i,1]);
    
    
# Replace customers' status with Boolean values

bc.Attrition_Flag = bc.Attrition_Flag.replace({'Attrited Customer': 1, 'Existing Customer': 0})


# Replace Income_Category with ordinal values (hard code)
# Option: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html

bc.Income_Category = bc.Income_Category.replace({'Less than RM40K': 1, 'RM40K - RM60K': 2,
                                                 'RM60K - RM80K': 3, 'RM80K - RM120K': 4, 'RM120K +': 5})


# Find Pearson's Correlation & Spearman's Rank Correlation among numeric features

bc.corr('pearson')
sns.heatmap(bc.corr('pearson'), cmap = 'PuOr', annot = True, vmin = -1, vmax = 1, center = 0)

bc.corr('spearman')
sns.heatmap(bc.corr('spearman'), cmap = 'PuOr', annot = True, vmin = -1, vmax = 1, center = 0)


# Modelling Libraries and Settings

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Prepare data for ML

X = bc[num]
Y = bc['Attrition_Flag']

X_train, X_test, Y_train, Y_test, = train_test_split(X, Y, test_size = 0.2, random_state = 999, stratify = Y)


# Run the logistic regression modelling and get accuracy score

logr = LogisticRegression()
logr.fit(X_train, Y_train)

predictions = logr.predict(X_test)

accuracy_score(Y_test, predictions)


