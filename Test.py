import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
import xgboost as xgb  # Import XGBoost

# Load Dataset
a1 = pd.read_excel(r'D:\CampusX\Credit Risk Modelling using ML\case_study1.xlsx')
a2 = pd.read_excel(r'D:\CampusX\Credit Risk Modelling using ML\case_study2.xlsx')

df1 = a1.copy()
df2 = a2.copy()

# Remove null
df1 = df1.loc[df1['Age_Oldest_TL'] != -99999]

columns_to_be_removed = []
for i in df2.columns:
    if df2.loc[df2[i] == -99999].shape[0] > 10000:
        columns_to_be_removed.append(i)

df2 = df2.drop(columns_to_be_removed, axis=1)

for i in df2.columns:
    df2 = df2.loc[df2[i] != -99999]

# Checking for common columns
for i in list(df1.columns):
    for i in list(df2.columns):
        print(i)

# Merge the two dataframes using an inner join to remove null values
df = pd.merge(df1, df2, how='inner', left_on='PROSPECTID', right_on='PROSPECTID')

# Check how many columns are categorical
for i in df.columns:
    if df[i].dtype == 'object':
        print(i)

# Chi-square test
for i in ['MARITALSTATUS', 'EDUCATION', 'last_prod_enq2', 'first_prod_enq2']:
    chi2, pval, _, _, = chi2_contingency(pd.crosstab(df[i], df['Approved_Flag']))
    print(i, '---', pval)

# VIF for numerical columns
numeric_columns = [col for col in df.columns if df[col].dtype != 'object']

vif_data = df[numeric_columns]
total_columns = vif_data.shape[1]
columns_to_be_kept = []
column_index = 0

for i in range(0, total_columns):
    vif_value = variance_inflation_factor(vif_data.values, i)
    print(numeric_columns[i], '---', vif_value)

    if vif_value <= 6:
        columns_to_be_kept.append(numeric_columns[i])
        column_index = column_index + 1
    else:
        vif_data = vif_data.drop([numeric_columns[i]], axis=1)

# Check ANOVA columns_to_be_kept
from scipy.stats import f_oneway

columns_to_be_kept_numerical = []

for i in columns_to_be_kept:
    a = list(df[i])
    b = list(df['Approved_Flag'])

    group_P1 = [value for value, group in zip(a, b) if group == 'P1']
    group_P2 = [value for value, group in zip(a, b) if group == 'P2']
    group_P3 = [value for value, group in zip(a, b) if group == 'P3']
    group_P4 = [value for value, group in zip(a, b) if group == 'P4']

    f_statistic, p_value = f_oneway(group_P1, group_P2, group_P3, group_P4)

    if p_value <= 0.05:
        columns_to_be_kept_numerical.append(i)

# Feature selection is done for cat and num features

# Listing all the final features
features = columns_to_be_kept_numerical + ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']
df = df[features + ['Approved_Flag']]

# Listing encoding for final features

print(df['MARITALSTATUS'].unique())
print(df['EDUCATION'].unique())
print(df['GENDER'].unique())
print(df['last_prod_enq2'].unique())
print(df['first_prod_enq2'].unique())

# Convert EDUCATION column to ordinal values
df.loc[df['EDUCATION'] == 'SSC', 'EDUCATION'] = 1
df.loc[df['EDUCATION'] == '12TH', 'EDUCATION'] = 2
df.loc[df['EDUCATION'] == 'GRADUATE', 'EDUCATION'] = 3
df.loc[df['EDUCATION'] == 'UNDER GRADUATE', 'EDUCATION'] = 3
df.loc[df['EDUCATION'] == 'POST-GRADUATE', 'EDUCATION'] = 4
df.loc[df['EDUCATION'] == 'OTHERS', 'EDUCATION'] = 1
df.loc[df['EDUCATION'] == 'PROFESSIONAL', 'EDUCATION'] = 3

# Check the value counts after conversion
print(df['EDUCATION'].value_counts())
df['EDUCATION'] = df['EDUCATION'].astype(int)

df_encoded = pd.get_dummies(df, columns=['MARITALSTATUS', 'GENDER', 'last_prod_enq2', 'first_prod_enq2'])

# Machine Learning model fitting

# XGBoost
from sklearn.preprocessing import LabelEncoder

xgbc = xgb.XGBClassifier(objective='multi:softmax', num_class=4)

y = df_encoded['Approved_Flag']
x = df_encoded.drop(['Approved_Flag'], axis=1)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)

xgbc.fit(x_train, y_train)

y_pred = xgbc.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print()
print(f'Accuracy: {accuracy}')
print()
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)

for i, v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f'Class {v}')
    print(f'Precision: {precision[i]}')
    print(f'Recall: {recall[i]}')
    print(f'F1 score: {f1_score[i]}')
    print()

# Predict for Unseen Data

a3 = pd.read_excel(r'D:\CampusX\Credit Risk Modelling using ML\session 4\Unseen_Dataset.xlsx')

cols_in_df = list(df.columns)
cols_in_df.pop(42)

df_unseen = a3[cols_in_df]

print(df_unseen['MARITALSTATUS'].unique())
print(df_unseen['EDUCATION'].unique())
print(df_unseen['GENDER'].unique())
print(df_unseen['last_prod_enq2'].unique())
print(df_unseen['first_prod_enq2'].unique())

df_unseen.loc[df_unseen['EDUCATION'] == 'SSC', 'EDUCATION'] = 1
df_unseen.loc[df_unseen['EDUCATION'] == '12TH', 'EDUCATION'] = 2
df_unseen.loc[df_unseen['EDUCATION'] == 'GRADUATE', 'EDUCATION'] = 3
df_unseen.loc[df_unseen['EDUCATION'] == 'UNDER GRADUATE', 'EDUCATION'] = 3
df_unseen.loc[df_unseen['EDUCATION'] == 'POST-GRADUATE', 'EDUCATION'] = 4
df_unseen.loc[df_unseen['EDUCATION'] == 'OTHERS', 'EDUCATION'] = 1
df_unseen.loc[df_unseen['EDUCATION'] == 'PROFESSIONAL', 'EDUCATION'] = 3

# Check the value counts after conversion
print(df_unseen['EDUCATION'].value_counts())
df_unseen['EDUCATION'] = df_unseen['EDUCATION'].astype(int)

df_encoded_unseen = pd.get_dummies(df_unseen, columns=['MARITALSTATUS', 'GENDER', 'last_prod_enq2', 'first_prod_enq2'])

model = xgb.XGBClassifier(objective='multi:softmax', num_class=4, colsample_bytree=0.9, learning_rate=1, max_depth=3, alpha=10, n_estimators=100)

model.fit(x_train, y_train)

y_pred_unseen = model.predict(df_encoded_unseen)

a3['Target_Variable'] = y_pred_unseen

a3.to_excel(r'D:\Final_Prediction.xlsx', index=False)
