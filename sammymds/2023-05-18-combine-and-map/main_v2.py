import os
import re
import numpy as np
import pandas as pd

CURRENT_YEAR = 2023
os.makedirs('processed-files-v2', exist_ok=True)

if not os.path.exists('processed-files-v2/base.csv'):
    df_base = pd.read_csv('client-files-v2/base.csv', low_memory=False).dropna(how='all')
    # df_base['PARID'] = df_base['PARID'].apply(lambda x: re.findall('\d+', x)[0]).astype(np.int64)
    df_base.to_csv('processed-files-v2/base.csv', index=False)
else:
    df_base = pd.read_csv('processed-files-v2/base.csv', low_memory=False).dropna(how='all')
df_base.sort_values(by='PARID', inplace=True, ignore_index=True)
print('[df_base.shape]', df_base.shape)
print(df_base.dtypes)
# print(df_base['PARID'].isna().sum())

if not os.path.exists('processed-files-v2/abatement.csv'):
    df_abatement = pd.read_csv('client-files-v2/abatement.csv', low_memory=False).dropna(how='all')
    # df_abatement['PARID'] = df_abatement['PARID'].astype(np.int64)
    df_abatement.to_csv('processed-files-v2/abatement.csv', index=False)
else:
    df_abatement = pd.read_csv('processed-files-v2/abatement.csv', low_memory=False).dropna(how='all')
df_abatement.sort_values(by='PARID', inplace=True, ignore_index=True)
print('[df_abatement.shape]', df_abatement.shape)
print(df_abatement.dtypes)
# print(df_abatement['PARID'].isna().sum())

if not os.path.exists('processed-files-v2/exemption.csv'):
    df_exemption = pd.read_csv('client-files-v2/exemption.csv', low_memory=False).dropna(how='all')
    # df_exemption['PARID'] = df_exemption['PARID'].astype(np.int64)
    df_exemption.to_csv('processed-files-v2/exemption.csv', index=False)
else:
    df_exemption = pd.read_csv('processed-files-v2/exemption.csv', low_memory=False).dropna(how='all')
df_exemption.sort_values(by='PARID', inplace=True, ignore_index=True)
print('[df_exemption.shape]', df_exemption.shape)
print(df_exemption.dtypes)
# print(df_exemption['PARID'].isna().sum())

df_merged = pd.merge(left=df_abatement, right=df_exemption, how='outer', left_on='PARID', right_on='PARID', suffixes=('_base', '_abatement'))
df_merged = pd.merge(left=df_merged, right=df_base, how='outer', left_on='PARID', right_on='PARID', suffixes=('_base', '_abatement'))
# df_merged = df_merged.iloc[:10000]
# print('[df_merged.shape]', df_merged.shape)
# print('[df_merged.columns]', df_merged.columns)
# print('[df_merged.dtypes]', df_merged.dtypes != object)
zero_fill_columns = df_merged.columns[(df_merged.isna().sum(axis=0) > 0) & (df_merged.dtypes != object)]
# print('[zero_fill_columns]', zero_fill_columns)
df_merged[zero_fill_columns] = df_merged[zero_fill_columns].fillna(0)
print('[df_merged]\n', df_merged)

value_columns = [
    'BBL', 
    'tax_rate', 
    'Billable Assessed Value', 
    'Annual Tax Before Abatements and Exemptions', 
    'Monthly Tax Before Abatements and Exemptions', 
    'Total Exemption Value', 
    'Total Abatement Value', 
    'Annual Tax with Exemptions', 
    'Annual Tax with Abatements and Exemptions', 
    'Monthly Tax with Abatements and Exemptions'
]
df_values = pd.DataFrame(columns=value_columns)
df_values['BBL'] = df_merged['PARID']
df_values['tax_rate'] = df_merged['rate']
df_values['Billable Assessed Value'] = df_merged['BAV']
df_values['Annual Tax Before Abatements and Exemptions'] = df_merged['BAV'] * df_merged['rate']
df_values['Monthly Tax Before Abatements and Exemptions'] = df_values['Annual Tax Before Abatements and Exemptions'] / 12
for group_id, group in df_merged.groupby('PARID'):
    rows = df_merged['PARID'] == group_id
    df_values.loc[rows, 'Total Exemption Value'] = group['CUREX'].sum()
    df_values.loc[rows, 'Total Abatement Value'] = group['APPLIEDABT'].sum()
df_values['Annual Tax with Exemptions'] = df_values['Billable Assessed Value'] - df_values['Total Exemption Value'] * df_values['tax_rate']
df_values['Annual Tax with Abatements and Exemptions'] = df_values['Annual Tax with Exemptions'] - df_values['Total Abatement Value']
df_values['Monthly Tax with Abatements and Exemptions'] = df_values['Annual Tax with Abatements and Exemptions'] / 12
print('[df_values.shape]', df_values.shape)
# print(df_values.isna().sum(axis=0))
columns_to_round = ['Billable Assessed Value', 'Annual Tax Before Abatements and Exemptions', 'Monthly Tax Before Abatements and Exemptions', 'Total Exemption Value', 'Total Abatement Value', 'Annual Tax with Abatements and Exemptions', 'Monthly Tax with Abatements and Exemptions']
df_values[columns_to_round] = df_values[columns_to_round].round().astype(np.int64)
print('[df_values.dtypes]\n', df_values.dtypes)
print('[df_values]\n', df_values)
# df_values.iloc[:10000].to_csv('output-outer2.csv', index=False)
df_values.to_csv('values.csv', index=False)

benefit_columns = [
    'BBL', 
    'Benefit Name', 
    'Benefit Value', 
    'Benefit End Date', 
    'Benefit Years Remaining', 
]
df_benefits = pd.DataFrame(columns=benefit_columns)
df_benefits['BBL'] = df_merged['PARID']
df_benefits['Benefit Name'] = df_merged['EXMP_CODE']
# df_benefits['Benefit Name'] = df_merged['TCCODE']
df_benefits['Benefit Value'] = df_merged['CUREX']
# df_benefits['Benefit Value'] = df_merged['APPLIEDABT']
df_benefits['Benefit End Date'] = df_merged['BENFTSTART'] + df_merged['NO_YEARS'] - 1
# df_benefits['Benefit End Date'] = df_merged['YRENDDT']
df_benefits['Benefit Years Remaining'] = df_benefits['Benefit End Date'] - CURRENT_YEAR
print('[df_benefits.shape]', df_benefits.shape)
print('[df_benefits.dtypes]\n', df_benefits.dtypes)
# print(df_benefits.isna().sum(axis=0))
columns_to_round = ['Benefit Value', 'Benefit End Date', 'Benefit Years Remaining']
df_benefits[columns_to_round] = df_benefits[columns_to_round].round().astype(np.int64)
print('[df_benefits]\n', df_benefits)
# df_benefits.iloc[:10000].to_csv('output-outer2.csv', index=False)
df_benefits.to_csv('benefits.csv', index=False)
