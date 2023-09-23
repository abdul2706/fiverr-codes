import os
import re
import numpy as np
import pandas as pd

CURRENT_YEAR = 2023

if not os.path.exists('processed-files/base.csv'):
    df_base = pd.read_csv('original-files/base.csv', low_memory=False).dropna(how='all')
    df_base['PARID'] = df_base['PARID'].apply(lambda x: re.findall('\d+', x)[0]).astype(np.int64)
    df_base.to_csv('processed-files/base.csv', index=False)
else:
    df_base = pd.read_csv('processed-files/base.csv', low_memory=False).dropna(how='all')
df_base.sort_values(by='PARID', inplace=True, ignore_index=True)
print('[df_base.shape]', df_base.shape)
# print(df_base.dtypes)
# print(df_base['PARID'].isna().sum())

if not os.path.exists('processed-files/abatement.csv'):
    df_abatement = pd.read_csv('original-files/abatement.csv', low_memory=False).dropna(how='all')
    df_abatement['PARID'] = df_abatement['PARID'].astype(np.int64)
    df_abatement.to_csv('processed-files/abatement.csv', index=False)
else:
    df_abatement = pd.read_csv('processed-files/abatement.csv', low_memory=False).dropna(how='all')
df_abatement.sort_values(by='PARID', inplace=True, ignore_index=True)
print('[df_abatement.shape]', df_abatement.shape)
# print(df_abatement.dtypes)
# print(df_abatement['PARID'].isna().sum())

if not os.path.exists('processed-files/exemption.csv'):
    df_exemption = pd.read_csv('original-files/exemption.csv', low_memory=False).dropna(how='all')
    df_exemption['PARID'] = df_exemption['PARID'].astype(np.int64)
    df_exemption.to_csv('processed-files/exemption.csv', index=False)
else:
    df_exemption = pd.read_csv('processed-files/exemption.csv', low_memory=False).dropna(how='all')
df_exemption.sort_values(by='PARID', inplace=True, ignore_index=True)
print('[df_exemption.shape]', df_exemption.shape)
# print(df_exemption.dtypes)
# print(df_exemption['PARID'].isna().sum())

df_merged = pd.merge(left=df_abatement, right=df_exemption, how='outer', left_on='PARID', right_on='PARID', suffixes=('_base', '_abatement'))
df_merged = pd.merge(left=df_merged, right=df_base, how='outer', left_on='PARID', right_on='PARID', suffixes=('_base', '_abatement'))
# df_merged = df_merged.iloc[:10000]
print('[df_merged.shape]', df_merged.shape)
zero_fill_columns = ['rate', 'FINACTTOT', 'FINTRNTOT', 'CUREXMPTOT', 'CUREXMPTRN', 'APPLIEDABT']
df_merged[zero_fill_columns] = df_merged[zero_fill_columns].fillna(0)

columns = [
    'BBL', 
    'tax_rate', 
    'Billable Assessed Value', 
    'Annual Tax Before Abatements and Exemptions', 
    'Monthly Tax Before Abatements and Exemptions', 
    'Exemption Name', 
    'Exemption Value', 
    'Exemption End Date', 
    'Exemption Years Remaining', 
    'Abatement Name', 
    'Abatement End Date', 
    'Abatement Value', 
    'Annual Tax with Abatements and Exemptions', 
    'Monthly Tax with Abatements and Exemptions'
]
df_output = pd.DataFrame(columns=columns)
df_output['BBL'] = df_merged['PARID']
df_output['tax_rate'] = df_merged['rate']
df_output['Billable Assessed Value'] = df_merged[['FINACTTOT', 'FINTRNTOT']].min(axis=1)
df_output['Annual Tax Before Abatements and Exemptions'] = df_output['Billable Assessed Value'] * df_output['tax_rate'] / 100
df_output['Monthly Tax Before Abatements and Exemptions'] = df_output['Annual Tax Before Abatements and Exemptions'] / 12
df_output['Exemption Name'] = df_merged['EXMP_CODE']
df_output['Exemption Value'] = df_merged[['CUREXMPTOT', 'CUREXMPTRN']].min(axis=1)
df_output['Exemption End Date'] = df_merged['BENFTSTART'] + df_merged['NO_YEARS'] - 1
df_output['Exemption Years Remaining'] = df_output['Exemption End Date'] - CURRENT_YEAR
df_output['Abatement Name'] = df_merged['TCCODE']
df_output['Abatement End Date'] = df_merged['YRENDDT']
df_output['Abatement Value'] = df_merged['APPLIEDABT']
df_output['Annual Tax with Abatements and Exemptions'] = df_output['Billable Assessed Value'] - df_output['Exemption Value'] * df_output['tax_rate'] - df_output['Abatement Value']
df_output['Monthly Tax with Abatements and Exemptions'] = df_output['Annual Tax with Abatements and Exemptions'] / 12
print('[df_output.shape]', df_output.shape)
# print(df_output.isna().sum(axis=0))
columns_to_round = ['Billable Assessed Value', 'Annual Tax Before Abatements and Exemptions', 'Monthly Tax Before Abatements and Exemptions', 'Exemption Value', 'Abatement Value', 'Annual Tax with Abatements and Exemptions', 'Monthly Tax with Abatements and Exemptions']
df_output[columns_to_round] = df_output[columns_to_round].round().astype(np.int64)
# print(df_output.dtypes)
# df_output.iloc[:10000].to_csv('output-outer2.csv', index=False)
df_output.to_csv('output.csv', index=False)
