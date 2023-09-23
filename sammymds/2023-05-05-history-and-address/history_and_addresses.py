import pandas as pd

df_addresses = pd.read_csv('addresses.csv')
df_addresses = df_addresses[['boro', 'block', 'lot', 'street_number', 'street']]
print('[df_addresses]\n', df_addresses)

history_cols = ['id', 'boro', 'block', 'lot', 'bin', 'street_number', 'street', 'full_address']
df_history = pd.read_csv('history_1.csv')
# total_rows = 16,666,234
print('[df_history]\n', df_history)

df_merged = pd.merge(left=df_history, right=df_addresses, how='inner', 
         left_on=['ml_Legal_BOROUGH', 'ml_Legal_STREET_NUMBER', 'ml_Legal_STREET_NAME'],
         right_on=['boro', 'street_number', 'street'],
         suffixes=('_history', '_addresses'))
print('[df_merged]\n', df_merged)

####### Verify that merge is correct #######
print('[df_addresses]', df_addresses['boro'].nunique())
print('[df_history]', df_history['ml_Legal_BOROUGH'].nunique())
print('[df_merged]', df_merged['boro'].nunique(), df_merged['ml_Legal_BOROUGH'].nunique())

print('[df_addresses]', df_addresses['street_number'].nunique())
print('[df_history]', df_history['ml_Legal_STREET_NUMBER'].nunique())
print('[df_merged]', df_merged['street_number'].nunique(), df_merged['ml_Legal_STREET_NUMBER'].nunique())

print('[df_addresses]', df_addresses['street'].nunique())
print('[df_history]', df_history['ml_Legal_STREET_NAME'].nunique())
print('[df_merged]', df_merged['street'].nunique(), df_merged['ml_Legal_STREET_NAME'].nunique())

print(df_merged.shape)
print(sum(df_merged['boro'] == df_merged['ml_Legal_BOROUGH']))
print(sum(df_merged['street_number'] == df_merged['ml_Legal_STREET_NUMBER']))
print(sum(df_merged['street'] == df_merged['ml_Legal_STREET_NAME']))
print(sum((df_merged['ml_Legal_BOROUGH'] == df_merged['boro']) & 
      (df_merged['ml_Legal_STREET_NUMBER'] == df_merged['street_number']) & 
      (df_merged['ml_Legal_STREET_NAME'] == df_merged['street'])))
#######

df_merged.drop(columns=['ml_Legal_BOROUGH', 'ml_Legal_STREET_NUMBER', 'ml_Legal_STREET_NAME'], inplace=True)
print('[df_merged]\n', df_merged)

df_output = df_merged[(df_merged['ml_Legal_BLOCK'] != df_merged['block']) | (df_merged['ml_Legal_LOT'] != df_merged['lot'])]
print('[df_output]\n', df_output)
df_output = df_output.iloc[:1000]
df_output.to_csv('output.csv', index=False)

