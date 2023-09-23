from tqdm import tqdm
import pandas as pd

df_history = pd.read_csv('history_1_grouped.csv')
# df_history.loc[df_history['ml_Legal_STREET_NAME'] == 'BANK', 'ml_Legal_STREET_NAME'] = 'BANK '
df_history['ml_Legal_STREET_NAME2'] = df_history['ml_Legal_STREET_NAME'].replace('BANK STREET', 'BANK')
df_history['group_cols'] = df_history[['ml_Legal_BOROUGH', 'ml_Legal_STREET_NUMBER', 'ml_Legal_STREET_NAME2']].apply(lambda x: ','.join(x.dropna().astype(str)), axis=1)
print('[df_history]\n', df_history)

# (ml_Legal_BOROUGH == ml_Legal_BOROUGH) AND 
# (ml_Legal_STREET_NUMBER == ml_Legal_STREET_NUMBER) AND 
# (ml_Legal_STREET_NAME LIKE LIKE ml_Legal_STREET_NAME) AND 
# ((ml_Legal_BLOCK != ml_Legal_BLOCK) OR (ml_Legal_LOT != ml_Legal_LOT))

# df_groups = df_history.groupby(by=['ml_Legal_BOROUGH', 'ml_Legal_STREET_NUMBER', 'ml_Legal_STREET_NAME'])
df_groups = df_history.groupby(by='group_cols')
print('[df_groups]\n', len(df_groups))

df_output = None

for i, (group_index, df_group) in tqdm(enumerate(df_groups), total=len(df_groups)):
    if len(df_group) > 1:
        blocks = df_group['ml_Legal_BLOCK'].nunique()
        lots = df_group['ml_Legal_LOT'].nunique()
        if blocks > 1 or lots > 1:
            # print('[group_index]', group_index)
            # print('[df_group]\n', df_group)
            if df_output is None:
                df_output = df_group
            else:
                df_output = pd.concat([df_output, df_group], axis=0, ignore_index=True)

df_output = df_output.sort_values(by='group_cols', ignore_index=True)
df_output = df_output.drop(columns=['group_cols', 'ml_Legal_STREET_NAME2'])
print('[df_output]\n', df_output)
df_output.to_csv('output.csv', index=False)

