import re
import pandas as pd

df = pd.read_csv('names sample.txt')
df['name_cleaned'] = df['name']
print('[df]\n', df)
df['name_cleaned'] = df['name_cleaned'].apply(lambda x: re.sub(r'C/0', 'C/O', x))
df['name_cleaned'] = df['name_cleaned'].apply(lambda x: re.sub(r'([\d]+)TH', r'\1', x))
df['name_cleaned'] = df['name_cleaned'].apply(lambda x: re.sub(r'([\d]+)ST', r'\1', x))
df['name_cleaned'] = df['name_cleaned'].apply(lambda x: re.sub(r'[^\w\s]', ' ', x))
df['name_cleaned'] = df['name_cleaned'].apply(lambda x: re.sub(r'\s{2,}', ' ', x))
df['name_cleaned'] = df['name_cleaned'].apply(lambda x: re.sub(r'C O', r'CO', x))
df['name_cleaned'] = df['name_cleaned'].apply(lambda x: re.sub(r'c o', r'co', x))
df['name_cleaned'] = df['name_cleaned'].apply(lambda x: re.sub(r'L L C', r'LLC', x))
df['name_cleaned'] = df['name_cleaned'].apply(lambda x: re.sub(r'^\s*\b((.+?))\b\s*$', r'\1', x))

df['name_cleaned'] = df['name_cleaned'].apply(lambda x: re.sub(r'\b((?:\d+))\b N ', r'\1 NORTH ', x))
df['name_cleaned'] = df['name_cleaned'].apply(lambda x: re.sub(r'\b((?:\d+))\b S ', r'\1 SOUTH ', x))
df['name_cleaned'] = df['name_cleaned'].apply(lambda x: re.sub(r'\b((?:\d+))\b E ', r'\1 EAST ', x))
df['name_cleaned'] = df['name_cleaned'].apply(lambda x: re.sub(r'\b((?:\d+))\b W ', r'\1 WEST ', x))

df['name_cleaned'] = df['name_cleaned'].apply(lambda x: re.sub(r' ST ', r' STREET ', x))
df['name_cleaned'] = df['name_cleaned'].apply(lambda x: re.sub(r' AVE ', r' AVENUE ', x))
df['name_cleaned'] = df['name_cleaned'].apply(lambda x: re.sub(r' AV ', r' AVENUE ', x))

df.to_csv('names-sample-clean.csv', index=False)
df['name_cleaned'].to_csv('name_only2.csv', index=False)
