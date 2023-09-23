import os
import re
import json
import pandas as pd

def process(value):
    TAG2 = TAG + '[process]'
    gram_conversions = {'m': 1e-3, 'µ': 1e-6}
    value = value.strip()

    # search if value corresponds to mass
    re_search = re.findall('^(\d+.?\d+).*g$', value)
    if re_search:
        if value[-2] in gram_conversions:
            value = float(re_search[0]) * gram_conversions[value[-2]]    
        return value

    # search if value corresponds to energy
    re_search = re.findall('^\d+ kJ \((.+)\)$', value)
    if re_search:
        value = re_search[0]
        return value

def json_to_csv(json_path, csv_path):
    TAG2 = TAG + '[json_to_csv]'
    string_columns = ['product_title', 'product_url', 'nutri_score']
    csv_columns_set = []
    products_json = json.load(open(json_path, 'r'))
    print(TAG2, '[products_json]', len(products_json))

    # extract unique column names
    for product_data in products_json:
        for key, value in product_data.items():
            if key not in csv_columns_set:
                csv_columns_set.append(key)

    # convert each product data to csv format
    products_data = {key: [] for key in csv_columns_set}
    for product_data in products_json:
        for key in csv_columns_set:
            if key in product_data:
                value = product_data[key]
                value = process(value) if key not in string_columns else value
                products_data[key].append(value)
            else:
                products_data[key].append(None)

    df_products_data = pd.DataFrame(products_data)
    df_products_data.to_csv(csv_path, index=False)
    print(TAG2, '[df_products_data]', df_products_data.shape)

print('[starts]')

TAG = '[scrap-products-multi-thread]'
BASE_PATH_OUTPUT = 'output2'
product_categories_csv_path = os.path.join(BASE_PATH_OUTPUT, 'product-categories.csv')

df_product_categories = pd.read_csv(product_categories_csv_path)

for i, row in df_product_categories.iterrows():
    category_path = os.path.join(BASE_PATH_OUTPUT, f"{i+1:02}-{row['folder_name']}")
    if not os.path.exists(category_path): continue
    if len(os.listdir(category_path)) == 0: continue
    for nutri_score in ['A', 'B', 'C', 'D', 'E']:
        nutri_score_json_path = os.path.join(category_path, f'{nutri_score}.json')
        nutri_score_csv_path = os.path.join(category_path, f'{nutri_score}.csv')
        if not os.path.exists(nutri_score_json_path): continue
        print(TAG, '[nutri_score_json_path]', nutri_score_json_path)
        print(TAG, '[nutri_score_csv_path]', nutri_score_csv_path)
        json_to_csv(nutri_score_json_path, nutri_score_csv_path)
        print()

print('[end]')
