import os
import json
import time
import threading
import pandas as pd

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def make_sure_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def scrap_products(nutri_score):
    global df_all_category_items
    TAG2 = TAG + '[scrap_products]' + f'[{nutri_score}]'
    service = Service('C:\\Python3912\\chromedriver.exe')
    options = webdriver.ChromeOptions()
    options.add_argument('headless')
    driver = webdriver.Chrome(service=service, options=options)
    driver_wait = WebDriverWait(driver, 2)
    product_table_wait_condition = EC.presence_of_element_located((By.CSS_SELECTOR, 'table.product-info-nutrition_table__1PDio > tbody'))

    # subset of df_all_category_items for particular nutri_score
    df_category_items_nutri_score = df_all_category_items.loc[df_all_category_items['nutri_score'] == nutri_score].reset_index(drop=True)
    category_paths = df_category_items_nutri_score['category_path'].value_counts().sort_values(ascending=True)

    # iterate over all categories related to the given nutri_score
    for category_path, products_count in category_paths.items():
        # select items belonging to current category_path
        df_category_items = df_category_items_nutri_score.loc[df_category_items_nutri_score['category_path'] == category_path].reset_index(drop=True)
        print(TAG2, '[items to scrap]', len(df_category_items), category_path)

        product_tables_list = []
        nutri_score_json_path = os.path.join(category_path, f'{nutri_score}.json')
        # load already scrapped data
        if os.path.exists(nutri_score_json_path):
            product_tables_list = json.load(open(nutri_score_json_path, 'r'))
        print(TAG2, '[items scrapped]', len(product_tables_list), category_path)
        # skip this category if already scrapped
        if len(product_tables_list) == products_count: continue
        # grab list of URLs already scrapped
        scrapped_urls = [prod['product_url'] for prod in product_tables_list]

        # iterate over all product items that have nutri_score (given in argument) and category related to category_path
        for i, product_info in df_category_items.iterrows():
            # skip tihs product if already scrapped
            if product_info['product_url'] in scrapped_urls: continue
            # goto product_url webpage
            driver.get(product_info['product_url'])
            # accept cookies if popup appears
            accept_cookies_btn = driver.find_elements(By.CSS_SELECTOR, 'button#accept-cookies')
            if accept_cookies_btn: accept_cookies_btn[0].click()

            # if table is available for current product item then scrap it
            product_table_dict = {'product_title': product_info['product_title'], 
                                  'product_url': product_info['product_url'], 
                                  'nutri_score': nutri_score}
            try:
                product_info_table = driver_wait.until(product_table_wait_condition)
                tr_tags = product_info_table.find_elements(By.TAG_NAME, 'tr')
                for tr_tag in tr_tags:
                    td_tags = tr_tag.find_elements(By.TAG_NAME, 'td')
                    key = td_tags[0].text
                    value = td_tags[1].text
                    product_table_dict[key] = value
            except Exception as e:
                # print(TAG2, '[e]', e)
                print(TAG2, '[table not found]', category_path, product_info['product_url'])
            finally:
                product_tables_list.append(product_table_dict)

            # dump data as json after every 10 iterations
            if (i + 1) % 10 == 0:
                json.dump(product_tables_list, open(nutri_score_json_path, 'w'), indent=4)

        # save all scrapped data for nutri_score (given in argument)
        json.dump(product_tables_list, open(nutri_score_json_path, 'w'), indent=4)

print('[starts]')

TAG = '[scrap-products-multi-thread]'
BASE_URL = 'https://www.ah.nl/'
BASE_PATH_OUTPUT = make_sure_exists('output2')
product_categories_csv_path = os.path.join(BASE_PATH_OUTPUT, 'product-categories.csv')

df_all_category_items = pd.DataFrame()
df_product_categories = pd.read_csv(product_categories_csv_path)
print('[df_product_categories]\n', df_product_categories)

for i, row in df_product_categories.iterrows():
    category_path = make_sure_exists(os.path.join(BASE_PATH_OUTPUT, f"{i+1:02}-{row['folder_name']}"))
    if not os.path.exists(category_path):
        raise Exception(f'Path ({category_path}) does not exists. Run script `scrap-categories.py` before running this script.')
    if len(os.listdir(category_path)) == 0:
        continue

    category_items_csv_path = os.path.join(category_path, 'category_items.csv')
    if not os.path.exists(category_items_csv_path):
        raise Exception(f'CSV file ({category_items_csv_path}) does not exists. Run script `scrap-categories.py` before running this script.')

    df_category_items = pd.read_csv(category_items_csv_path)
    df_category_items['category_path'] = category_path
    print(TAG, i, '[category_items_csv_path]', category_items_csv_path, df_category_items.shape)
    df_all_category_items = pd.concat((df_all_category_items, df_category_items), ignore_index=True)

print(TAG, '[df_all_category_items]')
print(df_all_category_items)
print(df_all_category_items.nutri_score.value_counts())
print(df_all_category_items.category_path.value_counts())
print(df_all_category_items[['category_path', 'nutri_score']].value_counts().sort_index().head(50))

thread_A = threading.Thread(target=scrap_products, args=('A',))
thread_B = threading.Thread(target=scrap_products, args=('B',))
thread_C = threading.Thread(target=scrap_products, args=('C',))
thread_D = threading.Thread(target=scrap_products, args=('D',))
thread_E = threading.Thread(target=scrap_products, args=('E',))

thread_A.start()
thread_B.start()
thread_C.start()
thread_D.start()
thread_E.start()

thread_A.join()
thread_B.join()
thread_C.join()
thread_D.join()
thread_E.join()

print('[end]')
