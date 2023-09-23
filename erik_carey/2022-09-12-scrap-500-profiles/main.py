import os
import time
import pandas as pd

from tqdm import tqdm

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

def make_sure_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def scrap_products(driver, category_path):
    TAG2 = TAG + '[scrap_products]'
    category_items_csv_path = os.path.join(category_path, 'category_items.csv')
    if not os.path.exists(category_items_csv_path):
        print(TAG2, 'CSV file does not exist:', category_items_csv_path)
        print(TAG2, 'Run `scrap-categories.py` file first')
        return

    df_product_items = pd.read_csv(category_items_csv_path)
    # df_nutri_scores_count = df_product_items['nutri_score'].value_counts()
    # print(TAG2, '[df_nutri_scores_count]\n', df_nutri_scores_count)
    nutri_scores = ['A', 'B', 'C', 'D', 'E']

    for nutri_score in nutri_scores:
        print(TAG2, '[nutri_score]', nutri_score)
        nutri_score_csv_path = os.path.join(category_path, f'{nutri_score}.csv')
        if os.path.exists(nutri_score_csv_path):
            print(TAG2, 'CSV file already exist:', nutri_score_csv_path)
            print(TAG2, 'If it is not scrapped completely then kindly delete the file to scrap again completely.')
            continue

        csv_columns_set = ['product_title', 'product_url', 'nutri_score']
        product_tables_list = []
        df_product_nutri_items = df_product_items.loc[df_product_items['nutri_score'] == nutri_score].reset_index(drop=True)
        # iterate over all product items related to nutri_score
        for i, row in tqdm(df_product_nutri_items.iterrows(), total=len(df_product_nutri_items)):
            # goto product_URL
            driver.get(row['product_url'])
            # accept cookies if container available
            accept_cookies_btn = driver.find_elements(By.CSS_SELECTOR, 'button#accept-cookies')
            if accept_cookies_btn: accept_cookies_btn[0].click()
            
            product_table_dict = {'product_title': row['product_title'], 'product_url': row['product_url'], 'nutri_score': nutri_score}
            product_info_table = driver.find_elements(By.CSS_SELECTOR, 'table.product-info-nutrition_table__1PDio > tbody')
            if len(product_info_table) > 0:
                product_info_table = product_info_table[0]
                tr_tags = product_info_table.find_elements(By.TAG_NAME, 'tr')
                for tr_tag in tr_tags:
                    td_tags = tr_tag.find_elements(By.TAG_NAME, 'td')
                    key = td_tags[0].text
                    value = td_tags[1].text
                    product_table_dict[key] = value
                    if key not in csv_columns_set:
                        csv_columns_set.append(key)

            product_tables_list.append(product_table_dict)
        
        print(TAG2, nutri_score, '[csv_columns_set]', len(csv_columns_set))
        print(csv_columns_set)
        products_data = {col: [] for col in csv_columns_set}
        for product_data in product_tables_list:
            for col in csv_columns_set:
                if col in product_data:
                    products_data[col].append(product_data[col])
                else:
                    products_data[col].append(None)
        
        df_products_data = pd.DataFrame(products_data)
        df_products_data.to_csv(nutri_score_csv_path, index=False)

print('[starts]')

TAG = f'[{os.path.basename(__file__)}]'
# SITES = [
#     'https://www.ihiremaintenanceandinstallation.com/',
#     'https://www.indeed.com/',
#     'https://ke.linkedin.com/',
#     'https://www.craigslist.org/',
#     'https://joinhandshake.com/',
# ]
keywords = ['HVAC Technician', 'Apartment Maintenance Technician', 'Maintenance Technician']
locations = ['Virginia, United States', 'North Carolina, United States']
data_keys = ['name', 'email', 'phone', 'resume', 'address']

service = Service('C:\\Python3912\\chromedriver.exe')
options = webdriver.ChromeOptions()
# options.add_argument('headless')
# options.add_argument('start-maximized')
# options.add_argument('--start-fullscreen')
driver = webdriver.Chrome(service=service, options=options)
driver.set_window_rect(10, 10, 1300, 700)

BASE_URL = r'https://www.linkedin.com/search/results/PEOPLE/?geoUrn=%5B%22101630962%22%2C%22103255397%22%5D&keywords=hvac%20technician&origin=FACETED_SEARCH&sid=SA8'
print(TAG, '[BASE_URL]', BASE_URL)

username = 'username'
password = 'password'

try:
    driver.get(BASE_URL)
    # driver.implicitly_wait(5)
    sign_in_button = driver.find_element(By.CSS_SELECTOR, 'a.main__sign-in-link')
    print(TAG, '[clicking signin]')
    sign_in_button.click()
    # driver.implicitly_wait(5)
    
    username_input = driver.find_element(By.CSS_SELECTOR, 'input#username')
    print(TAG, '[inputting username]', username)
    username_input.send_keys(username)
    
    password_input = driver.find_element(By.CSS_SELECTOR, 'input#password')
    print(TAG, '[inputting password]', password)
    password_input.send_keys(password)
    
    sign_in_button = driver.find_element(By.CSS_SELECTOR, 'button.btn__primary--large.from__button--floating')
    print(TAG, '[clicking signin]')
    sign_in_button.click()
except Exception as e:
    print(TAG, '[e]', e)

# BASE_PATH_OUTPUT = make_sure_exists('output')
# product_categories_csv_path = os.path.join(BASE_PATH_OUTPUT, 'product-categories.csv')
# df_product_categories = pd.read_csv(product_categories_csv_path)

# for i, row in df_product_categories.iterrows():
#     if i < 15: continue
#     category_path = make_sure_exists(os.path.join(BASE_PATH_OUTPUT, f"{i + 1:02}-{row['folder_name']}"))
#     print(TAG, i, '[category_path]', category_path)
#     # try:
#     scrap_products(driver, category_path)
#     # except Exception as e:
#     #     print(TAG, '[e]', e)
#     #     print(TAG, 'unable to scrap items for category:', row['category_title'])
#     print()

# driver.close()
print('[end]')
