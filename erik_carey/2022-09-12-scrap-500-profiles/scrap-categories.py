import os
import time
import pandas as pd

from tqdm import tqdm
from urllib import parse

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

def make_sure_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

# def expand_number(str_num):
#     postfix = str_num[-1] if str_num[-1] in ['K', 'M'] else ''
#     num = float(str_num[:-1]) if str_num[-1] in ['K', 'M'] else float(str_num)
#     if postfix == 'K':
#         num = num * 1e3
#     elif postfix == 'M':
#         num = num * 1e6
#     return int(num)

def scrap_grid_items(items):
    TAG2 = TAG + '[scrap_grid_items]'
    grid_items = {'title': [], 'url': []}

    for item in items:
        item_title = item.find_element(By.CSS_SELECTOR, '.grid-view-item__title')
        grid_items['title'].append(item_title.text.strip())
        grid_items['url'].append(item_title.get_attribute('href').strip())

    df_grid_items = pd.DataFrame(grid_items)
    print(TAG2, '[df_grid_items]', df_grid_items.shape)
    df_grid_items.to_csv('scrap_bonanzastrangi/.csv', index=False)

    return df_grid_items

def scrap_category_items2(driver, URL, category_path):
    TAG2 = TAG + '[scrap_category_items]'
    category_items_csv_path = os.path.join(category_path, 'category_items.csv')
    df_product_items_data = pd.read_csv(category_items_csv_path) if os.path.exists(category_items_csv_path) else None

    print(TAG2, '[URL]', URL)
    driver.get(URL)
    # accept cookies if container available
    accept_cookies_btn = driver.find_elements(By.CSS_SELECTOR, 'button#accept-cookies')
    if accept_cookies_btn: accept_cookies_btn[0].click()

    # load all items on the page
    load_more_div = driver.find_elements(By.CSS_SELECTOR, 'div.load-more_root__9MiHC')
    if len(load_more_div) > 0:
        load_more_paragraph = load_more_div[0].find_element(By.CSS_SELECTOR, 'span.load-more_paragraph__1mzoT')
        items_count = int(load_more_paragraph.text.split(' ')[0])
        total_count = int(load_more_paragraph.text.split(' ')[3])
        if df_product_items_data is not None and len(df_product_items_data) >= total_count:
            print(TAG2, category_items_csv_path, 'already scrapped')
            return

        page = int(total_count / 36 + 1)
        print(TAG2, '[items_count, total_count, page]', items_count, total_count, page)
        # load upto 1000 items
        URL2 = URL + '?page=' + str(page if page < 26 else 26)
        print(TAG2, '[URL2][1]', URL2)
        driver.get(URL2)
        # accept cookies if container available
        accept_cookies_btn = driver.find_elements(By.CSS_SELECTOR, 'button#accept-cookies')
        if accept_cookies_btn: accept_cookies_btn[0].click()

        # counter is for re-try
        counter = 0
        # re-try at least 10 times to make sure all items are loaded and load more button is no more available
        while True:
            # at last re-try check if the total product-cards on the page are as much as total_count written no botton of page
            if counter == 10:
                product_grid_items = driver.find_elements(By.CSS_SELECTOR, 'article.product-card-portrait_root__sZL4I.product-grid-lane_gridItem__eqh9g')
                print(TAG2, '[product_grid_items]', len(product_grid_items))
                if len(product_grid_items) == total_count:
                    # all items are loaded
                    break
                else:
                    # re-start counter
                    counter = 0
            
            print('[counter]', counter, driver.current_url)
            # find the load more container at bottom of page
            load_more_div = driver.find_elements(By.CSS_SELECTOR, 'div.load-more_root__9MiHC')
            if len(load_more_div) == 0:
                counter += 1
                continue
            # if load more container is available then find the load more button
            more_results_btn = load_more_div[0].find_elements(By.CSS_SELECTOR, 'button.button-default_root__2DBX1')
            if len(more_results_btn) == 0:
                counter += 1
                continue
            # if load more button is available then click that button and reset the counter
            counter = 0
            more_results_btn[0].click()
            print(TAG2, '[more_results_btn][clicked]')
            # time.sleep(0.5)

    # scrap and store items data from the page
    product_items_data = {'product_title': [], 'product_url': []}
    for product_grid_item in tqdm(product_grid_items, total=len(product_grid_items)):
        product_item = product_grid_item.find_element(By.CSS_SELECTOR, 'div.header_root__22e61')
        item_url_tag = product_item.find_element(By.CSS_SELECTOR, 'a.link_root__65rmW')
        product_title = item_url_tag.get_attribute('title')
        product_url = item_url_tag.get_attribute('href')
        if product_url:
            product_url = product_url.strip()
            product_title = product_title.strip() if product_title else ''
            product_items_data['product_title'].append(product_title)
            product_items_data['product_url'].append(product_url)
    
    df_product_items_data = pd.DataFrame(product_items_data)
    df_product_items_data.to_csv(category_items_csv_path, index=False)
    print('[df_product_items_data]')
    print(df_product_items_data)

def scrap_category_items(driver, URL, category_path):
    """
    Function to scrap info (Name, URL, Nutri-Score) of items related to given category URL.
    This function will only scrap if the CSV is not present in the category_path location.
    
    Parameters:
    -----------
        driver: Chrome Web Driver
        URL: URL of category specific web page
        category_path: local path of directory where scrapped data will be saved in CSV format
    
    Brief Algorithm:
    ----------------
        1. Iterate over nutri-scores (A, B, C, D, E)
        2. Go to URL (@param).
        3. Save nutri-scores checkboxes with total count of items in a dict.
        4. Click checkbox related to current iteration nutri-score.
        5. Load page completely, repeat these steps at least 5 times to make sure all items are loaded on page.
            1. Find load more button at bottom of page
            2. If button is available then click it
            3. If button is not available then increment counter.
            4. If counter reached value 5 then select all product items boxes
            5. If selected items are as many as total-count then goto step 6.
            6. else reset counter and repeat step 5.
            7. Also check for error container, if it's present on page then reload page for same nutri-score and go to step 1.
        6. After page is loaded scrap all the product items info.

    """
    TAG2 = TAG + '[scrap_category_items]'
    category_items_csv_path = os.path.join(category_path, 'category_items.csv')
    if os.path.exists(category_items_csv_path):
        print(TAG2, 'Already Scrapped:', category_items_csv_path)
        print(TAG2, 'If it is not scrapped completely then kindly delete the file to scrap again completely.')
        return

    product_items_data = {'product_title': [], 'product_url': [], 'nutri_score': []}
    nutri_scores = ['A', 'B', 'C', 'D', 'E']
    nutri_idx = 0

    while nutri_idx < len(nutri_scores):
        print(TAG2, '[URL]', URL)
        driver.get(URL)
        # accept cookies if container available
        accept_cookies_btn = driver.find_elements(By.CSS_SELECTOR, 'button#accept-cookies')
        if accept_cookies_btn: accept_cookies_btn[0].click()

        # get checkboxes
        show_more_nutri_scores = driver.find_elements(By.CSS_SELECTOR, 'a.show-more_root__22q8j')
        show_more_nutri_scores[0].click()
        checkboxes = driver.find_elements(By.CSS_SELECTOR, 'a.product-filter_root__3n45B')
        nutri_score_checkboxes = {}
        # print(TAG2, '[checkboxes]', len(checkboxes))
        for i in range(len(checkboxes)):
            aria_label_text = checkboxes[i].get_attribute('aria-label')
            # print(TAG2, '[aria_label_text]', aria_label_text)
            if 'Filter op Nutri-Score: Score' in aria_label_text:
                total_count = checkboxes[i].find_element(By.CSS_SELECTOR, 'span.product-filter_count__u72_w').text
                total_count = int(eval(total_count))
                print(TAG2, '[Filter op Nutri-Score: Score]', aria_label_text, total_count)
                nutri_score_checkboxes[aria_label_text[-1]] = [checkboxes[i], total_count]

        # some category pages don't have nutri-scores so ignore those pages
        print(TAG2, '[nutri_score_checkboxes]', nutri_score_checkboxes)
        if len(nutri_score_checkboxes) == 0:
            return

        # some cateogry pages don't have all five nutri-scores so update nutri_scores list
        if len(nutri_score_checkboxes.keys()) != 5:
            nutri_scores = sorted(list(nutri_score_checkboxes.keys()))
            print(TAG2, '[nutri_scores]', nutri_scores)

        nutri_score = nutri_scores[nutri_idx]
        checkbox, total_count = nutri_score_checkboxes[nutri_score]
        print(TAG2, '[checkbox][clicked][nutri_score, total_count]', nutri_score, total_count)
        checkbox.click()
        time.sleep(1)

        # load all items on the page
        counter = 0
        reload_page = False
        # re-try at least 5 times to make sure all items are loaded and load more button is no more available
        while True:
            print('[counter]', counter, driver.current_url)
            error_container = driver.find_elements(By.CSS_SELECTOR, 'div.error-message_root__1kiWb.search-error')
            if len(error_container) > 0:
                reload_page = True
                break
            # at last re-try check if the total product-cards on the page are as much as total_count written no botton of page
            if counter == 5:
                product_grid_items = driver.find_elements(By.CSS_SELECTOR, 'article.product-card-portrait_root__sZL4I.product-grid-lane_gridItem__eqh9g')
                print(TAG2, '[product_grid_items]', len(product_grid_items))
                if len(product_grid_items) == total_count:
                    # all items are loaded
                    break
                else:
                    # re-start counter
                    counter = 0

            # find the load more container at bottom of page
            load_more_div = driver.find_elements(By.CSS_SELECTOR, 'div.load-more_root__9MiHC')
            if len(load_more_div) == 0:
                counter += 1
                time.sleep(1)
                continue

            # if load more container is available then find the load more button
            more_results_btn = load_more_div[0].find_element(By.CSS_SELECTOR, 'button.button-default_root__2DBX1')
            # if load more button is available then click that button and reset the counter
            print(TAG2, '[more_results_btn][clicked]')
            more_results_btn.click()
            counter = 0
            time.sleep(0.5)
        
        if reload_page:
            continue

        # scrap and store items data from the page
        for product_grid_item in tqdm(product_grid_items, total=len(product_grid_items)):
            product_item = product_grid_item.find_element(By.CSS_SELECTOR, 'div.header_root__22e61')
            item_url_tag = product_item.find_element(By.CSS_SELECTOR, 'a.link_root__65rmW')
            product_title = item_url_tag.get_attribute('title')
            product_url = item_url_tag.get_attribute('href')
            if product_url:
                product_url = product_url.strip()
                product_title = product_title.strip() if product_title else ''
                product_items_data['product_title'].append(product_title)
                product_items_data['product_url'].append(product_url)
                product_items_data['nutri_score'].append(nutri_score)
        
        df_product_items_data = pd.DataFrame(product_items_data)
        df_product_items_data.to_csv(category_items_csv_path, index=False)
        print(TAG2, '[df_product_items_data]', df_product_items_data.shape)
        nutri_idx += 1
        print()

def scrap_product_categories(driver):
    TAG2 = TAG + '[scrap_product_categories]'
    if os.path.exists(product_categories_csv_path):
        df_product_categories = pd.read_csv(product_categories_csv_path)
        return df_product_categories
    
    PRODUCTS_URL = parse.urljoin(BASE_URL, 'producten')
    driver.get(PRODUCTS_URL)
    print(TAG2, '[PRODUCTS_URL]', PRODUCTS_URL)
    # accept cookies if container available
    accept_cookies_btn = driver.find_element(By.CSS_SELECTOR, 'button#accept-cookies')
    if accept_cookies_btn: accept_cookies_btn.click()
    
    product_categories = {'category_title': [], 'category_url': [], 'folder_name': []}
    product_categories_items = driver.find_elements(By.CSS_SELECTOR, 'a.taxonomy-card_imageLink__13VS1')
    print(TAG2, '[product_categories_items]', len(product_categories_items))
    for product_category_item in product_categories_items:
        category_title = product_category_item.get_attribute('title').strip()
        category_url = parse.urljoin(BASE_URL, product_category_item.get_attribute('href').strip())
        product_categories['category_title'].append(category_title)
        product_categories['category_url'].append(category_url)
        product_categories['folder_name'].append(category_url[category_url.rfind('/') + 1:])

    df_product_categories = pd.DataFrame(product_categories)
    df_product_categories.to_csv(product_categories_csv_path, index=False)
    print('[df_product_categories]')
    print(df_product_categories)
    
    return df_product_categories

print('[starts]')

TAG = '[main]'
BASE_URL = 'https://www.ah.nl/'
BASE_PATH_OUTPUT = make_sure_exists('output2')
product_categories_csv_path = os.path.join(BASE_PATH_OUTPUT, 'product-categories.csv')

service = Service('C:\\Python3912\\chromedriver.exe')
options = webdriver.ChromeOptions()
options.add_argument('headless')
# options.add_argument('start-maximized')
# options.add_argument('--start-fullscreen')
driver = webdriver.Chrome(service=service, options=options)
driver.set_window_rect(10, 10, 1300, 700)

df_product_categories = scrap_product_categories(driver)

for i, row in df_product_categories.iterrows():
    category_path = make_sure_exists(os.path.join(BASE_PATH_OUTPUT, f"{i + 1:02}-{row['folder_name']}"))
    print(TAG, i, '[category_path]', category_path)
    # try:
    scrap_category_items(driver, row['category_url'], category_path)
    # except Exception as e:
    #     print(TAG, '[e]', e)
    #     print(TAG, 'unable to scrap items for category:', row['category_title'])
    print()

driver.close()
print('[end]')
