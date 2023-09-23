import os
import time
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
# from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.support.ui import WebDriverWait

download_dir = os.path.abspath('./downloaded_pdfs')
print('[download_dir]', download_dir)
os.makedirs(download_dir, exist_ok=True)
service = Service(executable_path='/home/abdul2706/Downloads/chromedriver_linux64/chromedriver')
options = webdriver.ChromeOptions()
# options.add_argument('--test-type')
# options.add_argument('--disable-extensions')
options.add_experimental_option('prefs', {
    'download.default_directory': download_dir, # Change default directory for downloads
    'download.prompt_for_download': False, # To auto download the file
    'download.directory_upgrade': True,
    'plugins.always_open_pdf_externally': True # It will not show PDF directly in chrome
})
# options.add_argument('--headless')

# Below are a few sample links. We want to extract sections that contain "Grantee", "Buyer", or "Owners Approval"
# - https://a836-acris.nyc.gov/DS/DocumentSearch/DocumentImageView?doc_id=2012061500126001
# - https://a836-acris.nyc.gov/DS/DocumentSearch/DocumentImageView?doc_id=2003012702232001
# - https://a836-acris.nyc.gov/DS/DocumentSearch/DocumentImageView?doc_id=2023040500019001
# - https://a836-acris.nyc.gov/DS/DocumentSearch/DocumentImageView?doc_id=2023040500203001
# - https://a836-acris.nyc.gov/DS/DocumentSearch/DocumentImageView?doc_id=2023041000684002
# - https://a836-acris.nyc.gov/DS/DocumentSearch/DocumentImageView?doc_id=2018050100774001

urls_list = [
    'https://a836-acris.nyc.gov/DS/DocumentSearch/DocumentImageView?doc_id=2012061500126001',
    'https://a836-acris.nyc.gov/DS/DocumentSearch/DocumentImageView?doc_id=2003012702232001',
    'https://a836-acris.nyc.gov/DS/DocumentSearch/DocumentImageView?doc_id=2023040500019001',
    'https://a836-acris.nyc.gov/DS/DocumentSearch/DocumentImageView?doc_id=2023040500203001',
    'https://a836-acris.nyc.gov/DS/DocumentSearch/DocumentImageView?doc_id=2023041000684002',
    'https://a836-acris.nyc.gov/DS/DocumentSearch/DocumentImageView?doc_id=2018050100774001',
]

pdfs_downloaded = list(map(lambda x: x.split('&')[0], os.listdir(download_dir)))
print('[pdfs_downloaded]', pdfs_downloaded)

# for url in urls_list:
#     if url.split('?doc_id=')[1] in pdfs_downloaded:
#         print('pdf downloaded for url:', url)

# exit()

print('[start]')

driver = webdriver.Chrome(service=service, options=options)
driver.maximize_window()

for url in urls_list:
    if url.split('?doc_id=')[1] in pdfs_downloaded:
        print('pdf downloaded for url:', url)
        continue

    # url = 'https://a836-acris.nyc.gov/DS/DocumentSearch/DocumentImageView?doc_id=2012061500126001'
    print(f'getting url:', url)
    driver.get(url)
    iframe = driver.find_elements(by=By.TAG_NAME, value='iframe')[0]
    # print('[iframe]', iframe)

    iframe_url = iframe.get_attribute('src')
    driver.get(iframe_url)
    save_button = driver.find_elements(by=By.CSS_SELECTOR, value='td.vtm_buttonCell')[-1]
    # print('[save_button]', save_button)
    save_button.click()
    print('clicked save button')

    ok_button = driver.find_elements(by=By.CSS_SELECTOR, value='span.vtmBtn')[0]
    # print('[ok_button]', ok_button)
    ok_button.click()
    print('clicked ok button')

    time.sleep(3)
    while True:
        text_box = driver.find_elements(by=By.CSS_SELECTOR, value='div.vtm_msg')[0]
        if text_box.get_attribute('style') == 'visibility: hidden;':
            print('[text_box]', text_box.get_attribute('style'))
            break

    wait_time = 5 + random.randint(0, 5)
    print(f'waiting for {wait_time} seconds')
    time.sleep(wait_time)

print('[end]')
