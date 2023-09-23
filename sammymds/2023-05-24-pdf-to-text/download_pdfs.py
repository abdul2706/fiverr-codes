import os
import time
import random
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service

# path of directory where pdfs will be saved
download_dir = os.path.abspath('./downloaded_pdfs')
print('[download_dir]', download_dir)
os.makedirs(download_dir, exist_ok=True)

# path of chrome driver (download it according to your chrome browser version)
service = Service(executable_path='/home/abdul2706/programming/python/chromedriver_linux64/chromedriver')
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

# list down already downloaded pdfs
pdfs_downloaded = list(map(lambda x: x.split('&')[0], os.listdir(download_dir)))
print('[pdfs_downloaded]', pdfs_downloaded)

# common url in all pdf urls
base_url = 'https://a836-edms.nyc.gov/dctm-rest/repositories/dofedmspts/StatementSearch'
# set these two variables according to your requirement
stmtDate = '20230603'
stmtType = 'SOA'

# load csv file that contains the bbl values for all pdfs
df_bbls = pd.read_csv('client-files/input.csv')
# comment this line to download all pdf files, for testing it will download first 100 pdf files
bbls = df_bbls['bbl'].values[:200]
print('[bbls]\n', bbls)

driver = webdriver.Chrome(service=service, options=options)
driver.maximize_window()

# iterate over each bbl value
for bbl in bbls:
    # create src_path and dst_path
    # the downloaded pdf is saved as 'StatmentSearch.pdf', so need to rename it using dst_path
    src_path = os.path.join(download_dir, 'StatmentSearch.pdf')
    dst_path = os.path.join(download_dir, f'{bbl}.pdf')
    # if pdf already exists then skip this bbl value
    if os.path.exists(dst_path):
        print(f'pdf already exists:', dst_path)
        continue
    
    # create url for pdf file to download
    url = f'{base_url}?bbl={bbl}&stmtDate={stmtDate}&stmtType={stmtType}'
    print(f'getting url:', url)
    # load the pdf file, it will automatically download when the page loads completely
    driver.get(url)

    # wait between 5 to 9 seconds
    wait_time = 5 + random.randint(0, 5)
    print(f'waiting for {wait_time} seconds')
    time.sleep(wait_time)

    # if the pdf is downloaded completely then 'StatmentSearch.pdf' or src_path will exist in the `download_dir` and loop break
    # else keep waiting, which means that page hasn't loaded yet or file hasn't downloaded yet
    while not os.path.exists(src_path):
        time.sleep(1)
    # rename the downloaded file to match its name with bbl value
    os.rename(src_path, dst_path)

print('[end]')
