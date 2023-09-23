from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.expected_conditions import presence_of_element_located

import numpy as np
import pandas as pd

def get_school_info(URL):
    global driver
    # URL = 'https://www.eani.org.uk/parents/types-of-school/school-type/primary/st-marys-ps-killyleagh'
    driver.get(URL)

    top_info = driver.find_element(By.CSS_SELECTOR, 'div.text.col-xs-24.col-md-16').text
    print('[top_info]', top_info)
    top_info = top_info.split('\n')
    school_name = top_info[0]
    addresses = ''.join(top_info[1:-1]).split(',')[:4]
    addresses = [address.strip() for address in addresses]
    addresses = addresses + [''] * (4 - len(addresses))
    print('[addresses]', addresses)
    postcode = top_info[-1]

    contact_details = driver.find_element(By.CSS_SELECTOR, 'div.block.contact-details').text
    print('[contact_details]', contact_details)
    contact_details = contact_details.split('\n')
    designation = contact_details[1].split(', ')[0]
    fullname = contact_details[1].split(', ')[1]
    fullname = fullname.replace('Mr ', '').replace('Mrs ', '')
    firstname = fullname.split(' ')[0].strip()
    lastname = ' '.join(fullname.split(' ')[1:]).strip()
    email = contact_details[2].strip()
    contact = contact_details[3].strip()

    about_school = driver.find_elements(By.CSS_SELECTOR, 'div.primary.col-xs-24.col-md-13.no-padding div.block')[1].text
    print('[about_school]', about_school)
    about_school = about_school.split('\n')
    de_number = about_school[1].split(': ')[1].strip()
    school_type = about_school[2].split(': ')[1].strip()
    
    return [school_name, *addresses, postcode, designation, firstname, lastname, email, contact, de_number, school_type]

service = Service('/Users/abdulrehmankhan/Programs/chromedriver')
options = webdriver.ChromeOptions()
options.add_argument('headless')
driver = webdriver.Chrome(service=service) #, options=options)
page = 0
URL = f'https://www.eani.org.uk/school-search-2020?search_api_fulltext=&field_geodata%5Bdistance%5D%5Bfrom%5D=3.2&field_geodata%5Bvalue%5D=&page={page}'
COLUMNS = ['School Name', 'Address1', 'Address2', 'Address3', 'Address4', 'Postcode', 'Designation', 'Firstname', 'Lastname', 'Email', 'Contact', 'De number', 'School type']
# COLUMNS = ['School Name', 'Addresses', 'Postcode', 'Designation', 'Fullname', 'Email', 'Contact', 'De number', 'School type']
df_schools_info = pd.DataFrame(columns=COLUMNS)

driver.get(URL)
school_urls = driver.find_elements(By.CSS_SELECTOR, 'ul.search-list li.entry h2 a')
school_urls = [school_url.get_attribute('href') for school_url in school_urls]
print('[school_urls]', len(school_urls))
for i, school_url in enumerate(school_urls):
    # if i > 4: break
    print('[school_url]', i, school_url)
    school_info = get_school_info(school_url)
    df_school_info = pd.Series(school_info, index=COLUMNS)
    print('[school_info]', df_school_info)
    df_schools_info = df_schools_info.append(df_school_info, ignore_index=True)

df_schools_info.to_csv(f'df_schools_info_page{page}.csv', index=False)
