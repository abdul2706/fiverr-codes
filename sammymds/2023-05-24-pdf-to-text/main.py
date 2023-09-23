import gc
import os
import re
import fitz
import numpy as np
import pandas as pd
from pprint import pprint

# this function takes data (dictionary) and filename (string) and saves it as CSV file
def save_as_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_dir, filename), index=False)

# this function takes pdf_path (string) as input, reads it and extracts the text from it
def extract_text_with_location(pdf_path):
    # open pdf file
    doc = fitz.open(pdf_path)
    # dictionary to store required items from pdf in structured form
    text_dict = {'page': [], 'x1': [], 'y1': [], 'x2': [], 'y2': [], 'content': []}

    # iterate page by page
    for page_idx, page in enumerate(doc):
        # read blocks from page
        text_instances = page.get_text('blocks', sort=True)  # Get text instances as dictionaries
        # iterate over all text blocks
        for inst in text_instances:
            if inst[6] == 0:
                text_dict['page'].append(page_idx)
                text_dict['x1'].append(int(inst[0]))
                text_dict['y1'].append(int(inst[1]))
                text_dict['x2'].append(int(inst[2]))
                text_dict['y2'].append(int(inst[3]))
                text_dict['content'].append(inst[4])

    return pd.DataFrame(text_dict)

def process_template1(df_text_extracted, dict_benefits, dict_name_address, dict_values):
    # used in printing and debugging
    TAG = '[process_template1]'

    # combine mailing address row with block below it using nearest distance
    row_mailing_address = df_text_extracted.loc[df_text_extracted['content'] == 'Mailing address:\n']
    all_x1 = df_text_extracted['x1'].values
    all_y1 = df_text_extracted['y1'].values
    distances = np.sqrt(np.square(all_x1 - row_mailing_address['x1'].values) + np.square(all_y1 - row_mailing_address['y1'].values))
    # select rows from df that have distance > 0 and <= 20 from row containing 'Mailing address' value
    row_near_mailing_address = df_text_extracted.loc[(distances <= 20) & (distances > 0)]
    # merge `row_near_mailing_address` with `row_mailing_address` and also update x2, y2 to new values
    df_text_extracted.loc[row_mailing_address.index, 'content'] = row_mailing_address['content'].values[0] + row_near_mailing_address['content'].values[0]
    df_text_extracted.loc[row_mailing_address.index, 'x2'] = max(row_mailing_address['x2'].values[0], row_near_mailing_address['x2'].values[0])
    df_text_extracted.loc[row_mailing_address.index, 'y2'] = max(row_mailing_address['y2'].values[0], row_near_mailing_address['y2'].values[0])
    df_text_extracted = df_text_extracted.drop(index=row_near_mailing_address.index)
    # print(TAG, '[df_text_extracted]\n', df_text_extracted)

    # extract text from page1
    df_content_page1 = df_text_extracted.loc[df_text_extracted['page'] == 0]
    # print(TAG, '[df_content_page1]\n', df_content_page1)
    
    # using regex select only rows containing 'owner name'
    row_owner_name = df_content_page1.loc[df_content_page1['content'].apply(lambda x: any(re.findall(r'owner name', x.lower())))]
    # print(TAG, '[row_owner_name]\n', row_owner_name)
    
    # using regex select only rows containing 'mailing address'
    row_mailing_address = df_content_page1.loc[df_content_page1['content'].apply(lambda x: any(re.findall(r'mailing address', x.lower())))]
    # print(TAG, '[row_mailing_address]\n', row_mailing_address)
    
    # using regex select only rows containing 'property address'
    row_property_address = df_content_page1.loc[df_content_page1['content'].apply(lambda x: any(re.findall(r'property address', x.lower())))]
    # print(TAG, '[row_property_address]\n', row_property_address)

    # using regex select only rows containing 'borough, block & lot'
    row_borough_block_lot = df_content_page1.loc[df_content_page1['content'].apply(lambda x: any(re.findall(r'borough, block & lot', x.lower())))]
    # print(TAG, '[row_borough_block_lot]\n', row_borough_block_lot)

    # extract owner_name value
    owner_name = row_owner_name['content'].values[0].strip()
    # using regex remove extra characters from owner_name value, then split it into list of names
    owner_name = re.sub('owner name\W+?', '', owner_name, flags=re.IGNORECASE).strip().split('\n')
    # clean each string item in the list
    owner_name = list(map(lambda x: str.strip(x), owner_name))
    # print(TAG, '[owner_name]', owner_name)

    # extract mailing_address
    mailing_address = row_mailing_address['content'].values[0].strip()
    # print(TAG, '[mailing_address]', mailing_address)
    # using regex remove extra characters from mailing_address value, then split it into list of names
    mailing_address = re.sub('mailing address[\W]+?', '', mailing_address, flags=re.IGNORECASE).strip().split('\n')
    # clean each string item in the list
    mailing_address = list(map(lambda x: str.strip(x), mailing_address))
    # print(TAG, '[mailing_address]', mailing_address)
    
    # if mailing_address containins 3 values then manually add fourth value in the beginning to make the next code work
    if len(mailing_address) == 3:
        mailing_address = [''] + mailing_address
    # based on pattern in pdfs, swap 0th and 1st value in mailing_address array
    mailing_address[0], mailing_address[1] = mailing_address[1], mailing_address[0]
    # search for required value at the 2nd index in mailing_address
    mailing_address_2 = re.search(r' (\w+[.] [\d\w]+$)', mailing_address[2])
    # print(TAG, '[mailing_address_2]', mailing_address_2)
    # if the search is not successful in previous line then manually insert empty value in `mailing_address`
    # else extract the string found and insert it in `mailing_address`
    if mailing_address_2 is None:
        mailing_address.insert(3, '')
    else:
        mailing_address.insert(3, mailing_address[2][mailing_address_2.start() + 1:])
        mailing_address[2] = mailing_address[2][:mailing_address_2.start()]
    # split 4th index value in mailing_address
    mailing_address_4 = mailing_address[4].split(' ')
    # print(TAG, '[mailing_address_4]', mailing_address_4)
    # the mailing_address_4 should consist of 3 values, if it contains 4 values then join 0th and 1st index value and save it back in 0th index
    if len(mailing_address_4) == 4:
        mailing_address_4[0] = mailing_address_4[0] + ' ' + mailing_address_4[1]
        mailing_address_4.pop(1)
    # print(TAG, '[mailing_address_4]', mailing_address_4)
    # remove the 4th index value, as only 3 values are needed
    mailing_address.pop(4)
    mailing_address.extend(mailing_address_4)
    # print(TAG, '[mailing_address]', mailing_address)

    # extract property_address
    property_address = row_property_address['content'].values[0].strip()
    # using regex remove extra characters from property_address value, then split it into list of names
    property_address = re.sub('property address\W+?', '', property_address, flags=re.IGNORECASE).replace('\n', ' ').strip()
    # print(TAG, '[property_address]', property_address)
    # extract `borough_block_lot`
    borough_block_lot = row_borough_block_lot['content'].values[0].strip()
    # remove unwanted characters from `borough_block_lot`
    borough_block_lot = re.sub('[^\d,]+', '', borough_block_lot, flags=re.IGNORECASE).strip()
    # futher filter `borough_block_lot` list
    borough_block_lot = list(filter(lambda x: len(x) > 0, borough_block_lot.split(',')))
    # print(TAG, '[borough_block_lot]', borough_block_lot)
    # join all values in `borough_block_lot` list to create bbl
    bbl = ''.join(borough_block_lot)
    # print(TAG, '[bbl]', bbl)
    # convert each value in `borough_block_lot` list to int and save in borough, block, lot
    borough, block, lot = list(map(lambda x: int(x), borough_block_lot))
    # print(TAG, '[borough, block, lot]', borough, block, lot)

    # extract text from page2 and page3 (only in some pdfs)
    df_content_page2 = df_text_extracted.loc[df_text_extracted['page'] > 0]
    # for those pdfs containing page3, their y1, y2 values start from 0 but we need them to start after page2 max y1, y2 values, so addition is performed
    page_2_max_y1 = df_content_page2[df_content_page2['page'] == 1].y1.max()
    page_2_max_y2 = df_content_page2[df_content_page2['page'] == 1].y2.max()
    # print(TAG, '[page_2_max_y1]', page_2_max_y1)
    df_content_page2.loc[df_content_page2['page'] == 2, 'y1'] += page_2_max_y1
    df_content_page2.loc[df_content_page2['page'] == 2, 'y2'] += page_2_max_y2
    # print(TAG, '[df_content_page2]\n', df_content_page2)

    # using regex select rows containing 'current tax rate'
    row_current_tax_rate = df_content_page2.loc[df_content_page2['content'].apply(lambda x: any(re.findall(r'current tax rate', x.lower())))]
    # print(TAG, '[row_current_tax_rate]\n', row_current_tax_rate)
    
    # using regex select rows containing 'billable assessed value\n'
    row_billable_assessed_value = df_content_page2.loc[df_content_page2['content'].apply(lambda x: any(re.findall(r'billable assessed value\n', x.lower())))]
    # print(TAG, '[row_billable_assessed_value]\n', row_billable_assessed_value)
    
    # using regex select rows containing 'taxable value\n'
    row_taxable_value = df_content_page2.loc[df_content_page2['content'].apply(lambda x: any(re.findall(r'taxable value\n', x.lower())))]
    # print(TAG, '[row_taxable_value]\n', row_taxable_value)
    
    # using regex select rows containing 'tax before abatements and star\n'
    row_tax_before_abatement = df_content_page2.loc[df_content_page2['content'].apply(lambda x: any(re.findall(r'tax before abatements and star\n', x.lower())))]
    # if above regex is unsuccessful then consider 'row_taxable_value' as 'row_tax_before_abatement'
    if len(row_tax_before_abatement) == 0:
        row_tax_before_abatement = row_taxable_value
    # print(TAG, '[row_tax_before_abatement]\n', row_tax_before_abatement)

    # using regex select rows containing 'annual property tax\n'
    row_annual_property_tax = df_content_page2.loc[df_content_page2['content'].apply(lambda x: any(re.findall(r'annual property tax\n', x.lower())))]
    # print(TAG, '[row_annual_property_tax]\n', row_annual_property_tax)

    # extract and clean tax_rate value
    tax_rate = row_current_tax_rate['content'].values[0].strip()
    # using regex find the actual fraction number
    tax_rate = re.findall('\d+[.]\d+[%]', tax_rate, flags=re.IGNORECASE)[0]
    # print(TAG, '[tax_rate]', tax_rate)

    # extract and clean `billable_assessed_value`
    billable_assessed_value = row_billable_assessed_value['content'].values[0].strip()
    # using regex find the actual fraction number
    billable_assessed_value = re.findall('\d+[,\d+]+', billable_assessed_value, flags=re.IGNORECASE)[0].replace(',', '')
    billable_assessed_value = float(billable_assessed_value)
    # print(TAG, '[billable_assessed_value]', repr(billable_assessed_value))

    # in pdf the pattern is that benefit names are between `row_billable_assessed_value` and `row_taxable_value`
    # or between `row_tax_before_abatement` and `row_annual_property_tax`
    df_page2_benefit_name = pd.concat([
        df_content_page2.loc[
            (df_content_page2['y1'] > row_billable_assessed_value['y1'].values[0]) & 
            (df_content_page2['y2'] < row_taxable_value['y2'].values[0])
        ],
        df_content_page2.loc[
            (df_content_page2['y1'] > row_tax_before_abatement['y1'].values[0]) & 
            (df_content_page2['y2'] < row_annual_property_tax['y2'].values[0])
        ]
    ], axis=0)
    # print(TAG, '[df_page2_benefit_name]\n', df_page2_benefit_name)

    # iterate over benefit names found, then process, clean and store them in list
    benefits_list = df_page2_benefit_name['content'].apply(str.strip).values.tolist()
    benefit_names, benefit_amounts = [], []
    # print(TAG, '[benefits_list]', benefits_list)
    for benefit_item in benefits_list:
        benefit_name, benefit_amount = benefit_item.split('\n')
        # print(TAG, '[benefit_name, benefit_amount]', benefit_name, benefit_amount)
        benefit_amount = re.findall('[-+]\d+[.,\d+]+', benefit_amount, flags=re.IGNORECASE)[0].replace(',', '')
        benefit_amount = float(benefit_amount)
        # print(TAG, '[benefit_name, benefit_amount]', benefit_name, benefit_amount)
        benefit_names.append(benefit_name)
        benefit_amounts.append(np.abs(benefit_amount))

    # TODO: couldn't find in pdf file yet
    tax_commission_reduction = 0

    # print(TAG, '-' * 100)

    # store all required extracted data in the global dictionaries
    for i in range(len(benefits_list)):
        dict_benefits['bbl'].append(bbl)
        dict_benefits['borough'].append(borough)
        dict_benefits['block'].append(block)
        dict_benefits['lot'].append(lot)
        dict_benefits['benefit_name'].append(benefit_names[i])
        dict_benefits['benefit_amount'].append(benefit_amounts[i])
        dict_benefits['tax_commission_reduction'].append(tax_commission_reduction)

    dict_name_address['bbl'].append(bbl)
    dict_name_address['borough'].append(borough)
    dict_name_address['block'].append(block)
    dict_name_address['lot'].append(lot)
    dict_name_address['owner_name_1'].append(owner_name[0])
    dict_name_address['owner_name_2'].append(owner_name[1] if len(owner_name) > 1 else '')
    dict_name_address['property_address'].append(property_address)
    dict_name_address['mailing_address_1'].append(mailing_address[0])
    dict_name_address['mailing_address_2'].append(mailing_address[1] if len(mailing_address) > 1 else '')
    dict_name_address['mailing_address_3'].append(mailing_address[2] if len(mailing_address) > 2 else '')
    dict_name_address['mailing_address_4'].append(mailing_address[3] if len(mailing_address) > 3 else '')
    dict_name_address['mailing_address_5'].append(mailing_address[4] if len(mailing_address) > 4 else '')
    dict_name_address['mailing_address_6'].append(mailing_address[5] if len(mailing_address) > 5 else '')
    dict_name_address['mailing_address_7'].append(mailing_address[6] if len(mailing_address) > 6 else '')
    
    dict_values['bbl'].append(bbl)
    dict_values['borough'].append(borough)
    dict_values['block'].append(block)
    dict_values['lot'].append(lot)
    dict_values['billable_assessed_value'].append(billable_assessed_value)
    dict_values['tax_rate'].append(tax_rate)
    dict_values['link_to_tax_bill'].append(f'{base_url}?bbl={bbl}&stmtDate={stmtDate}&stmtType={stmtType}')

def process_template2(df_text_extracted, dict_benefits, dict_name_address, dict_values):
    # global dict_benefits, dict_name_address, dict_values
    TAG = '[process_template2]'

    # extract text from page1
    df_content_page1 = df_text_extracted.loc[df_text_extracted['page'] == 0]
    # print(TAG, '[df_content_page1]\n', df_content_page1)
    row_owner_name = df_content_page1.loc[df_content_page1['content'].apply(lambda x: any(re.findall(r'owner name', x.lower())))]
    # print(TAG, '[row_owner_name]\n', repr(row_owner_name.content.values[0]))
    row_borough_block_lot = df_content_page1.loc[df_content_page1['content'].apply(lambda x: any(re.findall(r'(borough|block|lot)', x.lower())))]
    # print(TAG, '[row_borough_block_lot]\n', row_borough_block_lot)
    row_mailing_address = df_content_page1.loc[df_content_page1['content'].apply(lambda x: any(re.findall(r'[#]\d{15}[#]', x.lower())))]
    # print(TAG, '[row_mailing_address]\n', row_mailing_address)

    # extract owner_name
    owner_name = row_owner_name['content'].values[0].strip()
    # print(TAG, '[owner_name]', repr(owner_name))
    owner_name = re.sub('owner name\W+?', '', owner_name, flags=re.IGNORECASE).strip().split('\n')
    # print(TAG, '[owner_name]', repr(owner_name))
    owner_name = list(map(lambda x: str.strip(x), owner_name))
    # print(TAG, '[owner_name]', owner_name)
    
    # extract property_address
    if owner_name[1].lower().startswith('property'):
        property_address = owner_name[1]
        owner_name.pop(1)
    else:
        row_property_address = df_content_page1.loc[df_content_page1['content'].apply(lambda x: any(re.findall(r'property address', x.lower())))]
        # print(TAG, '[row_property_address]\n', repr(row_property_address.content.values[0]))
        property_address = row_property_address['content'].values[0].strip()
    property_address = re.sub('property address\W+?', '', property_address, flags=re.IGNORECASE).replace('\n', ' ').strip()
    # print(TAG, '[owner_name]', owner_name)
    # print(TAG, '[property_address]', property_address)

    # extract mailing_address
    mailing_address = row_mailing_address['content'].values[0].strip().split('\n')[1:]
    # mailing_address = re.sub('mailing address[\W]+?', '', mailing_address, flags=re.IGNORECASE).strip().split('\n')
    mailing_address = list(map(lambda x: str.strip(x), mailing_address))
    # print(TAG, '[mailing_address]', mailing_address)
    if len(mailing_address) == 3:
        mailing_address.insert(1, '')
    # print(TAG, '[mailing_address]', mailing_address)
    mailing_address_2 = re.search(r' (\w+[.]? [\d\w]{1,3}$)', mailing_address[2])
    # print(TAG, '[mailing_address_2]', mailing_address_2)
    if mailing_address_2 is not None and ' BOX ' in mailing_address_2.string:
        mailing_address_2 = None
    if mailing_address_2 is None:
        mailing_address.insert(3, '')
    else:
        mailing_address.insert(3, mailing_address[2][mailing_address_2.start() + 1:])
        mailing_address[2] = mailing_address[2][:mailing_address_2.start()]
    mailing_address_4 = mailing_address[4].split(' ')
    # print(TAG, '[mailing_address_4]', mailing_address_4)
    if len(mailing_address_4) == 4:
        mailing_address_4[0] = mailing_address_4[0] + ' ' + mailing_address_4[1]
        mailing_address_4.pop(1)
    # print(TAG, '[mailing_address_4]', mailing_address_4)
    mailing_address.pop(4)
    mailing_address.extend(mailing_address_4)
    # print(TAG, '[mailing_address]', mailing_address)

    # extract borough_block_lot
    borough_block_lot = row_borough_block_lot['content'].values[0].strip().split('\n')
    # print(TAG, '[borough_block_lot]', borough_block_lot)
    borough, block, lot = borough_block_lot[1], borough_block_lot[3], borough_block_lot[5]
    borough = re.findall(r'\d+', borough)[0]
    borough, block, lot = int(borough), int(block), int(lot)
    # print(TAG, '[borough, block, lot]', borough, block, lot)
    # num_zeros = 10 - len(borough + block + lot)
    # bbl = borough + '0' * num_zeros + block + lot
    bbl = f"{borough:<02}{block:04}{lot:04}"
    # print(TAG, '[bbl]', bbl)

    # extract text from page2
    df_content_page2 = df_text_extracted.loc[df_text_extracted['page'] == 1]
    # print(TAG, '[df_content_page2]\n', df_content_page2)
    row_billable_assessed_value = df_text_extracted.loc[df_text_extracted['content'].apply(lambda x: any(re.findall(r'billable assessed value', x.lower())))]
    # print(TAG, '[row_billable_assessed_value]\n', row_billable_assessed_value)
    row_tax_rate = df_text_extracted.loc[df_text_extracted['content'].apply(lambda x: any(re.findall(r'times the tax rate', x.lower())))]
    # print(TAG, '[row_tax_rate]\n', row_tax_rate)
    row_exemptions = df_text_extracted.loc[df_text_extracted['content'].apply(lambda x: any(re.findall(r'^exemptions:\n', x.lower())))]
    # print(TAG, '[row_exemptions]\n', row_exemptions)
    row_abatements = df_text_extracted.loc[df_text_extracted['content'].apply(lambda x: any(re.findall(r'^abatements:\n', x.lower())))]
    # print(TAG, '[row_abatements]\n', row_abatements)
    row_billing_activity = df_text_extracted.loc[df_text_extracted['content'].apply(lambda x: any(re.findall(r'activity for this billing period', x.lower())))]
    # print(TAG, '[row_billing_activity]\n', row_billing_activity)

    tax_rate = row_tax_rate['content'].values[0].strip().split('\n')[1].strip()
    tax_rate = re.findall('\d+[.]\d+[%]', tax_rate, flags=re.IGNORECASE)[0]
    # print(TAG, '[tax_rate]', tax_rate)
    billable_assessed_value = row_billable_assessed_value['content'].values[0].strip().split('\n')[1]
    billable_assessed_value = re.findall('\d+[,\d+]+', billable_assessed_value, flags=re.IGNORECASE)[0].replace(',', '')
    billable_assessed_value = float(billable_assessed_value)
    # print(TAG, '[billable_assessed_value]', repr(billable_assessed_value))

    if len(row_exemptions) > 0 or len(row_abatements) > 0:
        if len(row_exemptions) == 0:
            row_exemptions = row_abatements
        elif len(row_abatements) == 0:
            row_abatements = row_exemptions
        df_page2_benefit_name = pd.concat([
            df_content_page2.loc[
                (df_content_page2['y1'] > row_exemptions['y2'].values[0]) & 
                (df_content_page2['y2'] < row_abatements['y1'].values[0]) & 
                (df_content_page2['x2'] < row_tax_rate['x1'].values[0])
            ],
            df_content_page2.loc[
                (df_content_page2['y1'] > row_abatements['y2'].values[0]) & 
                (df_content_page2['y2'] < row_billing_activity['y1'].values[0]) & 
                (df_content_page2['x2'] < row_tax_rate['x1'].values[0])
            ]
        ], axis=0)
        # print(TAG, '[df_page2_benefit_name]\n', df_page2_benefit_name)

        benefits_list = df_page2_benefit_name['content'].apply(str.strip).values.tolist()
        benefit_names, benefit_amounts = [], []
        # print(TAG, '[benefits_list]', benefits_list)
        for benefit_item in benefits_list:
            benefit_item_list = benefit_item.split('\n')
            if len(benefit_item_list) == 2:
                benefit_name, benefit_amount = benefit_item_list[0], benefit_item_list[1]
            else:
                benefit_name, benefit_amount = benefit_item_list[0], benefit_item_list[-1]
            # print(TAG, '[benefit_name, benefit_amount]', benefit_name, benefit_amount)
            benefit_amount = re.findall('[-+]?[$]\d+[.,\d+]+', benefit_amount, flags=re.IGNORECASE)
            # print(TAG, '[benefit_name, benefit_amount]', benefit_name, benefit_amount)
            benefit_amount = benefit_amount[0] if benefit_amount and len(benefit_amount) > 0 else '0'
            benefit_amount = benefit_amount.replace(',', '').replace('$', '')
            benefit_amount = float(benefit_amount)
            # print(TAG, '[benefit_name, benefit_amount]', benefit_name, benefit_amount)
            benefit_names.append(benefit_name)
            benefit_amounts.append(np.abs(benefit_amount))
    else:
        benefits_list = []
        benefit_names = []
        benefit_amounts = []
    # print(TAG, '[benefits_list]', benefits_list)

    # TODO: couldn't find in pdf file yet
    tax_commission_reduction = 0

    # print(TAG, '-' * 100)


    # store all required extracted data in the global dictionaries
    for i in range(len(benefits_list)):
        dict_benefits['bbl'].append(bbl)
        dict_benefits['borough'].append(borough)
        dict_benefits['block'].append(block)
        dict_benefits['lot'].append(lot)
        dict_benefits['benefit_name'].append(benefit_names[i])
        dict_benefits['benefit_amount'].append(benefit_amounts[i])
        dict_benefits['tax_commission_reduction'].append(tax_commission_reduction)

    dict_name_address['bbl'].append(bbl)
    dict_name_address['borough'].append(borough)
    dict_name_address['block'].append(block)
    dict_name_address['lot'].append(lot)
    dict_name_address['owner_name_1'].append(owner_name[0])
    dict_name_address['owner_name_2'].append(owner_name[1] if len(owner_name) > 1 else '')
    dict_name_address['property_address'].append(property_address)
    dict_name_address['mailing_address_1'].append(mailing_address[0])
    dict_name_address['mailing_address_2'].append(mailing_address[1] if len(mailing_address) > 1 else '')
    dict_name_address['mailing_address_3'].append(mailing_address[2] if len(mailing_address) > 2 else '')
    dict_name_address['mailing_address_4'].append(mailing_address[3] if len(mailing_address) > 3 else '')
    dict_name_address['mailing_address_5'].append(mailing_address[4] if len(mailing_address) > 4 else '')
    dict_name_address['mailing_address_6'].append(mailing_address[5] if len(mailing_address) > 5 else '')
    dict_name_address['mailing_address_7'].append(mailing_address[6] if len(mailing_address) > 6 else '')
    
    dict_values['bbl'].append(bbl)
    dict_values['borough'].append(borough)
    dict_values['block'].append(block)
    dict_values['lot'].append(lot)
    dict_values['billable_assessed_value'].append(billable_assessed_value)
    dict_values['tax_rate'].append(tax_rate)
    dict_values['link_to_tax_bill'].append(f'{base_url}?bbl={bbl}&stmtDate={stmtDate}&stmtType={stmtType}')

# Provide the path to your PDF file
# data_extracted = {
#     'bbl', 'borough', 'block', 'lot', 'owner_name_1', 'owner_name_2', 'property_address', 'mailing_address_1', 
#     'mailing_address_2', 'mailing_address_3', 'mailing_address_4', 'mailing_address_5', 'mailing_address_6', 
#     'mailing_address_7', 'billable_assessed_value', 'tax_rate', 'link_to_tax_bill', 'benefit_name', 'benefit_amount', 
#     'tax_commission_reduction'
# }

# common url in all pdf urls
base_url = 'https://a836-edms.nyc.gov/dctm-rest/repositories/dofedmspts/StatementSearch'
# set these two variables according to your requirement
stmtDate = '20230603'
stmtType = 'SOA'

# path of folder containing downloaded pdf files
download_dir = os.path.abspath('./downloaded_pdfs')
# list down all files
pdf_filenames = os.listdir(download_dir)
# filter only pdf files
pdf_filenames = list(sorted(filter(lambda x: x.endswith('.pdf'), pdf_filenames), key=lambda x: int(x[:-4])))
# pdf_filenames = pdf_filenames[:5]
# print('[pdf_filenames]\n', pdf_filenames)

# initialize variables that will contain data extracted from pdfs
# dict_benefits1, dict_name_address1, dict_values1 are for pdf of one format
# dict_benefits2, dict_name_address2, dict_values2 are for pdf of second format
cols_benefits = ['bbl', 'borough', 'block', 'lot', 'benefit_name', 'benefit_amount', 'tax_commission_reduction']
dict_benefits1 = {key: [] for key in cols_benefits}
dict_benefits2 = {key: [] for key in cols_benefits}
cols_name_address = ['bbl', 'borough', 'block', 'lot', 'owner_name_1', 'owner_name_2', 'property_address', 'mailing_address_1', 'mailing_address_2', 'mailing_address_3', 'mailing_address_4', 'mailing_address_5', 'mailing_address_6', 'mailing_address_7']
dict_name_address1 = {key: [] for key in cols_name_address}
dict_name_address2 = {key: [] for key in cols_name_address}
cols_values = ['bbl', 'borough', 'block', 'lot', 'billable_assessed_value', 'tax_rate', 'link_to_tax_bill']
dict_values1 = {key: [] for key in cols_values}
dict_values2 = {key: [] for key in cols_values}

# iterate over all pdf files
for i, pdf_filename in enumerate(pdf_filenames):
    # if pdf_filename not in ['1000161075.pdf', '1002461010.pdf', '1002461019.pdf']: continue
    # if pdf_filename not in ['1002110015.pdf', '1002050013.pdf']: continue
    
    # extract text from pdf file
    pdf_path = os.path.join(download_dir, pdf_filename)
    df_text_extracted = extract_text_with_location(pdf_path)
    # print('[df_text_extracted]\n')
    # pprint(df_text_extracted.loc[:5, 'content'].values)
    # print('[df_text_extracted]\n', df_text_extracted.dtypes)

    # check if pdf is for first format or second format, then use relavent function to process and extract data from it
    if re.findall(r'property tax bill[\n]quarterly statement', df_text_extracted.loc[0, 'content'].lower()):
        print('[pdf_path][1]', pdf_path)
        process_template1(df_text_extracted, dict_benefits1, dict_name_address1, dict_values1)
    elif re.findall(r'property tax bill quarterly statement', df_text_extracted.loc[0, 'content'].lower()):
        print('[pdf_path][2]', pdf_path)
        process_template2(df_text_extracted, dict_benefits2, dict_name_address2, dict_values2)
    # clean memory for memory efficiency
    gc.collect()

    # save extracted data after every 100 pdf files and at the end of all files as well
    if i > 0 and i % 100 == 0 or i == len(pdf_filenames) - 1:
        # save all output csv files
        print('saving data...')
        output_dir = 'outputs'
        os.makedirs(output_dir, exist_ok=True)
        save_as_csv(dict_benefits1, 'benefits1.csv')
        save_as_csv(dict_name_address1, 'name_address1.csv')
        save_as_csv(dict_values1, 'values1.csv')
        save_as_csv(dict_benefits2, 'benefits2.csv')
        save_as_csv(dict_name_address2, 'name_address2.csv')
        save_as_csv(dict_values2, 'values2.csv')
