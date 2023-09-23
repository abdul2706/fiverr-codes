import re
import fitz
import numpy as np
import pandas as pd

def extract_text_with_location(pdf_path):
    doc = fitz.open(pdf_path)
    text_dict = {'page': [], 'x1': [], 'y1': [], 'x2': [], 'y2': [], 'content': []}

    for page_idx, page in enumerate(doc):
        text_instances = page.get_text('blocks', sort=True)  # Get text instances as dictionaries
        for inst in text_instances:
            if inst[6] == 0:
                text_dict['page'].append(page_idx)
                text_dict['x1'].append(int(inst[0]))
                text_dict['y1'].append(int(inst[1]))
                text_dict['x2'].append(int(inst[2]))
                text_dict['y2'].append(int(inst[3]))
                text_dict['content'].append(inst[4])

    return pd.DataFrame(text_dict)

# Provide the path to your PDF file
data_extracted = {
    'bbl', 'borough', 'block', 'lot', 'owner_name_1', 'owner_name_2', 'property_address', 'mailing_address_1', 
    'mailing_address_2', 'mailing_address_3', 'mailing_address_4', 'mailing_address_5', 'mailing_address_6', 
    'mailing_address_7', 'billable_assessed_value', 'tax_rate', 'link_to_tax_bill', 'benefit_name', 'benefit_amount', 
    'tax_commission_reduction'
}
pdf_paths = [
    "client-files/1003781010.pdf",
    "client-files/3023481054.pdf",
    "client-files/1000181273.pdf"
]

cols_benefits = ['bbl', 'borough', 'block', 'lot', 'benefit_name', 'benefit_amount', 'tax_commission_reduction']
dict_benefits = {key: [] for key in cols_benefits}
cols_name_address = ['bbl', 'borough', 'block', 'lot', 'owner_name_1', 'owner_name_2', 'property_address', 'mailing_address_1', 'mailing_address_2', 'mailing_address_3', 'mailing_address_4', 'mailing_address_5', 'mailing_address_6', 'mailing_address_7']
dict_name_address = {key: [] for key in cols_name_address}
cols_values = ['bbl', 'borough', 'block', 'lot', 'billable_assessed_value', 'tax_rate', 'link_to_tax_bill']
dict_values = {key: [] for key in cols_values}

for pdf_path in pdf_paths:
    print('[pdf_path]', pdf_path)
    df_text_extracted = extract_text_with_location(pdf_path)
    # print('[df_text_extracted]\n', df_text_extracted)
    # print('[df_text_extracted]', df_text_extracted.dtypes)

    # combine mailing address row with block below it using nearest distance
    row_mailing_address = df_text_extracted.loc[df_text_extracted['content'] == 'Mailing address:\n']
    all_x1 = df_text_extracted['x1'].values
    all_y1 = df_text_extracted['y1'].values
    distances = np.sqrt(np.square(all_x1 - row_mailing_address['x1'].values) + np.square(all_y1 - row_mailing_address['y1'].values))
    row_near_mailing_address = df_text_extracted.loc[(distances <= 20) & (distances > 0)]
    df_text_extracted.loc[row_mailing_address.index, 'content'] = row_mailing_address['content'].values[0] + row_near_mailing_address['content'].values[0]
    df_text_extracted.loc[row_mailing_address.index, 'x2'] = max(row_mailing_address['x2'].values[0], row_near_mailing_address['x2'].values[0])
    df_text_extracted.loc[row_mailing_address.index, 'y2'] = max(row_mailing_address['y2'].values[0], row_near_mailing_address['y2'].values[0])
    df_text_extracted = df_text_extracted.drop(index=row_near_mailing_address.index)
    # print('[df_text_extracted]\n', df_text_extracted)

    # extract text from page1
    df_content_page1 = df_text_extracted.loc[df_text_extracted['page'] == 0]
    # print('[df_content_page1]\n', df_content_page1)
    row_owner_name = df_content_page1.loc[df_content_page1['content'].apply(lambda x: any(re.findall(r'owner name', x.lower())))]
    # print('[row_owner_name]\n', row_owner_name)
    row_mailing_address = df_content_page1.loc[df_content_page1['content'].apply(lambda x: any(re.findall(r'mailing address', x.lower())))]
    # print('[row_mailing_address]\n', row_mailing_address)
    row_property_address = df_content_page1.loc[df_content_page1['content'].apply(lambda x: any(re.findall(r'property address', x.lower())))]
    # print('[row_property_address]\n', row_property_address)
    row_borough_block_lot = df_content_page1.loc[df_content_page1['content'].apply(lambda x: any(re.findall(r'borough, block & lot', x.lower())))]
    # print('[row_borough_block_lot]\n', row_borough_block_lot)

    # extract owner_name
    owner_name = row_owner_name['content'].values[0].strip()
    owner_name = re.sub('owner name\W+?', '', owner_name, flags=re.IGNORECASE).strip().split('\n')
    owner_name = list(map(lambda x: str.strip(x), owner_name))
    print('[owner_name]', owner_name)
    # extract mailing_address
    mailing_address = row_mailing_address['content'].values[0].strip()
    mailing_address = re.sub('mailing address[\W]+?', '', mailing_address, flags=re.IGNORECASE).strip().split('\n')
    mailing_address = list(map(lambda x: str.strip(x), mailing_address))
    if len(mailing_address) == 3:
        if len(owner_name) == 2:
            mailing_address = [owner_name[1]] + mailing_address
        else:
            mailing_address = [''] + mailing_address
    mailing_address[0], mailing_address[1] = mailing_address[1], mailing_address[0]
    mailing_address_2 = re.search(r' (\w+. \d+$)', mailing_address[2])
    mailing_address.insert(3, mailing_address[2][mailing_address_2.start() + 1:])
    mailing_address[2] = mailing_address[2][:mailing_address_2.start()]
    mailing_address_4 = mailing_address[4].split(' ')
    print(mailing_address_4)
    if len(mailing_address_4) == 4:
        mailing_address_4[0] = mailing_address_4[0] + mailing_address_4[1]
        mailing_address_4.pop(1)
    print(mailing_address_4)
    mailing_address.pop(4)
    mailing_address.extend(mailing_address_4)
    print('[mailing_address]', mailing_address)
    # extract property_address
    property_address = row_property_address['content'].values[0].strip()
    property_address = re.sub('property address\W+?', '', property_address, flags=re.IGNORECASE).replace('\n', ' ').strip()
    print('[property_address]', property_address)
    # extract borough_block_lot
    borough_block_lot = row_borough_block_lot['content'].values[0].strip()
    borough_block_lot = re.sub('[^\d,]+', '', borough_block_lot, flags=re.IGNORECASE).strip()
    borough_block_lot = list(filter(lambda x: len(x) > 0, borough_block_lot.split(',')))
    print('[borough_block_lot]', borough_block_lot)
    bbl = ''.join(borough_block_lot)
    print('[bbl]', bbl)
    borough, block, lot = list(map(lambda x: int(x), borough_block_lot))
    print('[borough, block, lot]', borough, block, lot)

    # extract text from page2
    df_content_page2 = df_text_extracted.loc[df_text_extracted['page'] == 1]
    # print('[df_content_page2]\n', df_content_page2)
    row_current_tax_rate = df_content_page2.loc[df_content_page2['content'].apply(lambda x: any(re.findall(r'current tax rate', x.lower())))]
    # print('[row_current_tax_rate]\n', row_current_tax_rate)
    row_billable_assessed_value = df_content_page2.loc[df_content_page2['content'].apply(lambda x: any(re.findall(r'billable assessed value\n', x.lower())))]
    # print('[row_billable_assessed_value]\n', row_billable_assessed_value)
    row_taxable_value = df_content_page2.loc[df_content_page2['content'].apply(lambda x: any(re.findall(r'taxable value\n', x.lower())))]
    # print('[row_taxable_value]\n', row_taxable_value)
    row_tax_before_abatement = df_content_page2.loc[df_content_page2['content'].apply(lambda x: any(re.findall(r'tax before abatements and star\n', x.lower())))]
    # print('[row_tax_before_abatement]\n', row_tax_before_abatement)
    row_annual_property_tax = df_content_page2.loc[df_content_page2['content'].apply(lambda x: any(re.findall(r'annual property tax\n', x.lower())))]
    # print('[row_annual_property_tax]\n', row_annual_property_tax)

    tax_rate = row_current_tax_rate['content'].values[0].strip()
    tax_rate = re.findall('\d+[.]\d+[%]', tax_rate, flags=re.IGNORECASE)[0]
    print('[tax_rate]', tax_rate)
    billable_assessed_value = row_billable_assessed_value['content'].values[0].strip()
    billable_assessed_value = re.findall('\d+[,\d+]+', billable_assessed_value, flags=re.IGNORECASE)[0].replace(',', '')
    billable_assessed_value = float(billable_assessed_value)
    print('[billable_assessed_value]', repr(billable_assessed_value))

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
    # print('[df_page2_benefit_name]\n', df_page2_benefit_name)

    benefits_list = df_page2_benefit_name['content'].apply(str.strip).values.tolist()
    benefit_names, benefit_amounts = [], []
    print('[benefits_list]', benefits_list)
    for benefit_item in benefits_list:
        benefit_name, benefit_amount = benefit_item.split('\n')
        # print('[benefit_name, benefit_amount]', benefit_name, benefit_amount)
        benefit_amount = re.findall('[-+]\d+[.,\d+]+', benefit_amount, flags=re.IGNORECASE)[0].replace(',', '')
        benefit_amount = float(benefit_amount)
        print('[benefit_name, benefit_amount]', benefit_name, benefit_amount)
        benefit_names.append(benefit_name)
        benefit_amounts.append(benefit_amount)

    # TODO: couldn't find in pdf file yet
    tax_commission_reduction = 0

    print('-' * 100)

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
    dict_values['link_to_tax_bill'].append('https://www.url.com')

path_benefits = 'outputs/benefits.csv'
path_name_address = 'outputs/name_address.csv'
path_values = 'outputs/values.csv'
pd.DataFrame(dict_benefits).to_csv(path_benefits, index=False)
pd.DataFrame(dict_name_address).to_csv(path_name_address, index=False)
pd.DataFrame(dict_values).to_csv(path_values, index=False)
