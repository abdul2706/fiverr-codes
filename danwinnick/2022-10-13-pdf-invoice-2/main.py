import json
import requests
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from collections import OrderedDict

from fpdf import FPDF
from fpdf.enums import XPos, YPos

class JSONData():
    def __init__(self, json_path, debug=False):
        self.debug = debug
        self.data = json.load(open(json_path, 'r'), object_hook=OrderedDict)

        self.invoice_id = self.data['invoice_id']
        self.date = self.data['date']
        self.email = self.data['email']
        self.auth = self.data['auth']
        self.level = int(self.data['level'])
        self.total_charge = int(self.data['total_charge'])
        self.scheduling_discount_factor = int(self.data['scheduling_discount_factor'] * 100)
        # datetime.fromtimestamp(int(self.data['customer_since'])).strftime('%B %d, %Y')
        self.line_items = self.data['line_items']
        # for item in self.data['line_items']:
        #     item['total'] = item['qty'] * item['each']
        #     self.line_items.append(item)
        
        if self.debug: print('self.invoice_id:', self.invoice_id)
        if self.debug: print('self.date:', self.date)
        if self.debug: print('self.email:', self.email)
        if self.debug: print('self.auth:', self.auth)
        if self.debug: print('self.level:', self.level)
        if self.debug: print('self.total_charge:', self.total_charge)
        if self.debug: print('self.scheduling_discount_factor:', self.scheduling_discount_factor)
        if self.debug: print('self.line_items:', self.line_items)

def create_background(x, y, w, h, color):
    pdf.set_fill_color(*color)
    pdf.set_xy(x, y)
    pdf.cell(w, h, '', fill=1)

def create_text(x, y, text, color=0, style='', size=11):
    pdf.set_font('helvetica', style=style, size=size)
    pdf.set_text_color(color)
    pdf.set_xy(x, y)
    pdf.cell(pdf.get_string_width(text), 20, text, align='L')

##############################

BLUE_BG_COLOR = (9, 44, 218)
GRAY_BG_COLOR = (200, 200, 200)
json_path = 'monthly_invoice_template.json'
data = JSONData(json_path, debug=False)
# legal (W x H) -> 216mm x 356mm
# A4 (W x H) -> 210mm x 297mm
pdf = FPDF('P', 'mm', 'A4')
WIDTH, HEIGHT = 210, 297
pdf.set_margin(0)
pdf.add_page()

####### HEADER SECTION #######
# add invoice background image
pdf.image('invoice-bg.png', x=0, y=0, w=WIDTH, h=HEIGHT)
# create text "RainDrop"
create_text(14, 8, 'RainDrop', color=240, style='B', size=22)
# create text for billing email
create_text(22, 19, 'billing@raindroprdp.com', color=200, size=12)
# create text for website address
create_text(22, 26, 'www.raindroprdp.com', color=200, size=12)
# create text "INVOICE"
create_text(WIDTH - 60, 43, 'INVOICE', color=240, style='B', size=30)
# create text for invoice_id
create_text(14, 40, 'Invoice No:' + data.invoice_id, color=240, size=12)
# cover bottom section with white rectangle
create_background(0, 215, WIDTH, HEIGHT - 215, color=(255, 255, 255))

####### FIRST TABLE #######
# offset_y = 8
# create text "CUSTOMER"
create_text(20, 62, 'CUSTOMER', color=0, size=9)
# create text "TIER"
create_text(60, 62, 'TIER', color=0, size=9)
# create text "PAYMENT INFO"
create_text(100, 62, 'PAYMENT INFO', color=0, size=9)
# create text "CREATOR"
create_text(160, 62, 'CREATOR', color=0, size=9)

# # create text for data.date
# create_text(23, 76, data.date, color=0, style='B', size=11)
# # create text for data.email
# create_text(80, 76, data.email, color=0, style='B', size=11)
# # create text for data.level
# create_text(157, 76, str(data.level), color=0, style='B', size=11)

# create text for 'Test05'
create_text(22, 71, 'Test05', color=0, style='B', size=9)
# create text for 'Test05'
create_text(63, 71, str(data.level), color=0, style='B', size=9)
# create text for 'Account No: 2063'
create_text(98, 71, 'Account No: 2063', color=0, style='B', size=9)
# create text for 'RainDrop Automated Billing'
create_text(145, 71, 'RainDrop Automated Billing', color=0, style='B', size=9)

# create horizontal lines of table
x1, y1, x2, y2 = 12, 68, WIDTH - 12, 85
pdf.set_line_width(0.3)
pdf.line(x1, y1, x2, y1)
pdf.line(x1, (y1 + y2) / 2, x2, (y1 + y2) / 2)
pdf.line(x1, y2, x2, y2)
# create vertical lines of table
pdf.line(x1, y1, x1, y2)
pdf.line(x1+35, y1, x1+35, y2)
pdf.line(x1+70, y1, x1+70, y2)
pdf.line(x1+130, y1, x1+130, y2)
pdf.line(x2, y1, x2, y2)

####### SECOND TABLE #######
y1 = 94
for i, line_item in enumerate(data.line_items):
    # create gray background
    create_background(12, y1, WIDTH - 24, 9, color=GRAY_BG_COLOR)
    # create text for company_name
    create_text(15, y1 - 5, line_item['company_name'].upper(), color=50, style='B', size=10)
    # create text "Office Hours Discount:"
    create_text(WIDTH - 65, y1 - 5, f'Office Hours Discount: {data.scheduling_discount_factor}%', color=50, style='B', size=10)
    # create blue background
    y1 += 9
    create_background(12, y1, WIDTH - 24, 9, color=BLUE_BG_COLOR)
    # create text "ITEM DESCRIPTION"
    create_text(15, y1 - 5, 'ITEM DESCRIPTION', color=240, style='B', size=9)
    # create text "UNITS"
    create_text(98, y1 - 5, 'UNITS', color=240, style='B', size=9)
    # create text "PRICE PER ONE"
    create_text(120, y1 - 5, 'PRICE PER ONE', color=240, style='B', size=9)
    # create text "TOTAL PRICE"
    create_text(165, y1 - 5, 'TOTAL PRICE', color=240, style='B', size=9)

    # create table rows for each line_item
    y1 += 9
    for i, item_row in enumerate(line_item['company_usage']):
        create_background(12, y1, WIDTH - 24, 9, color=(255, 255, 255) if i % 2 == 0 else (240, 240, 240))
        # create text for item_row['description']
        create_text(12, y1 - 5, item_row['description'].capitalize(), size=9)
        # create text for item_row['description']
        create_text(100, y1 - 5, str(item_row['qty']), size=9)
        # create text for item_row['description']
        create_text(130, y1 - 5, f"${item_row['each']:.2f}", size=9)
        # create text for item_row['description']
        create_text(170, y1 - 5, f"${item_row['total']:.2f}", size=9)
        y1 += 9
    create_background(12, y1, WIDTH - 24, 9, color=(255, 255, 255) if i % 2 == 1 else (240, 240, 240))
    # create text "COMPANY TOTAL"
    create_text(WIDTH // 2 + 5, y1 - 5, 'COMPANY TOTAL:', style='B', size=9)
    # create text for line_item['company_total']
    create_text(WIDTH // 2 + 45, y1 - 5, f"${line_item['company_total']:.2f}", size=9)
    y1 += 9

# create gray background
create_background(12, y1, WIDTH - 24, 9, color=GRAY_BG_COLOR)
y1 += 9
# create_background(12, y1, WIDTH - 24, 9, color=(255, 255, 255))
create_text(12, y1 - 5, 'Account Overflow Credit', size=9)
create_text(100, y1 - 5, '1', size=9)
create_text(130, y1 - 5, '$205.00', size=9)
create_text(170, y1 - 5, '$205.00', size=9)

# create text "PATH IN FULL - AUTH:"
y1 += 9
create_background(12, y1, WIDTH - 24, 9, color=(240, 240, 240))
create_text(12, y1 - 5, 'PATH IN FULL - AUTH:', style='BI', size=9)
create_text(52, y1 - 5, data.auth, color=40, style='I', size=9)
create_text(130, y1 - 5, 'TOTAL', style='B', size=9)
create_text(170, y1 - 5, f"${data.total_charge:.2f}", style='B', size=9)

# create bottom section
# add image for "PAID" stamp
pdf.image('paid-stamp.jpg', x=WIDTH // 2 - 18, y=233, w=36, h=36)
text1 = 'Note: This is a                        , NO NEED to submit payment.'
w1 = pdf.get_string_width(text1)
x1 = (WIDTH - w1 + 10) // 2
y1 = 262
create_text(x1, y1, text1, size=9)
text1 = 'PAID invoice'
create_text(x1 + 22, y1, text1, size=9, style='B')

text1 = 'Thank you so much for choosing                   for your remote workstation solutions.'
w1 = pdf.get_string_width(text1)
x1 = (WIDTH - w1 + 10) // 2
y1 = 267
create_text(x1, y1, text1, size=9)
text1 = 'RainDrop'
create_text(x1 + 47, y1, text1, size=9, style='B')

# save pdf file
pdf.output('generated.pdf')
