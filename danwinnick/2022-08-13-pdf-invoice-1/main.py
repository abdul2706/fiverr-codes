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

        # page-1 data
        self.date = datetime.fromtimestamp(int(self.data['date'])).strftime('%d %B %Y')
        self.customer = self.data['customer']
        self.description = self.data['description']
        self.invoice_number = self.data['invoice_number']
        self.invoice_amount = float(self.data['invoice_amount']) / 100
        self.customer_since = datetime.fromtimestamp(int(self.data['customer_since'])).strftime('%B %d, %Y')
        # print('[self.date]', self.date)
        # print('[self.customer_since]', self.customer_since)

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

json_path = 'invoice_template.json'
data = JSONData(json_path)
# legal (W x H) -> 216mm x 356mm
# A4 (W x H) -> 210mm x 297mm
pdf = FPDF('P', 'mm', 'A4')
WIDTH, HEIGHT = 210, 297
pdf.set_margin(0)
pdf.add_page()

# add header at the top
pdf.image('header.png', x=0, y=0, w=WIDTH)
# create text "Customer:"
create_text(14, 95, 'Customer:', color=50, size=11)
# create text for data.customer
create_text(14, 100, data.customer, style='B', size=11)
# create text "Customer Since:"
create_text(14, 105, 'Customer Since:', color=50, size=11)
# create text for data.customer_since
create_text(14, 110, data.customer_since, style='B', size=11)
# create text "Invoice Number:"
create_text(112, 117, 'Invoice Number:', size=14)
# create text for data.invoice_number
create_text(112, 124, data.invoice_number, style='B', size=14)

# add gradient background in the middle
pdf.image('gradient.png', x=10, y=142, w=190)
# create text "DATE"
create_text(14, 138, 'DATE', color=255, style='B', size=10)
# create text "ITEM DESCRIPTION"
create_text(80, 138, 'ITEM DESCRIPTION', color=255, style='B', size=10)
# create text "PRICE"
create_text(176, 138, 'PRICE', color=255, style='B', size=10)

# create text for data.date
create_text(12, 151, data.date, size=10)
# create text data.description
create_text(70, 151, data.description, size=10)
# create text data.invoice_amount
create_text(174, 151, f'$ {data.invoice_amount:.2f}', size=10)

# create gray background for total section
create_background(10, 167, 190, 14, (242, 242, 242))
# create text "TOTAL"
create_text(148, 165, 'TOTAL', size=11)
# create text data.invoice_amount
create_text(172, 165, f'$ {data.invoice_amount:.2f}', style='B', size=12)

# create text "THANK YOU FOR CHOOSING FEEZABILITY - INVOICE PAID"
create_background(10, 196, 133, 14, (242, 242, 242))
create_text(11, 193, 'THANK YOU FOR CHOOSING FEEZABILITY - INVOICE PAID', color=50, style='I', size=10.5)

# create text "PAID"
create_background(143, 196, 57, 14, (50, 52, 168))
create_text(160, 194, 'PAID', color=255, style='B', size=20)

# create text "Invoice Number:"
x, y, gap = 76, 72, 3.5
create_text(x, HEIGHT - y, 'Invoice paid by the credit card on file. IMPORTANT:', color=50, size=8)
create_text(x, HEIGHT - y + gap * 1, 'We do not store payment details. We use', size=8)
create_text(x, HEIGHT - y + gap * 2, 'Stripe.com in order to keep your payment', size=8)
create_text(x, HEIGHT - y + gap * 3, 'information secure using the highest standard of', size=8)
create_text(x, HEIGHT - y + gap * 4, 'privacy and security.', size=8)

# create text "Customer:"
create_text(76, HEIGHT - 35, 'THANK YOU!', style='BI', size=24)

# save pdf file
pdf.output('generated.pdf')
