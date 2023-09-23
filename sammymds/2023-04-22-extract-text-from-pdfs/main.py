import cv2
import pytesseract
from PyPDF2 import PdfReader

def rows_containing_strings(value):
    return value.find('Signature') >= 0 or value.find('of') >= 0 or value.find('Grantee')

reader = PdfReader("page.pdf")
page = reader.pages[0]
count = 0
for image_file_object in page.images:
    image_path = str(count) + image_file_object.name
    print('[image_path]', image_path)
    with open(image_path, "wb") as fp:
        fp.write(image_file_object.data)
        count += 1

image = cv2.imread(image_path)
df = pytesseract.image_to_data(image, output_type='data.frame')
df = df.loc[~df['text'].isna()].reset_index(drop=True)
df2 = df.loc[df['text'].apply(rows_containing_strings) >= 0].reset_index(drop=True)
signatures_block_num = df2.loc[df2['text'].str.find('Signature') >= 0, 'block_num']
grantees_block_num = df2.loc[df2['text'].str.find('Grantee') >= 0, 'block_num']
print('[signatures_block_num]\n', signatures_block_num)
print('[grantees_block_num]\n', grantees_block_num)
common_block_nums = set(signatures_block_num).union(set(grantees_block_num))
print('[common_block_nums]\n', common_block_nums)
df2 = df2.loc[df2['block_num'].isin(common_block_nums)].reset_index(drop=True)
print('[df2]\n', df2)
df2['right'] = df2['left'] + df2['width']
df2['bottom'] = df2['top'] + df2['height']
bbox = [df2['left'].min(), df2['top'].min(), df2['right'].max(), df2['bottom'].max()]
bbox_width, bbox_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
large_bbox = [bbox[0] - bbox_width, bbox[1], bbox[2] + bbox_width, bbox[3] + 2*bbox_height]
# for i, row in df2.iterrows():
#     (x, y, w, h) = (row['left'], row['top'], row['width'], row['height'])
#     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)
cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 3)
cv2.rectangle(image, (large_bbox[0], large_bbox[1]), (large_bbox[2], large_bbox[3]), (255, 0, 0), 3)
# image = cv2.resize(image, dsize=None, fx=0.25, fy=0.25)
cv2.imwrite('out.jpg', image)



image = cv2.imread(image_path)
image = cv2.GaussianBlur(image, ksize=(25, 25), sigmaX=1.0)
# _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
image_crop = image[large_bbox[1]:large_bbox[3], large_bbox[0]:large_bbox[2]]
df3 = pytesseract.image_to_data(image_crop, output_type='data.frame')
print('[df3]\n', df3)
df3 = df3.loc[~df3['text'].isna()].reset_index(drop=True)
df3 = df3.loc[df3['text'].str.strip().str.len() > 0].reset_index(drop=True)
df3 = df3.loc[df3['text'].apply(rows_containing_strings) < 0].reset_index(drop=True)
df3 = df3.loc[df3['block_num'] == 1].reset_index(drop=True)
print('[df3]\n', df3)
for i, row in df3.iterrows():
    (x, y, w, h) = (row['left'], row['top'], row['width'], row['height'])
    cv2.rectangle(image_crop, (x, y), (x + w, y + h), (0, 0, 255), 3)
cv2.imwrite('out_crop.jpg', image_crop)
